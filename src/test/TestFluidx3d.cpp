#include "TestFluidx3d.h"
#include <cstdio>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iostream>
#include <regex> // 需要这个头文件进行数字提取
// VTK 核心
#include <vtkRendererCollection.h>
#include <vtkCamera.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCallbackCommand.h>
#include <vtkOutputWindow.h>
// 读取器
#include <vtkDataSetReader.h>
#include <vtkPointData.h>

// 等值面提取与显示
#include <vtkFlyingEdges3D.h> 
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>

// --- 新增的 GPU 体渲染相关头文件 ---
#include <vtkGPUVolumeRayCastMapper.h> // GPU 计算核心
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkContourValues.h>        // 用于控制等值面阈值
#include <vtkImageData.h>            // GPU Mapper 通常需要 ImageData
#include <vtkOpenGLRenderer.h> // 修复 SH 警告必须引用
// --- 必须添加的高级渲染头文件 ---
#include <vtkRenderStepsPass.h>
#include <vtkSSAOPass.h>
#include <vtkCameraPass.h>
#include <vtkOpenGLRenderer.h>

// --- 之前的 GPU 和 体渲染头文件 (如果没加也要加上) ---
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkContourValues.h>
#include <vtkImageData.h>

// --- 必须补充的头文件 ---
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkFlyingEdges3D.h> // 高速等值面提取
#include <vtkLookupTable.h>   // 颜色映射表

// --- 新增: 背景图与天空盒相关 ---
#include <vtkPNGReader.h>
#include <vtkTexture.h>
#include <vtkSkybox.h>
#include <vtkEquirectangularToCubeMapTexture.h> // 用于将全景图转为天空盒
#include <vtkOpenGLTexture.h> // 必须引用，因为转换需要 OpenGL 纹理


namespace fs = std::filesystem;

// 动画上下文结构体
struct AnimationContext {
    std::vector<std::string> filePaths;
    int currentFrame = 0;

    // 优化：将 Reader 放在上下文里复用，而不是每次都 New
    vtkSmartPointer<vtkDataSetReader> reader;
    vtkFlyingEdges3D* contourFilter = nullptr;
    vtkRenderWindow* renderWindow = nullptr;
    std::string fieldName = "data";
};

// 定时器回调
void TimerCallbackFunction(vtkObject* caller, unsigned long eventId, void* clientData, void* callData) {
    auto* context = static_cast<AnimationContext*>(clientData);
    if (context->filePaths.empty()) return;

    // 1. 获取当前帧路径
    std::string filePath = context->filePaths[context->currentFrame];

    // 2. 复用 reader 读取数据 (比每次 New 快)
    context->reader->SetFileName(filePath.c_str());
    context->reader->Update();

    vtkDataSet* output = context->reader->GetOutput();
    if (output) {
        output->GetPointData()->SetActiveScalars(context->fieldName.c_str());
        // 因为我们直接复用了 reader，而 Filter 已经连接了 reader 的 OutputPort，
        // 所以通常只需 Update reader，Filter 会自动感知管道变化。
        // 但为了保险，我们可以显式通知 Filter 输入变了（虽然在 Pipeline 机制里通常不需要）
        context->contourFilter->SetInputData(output);
    }

    // 3. 渲染
    context->renderWindow->Render();

    // 4. 循环帧数
    context->currentFrame = (context->currentFrame + 1) % context->filePaths.size();
}

void TestFluidx3d::visualizePhi()
{
    printf("Starting Optimized Isosurface Rendering...\n");

    // 1. 设置路径
    std::string folderPath = "D:/workspace/project1/c/fluidx3d_vtk_visualization/data/phi";

    std::vector<std::string> vtkFiles;
    if (!fs::exists(folderPath)) {
        std::cerr << "Error: Directory not found: " << folderPath << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".vtk") {
            vtkFiles.push_back(entry.path().string());
        }
    }
    std::sort(vtkFiles.begin(), vtkFiles.end());

    if (vtkFiles.empty()) {
        std::cerr << "Error: No .vtk files found." << std::endl;
        return;
    }
    printf("Found %zu files.\n", vtkFiles.size());

    // 2. 初始化 Reader (只创建一次)
    auto reader = vtkSmartPointer<vtkDataSetReader>::New();
    reader->SetFileName(vtkFiles[0].c_str());
    reader->Update();
    reader->GetOutput()->GetPointData()->SetActiveScalars("data");

    // 3. 创建等值面提取
    auto contourFilter = vtkSmartPointer<vtkFlyingEdges3D>::New();
    contourFilter->SetInputData(reader->GetOutput());

    // *** 关键设置：根据你的反馈，阈值设为 0.5 ***
    contourFilter->SetValue(0, 0.5);
    contourFilter->ComputeNormalsOn();

    // 4. 渲染管线
    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(contourFilter->GetOutputPort());
    mapper->ScalarVisibilityOff();

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // 设置水的材质
    vtkProperty* prop = actor->GetProperty();
    prop->SetColor(0.2, 0.6, 0.9);
    prop->SetSpecular(0.8);
    prop->SetSpecularPower(80.0);
    prop->SetAmbient(0.2);
    prop->SetDiffuse(0.7);

    // 5. 窗口设置
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(0.15, 0.15, 0.2);

    auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1280, 720);
    renderWindow->SetWindowName("FluidX3D Visualization (Interactive)");

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);

    // 6. 动画设置
    AnimationContext context;
    context.filePaths = vtkFiles;
    context.reader = reader; // 传入复用的 reader
    context.contourFilter = contourFilter;
    context.renderWindow = renderWindow;
    context.fieldName = "data";

    auto callback = vtkSmartPointer<vtkCallbackCommand>::New();
    callback->SetCallback(TimerCallbackFunction);
    callback->SetClientData(&context);

    interactor->Initialize();

    // 自动重置相机，防止一开始看不到物体
    renderer->ResetCamera();

    // *** 关键修改：将定时器改为 100ms (即每秒10帧) ***
    // 如果依然卡顿，可以尝试改为 200 或 300
    interactor->CreateRepeatingTimer(100);

    interactor->AddObserver(vtkCommand::TimerEvent, callback);

    renderWindow->Render();
    interactor->Start();
}


// GPU 动画专用的上下文结构体
struct GPUAnimationContext {
    std::vector<std::string> filePaths;
    int currentFrame = 0;

    vtkSmartPointer<vtkDataSetReader> reader;
    vtkSmartPointer<vtkGPUVolumeRayCastMapper> gpuMapper; // 控制 GPU Mapper
    vtkRenderWindow* renderWindow = nullptr;
    std::string fieldName = "data";
};

// GPU 动画定时器回调
void GPUTimerCallbackFunction(vtkObject* caller, unsigned long eventId, void* clientData, void* callData) {
    auto* context = static_cast<GPUAnimationContext*>(clientData);
    if (context->filePaths.empty()) return;

    // 1. 获取文件路径
    std::string filePath = context->filePaths[context->currentFrame];

    // 2. CPU 读取数据 (这是性能瓶颈，受限于硬盘速度)
    context->reader->SetFileName(filePath.c_str());
    context->reader->Update();

    vtkDataSet* output = context->reader->GetOutput();
    if (output) {
        output->GetPointData()->SetActiveScalars(context->fieldName.c_str());

        // 3. 将数据上传至 GPU
        // GPUVolumeRayCastMapper 需要 vtkImageData 格式
        // SetInputData 会触发纹理上传，之后的所有计算都在显卡中完成
        if (output->IsA("vtkImageData")) {
            context->gpuMapper->SetInputData(static_cast<vtkImageData*>(output));
        }
    }

    // 4. 触发渲染
    context->renderWindow->Render();

    // 5. 循环帧数
    context->currentFrame = (context->currentFrame + 1) % context->filePaths.size();
}

void TestFluidx3d::visualizePhiWithGPU()
{
    printf("Starting GPU-Accelerated Isosurface Rendering...\n");

    // ==========================================
    // 1. 扫描文件路径
    // ==========================================
    std::string folderPath = "D:/workspace/project1/c/fluidx3d_vtk_visualization/data/phi";
    std::vector<std::string> vtkFiles;

    if (!fs::exists(folderPath)) {
        std::cerr << "Error: Directory not found: " << folderPath << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".vtk") {
            vtkFiles.push_back(entry.path().string());
        }
    }
    std::sort(vtkFiles.begin(), vtkFiles.end());

    if (vtkFiles.empty()) {
        std::cerr << "Error: No .vtk files found." << std::endl;
        return;
    }
    printf("Found %zu files. GPU pipeline ready.\n", vtkFiles.size());

    // ==========================================
    // 2. 初始化 Reader
    // ==========================================
    auto reader = vtkSmartPointer<vtkDataSetReader>::New();
    reader->SetFileName(vtkFiles[0].c_str());
    reader->Update();
    reader->GetOutput()->GetPointData()->SetActiveScalars("data");

    // ==========================================
    // 3. 创建 GPU 渲染管线
    // ==========================================

    // 3.1 创建 GPU Mapper
    auto volumeMapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
    volumeMapper->SetInputData(reader->GetOutput());

    // *** 核心：设置混合模式为“等值面” ***
    // 这告诉 GPU 不要进行云雾状的体渲染，而是只渲染特定的数值表面
    volumeMapper->SetBlendModeToIsoSurface();

    // 3.2 设置 Volume Property (外观)
    auto volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    volumeProperty->ShadeOn();                      // 开启 Phong 光照，让流体有立体感
    volumeProperty->SetInterpolationTypeToLinear(); // 线性插值，表面更光滑

    // *** 关键：设置 GPU 等值面阈值 ***
    // 这里的 0.5 对应你之前测试出来的流体界面值
    volumeProperty->GetIsoSurfaceValues()->SetValue(0, 0.5);

    // 3.3 设置颜色 (Color Transfer Function)
    // 即使是等值面，GPU 也会根据这个函数决定表面的颜色
    auto colorFunc = vtkSmartPointer<vtkColorTransferFunction>::New();
    colorFunc->AddRGBPoint(0.0, 0.2, 0.6, 0.9); // 设定流体颜色 (水蓝色)
    colorFunc->AddRGBPoint(1.0, 0.2, 0.6, 0.9);

    // 3.4 设置不透明度
    // 必须设置为不透明 (1.0)，否则会有透视效果
    auto opacityFunc = vtkSmartPointer<vtkPiecewiseFunction>::New();
    opacityFunc->AddPoint(0.0, 1.0);
    opacityFunc->AddPoint(1.0, 1.0);

    volumeProperty->SetColor(colorFunc);
    volumeProperty->SetScalarOpacity(opacityFunc);

    // 3.5 创建 Volume Actor
    auto volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(volumeMapper);
    volume->SetProperty(volumeProperty);

    // ==========================================
    // 4. 渲染环境设置
    // ==========================================
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddVolume(volume);
    renderer->SetBackground(0.15, 0.15, 0.2); // 深灰蓝背景

    auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1280, 720);
    renderWindow->SetWindowName("FluidX3D GPU Visualization");

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);

    // ==========================================
    // 5. 动画循环配置
    // ==========================================
    GPUAnimationContext context;
    context.filePaths = vtkFiles;
    context.reader = reader;
    context.gpuMapper = volumeMapper; // 传入 GPU Mapper
    context.renderWindow = renderWindow;
    context.fieldName = "data";

    auto callback = vtkSmartPointer<vtkCallbackCommand>::New();
    callback->SetCallback(GPUTimerCallbackFunction); // 使用新的 GPU 回调
    callback->SetClientData(&context);

    interactor->Initialize();
    renderer->ResetCamera(); // 自动对焦

    // GPU 渲染速度很快，瓶颈主要在硬盘读取
    // 设置 33ms (约30FPS) 以获得更流畅的体验，如果卡顿可适当增加到 50-100
    interactor->CreateRepeatingTimer(33);
    interactor->AddObserver(vtkCommand::TimerEvent, callback);

    renderWindow->Render();
    interactor->Start();
}

void TestFluidx3d::visualizePhiOptimized()
{
    printf("Starting 3D-Game Style Liquid Rendering...\n");

    // ==========================================
    // 1. 资源路径与文件扫描
    // ==========================================
    std::string folderPath = "D:/workspace/project1/c/fluidx3d_vtk_visualization/data/phi";
    std::string skyboxPath = "D:/workspace/project/fluid_solid/FluidX3d/skybox/skybox8k.png";

    std::vector<std::string> vtkFiles;
    if (fs::exists(folderPath)) {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.path().extension() == ".vtk") vtkFiles.push_back(entry.path().string());
        }
    }
    std::sort(vtkFiles.begin(), vtkFiles.end()); // 文件名是补零的，直接排序即可

    if (vtkFiles.empty()) {
        std::cerr << "Error: No .vtk files found in " << folderPath << std::endl;
        return;
    }

    // ==========================================
    // 2. 初始化 Reader
    // ==========================================
    auto reader = vtkSmartPointer<vtkDataSetReader>::New();
    reader->SetFileName(vtkFiles[0].c_str());
    reader->Update();
    reader->GetOutput()->GetPointData()->SetActiveScalars("data");

    // ==========================================
    // 3. 准备渲染器
    // ==========================================
    auto renderer = vtkSmartPointer<vtkRenderer>::New();

    // 显式关闭球谐系数计算，消除警告
    if (auto glRenderer = vtkOpenGLRenderer::SafeDownCast(renderer)) {
        glRenderer->UseSphericalHarmonicsOff();
    }

    // ==========================================
    // 4. 加载 Skybox (游戏画质的关键：环境反射)
    // ==========================================
    if (fs::exists(skyboxPath)) {
        printf("Loading Skybox: %s ...\n", skyboxPath.c_str());
        auto texReader = vtkSmartPointer<vtkPNGReader>::New();
        texReader->SetFileName(skyboxPath.c_str());
        texReader->Update();

        auto texture = vtkSmartPointer<vtkTexture>::New();
        texture->SetInputConnection(texReader->GetOutputPort());
        texture->InterpolateOn();
        texture->MipmapOn();

        auto cubemap = vtkSmartPointer<vtkEquirectangularToCubeMapTexture>::New();
        cubemap->SetInputTexture(vtkOpenGLTexture::SafeDownCast(texture));
        cubemap->MipmapOn();
        cubemap->InterpolateOn();

        auto skyboxActor = vtkSmartPointer<vtkSkybox>::New();
        skyboxActor->SetTexture(cubemap);
        renderer->AddActor(skyboxActor);

        renderer->UseImageBasedLightingOn();
        renderer->SetEnvironmentTexture(cubemap);
        renderer->SetEnvironmentUp(0, 1, 0);
    } else {
        std::cerr << "Warning: Skybox not found! Water will look dull." << std::endl;
        renderer->SetBackground(0.1, 0.1, 0.15);
    }

    // ==========================================
    // 5. GPU 流体渲染管线 (Game Water Style)
    // ==========================================
    auto volumeMapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
    volumeMapper->SetInputData(reader->GetOutput());
    volumeMapper->SetBlendModeToIsoSurface(); // 保持等值面模式，表面最清晰

    // [核心调整 1] 光线穿透与散射
    // GlobalIlluminationReach: 0.9 (接近 1.0)，让光线穿透整个流体，实现"通透"感
    volumeMapper->SetGlobalIlluminationReach(0.9);
    // Scattering: 0.2 (较低)，减少内部浑浊感，让它像清水而不是牛奶
    volumeMapper->SetVolumetricScatteringBlending(0.2);

    auto volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    volumeProperty->ShadeOn();
    volumeProperty->SetInterpolationTypeToLinear();

    // [核心调整 2] 材质：模拟湿润反光表面
    volumeProperty->SetAmbient(0.2);         // 环境光适中
    volumeProperty->SetDiffuse(0.2);         // 漫反射很低（因为水是透明的，不怎么漫反射）
    volumeProperty->SetSpecular(3.0);        // 【极强】高光，游戏风格通常会夸大高光
    volumeProperty->SetSpecularPower(150.0); // 【极锐利】高光范围很小，像打磨过的玻璃

    volumeProperty->GetIsoSurfaceValues()->SetValue(0, 0.5);

    // [核心调整 3] 配色：游戏风格的“清澈蓝”
    auto colorFunc = vtkSmartPointer<vtkColorTransferFunction>::New();
    // 0.0 (深处): 深海蓝/墨色，提供体积感
    colorFunc->AddRGBPoint(0.0, 0.0, 0.1, 0.3);
    // 0.5 (主体): 鲜艳的青蓝色 (Cyan)，这是最典型的"游戏水"颜色
    colorFunc->AddRGBPoint(0.5, 0.0, 0.8, 0.9);
    // 1.0 (高光/薄处): 透亮的白青色
    colorFunc->AddRGBPoint(1.0, 0.8, 1.0, 1.0);

    // [核心调整 4] 透明度：极低 (Crystal Clear)
    auto opacityFunc = vtkSmartPointer<vtkPiecewiseFunction>::New();
    opacityFunc->AddPoint(0.0, 0.0);
    opacityFunc->AddPoint(0.45, 0.0); // 空气完全透明
    // 0.5 处只有 0.1 的不透明度。这意味着你要重叠 10 层水才能完全挡住背景。
    // 这种设置能最大程度展示天空盒的折射和反射。
    opacityFunc->AddPoint(0.5, 0.1);
    opacityFunc->AddPoint(1.0, 0.15);

    volumeProperty->SetColor(colorFunc);
    volumeProperty->SetScalarOpacity(opacityFunc);

    auto volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(volumeMapper);
    volume->SetProperty(volumeProperty);

    renderer->AddVolume(volume);

    // ==========================================
    // 6. 窗口与交互
    // ==========================================
    auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1280, 720);
    renderWindow->SetWindowName("FluidX3D Game-Style Water");

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);

    GPUAnimationContext context;
    context.filePaths = vtkFiles;
    context.reader = reader;
    context.gpuMapper = volumeMapper;
    context.renderWindow = renderWindow;
    context.fieldName = "data";

    auto callback = vtkSmartPointer<vtkCallbackCommand>::New();
    callback->SetCallback(GPUTimerCallbackFunction);
    callback->SetClientData(&context);

    interactor->Initialize();

    renderer->ResetCamera();
    renderer->GetActiveCamera()->Zoom(1.1);

    interactor->CreateRepeatingTimer(33); // 30 FPS
    interactor->AddObserver(vtkCommand::TimerEvent, callback);

    renderWindow->Render();
    interactor->Start();
}

// 专用于 Phi+U 可视化的上下文
struct PhiUContext {
    std::vector<std::string> phiFiles;
    std::vector<std::string> uFiles;
    int currentFrame = 0;

    vtkSmartPointer<vtkDataSetReader> phiReader;
    vtkSmartPointer<vtkDataSetReader> uReader;
    vtkSmartPointer<vtkFlyingEdges3D> isosurface;
    vtkRenderWindow* renderWindow = nullptr;
};

#include <regex> // 需要这个头文件进行数字提取

// --- 辅助函数：提取文件名中的数字进行排序 ---
bool NaturalSort(const std::string& a, const std::string& b) {
    // 提取字符串中的数字部分
    std::regex re("(\\d+)");
    std::smatch matchA, matchB;

    bool foundA = std::regex_search(a, matchA, re);
    bool foundB = std::regex_search(b, matchB, re);

    if (foundA && foundB) {
        // 将提取到的数字字符串转为整数进行比较
        long long numA = std::stoll(matchA.str());
        long long numB = std::stoll(matchB.str());
        if (numA != numB) {
            return numA < numB;
        }
    }
    // 如果没数字或数字相同，按默认字符串排
    return a < b;
}

// 动画回调：加入日志和防消失机制
void PhiUTimerCallback(vtkObject* caller, unsigned long eventId, void* clientData, void* callData) {
    auto* ctx = static_cast<PhiUContext*>(clientData);
    if (ctx->phiFiles.empty() || ctx->uFiles.empty()) return;

    // 1. 获取文件名
    std::string phiPath = ctx->phiFiles[ctx->currentFrame];
    std::string uPath = ctx->uFiles[ctx->currentFrame];

    // --- Debug: 打印当前帧，如果这里卡住或文件名乱跳，你就知道原因了 ---
    // printf("Rendering Frame %d: %s\n", ctx->currentFrame, std::filesystem::path(phiPath).filename().string().c_str());

    // 2. 读取数据
    ctx->phiReader->SetFileName(phiPath.c_str());
    ctx->phiReader->Update();
    ctx->uReader->SetFileName(uPath.c_str());
    ctx->uReader->Update();

    vtkDataSet* phiData = ctx->phiReader->GetOutput();
    vtkDataSet* uData = ctx->uReader->GetOutput();

    if (phiData && uData) {
        // 3. 计算速度
        vtkDataArray* velocityVectors = uData->GetPointData()->GetVectors();
        if (!velocityVectors) velocityVectors = uData->GetPointData()->GetArray("data");

        if (velocityVectors) {
            vtkIdType numPoints = phiData->GetNumberOfPoints();
            auto magArray = vtkSmartPointer<vtkFloatArray>::New();
            magArray->SetName("VelocityMagnitude");
            magArray->SetNumberOfComponents(1);
            magArray->SetNumberOfTuples(numPoints);

            float* vecPtr = static_cast<float*>(velocityVectors->GetVoidPointer(0));
            float* magPtr = static_cast<float*>(magArray->GetVoidPointer(0));

            for (vtkIdType i = 0; i < numPoints; ++i) {
                float vx = vecPtr[3*i];
                float vy = vecPtr[3*i+1];
                float vz = vecPtr[3*i+2];
                magPtr[i] = std::sqrt(vx*vx + vy*vy + vz*vz);
            }
            phiData->GetPointData()->AddArray(magArray);
            phiData->GetPointData()->SetActiveScalars("VelocityMagnitude");
        }

        // 4. 更新表面
        ctx->isosurface->SetInputData(phiData);
    }

    // --- 关键修复: 每一帧都重置相机的“裁剪范围” ---
    // 这不会改变相机位置，只会确保远近物体都能被看见，防止“消失”
    ctx->renderWindow->GetRenderers()->GetFirstRenderer()->ResetCameraClippingRange();

    // 5. 渲染
    ctx->renderWindow->Render();
    ctx->currentFrame = (ctx->currentFrame + 1) % std::min(ctx->phiFiles.size(), ctx->uFiles.size());
}

void TestFluidx3d::visualizePhiAndU()
{
    printf("Starting Sorted Visualization: Phi + U...\n");

    vtkOutputWindow::SetGlobalWarningDisplay(0);
    std::string phiFolder = "D:/workspace/project1/c/fluidx3d_vtk_visualization/data/phi";
    std::string uFolder   = "D:/workspace/project1/c/fluidx3d_vtk_visualization/data/u";
    std::string skyboxPath = "D:/workspace/project/fluid_solid/FluidX3d/skybox/skybox8k.png";

    // --- 1. 扫描文件 ---
    std::vector<std::string> phiFiles, uFiles;
    if (fs::exists(phiFolder)) {
        for (const auto& entry : fs::directory_iterator(phiFolder))
            if (entry.path().extension() == ".vtk") phiFiles.push_back(entry.path().string());
    }
    if (fs::exists(uFolder)) {
        for (const auto& entry : fs::directory_iterator(uFolder))
            if (entry.path().extension() == ".vtk") uFiles.push_back(entry.path().string());
    }

    // *** 关键修复：使用自然数字排序 ***
    // 这样 phi_2.vtk 就会排在 phi_10.vtk 前面，而不是后面
    std::sort(phiFiles.begin(), phiFiles.end(), NaturalSort);
    std::sort(uFiles.begin(), uFiles.end(), NaturalSort);

    if (phiFiles.empty() || uFiles.empty()) {
        std::cerr << "Error: No files found!" << std::endl;
        return;
    }
    printf("Found %zu phi files. First: %s\n", phiFiles.size(), fs::path(phiFiles[0]).filename().string().c_str());

    // --- 2. 初始化管线 ---
    auto phiReader = vtkSmartPointer<vtkDataSetReader>::New();
    auto uReader = vtkSmartPointer<vtkDataSetReader>::New();

    // 预读第一帧
    phiReader->SetFileName(phiFiles[0].c_str());
    phiReader->Update();
    uReader->SetFileName(uFiles[0].c_str());
    uReader->Update();

    // --- 3. 等值面 ---
    auto contour = vtkSmartPointer<vtkFlyingEdges3D>::New();
    contour->SetInputData(phiReader->GetOutput());
    contour->SetValue(0, 0.5);
    contour->ComputeNormalsOn();

    // --- 4. 颜色与材质 ---
    auto lut = vtkSmartPointer<vtkLookupTable>::New();
    lut->SetHueRange(0.66, 0.5);
    lut->SetSaturationRange(1.0, 0.2);
    lut->SetValueRange(0.3, 1.0);
    lut->SetTableRange(0.0, 0.15);
    lut->Build();

    auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(contour->GetOutputPort());
    mapper->SetLookupTable(lut);
    mapper->SetScalarRange(0.0, 0.15);
    mapper->ScalarVisibilityOn();

    auto actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkProperty* prop = actor->GetProperty();
    prop->SetInterpolationToPBR();
    prop->SetMetallic(0.1);
    prop->SetRoughness(0.05);

    // --- 5. 渲染器 ---
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    if (auto glRenderer = vtkOpenGLRenderer::SafeDownCast(renderer)) {
        glRenderer->UseSphericalHarmonicsOff();
    }

    if (fs::exists(skyboxPath)) {
        auto texReader = vtkSmartPointer<vtkPNGReader>::New();
        texReader->SetFileName(skyboxPath.c_str());
        texReader->Update();

        auto texture = vtkSmartPointer<vtkTexture>::New();
        texture->SetInputConnection(texReader->GetOutputPort());
        texture->InterpolateOn();
        texture->MipmapOn();

        auto cubemap = vtkSmartPointer<vtkEquirectangularToCubeMapTexture>::New();
        cubemap->SetInputTexture(vtkOpenGLTexture::SafeDownCast(texture));
        cubemap->MipmapOn();
        cubemap->InterpolateOn();

        renderer->UseImageBasedLightingOn();
        renderer->SetEnvironmentTexture(cubemap);
        renderer->SetEnvironmentUp(0, 1, 0);

        auto skybox = vtkSmartPointer<vtkSkybox>::New();
        skybox->SetTexture(cubemap);
        renderer->AddActor(skybox);
    } else {
        renderer->SetBackground(0.1, 0.1, 0.15);
    }

    renderer->AddActor(actor);

    // --- 6. 窗口 ---
    auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1280, 720);
    renderWindow->SetWindowName("Sorted PBR Fluid Visualization");

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);

    PhiUContext context;
    context.phiFiles = phiFiles;
    context.uFiles = uFiles;
    context.phiReader = phiReader;
    context.uReader = uReader;
    context.isosurface = contour;
    context.renderWindow = renderWindow;

    auto callback = vtkSmartPointer<vtkCallbackCommand>::New();
    callback->SetCallback(PhiUTimerCallback);
    callback->SetClientData(&context);

    interactor->Initialize();

    // 初始重置相机
    renderer->ResetCamera();
    renderer->GetActiveCamera()->Zoom(1.2);

    interactor->CreateRepeatingTimer(33);
    interactor->AddObserver(vtkCommand::TimerEvent, callback);

    renderWindow->Render();
    interactor->Start();
}
