cmake_minimum_required(VERSION 4.0)
project(fluidx3d_vtk_visualization)

set(CMAKE_CXX_STANDARD 20)


set(THIRD_PARTY_LIB_DIR ${CMAKE_SOURCE_DIR}/3rdparty)



find_package(VTK
        COMPONENTS
        CommonCore
        CommonColor
        FiltersSources
        InteractionStyle
        RenderingCore
        RenderingOpenGL2
        RenderingVolume         # 新增: 体渲染通用
        RenderingVolumeOpenGL2  # 新增: OpenGL体渲染实现
        IOLegacy                # 新增: 读取旧版 .vtk 文件
        IOXML                   # 新增: 读取 .vti/.vtp 文件
        REQUIRED
)

if (NOT VTK_FOUND)
    message(FATAL_ERROR ${PROJECT_NAME}": Unable to find VTK.")
endif()



file(GLOB_RECURSE SOURCE "src/*.cpp" "src/*.h")
add_executable(${PROJECT_NAME} ${SOURCE})


# 链接 VTK 模块
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${VTK_LIBRARIES}
)

# VTK 9 的自动初始化，用于渲染相关模块
vtk_module_autoinit(
        TARGETS ${PROJECT_NAME}
        MODULES ${VTK_LIBRARIES}
)