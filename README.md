1. 开发工具Clion，平台Windows 
2. 把VTK编译出来之后，配置到环境变量  
3. 运行本代码需要加上Cmake option,路径根据实际情况修改
```
-DVTK_DIR="path/vtk/lib/cmake/vtk-9.5"
```
4. 代码中的phi文件夹路径和背景图片路径根据实际情况修改；如果写相对路径，以编译之后的exe文件来确定位置