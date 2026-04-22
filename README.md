# YangJiwei
An Mechanical Engineering Student
keep on learning

## COMSOL 调用 MATLAB `.m` 文件（LiveLink for MATLAB）

### 1) 环境确认
- 已安装并授权 COMSOL 与 **LiveLink for MATLAB**
- MATLAB 版本与 COMSOL 版本兼容
- 目标 `.m` 文件可在 MATLAB 中单独运行

### 2) 在 COMSOL 中添加 MATLAB 函数
- 打开 **Global Definitions / Component Definitions → Functions**
- 新建 **MATLAB Function**（不同版本名称可能略有差异）
- 配置函数名、输入参数、输出参数
- 将 `.m` 文件所在目录加入 MATLAB path

### 3) 在模型表达式中调用
- 在材料参数、边界条件、源项等位置直接调用该函数
- 保证输入输出维度、单位与 COMSOL 设置一致

### 4) 联调与求解
- 先用简单输入测试函数调用是否成功
- 再进行完整求解，观察收敛与计算性能

### 5) 常见问题
- **找不到函数**：通常是 MATLAB path 未正确配置
- **计算很慢**：函数调用过于频繁，可考虑减少调用或做插值/缓存
- **维度错误**：函数返回值形状与 COMSOL 期望不一致
