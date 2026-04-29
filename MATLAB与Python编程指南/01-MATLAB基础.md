# 第一章 MATLAB 基础与矩阵运算

## 1.1 MATLAB 快速入门

### 基本运算

```matlab
% 变量赋值（不需要声明类型）
a = 5;
b = 3.14;
c = a + b;        % 加减乘除
d = a^2;          % 幂运算
e = sqrt(16);     % 开方 → 4

% 常用常量
pi          % 3.14159...
exp(1)      % 自然常数 e = 2.71828...
inf         % 无穷大
```

### 向量与矩阵

```matlab
% 行向量
v = [1, 2, 3, 4, 5];

% 列向量
w = [1; 2; 3; 4; 5];

% 矩阵
A = [1 2 3;
     4 5 6;
     7 8 9];

% 快速生成
x = 0:0.1:10;        % 0到10，步长0.1
x = linspace(0, 10, 100);  % 0到10，100个点
I = eye(3);           % 3×3 单位矩阵
Z = zeros(3, 3);      % 3×3 零矩阵
O = ones(3, 3);       % 3×3 全1矩阵
```

---

## 1.2 矩阵运算（力学核心！）

```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];

% 矩阵加减
C = A + B;

% 矩阵乘法
C = A * B;          % 矩阵乘法
C = A .* B;         % 逐元素乘法（点乘）

% 转置
AT = A';            % 转置（共轭转置）
AT = A.';           % 转置（不共轭）

% 逆矩阵
Ainv = inv(A);      % A 的逆矩阵
% 更推荐：解方程 A*x = b
x = A \ b;          % 左除（等价于 inv(A)*b，但更高效）

% 行列式
d = det(A);

% 特征值
[V, D] = eig(A);   % V: 特征向量, D: 特征值对角阵

% 矩阵大小
[m, n] = size(A);   % m行 n列
```

### 力学中的矩阵运算示例

```matlab
% 求解 K * u = F（有限元基本方程）
K = [300 -100; -100 200];    % 刚度矩阵
F = [0; 500];                % 载荷向量

u = K \ F;                   % 求解位移！
% u = [0.5; 2.75]

% 应力计算
E = 200e3;                   % MPa
L = 100;                     % mm
epsilon = (u(2) - u(1)) / L; % 应变
sigma = E * epsilon;          % 应力
```

---

## 1.3 流程控制

```matlab
% if-else
if sigma > sigma_s
    disp('材料屈服！');
elseif sigma > 0.8 * sigma_s
    disp('接近屈服，注意！');
else
    disp('安全');
end

% for 循环
for i = 1:10
    x(i) = i^2;
end

% while 循环
iter = 0;
error = 1;
while error > 1e-6
    iter = iter + 1;
    % ... 迭代计算 ...
end

% 向量化操作（比循环快得多！）
x = 1:1000;
y = x.^2 + 2*x + 1;    % 一行代替循环
```

---

## 1.4 函数

```matlab
% 定义函数（保存为 beam_deflection.m）
function w = beam_deflection(F, L, E, I, x)
    % 简支梁跨中集中力的挠度
    % F: 集中力, L: 跨度, E: 弹性模量, I: 惯性矩
    % x: 位置坐标向量
    w = zeros(size(x));
    for i = 1:length(x)
        if x(i) <= L/2
            w(i) = F*x(i)*(3*L^2 - 4*x(i)^2) / (48*E*I);
        else
            xi = L - x(i);
            w(i) = F*xi*(3*L^2 - 4*xi^2) / (48*E*I);
        end
    end
end

% 调用
x = linspace(0, 1000, 100);
w = beam_deflection(1000, 1000, 200e3, 1e6, x);

% 匿名函数（简单函数的快捷方式）
stress = @(F, A) F / A;
sigma = stress(5000, 100);   % σ = 50 MPa
```

---

## 1.5 绘图

```matlab
% 基本绘图
x = linspace(0, 2*pi, 100);
y = sin(x);

figure;
plot(x, y, 'b-', 'LineWidth', 2);
xlabel('x (rad)');
ylabel('sin(x)');
title('正弦函数');
grid on;

% 多条曲线
hold on;
plot(x, cos(x), 'r--', 'LineWidth', 2);
legend('sin(x)', 'cos(x)');

% 子图
figure;
subplot(2, 1, 1);   % 2行1列，第1个
plot(x, sin(x));
title('正弦');

subplot(2, 1, 2);   % 2行1列，第2个
plot(x, cos(x));
title('余弦');

% 应力-应变曲线绘制
epsilon = linspace(0, 0.3, 100);
sigma = 235 * (1 - exp(-50*epsilon));  % 近似曲线

figure;
plot(epsilon, sigma, 'b-', 'LineWidth', 2);
xlabel('\epsilon (应变)');
ylabel('\sigma (MPa)');
title('应力-应变曲线');
grid on;
```

### 力学常用图表

```matlab
% 绘制弯矩图
x = [0, 2, 2, 5, 5];
M = [0, 6, 6, 0, 0];   % kN·m

figure;
fill([x, 5, 0], [M, 0, 0], [0.8 0.9 1], 'EdgeColor', 'b', 'LineWidth', 2);
xlabel('x (m)');
ylabel('M (kN·m)');
title('弯矩图');
grid on;

% 绘制云图（伪彩色图）
[X, Y] = meshgrid(0:0.1:10, 0:0.1:5);
stress_field = 100 * sin(pi*X/10) .* sin(pi*Y/5);

figure;
pcolor(X, Y, stress_field);
shading interp;
colorbar;
colormap('jet');
xlabel('x (mm)'); ylabel('y (mm)');
title('应力分布云图 (MPa)');
```

---

## 1.6 文件操作

```matlab
% 保存数据
save('results.mat', 'u', 'sigma');

% 加载数据
load('results.mat');

% 读写文本文件
data = readmatrix('input.csv');
writematrix(results, 'output.csv');

% 读取 Excel
data = readtable('data.xlsx');
```

---

## 本章练习

```matlab
% 练习1：求解桁架的节点位移
% 3个弹簧串联：k1=100, k2=200, k3=150 N/mm
% 节点1固定，节点4施加 F=600 N
% 组装刚度矩阵并求解

K = [100  -100    0      0;
    -100   300  -200     0;
      0   -200   350  -150;
      0      0  -150   150];

F = [0; 0; 0; 600];

% 施加边界条件（删去第1行列）
Kred = K(2:4, 2:4);
Fred = F(2:4);

u_free = Kred \ Fred;
u = [0; u_free];

fprintf('节点位移：\n');
for i = 1:4
    fprintf('  u%d = %.4f mm\n', i, u(i));
end
```

---

[下一章：Python 科学计算 →](./02-Python科学计算.md)
