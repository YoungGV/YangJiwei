# 第二章 Python 科学计算基础

## 2.1 Python 环境搭建

```
  推荐安装 Anaconda（科学计算全家桶）：
  https://www.anaconda.com
  
  包含：Python + NumPy + SciPy + Matplotlib + Jupyter
  
  或使用 pip 安装：
  pip install numpy scipy matplotlib jupyter
```

### Python vs MATLAB

```
  Python 优势：
  ✓ 免费开源
  ✓ 生态丰富（机器学习、深度学习、Web等）
  ✓ Abaqus 脚本语言就是 Python
  ✓ 工业界更广泛使用
  
  MATLAB 优势：
  ✓ 矩阵运算语法更简洁
  ✓ 工具箱丰富（信号处理、控制、优化等）
  ✓ 学术界使用广泛
  ✓ 调试方便
```

---

## 2.2 NumPy 基础（矩阵运算）

### 数组创建

```python
import numpy as np

# 一维数组
a = np.array([1, 2, 3, 4, 5])

# 二维数组（矩阵）
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 快速生成
x = np.arange(0, 10, 0.1)         # 等间隔
x = np.linspace(0, 10, 100)        # 均匀100个点
I = np.eye(3)                       # 单位矩阵
Z = np.zeros((3, 3))               # 零矩阵
O = np.ones((3, 3))                # 全1矩阵
```

### 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = A @ B                  # 推荐写法
C = np.dot(A, B)           # 等价写法
C = A * B                  # 注意！这是逐元素乘法

# 转置
AT = A.T

# 逆矩阵
Ainv = np.linalg.inv(A)

# 解线性方程组 Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)   # 比 inv(A)@b 更高效

# 行列式
d = np.linalg.det(A)

# 特征值
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### MATLAB 与 Python 对照

```
  MATLAB              Python (NumPy)
  ────────────────    ────────────────
  A * B               A @ B
  A .* B              A * B
  A'                  A.T
  inv(A)              np.linalg.inv(A)
  A \ b               np.linalg.solve(A, b)
  det(A)              np.linalg.det(A)
  eig(A)              np.linalg.eig(A)
  zeros(3,3)          np.zeros((3,3))
  eye(3)              np.eye(3)
  size(A)             A.shape
  linspace(0,1,100)   np.linspace(0,1,100)
```

### 力学示例：求解有限元方程

```python
import numpy as np

# 刚度矩阵
K = np.array([[300, -100],
              [-100, 200]])

# 载荷向量
F = np.array([0, 500])

# 求解位移
u = np.linalg.solve(K, F)
print(f"节点位移: u = {u}")

# 应力计算
E = 200e3    # MPa
L = 100      # mm
epsilon = (u[1] - u[0]) / L
sigma = E * epsilon
print(f"应变: ε = {epsilon:.6f}")
print(f"应力: σ = {sigma:.2f} MPa")
```

---

## 2.3 Matplotlib 绘图

### 基础绘图

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
plt.xlabel('x (rad)', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('三角函数', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('trig.png', dpi=150)
plt.show()
```

### 应力-应变曲线

```python
import matplotlib.pyplot as plt
import numpy as np

# 模拟低碳钢拉伸曲线
epsilon = np.linspace(0, 0.25, 500)
sigma = np.zeros_like(epsilon)

E = 200e3   # MPa
sigma_s = 235  # MPa
sigma_b = 400  # MPa
eps_s = sigma_s / E

for i, eps in enumerate(epsilon):
    if eps <= eps_s:
        sigma[i] = E * eps
    elif eps <= 0.02:
        sigma[i] = sigma_s
    else:
        sigma[i] = sigma_s + (sigma_b - sigma_s) * (1 - np.exp(-20*(eps - 0.02)))

plt.figure(figsize=(10, 6))
plt.plot(epsilon, sigma, 'b-', linewidth=2)
plt.axhline(y=sigma_s, color='g', linestyle='--', label=f'σ_s = {sigma_s} MPa')
plt.xlabel('应变 ε', fontsize=14)
plt.ylabel('应力 σ (MPa)', fontsize=14)
plt.title('低碳钢应力-应变曲线', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('stress_strain.png', dpi=150)
plt.show()
```

### 二维云图

```python
import matplotlib.pyplot as plt
import numpy as np

# 应力场
x = np.linspace(0, 10, 100)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)
stress = 100 * np.sin(np.pi*X/10) * np.sin(np.pi*Y/5)

plt.figure(figsize=(12, 5))
plt.pcolormesh(X, Y, stress, cmap='jet', shading='auto')
plt.colorbar(label='应力 σ (MPa)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('应力分布云图')
plt.axis('equal')
plt.savefig('stress_contour.png', dpi=150)
plt.show()
```

---

## 2.4 SciPy 科学计算

### 线性代数

```python
from scipy import linalg
import numpy as np

A = np.array([[4, 2], [2, 3]])
b = np.array([1, 2])

# LU 分解
P, L, U = linalg.lu(A)

# Cholesky 分解（对称正定矩阵）
L_chol = linalg.cholesky(A, lower=True)

# 稀疏矩阵求解（大型有限元常用）
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

K_sparse = csr_matrix(K_large)  # 大型刚度矩阵
u = spsolve(K_sparse, F)
```

### 数值积分

```python
from scipy import integrate

# 定积分
result, error = integrate.quad(lambda x: x**2, 0, 1)
# result = 0.3333...

# 高斯积分（力学核心）
# 2点高斯积分
xi = [-1/np.sqrt(3), 1/np.sqrt(3)]
wi = [1, 1]

def f(x):
    return x**3 + 2*x + 1

integral = sum(w * f(x) for x, w in zip(xi, wi))
```

### 常微分方程求解

```python
from scipy.integrate import solve_ivp
import numpy as np

# 单摆运动方程：θ'' + (g/l)sinθ = 0
# 转化为一阶方程组：y = [θ, ω]
g, l = 9.81, 1.0

def pendulum(t, y):
    theta, omega = y
    return [omega, -g/l * np.sin(theta)]

# 初始条件：θ₀ = 30°, ω₀ = 0
y0 = [np.radians(30), 0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

sol = solve_ivp(pendulum, t_span, y0, t_eval=t_eval)

# 绘图
plt.plot(sol.t, np.degrees(sol.y[0]))
plt.xlabel('时间 (s)')
plt.ylabel('角度 (°)')
plt.title('单摆运动')
plt.grid(True)
plt.show()
```

### 优化

```python
from scipy.optimize import minimize

# 寻找函数最小值（结构优化常用）
def objective(x):
    return (x[0]-1)**2 + (x[1]-2.5)**2

result = minimize(objective, x0=[0, 0], method='BFGS')
print(f"最优解: {result.x}")
print(f"最小值: {result.fun}")
```

---

## 本章练习

```python
# 练习：用 Python 求解弹簧系统
import numpy as np

# 3个弹簧串联
k1, k2, k3 = 100, 200, 150  # N/mm

# 总刚度矩阵（4×4）
K = np.array([
    [ k1,    -k1,      0,      0],
    [-k1, k1+k2,    -k2,      0],
    [  0,    -k2, k2+k3,    -k3],
    [  0,      0,    -k3,     k3]
])

F = np.array([0, 0, 0, 600])

# 边界条件：u1=0 → 删去第0行列
K_red = K[1:, 1:]
F_red = F[1:]

u_free = np.linalg.solve(K_red, F_red)
u = np.concatenate([[0], u_free])

print("节点位移:")
for i, ui in enumerate(u):
    print(f"  u{i+1} = {ui:.4f} mm")
```

---

[← 上一章：MATLAB 基础](./01-MATLAB基础.md) | [下一章：力学数值计算 →](./03-力学数值计算.md)
