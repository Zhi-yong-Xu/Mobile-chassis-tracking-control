# Mobile-chassis-tracking-control

---

## 目录

- [0. 项目概述](#0-项目概述)
- [1. 统一误差框架](#1-统一误差框架)
- [2. 差速轮模型](#2-差速轮模型)
- [3. 麦克纳姆轮模型](#3-麦克纳姆轮模型)
- [4. 各文件控制方法](#4-各文件控制方法)
- [5. 总结与横向对照](#5-总结与横向对照)

---

## 0. 项目概述

本项目包含一组可独立运行的 MuJoCo / MATLAB 仿真脚本，验证两类移动底盘的轨迹跟踪控制：

- **差速/滑移转向底盘**（Husky）：`husky_volPID.py`、`husky_EW_PID.py`、`husky_EB_PID.py`、`husky_EB_DSC.py`
- **麦克纳姆轮底盘**（Summit XL）：`Mac_PID.py`、`Mac_DSC.py`、`Macdscpid.py`
- **非完整质点车参考实现**：`NonholonomicMobileRobotPIDControl.m`

### 0.1 依赖与运行

```bash
pip install mujoco numpy matplotlib
python husky_EB_DSC.py     # 任一脚本独立运行
```

MATLAB：`NonholonomicMobileRobotPIDControl.m`。

### 0.2 文件速查

| 文件 | 底盘 | 误差坐标系 | 航向生成 | 上层方法 | 底层接口 |
|---|---|---|---|---|---|
| `husky_volPID.py` | 差速 | 世界系 | 世界系制导方向 | PID + 制导 | 轮速/力矩环 |
| `husky_EW_PID.py` | 差速 | 世界系 | 世界系制导方向 | PID + 制导 | 轮速/力矩环 |
| `husky_EB_PID.py` | 差速 | 车体系 | 车体系制导方向 | PID + 方位制导 | 轮速/力矩环 |
| `husky_EB_DSC.py` | 差速 | 车体系 | 滤波虚拟控制量方向 | DSC + 积分/微分 | 轮速伺服 + 前馈 |
| `Mac_PID.py` | 麦轮 | 世界系+车体系 | 显式轨迹切线 | 外环位置 PID + 内环速度 PID | 力矩 |
| `Mac_DSC.py` | 麦轮 | 车体系 | 显式轨迹切线 | 纯 DSC | 力矩 |
| `Macdscpid.py` | 麦轮 | 车体系 | 显式轨迹切线 | DSC + 积分/微分 | 力矩 |
| `NonholonomicMobileRobotPIDControl.m` | 质点车 | 世界系 | 世界系制导方向 | 外环 PID + 内环姿态 PID | 推力 + 力矩 |

---

## 1. 统一误差框架

本节汇总所有脚本共用的误差、旋转与航向符号，后续各文件的控制方法均引用本节公式。

### 1.1 坐标系与旋转

世界系位置 $p=[x,y]^T$，航向角 $\phi$，车体系速度 $\nu=[v_x^b,v_y^b,\omega]^T$。

车体系到世界系的旋转矩阵 $R(\phi)$ 与反对称矩阵 $[\omega]_\times$：

$$
R(\phi)=\begin{bmatrix}\cos\phi & -\sin\phi \\ \sin\phi & \cos\phi\end{bmatrix},
\qquad
[\omega]_\times=\begin{bmatrix}0 & -\omega \\ \omega & 0\end{bmatrix}
$$

旋转矩阵及其转置对时间的导数：

$$
\dot R=R\,[\omega]_\times,
\qquad
\frac{d}{dt}R^T=-[\omega]_\times R^T
$$

### 1.2 世界系误差

参考位置 $p_d=[x_d,y_d]^T$，世界系误差及其导数：

$$
e_x=x_d-x,
\qquad
e_y=y_d-y
$$

$$
\dot e_x=v_{x,d}-v_x,
\qquad
\dot e_y=v_{y,d}-v_y
$$

世界系误差导数无旋转耦合项，是 `husky_volPID.py` / `husky_EW_PID.py` / MATLAB 直接使用解析微分 $K_d(\dot p_d-\dot p)$ 的依据。

### 1.3 车体系误差

$$
e_x^b=e_x\cos\phi+e_y\sin\phi,
\qquad
e_y^b=-e_x\sin\phi+e_y\cos\phi
$$

**代码中的车体系误差导数（含旋转耦合项）**：

`husky_EB_PID.py` 实现：

$$
v_{x,d}^b=v_{x,d}\cos\phi+v_{y,d}\sin\phi,
\qquad
v_{y,d}^b=-v_{x,d}\sin\phi+v_{y,d}\cos\phi
$$

$$
v_x^b=v_{x}\cos\phi+v_{y}\sin\phi,
\qquad
v_y^b=-v_{x}\sin\phi+v_{y}\cos\phi
$$

$$
\dot e_x^b=(v_{x,d}^b-v_x^b)+e_y^b\,\omega
$$

$$
\dot e_y^b=(v_{y,d}^b-v_y^b)-e_x^b\,\omega
$$

`husky_EB_DSC.py` / `Macdscpid.py` 实现：

$$
v_{x,ff}=v_{x,d}\cos\phi+v_{y,d}\sin\phi,
\qquad
v_{y,ff}=-v_{x,d}\sin\phi+v_{y,d}\cos\phi
$$

$$
D_{exb}=(v_{x,ff}-v_x^b)+e_y^b\,\omega
$$

$$
D_{eyb}=-(v_{y,ff}-v_y^b)-e_x^b\,\omega
$$

### 1.4 航向误差的三种生成方式

角度归一化 $\mathrm{wrap}(\alpha)=\mathrm{atan2}(\sin\alpha,\cos\alpha)$。

| 方式 | 表达式 | 使用文件 |
|---|---|---|
| 世界系制导 | $\phi_d=\mathrm{atan2}(u_{Gy},u_{Gx})$，$\tilde\phi=\mathrm{wrap}(\phi_d-\phi)$ | `husky_volPID.py` / `husky_EW_PID.py` / MATLAB |
| 车体系制导 | $\tilde\phi=\mathrm{atan2}(u_{by},u_{bx})$ | `husky_EB_PID.py` |
| 滤波面构造 | $\tilde\phi=\mathrm{atan2}(\alpha_{f,y},\alpha_{f,x})$ | `husky_EB_DSC.py` |
| 轨迹切线 | $\phi_d=\omega_{traj}t+\pi/2$，$\tilde\phi=\mathrm{wrap}(\phi_d-\phi)$ | `Mac_PID.py` / `Mac_DSC.py` / `Macdscpid.py` |

### 1.5 航向误差动力学

对任意定义：

$$
\dot{\tilde\phi}=\dot\phi_d-\omega
$$

由此角速度指令一般形式为

$$
\omega_{cmd}=K_{att}\tilde\phi-K_{d,att}\omega+w_{ff},
\qquad
w_{ff}=\dot\phi_d
$$

---

## 2. 差速轮模型

### 2.1 运动学

半轮距 $d_{car}$、轮半径 $r$。逆解（由车体速度到左右轮角速度）：

$$
v_L=v-\omega\,d_{car},
\qquad
v_R=v+\omega\,d_{car}
$$

$$
\omega_L=\frac{v_L}{r},
\qquad
\omega_R=\frac{v_R}{r}
$$

四轮同侧同速分配：`[w_L, w_R, w_L, w_R]`。

### 2.2 动力学（仅 `husky_EB_DSC.py` 使用）

等效惯量：

$$
M_{11}=m+\frac{2 I_w}{r^2},
\qquad
M_{33}=I_{zz}+\frac{2 I_w (a^2+b^2)}{r^2}
$$

其中 $I_w=0.5\,m_w r^2$ 由估计轮质量 $m_w$ 计算。

科氏项与广义力：

$$
C_x=-m\,\omega\,v_y^b,
\qquad
F_x=M_{11}a_x+C_x,
\qquad
T_z=M_{33}a_\phi
$$

共模/差模前馈力矩：

$$
\tau_{ff,c}=\frac{F_x r}{4},
\qquad
\tau_{ff,d}=\frac{T_z r}{4 d_{car}}
$$

### 2.3 底层执行器接口

`husky_EB_DSC.py` 自动检测执行器类型：

**速度伺服模式**：`ctrl[i] = ω_{i,cmd}`。

**力矩模式**（本地速度环 + DSC 前馈）：

$$
\tau_L=k_v(\omega_{L,cmd}-\omega_{L,meas})+\tau_{ff,c}-\tau_{ff,d}
$$

$$
\tau_R=k_v(\omega_{R,cmd}-\omega_{R,meas})+\tau_{ff,c}+\tau_{ff,d}
$$

限幅到 $\pm\tau_{\max}$。

其它 Husky 脚本（`husky_volPID.py` / `husky_EW_PID.py` / `husky_EB_PID.py`）在力矩模式下仅使用本地速度环：

$$
\tau_i=k_v(\omega_{i,cmd}-\omega_{i,meas})
$$

---

## 3. 麦克纳姆轮模型

### 3.1 运动学

半轴距 $a$、半轮距 $b$、轮半径 $r$。代码中力矩分配矩阵（`J_pinv`）：

$$
J_{pinv}=\frac{1}{r}
\begin{bmatrix}
1 &  1 &  (a+b) \\
1 & -1 & -(a+b) \\
1 & -1 &  (a+b) \\
1 &  1 & -(a+b)
\end{bmatrix}
$$

### 3.2 动力学

等效惯量矩阵：

$$
M_{mat}=\mathrm{diag}\!\left(
m+\frac{2 I_w}{r^2},\;
m+\frac{2 I_w}{r^2},\;
I_{zz}+\frac{2 I_w(a^2+b^2)}{r^2}
\right)
$$

科氏力补偿：

$$
C_{force}=\begin{bmatrix}
-m\,\omega\,v_y^b \\
m\,\omega\,v_x^b \\
0
\end{bmatrix}
$$

从期望加速度到力矩：

$$
F_{cmd}=M_{mat}\begin{bmatrix}a_x \\ a_y \\ a_\phi\end{bmatrix}+C_{force},
\qquad
\tau_{cmd}=J_{pinv}\,F_{cmd}
$$

`Mac_PID.py` / `Macdscpid.py` 中 `tau_cmd = np.clip(tau_cmd, -50, 50)`；`Mac_DSC.py` 同样限幅到 $\pm 50$。

---

## 4. 各文件控制方法

### 4.1 `husky_volPID.py` —— 世界系 PID + 速度制导

**误差坐标系**：世界系。**航向生成**：世界系制导方向。

外环 PID：

$$
u_{Gx}=K_p e_x+K_i I_x+K_d(v_{x,d}-v_x)
$$

$$
u_{Gy}=K_p e_y+K_i I_y+K_d(v_{y,d}-v_y)
$$

期望航向：

$$
\phi_d=\mathrm{atan2}(u_{Gy},u_{Gx}),
\qquad
\tilde\phi=\mathrm{wrap}(\phi_d-\phi)
$$

前向速度（位置误差投影）：

$$
e_{fwd}=e_x\cos\phi+e_y\sin\phi
$$

$$
v_{cmd}=\mathrm{clip}(K_v e_{fwd},-v_{max},v_{max})\cdot\max(\cos\tilde\phi,0)
$$

角速度（P + D + 曲率前馈）：

$$
w_{ff}=v_{cmd}/R,
\qquad
\omega_{cmd}=\mathrm{clip}(K_{att}\tilde\phi-K_{d,att}\omega+w_{ff},-w_{max},w_{max})
$$

经差速逆解写入执行器。

### 4.2 `husky_EW_PID.py` —— 世界系 PID（改进版）

结构与 4.1 相同，主要差别是前向速度改用**制导向量投影**：

$$
e_{fwd}=u_{Gx}\cos\phi+u_{Gy}\sin\phi
$$

角速度前馈为 $w_{ff}=v_{cmd}\,\kappa$，其中 $\kappa=0$ 或 $1/R$（由 `USE_STRAIGHT_TEST` 决定）。

### 4.3 `husky_EB_PID.py` —— 车体系误差 PID + 方位制导

**误差坐标系**：车体系。**航向生成**：车体系制导方向。

车体系误差与其导数（见 §1.3）。

车体系 PID 制导：

$$
u_{bx}=K_p e_x^b+K_i I_x^b+K_d\dot e_x^b
$$

$$
u_{by}=K_p e_y^b+K_i I_y^b+K_d\dot e_y^b
$$

航向误差由制导向量方向给出（无需求归一化）：

$$
\tilde\phi=\mathrm{atan2}(u_{by},u_{bx})\quad (\|u_b\|>10^{-3})
$$

速度与角速度：

$$
v_{cmd}=\mathrm{clip}(K_v u_{bx},-v_{max},v_{max})\cdot\max(\cos\tilde\phi,0)
$$

$$
\omega_{cmd}=\mathrm{clip}(K_{att}\tilde\phi-K_{d,att}\omega+v_{cmd}/R,-w_{max},w_{max})
$$

经差速逆解与本地速度环写入执行器。

### 4.4 `husky_EB_DSC.py` —— 车体系 DSC + 轮速伺服前馈

**误差坐标系**：车体系。**航向生成**：滤波虚拟控制量方向。

**第一动态面（前向/横向通道）带积分/微分**：

$$
z_{1,x}=-e_x^b,
\qquad
z_{1,y}=-e_y^b
$$

$$
\alpha_x=-k_{1x}z_{1,x}-k_{1i}I_{z1x}+k_{1d}D_{exb}
$$

$$
\alpha_y=-k_{1y}z_{1,y}-k_{1i}I_{z1y}+k_{1d}D_{eyb}
$$

**一阶低通滤波**：

$$
\dot\alpha_{f,i}=-(\alpha_{f,i}-\alpha_i)/\tau,
\qquad
\alpha_{f,i}\leftarrow\alpha_{f,i}+\dot\alpha_{f,i}\,\Delta t
$$

**航向由滤波虚拟控制量构造**：

$$
\tilde\phi=\mathrm{atan2}(\alpha_{f,y},\alpha_{f,x})\quad (\|\alpha_f\|>10^{-3})
$$

**前向速度调度与曲率前馈**：

$$
s=\max(\cos\tilde\phi,0),
\qquad
v_{ref}=\alpha_{f,x}\,s
$$

$$
\omega_{ff}=\alpha_{f,x}/R\quad (\text{圆轨迹})
$$

**航向通道第一面与滤波**：

$$
z_{1,\phi}=-\tilde\phi,
\qquad
\alpha_\phi=-k_{1,\phi}z_{1,\phi}+\omega_{ff}
$$

$$
\dot\alpha_{f,\phi}=-(\alpha_{f,\phi}-\alpha_\phi)/\tau
$$

**第二动态面与加速度指令**：

$$
z_{2,x}=v_x^b-v_{ref},
\qquad
z_{2,\phi}=\omega-\alpha_{f,\phi}
$$

$$
a_x=-k_2 z_{2,x}+\dot\alpha_{f,x}-z_{1,x}
$$

$$
a_\phi=-k_2 z_{2,\phi}+\dot\alpha_{f,\phi}-z_{1,\phi}
$$

**动力学前馈**（§2.2）得到 $\tau_{ff,c},\tau_{ff,d}$。

**底层接口**：

$$
\omega_{L,cmd}=\frac{v_{ref}-\alpha_{f,\phi}\,d_{car}}{r},
\qquad
\omega_{R,cmd}=\frac{v_{ref}+\alpha_{f,\phi}\,d_{car}}{r}
$$

速度伺服直接写 $\omega_{L/R,cmd}$；力矩模式按 §2.3 写入。

### 4.5 `Mac_PID.py` —— 外环位置 PID + 内环速度 PID

**误差坐标系**：世界系与车体系同时计算。**航向生成**：轨迹切线。

期望航向：

$$
\phi_d=\omega_{traj}t+\pi/2,
\qquad
\tilde\phi=\mathrm{wrap}(\phi_d-\phi)
$$

外环位置 PID（世界系与车体系）：

$$
u_{Gx}^{world}=K_p e_x+K_i I_x+K_d D_{ex},
\qquad
u_{Gx}^{body}=K_p e_x^b+K_i I_x^b+K_d D_{exb}
$$

$y$ 方向同理。

期望车体速度 = 车体系 PID + 参考速度前馈：

$$
v_{x,cmd}^b=u_{bx}^{body}+(v_{x,d}^w\cos\phi+v_{y,d}^w\sin\phi)
$$

$$
v_{y,cmd}^b=u_{by}^{body}+(-v_{x,d}^w\sin\phi+v_{y,d}^w\cos\phi)
$$

航向角速度指令：

$$
v_{\phi,cmd}=K_p\tilde\phi+K_i I_\phi+K_d D_{\tilde\phi}
$$

内环速度环（含前馈加速度）：

$$
a_x=a_{x,ff}+K_{p,v}(v_{x,cmd}^b-v_x^b)+K_{i,v}I_{vx}+K_{d,v}D_{vx}
$$

$$
a_y=a_{y,ff}+K_{p,v}(v_{y,cmd}^b-v_y^b)+K_{i,v}I_{vy}+K_{d,v}D_{vy}
$$

$$
a_\phi=K_{p,v}(v_{\phi,cmd}-\omega)+K_{i,v}I_{v\phi}+K_{d,v}D_{v\phi}
$$

经 §3.2 麦轮动力学与 $J_{pinv}$ 分配写入四轮力矩。

### 4.6 `Mac_DSC.py` —— 纯 DSC 动态面

**误差坐标系**：车体系。**航向生成**：轨迹切线。

第一动态面：

$$
z_{1,x}=-e_x^b,
\quad
z_{1,y}=-e_y^b,
\quad
z_{1,\phi}=-\tilde\phi
$$

虚拟控制律（含参考前馈）：

$$
\alpha_x=-k_1 z_{1,x}+v_{x,ff},
\qquad
\alpha_y=-k_1 z_{1,y}+v_{y,ff}
$$

$$
\alpha_\phi=-k_{1,\phi}z_{1,\phi}+\omega_{ff}
$$

一阶低通滤波（同 DSC 通用形式）。

第二动态面与加速度指令：

$$
z_{2,x}=v_x^b-\alpha_{f,x},
\quad
z_{2,y}=v_y^b-\alpha_{f,y},
\quad
z_{2,\phi}=\omega-\alpha_{f,\phi}
$$

$$
a_x=-k_2 z_{2,x}+\dot\alpha_{f,x}-z_{1,x}
$$

$$
a_y=-k_2 z_{2,y}+\dot\alpha_{f,y}-z_{1,y}
$$

$$
a_\phi=-k_{2,\phi}z_{2,\phi}+\dot\alpha_{f,\phi}-z_{1,\phi}
$$

经 §3.2 动力学与 $J_{pinv}$ 分配得到力矩。

### 4.7 `Macdscpid.py` —— DSC + 积分/微分增强

结构与 `Mac_DSC.py` 相同，唯一区别是**第一动态面虚拟控制律加入积分/微分项**：

$$
D_{exb}=(v_{x,ff}-v_x^b)+e_y^b\,\omega
$$

$$
D_{eyb}=-(v_{y,ff}-v_y^b)-e_x^b\,\omega
$$

$$
D_{e\phi}=\omega_{ff}-\omega
$$

$$
I_{z1,i}\leftarrow\mathrm{clip}(I_{z1,i}+z_{1,i}\,\Delta t,\,-I_{lim},\,I_{lim})
$$

$$
\alpha_x=-k_1 z_{1,x}-k_{1i}I_{z1x}+k_{1d}D_{exb}
$$

$$
\alpha_y=-k_1 z_{1,y}-k_{1i}I_{z1y}+k_{1d}D_{eyb}
$$

$$
\alpha_\phi=-k_{1,\phi}z_{1,\phi}-k_{1i}I_{z1\phi}+k_{1d}D_{e\phi}
$$

其余（滤波、第二面、动力学补偿、伪逆分配）与 `Mac_DSC.py` 相同。

### 4.8 `NonholonomicMobileRobotPIDControl.m` —— MATLAB 参考实现

被控对象为非完整质点车，状态向量 $[x,\dot x,y,\dot y,\phi,\dot\phi]^T$。

外环位置 PID：

$$
u_{Gx}=K_{p,x}e_x+K_{i,x}I_x+K_{d,x}D_{ex}
$$

$$
u_{Gy}=K_{p,y}e_y+K_{i,y}I_y+K_{d,y}D_{ey}
$$

微分项由差分实现 $D_{ex}=(e_x-e_{x,last})/\Delta t$。

期望航向：

$$
\phi_d=\mathrm{atan2}(u_{Gy},u_{Gx}),
\qquad
\tilde\phi=\mathrm{wrap}(\phi_d-\phi)
$$

推力取制导向量在车头方向投影：

$$
F_{p,cmd}=m(u_{Gx}\cos\phi+u_{Gy}\sin\phi)
$$

$$
F_p=\mathrm{clip}(F_{p,cmd},-F_{p,max},F_{p,max})
$$

内环姿态 PID：

$$
\ddot\phi_{ref}=K_{p,\phi}\tilde\phi+K_{i,\phi}I_\phi+K_{d,\phi}(-\dot\phi)
$$

$$
T_p=\mathrm{clip}(J\ddot\phi_{ref},-T_{p,max},T_{p,max})
$$

动力学积分：

$$
a_x=(F_p/m)\cos\phi,
\qquad
a_y=(F_p/m)\sin\phi,
\qquad
\alpha=T_p/J
$$

---
