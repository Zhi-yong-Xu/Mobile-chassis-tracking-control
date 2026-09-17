%% 无人车分层PID控制器仿真 (性能优化版)
clear; clc; close all;

%% 1. 系统参数初始化 (去除全局变量，提升运行速度)
m = 50;         % 整车质量
J = 10;         % 转动惯量
Fp_max = 200;   % 最大推力
Tp_max = 1000;   % 最大力矩

dt = 0.01;     % 离散时间步长
T_end = 600;     % 仿真总时长
t_vec = 0:dt:T_end;
N = length(t_vec);

% 状态变量初始化 [x, dx, y, dy, phi, dphi]
State = zeros(6, 1);
State(5) = pi / 2; 

% 控制器参数
% Kp_pos = 1; Ki_pos = 0.0; Kd_pos = 0; 
% Kp_att = 1; Ki_att = 0.0; Kd_att = 0;

% 积分限幅阈值
I_limit = 10; 

% 将PID状态变量全部转为局部标量 (比struct快得多)
Ix = 0; Iy = 0; Iphi = 0;
ex_last = 0; ey_last = 0; ephi_last = 0;

%% 2. 主循环仿真 (预分配内存)
history_pos = zeros(N, 2);
history_phi = zeros(N, 1);
history_uG = zeros(N, 2);
history_F_T = zeros(N, 2);

phi_d_actual = zeros(N,1);

% 目标轨迹：圆形轨迹测试
radius = 50; 
x_d_arr = radius * cos(t_vec * 0.1) - 50;
y_d_arr = radius * sin(t_vec * 0.1);

a_max = Fp_max / m; % 预计算最大允许加速度

% 控制器参数 (建议调整)
Kp_pos = 1.0; Ki_pos = 0.0; Kd_pos = 1.0; % 外环
Kp_att = 5.0; Ki_att = 0.0; Kd_att = 2.0; % 内环需更快

for i = 1:N-1
    % --- Step 1: 采集反馈 ---
    x = State(1); dx = State(2);
    y = State(3); dy = State(4);
    phi = State(5); dphi = State(6);
    
    x_d = x_d_arr(i); 
    y_d = y_d_arr(i);
    
    % --- Step 2: 外环位置PID ---
    ex = x_d - x;
    ey = y_d - y;
    
    % 积分与限幅
    Ix = Ix + ex * dt; Ix = max(min(Ix, I_limit), -I_limit);
    Iy = Iy + ey * dt; Iy = max(min(Iy, I_limit), -I_limit);
    
    % 微分项
    Dex = (ex - ex_last) / dt;
    Dey = (ey - ey_last) / dt;
    
    % 期望加速度
    u_Gx = Kp_pos * ex + Ki_pos * Ix + Kd_pos * Dex;
    u_Gy = Kp_pos * ey + Ki_pos * Iy + Kd_pos * Dey;
    
    ex_last = ex; ey_last = ey;
    
    % --- Step 3: 推力分配 (核心修正) ---
    phi_d = atan2(u_Gy, u_Gx);
    phi_d_actual(i) = phi_d;

    % 推力 = 需求加速度向量 在 车身方向 的投影
    % 这能确保：当车身角度偏差大时，推力自动减小，避免冲出跑道
    F_p_cmd = m * (u_Gx * cos(phi) + u_Gy * sin(phi));
    F_p = max(min(F_p_cmd, Fp_max), -Fp_max);
    
    % --- Step 4: 内环姿态PID ---
    e_phi = phi_d - phi;
    e_phi = atan2(sin(e_phi), cos(e_phi)); % 角度归一化
    
    Iphi = Iphi + e_phi * dt;
    Iphi = max(min(Iphi, I_limit), -I_limit);
    
    De_phi = -dphi; % 此处简化处理，假设期望角速度为0
    phi_ddot_ref = Kp_att * e_phi + Ki_att * Iphi + Kd_att * De_phi;
    
    T_p_cmd = J * phi_ddot_ref;
    T_p = max(min(T_p_cmd, Tp_max), -Tp_max);
    
    % --- Step 5: 动力学积分 ---
    ax = (F_p / m) * cos(phi);
    ay = (F_p / m) * sin(phi);
    alpha = T_p / J;
    
    State(1) = State(1) + dx * dt;       
    State(2) = State(2) + ax * dt;       
    State(3) = State(3) + dy * dt;       
    State(4) = State(4) + ay * dt;       
    State(5) = State(5) + dphi * dt;     
    State(6) = State(6) + alpha * dt;    
    
    State(5) = atan2(sin(State(5)), cos(State(5)));
    
    % 记录数据...
    history_pos(i, :) = [x, y];
    history_phi(i) = phi;
    history_uG(i, :) = [u_Gx, u_Gy];
    history_F_T(i, :) = [F_p, T_p];
end

%% 3. 结果绘图
figure('Color', 'w', 'Position', [100, 100, 1000, 800]);

subplot(2, 2, 1);
plot(history_pos(:, 1), history_pos(:, 2), 'b', 'LineWidth', 1.5); hold on;
plot(x_d_arr, y_d_arr, 'r--', 'LineWidth', 1.5);
legend('实际轨迹', '期望轨迹');
xlabel('X (m)'); ylabel('Y (m)');
title('XY 平面轨迹跟踪');
axis equal; grid on;

subplot(2, 2, 2);
ex_arr = x_d_arr(:) - history_pos(:, 1);
ey_arr = y_d_arr(:) - history_pos(:, 2);
plot(t_vec(1:end-1), ex_arr(1:end-1), 'r'); hold on;
plot(t_vec(1:end-1), ey_arr(1:end-1), 'b');
legend('X误差', 'Y误差');
xlabel('时间');
title('位置跟踪误差');
grid on;



subplot(2, 2, 3);
plot(t_vec(1:end-1), history_phi(1:end-1), 'b', 'LineWidth', 1.5); hold on;
plot(t_vec(1:end-1), phi_d_actual(1:end-1), 'r--');
legend('航向误差', '期望航向');
xlabel('时间'); ylabel('角度');
title('航向角跟踪');
grid on;

subplot(2, 2, 4);
yyaxis left;
plot(t_vec, history_F_T(:, 1), 'k');
ylabel('推力 F_p (N)');
yyaxis right;
plot(t_vec, history_F_T(:, 2), 'm');
ylabel('力矩 T_p (Nm)');
xlabel('时间');
title('控制输入 (推力与力矩)');
grid on;


figure() ;hold on
subplot(3, 1, 1);
plot(t_vec, x_d_arr, 'r'); hold on;
plot(t_vec, history_pos(:, 1), 'b');
legend('期望X', '实际X');
xlabel('时间');
title('X轴位置跟踪效果');
grid on;

subplot(3, 1, 2);
plot(t_vec, y_d_arr, 'r'); hold on;
plot(t_vec, history_pos(:, 2), 'b');
legend('期望Y', '实际Y');
xlabel('时间');
title('Y轴位置跟踪效果');
grid on;



subplot(3, 1, 3);
plot(t_vec(1:end-1), history_phi(1:end-1), 'b', 'LineWidth', 1.5); hold on;
plot(t_vec(1:end-1), phi_d_actual(1:end-1), 'r--');
legend('实际航向', '期望航向');
xlabel('时间'); ylabel('角度');
title('航向角跟踪');
grid on;

sgtitle('无人车分层PID控制器仿真验证结果');
