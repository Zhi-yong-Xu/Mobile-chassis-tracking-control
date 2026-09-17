import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
import threading
import time

# ==========================================
# 全局变量与锁 (用于线程间通信)
# ==========================================
plot_data_lock = threading.Lock()
shared_data = {
    "time": [],
    "err_x": [],
    "err_y": [],
    "err_norm": [],
    "tau_L": [],
    "tau_R": []
}
stop_plotting = False

# ==========================================
# 1. 绘图线程函数
# ==========================================
def plot_thread_func():
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    
    # 初始化线条
    line_err_x, = ax1.plot([], [], 'r', label='Error X')
    line_err_y, = ax1.plot([], [], 'b', label='Error Y')
    line_norm, = ax2.plot([], [], 'k', label='Error Norm')
    line_tau_L, = ax3.plot([], [], 'g', label='Torque Left')
    line_tau_R, = ax3.plot([], [], 'm', label='Torque Right')

    ax1.set_title('Handpoint Tracking Error (Real-time)')
    ax1.set_ylabel('Position Error (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax2.set_ylabel('Norm Error (m)')
    ax2.grid(True)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (Nm)')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    plt.tight_layout()

    print("绘图线程已启动...")

    while not stop_plotting:
        start_plot_time = time.time()
        
        # 1. 获取数据 (加锁，防止数据竞争)
        with plot_data_lock:
            # 拷贝数据，避免长时间占用锁
            t_data = list(shared_data["time"])
            ex_data = list(shared_data["err_x"])
            ey_data = list(shared_data["err_y"])
            en_data = list(shared_data["err_norm"])
            tl_data = list(shared_data["tau_L"])
            tr_data = list(shared_data["tau_R"])

        # 2. 更新绘图
        if len(t_data) > 0:
            line_err_x.set_data(t_data, ex_data)
            line_err_y.set_data(t_data, ey_data)
            line_norm.set_data(t_data, en_data)
            line_tau_L.set_data(t_data, tl_data)
            line_tau_R.set_data(t_data, tr_data)

            for ax in [ax1, ax2, ax3]:
                ax.relim()
                ax.autoscale_view()
            
            plt.draw()
            plt.pause(0.001) # 在这个线程里暂停，不影响主线程仿真
        
        # 简单的频率控制，避免绘图占用过多 CPU，例如 20FPS
        elapsed = time.time() - start_plot_time
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)

    plt.close(fig)
    print("绘图线程已关闭。")

# ==========================================
# 1. 加载模型
# ==========================================
model = mujoco.MjModel.from_xml_path(r'robots\husky\husky.xml')
data = mujoco.MjData(model)

# ==========================================
# 2. 机器人物理参数提取
# ==========================================
car_body_name = 'base_link'
wheel_names = ['front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link']
joint_names = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']

car_body_id = model.body(car_body_name).id
m_car = model.body_mass[car_body_id]
I_car = model.body_inertia[car_body_id][2] # Z轴惯量

# 获取轮距 (Track Width)
# 左轮 y坐标约为 0.2775，右轮 y坐标约为 -0.2775
fl_pos = model.body_pos[model.body(wheel_names[0]).id]
fr_pos = model.body_pos[model.body(wheel_names[1]).id]
L_wheel = abs(fl_pos[1] - fr_pos[1])

# 获取轮子半径
# robot.xml 中 geom type="cylinder" size="0.1651 0.05715"，半径为第一个元素
fl_geom_id = model.body_geomadr[model.body(wheel_names[0]).id]
r = model.geom_size[fl_geom_id][0]

# 获取单个轮子的质量和转动惯量
m_w = model.body_mass[model.body(wheel_names[0]).id]
I_w = 0.5 * m_w * r**2

# ==========================================
# 3. 控制器参数 (参考 formation3-perfect.py)
# ==========================================
# 动力学预计算
L_hand = 0.01 # 虚拟手点偏移
d_car = L_wheel / 2.0
M11 = m_car + 4 * I_w / (r**2) # 包含4个轮子惯量的等效质量
M22 = I_car + 4 * I_w * (d_car**2) / (r**2) # 包含4个轮子惯量的等效转动惯量

# 速度控制器增益 (来自 XML kv="30")
kv = 30.0

# DSC 控制器参数
k1_car = 4.0
k2_car = 4.0
tau_car = 0.05
F_max_car = 150.0 # 力矩限制

# 轨迹参数
car_traj_radius = 8
car_traj_omega = 0.01
car_traj_center = [-8, 0]

# 系数
coef_v = r / 2.0
coef_w = r / (2.0 * d_car)

# 滤波器状态
alpha_fx = 0.0
alpha_fy = 0.0

# ==========================================
# 4. 获取 Joint 索引
# ==========================================
joint_ids = [model.joint(name).id for name in joint_names]
joint_qvel_adrs = [model.body_jntadr[jid] for jid in joint_ids]


# 启动绘图线程
plot_thread = threading.Thread(target=plot_thread_func)
plot_thread.daemon = True # 设置为守护线程，主程序退出时它也会自动退出
plot_thread.start()

# ==========================================
# 5. 主循环
# ==========================================
dt = model.opt.timestep
frame_count = 0
DATA_SAVE_FREQ = 10 # 每10帧保存一次数据到共享变量
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("=== Husky Robot Control Started ===")
    
    while viewer.is_running():
        t = data.time
        
        # =========================================
        # A. 状态获取
        # =========================================
        car_pos = data.xpos[car_body_id].copy()
        car_quat = data.xquat[car_body_id]
        car_rot = R.from_quat([car_quat[1], car_quat[2], car_quat[3], car_quat[0]])
        car_yaw = car_rot.as_euler('xyz', degrees=False)[2]
        
        # 获取 Base Link 的速度
        car_joint_id = model.body_jntadr[car_body_id]
        car_qvel_adr = model.jnt_qposadr[car_joint_id]
        vx_glob = data.qvel[car_qvel_adr]
        vy_glob = data.qvel[car_qvel_adr + 1]
        wz_glob = data.qvel[car_qvel_adr + 5]
        
        # 转换到局部坐标系速度
        rotation_matrix = car_rot.as_matrix()
        v_local = rotation_matrix.T @ np.array([vx_glob, vy_glob, 0])
        v_car = v_local[0]
        w_car = wz_glob
        
        # =========================================
        # B. 参考轨迹生成
        # =========================================
        ref_pos = np.array([
            car_traj_center[0] + car_traj_radius * math.cos(car_traj_omega * t),
            car_traj_center[1] + car_traj_radius * math.sin(car_traj_omega * t)
        ])
        ref_vel = np.array([
            -car_traj_radius * car_traj_omega * math.sin(car_traj_omega * t),
            car_traj_radius * car_traj_omega * math.cos(car_traj_omega * t)
        ])
        ref_acc = np.array([
            -car_traj_radius * (car_traj_omega**2) * math.cos(car_traj_omega * t),
            -car_traj_radius * (car_traj_omega**2) * math.sin(car_traj_omega * t)
        ])
        
        # =========================================
        # C. DSC 控制律计算
        # =========================================
        # 1. 手点状态
        xc = car_pos[0] + L_hand * math.cos(car_yaw)
        yc = car_pos[1] + L_hand * math.sin(car_yaw)
        vc_x = v_car * math.cos(car_yaw) - L_hand * w_car * math.sin(car_yaw)
        vc_y = v_car * math.sin(car_yaw) + L_hand * w_car * math.cos(car_yaw)
        
        # 2. 非线性项
        f_x = -v_car * w_car * math.sin(car_yaw) - L_hand * (w_car**2) * math.cos(car_yaw)
        f_y = v_car * w_car * math.cos(car_yaw) - L_hand * (w_car**2) * math.sin(car_yaw)
        
        # 3. 误差与滤波
        z1_x = xc - ref_pos[0]
        z1_y = yc - ref_pos[1]
        
        alpha_x = -k1_car * z1_x + ref_vel[0]
        alpha_y = -k1_car * z1_y + ref_vel[1]
        
        alpha_f_dot_x = (alpha_x - alpha_fx) / tau_car
        alpha_f_dot_y = (alpha_y - alpha_fy) / tau_car
        
        z2_x = vc_x - alpha_fx
        z2_y = vc_y - alpha_fy
        
        # 4. 加速度指令
        acc_cmd_x = -k2_car * z2_x - z1_x + alpha_f_dot_x + ref_acc[0]
        acc_cmd_y = -k2_car * z2_y - z1_y + alpha_f_dot_y + ref_acc[1]
        
        u_x = acc_cmd_x - f_x
        u_y = acc_cmd_y - f_y
        
        # 5. 逆变换 (全局加速度 -> 局部加速度)
        dv = u_x * math.cos(car_yaw) + u_y * math.sin(car_yaw)
        dw =  (-u_x * math.sin(car_yaw) + u_y * math.cos(car_yaw)) / L_hand
        
        # 6. 逆动力学 (局部加速度 -> 左右侧总力矩)
        tau_L = coef_v * M11 * dv - coef_w * M22 * dw
        tau_R = coef_v * M11 * dv + coef_w * M22 * dw
        
        # 7. 力矩限幅
        tau_L = max(-F_max_car, min(F_max_car, tau_L))
        tau_R = max(-F_max_car, min(F_max_car, tau_R))
        
        # =========================================
        # D. 执行器映射 (关键部分)
        # =========================================
        # 要求：同侧车轮控制输入力矩相等。
        # 因为 robot.xml 有 4 个轮子，且动力学模型中的 tau_L/R 代表左右侧的总力矩，
        # 我们将总力矩平分给同侧的两个轮子，以保持运动一致性。
        
        # 计算每个轮子的目标力矩
        # tau_L / 2.0 分配给左前和左后
        # tau_R / 2.0 分配给右前和右后
        torque_per_wheel_left = tau_L / 2.0
        torque_per_wheel_right = tau_R / 2.0
        
        # 由于 robot.xml 使用的是 velocity 类型的执行器，我们需要将力矩命令转换为速度命令。
        # 公式: Torque = kv * (target_vel - current_vel)
        # => target_vel = current_vel + Torque / kv
        
        # 获取当前轮子角速度
        vel_fl = data.qvel[joint_qvel_adrs[0]]
        vel_fr = data.qvel[joint_qvel_adrs[1]]
        vel_rl = data.qvel[joint_qvel_adrs[2]]
        vel_rr = data.qvel[joint_qvel_adrs[3]]
        

        data.ctrl[0] = torque_per_wheel_left
        data.ctrl[1] = torque_per_wheel_right
        data.ctrl[2] = torque_per_wheel_left
        data.ctrl[3] = torque_per_wheel_right
        
        # =========================================
        # E. 更新滤波器状态
        # =========================================
        alpha_fx = alpha_fx + alpha_f_dot_x * dt
        alpha_fy = alpha_fy + alpha_f_dot_y * dt


        # --- 数据记录 (加锁) ---
        if frame_count % DATA_SAVE_FREQ == 0:
            with plot_data_lock:
                shared_data["time"].append(t)
                shared_data["err_x"].append(z1_x)
                shared_data["err_y"].append(z1_y)
                shared_data["err_norm"].append(np.sqrt(z1_x**2 + z1_y**2))
                shared_data["tau_L"].append(tau_L)
                shared_data["tau_R"].append(tau_R)
        
        # 仿真步进
        mujoco.mj_step(model, data)
        try:
            marker_site_id = model.site('trajectory_marker').id
            data.site_xpos[marker_site_id][:2] = ref_pos
        except KeyError:
            pass
        viewer.sync()
        frame_count += 1

# 结束处理
stop_plotting = True
plot_thread.join()
