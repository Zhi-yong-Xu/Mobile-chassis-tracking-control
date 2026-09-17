import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import matplotlib.pyplot as plt


# ==========================================
# 辅助函数
# ==========================================
def get_state_from_mjdata(model, data, base_id, qpos_adr, dof_adr):
    """
    从MuJoCo数据对象中提取当前状态
    返回: 世界坐标系下的位置, 速度(6维), 航向角
    """
    pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
    vel = data.qvel[dof_adr:dof_adr + 6].copy()

    # 航向角 Phi (从四元数转换)
    quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    R = mat.reshape(3, 3)
    phi = np.arctan2(R[1, 0], R[0, 0])
    return pos, vel, phi


def add_line(scn, p1, p2, radius, rgba):
    """在 viewer 场景中添加一条圆柱线段 (用于绘制参考轨迹)"""
    if scn.ngeom >= scn.maxgeom:
        return
    mid = (p1 + p2) / 2
    diff = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-10:
        return

    direction = diff / length
    z_axis = np.array([0, 0, 1])

    if np.abs(np.dot(direction, z_axis)) > 0.999:
        mat = np.eye(3) if np.dot(direction, z_axis) > 0 else np.diag([1, -1, -1]).astype(float)
    else:
        v = np.cross(z_axis, direction)
        v = v / np.linalg.norm(v)
        c = np.dot(z_axis, direction)
        s = np.sqrt(1 - c * c)
        mat = np.array([
            [c + v[0] * v[0] * (1 - c),      v[0] * v[1] * (1 - c) - v[2] * s, v[0] * v[2] * (1 - c) + v[1] * s],
            [v[1] * v[0] * (1 - c) + v[2] * s, c + v[1] * v[1] * (1 - c),      v[1] * v[2] * (1 - c) - v[0] * s],
            [v[2] * v[0] * (1 - c) - v[1] * s, v[2] * v[1] * (1 - c) + v[0] * s, c + v[2] * v[2] * (1 - c)]
        ])

    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, length / 2, 0]),
        mid,
        mat.flatten(),
        rgba
    )
    scn.ngeom += 1


def main():
    # ------------------------------------------
    # 1. 加载模型与参数提取
    # ------------------------------------------
    model = mujoco.MjModel.from_xml_path(r"robots\husky\husky.xml")
    data = mujoco.MjData(model)

    # --- 底盘 body / joint ---
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    wheel_names = ['front_left_wheel_link', 'front_right_wheel_link',
                   'rear_left_wheel_link', 'rear_right_wheel_link']
    joint_names = ['front_left_wheel', 'front_right_wheel',
                   'rear_left_wheel', 'rear_right_wheel']

    wheel_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in wheel_names]
    wheel_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    wheel_dof_adrs = [model.jnt_dofadr[j] for j in wheel_joint_ids]

    # --- 几何参数: 轮距 / 轮半径 ---
    pos_fl = model.body_pos[wheel_body_ids[0]]
    pos_fr = model.body_pos[wheel_body_ids[1]]
    L_wheel = abs(pos_fl[1] - pos_fr[1])
    d_car = L_wheel / 2.0
    fl_geom_id = model.body_geomadr[wheel_body_ids[0]]
    r = model.geom_size[fl_geom_id][0]

    # --- 底盘自由关节 qpos/qvel 地址 ---
    base_jnt_id = model.body_jntadr[base_id]
    base_qpos_adr = model.jnt_qposadr[base_jnt_id]
    base_dof_adr = model.jnt_dofadr[base_jnt_id]

    # ------------------------------------------
    # 2. 执行器类型自动检测 (关键!)
    # ------------------------------------------
    # velocity 伺服特征: biastype=AFFINE 且 biasprm[2] = -kv < 0
    IS_VELOCITY_SERVO = (model.actuator_biastype[0] == int(mujoco.mjtBias.mjBIAS_AFFINE)
                         and model.actuator_biasprm[0][2] < 0)
    kv_servo = abs(model.actuator_biasprm[0][2]) if IS_VELOCITY_SERVO else 0.0

    ctrl_lo = model.actuator_ctrlrange[:, 0] if model.actuator_ctrllimited[0] else -np.inf
    ctrl_hi = model.actuator_ctrlrange[:, 1] if model.actuator_ctrllimited[0] else np.inf

    print("=" * 60)
    if IS_VELOCITY_SERVO:
        print(f"执行器模式: VELOCITY伺服 (ctrl=目标轮速, kv={kv_servo:.1f})")
    else:
        print("执行器模式: MOTOR/力矩 (ctrl=力矩)")
    print(f"轮半径 r={r:.4f} m, 半轮距 d={d_car:.4f} m")
    print("=" * 60)

    # ------------------------------------------
    # 3. 控制器参数 (外环PID + 速度制导 + 内环角速度)
    # ------------------------------------------
    # 外环位置 PID (世界系)
    Kp_pos, Ki_pos, Kd_pos = 5.0, 0.2, 1.0
    I_limit = 10.0

    # 速度指令层
    Kv_fwd = 1.0      # 前向误差 -> 线速度增益
    v_max = 2.0       # 线速度上限 (m/s), 先低速调通再加
    w_max = 2.0       # 角速度上限

    # 内环: 角速度 P/D
    Kp_att = 3.0
    Kd_att = 0.3

    # motor 模式下的本地速度环
    kv_pd = 30.0
    tau_max = 100.0   # 每个轮子的力矩限幅

    # ------------------------------------------
    # 4. 仿真设置与参考轨迹
    # ------------------------------------------
    duration = 60.0
    dt = model.opt.timestep

    radius = 8.0
    omega_traj = 0.25        # 线速度 = radius * omega = 2 m/s
    center = np.array([-radius, 0.0])

    # PID / 制导层状态变量
    Ix = 0.0
    Iy = 0.0
    phi_d_prev = 0.0

    # 数据记录 (预分配用 list, 结束后转 array)
    history = {k: [] for k in
               ['t', 'x', 'y', 'phi', 'xd', 'yd', 'phid',
                'err_x', 'err_y', 'err_norm', 'v_cmd', 'w_cmd']}

    # 预生成用于 Viewer 显示的参考轨迹点
    traj_vis_points = []
    for th in np.linspace(0, 2 * np.pi, 73):
        traj_vis_points.append(np.array([
            center[0] + radius * np.cos(th),
            center[1] + radius * np.sin(th),
            0.2           # 显示高度
        ]))

    # ------------------------------------------
    # 5. 启动可视化与控制循环
    # ------------------------------------------
    print("Launching MuJoCo Viewer (PID + Velocity Guidance)...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_step = 0

        while viewer.is_running() and data.time < duration:
            t = data.time

            # ==================================================
            # 1. 目标轨迹 (世界坐标系)
            # ==================================================
            x_d = center[0] + radius * np.cos(omega_traj * t)
            y_d = center[1] + radius * np.sin(omega_traj * t)

            vx_d = -radius * omega_traj * np.sin(omega_traj * t)
            vy_d = radius * omega_traj * np.cos(omega_traj * t)
            omega_d = omega_traj

            # ==================================================
            # 2. 状态反馈
            # ==================================================
            pos, vel, phi = get_state_from_mjdata(
                model, data, base_id, base_qpos_adr, base_dof_adr)
            x, y = pos[0], pos[1]
            vx = vel[0]
            vy = vel[1]
            wz = vel[5]

            # ==================================================
            # 3. 外环位置 PID (世界系)
            # ==================================================
            ex = x_d - x
            ey = y_d - y

            # 积分与抗饱和
            Ix = max(min(Ix + ex * dt, I_limit), -I_limit)
            Iy = max(min(Iy + ey * dt, I_limit), -I_limit)

            # 微分用 (参考速度 - 实际速度) 代替差分, 避免物理步长噪声
            Dex = vx_d - vx
            Dey = vy_d - vy

            u_Gx = Kp_pos * ex + Ki_pos * Ix + Kd_pos * Dex
            u_Gy = Kp_pos * ey + Ki_pos * Iy + Kd_pos * Dey

            # ==================================================
            # 4. 速度制导层
            # ==================================================
            # 方向: 期望航向 = u_G 的方向 (u_G 过小时保持上一帧, 防 atan2 跳变)
            if math.hypot(u_Gx, u_Gy) > 1e-3:
                phi_d = math.atan2(u_Gy, u_Gx)
            else:
                phi_d = phi_d_prev
            phi_d_prev = phi_d

            ephi = math.atan2(math.sin(phi_d - phi), math.cos(phi_d - phi))

            # 大小: 车体系前向误差 P 控制, 航向偏差大时先减速转向
            err_fwd = ex * np.cos(phi) + ey * np.sin(phi)
            v_cmd = max(min(Kv_fwd * err_fwd, v_max), -v_max)
            v_cmd *= max(math.cos(ephi), 0.0)

            # 角速度指令: P + D (用真实 wz) + 圆弧曲率前馈 w_ff = v/R
            w_ff = v_cmd / radius
            w_cmd = Kp_att * ephi - Kd_att * wz + w_ff
            w_cmd = max(min(w_cmd, w_max), -w_max)

            # ==================================================
            # 5. 差速运动学逆解 (v_cmd, w_cmd -> 左右轮角速度)
            # ==================================================
            v_L = v_cmd - w_cmd * d_car
            v_R = v_cmd + w_cmd * d_car
            w_L = v_L / r
            w_R = v_R / r

            # ==================================================
            # 6. 执行器写入 (双模式)
            # ==================================================
            w_wheel_cmds = [w_L, w_R, w_L, w_R]
            if IS_VELOCITY_SERVO:
                # velocity 伺服: 直接命令轮角速度,
                # 底层 kv 伺服自动克服摩擦/惯量
                for i, wc in enumerate(w_wheel_cmds):
                    data.ctrl[i] = max(min(wc, ctrl_hi[i]), ctrl_lo[i])
            else:
                # motor 模式: 本地 PD 速度环产生力矩
                for i, wc in enumerate(w_wheel_cmds):
                    w_meas = data.qvel[wheel_dof_adrs[i]]
                    tau = kv_pd * (wc - w_meas)
                    data.ctrl[i] = max(min(tau, tau_max), -tau_max)

            # ==================================================
            # 7. 步进仿真
            # ==================================================
            mujoco.mj_step(model, data)

            # ==================================================
            # 8. 数据记录
            # ==================================================
            history['t'].append(t)
            history['x'].append(x)
            history['y'].append(y)
            history['phi'].append(phi)
            history['xd'].append(x_d)
            history['yd'].append(y_d)
            history['phid'].append(phi_d)
            history['err_x'].append(ex)
            history['err_y'].append(ey)
            history['err_norm'].append(math.hypot(ex, ey))
            history['v_cmd'].append(v_cmd)
            history['w_cmd'].append(w_cmd)

            # ==================================================
            # 9. 绘制参考轨迹和当前目标点 (user_scn)
            # ==================================================
            viewer.user_scn.ngeom = 0

            # 参考轨迹圆 (半透明红色)
            for i in range(len(traj_vis_points) - 1):
                add_line(viewer.user_scn,
                         traj_vis_points[i],
                         traj_vis_points[i + 1],
                         0.02,
                         np.array([1.0, 0.0, 0.0, 0.5]))

            # 当前目标点 (黄色球)
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.1, 0, 0]),
                    np.array([x_d, y_d, 0.2]),
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 0.0, 1.0]))
                viewer.user_scn.ngeom += 1

            # ==================================================
            # 10. 同步可视化与实时控制
            # ==================================================
            viewer.sync()

            expected_time = start_time + simulation_step * dt
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            simulation_step += 1

    # ------------------------------------------
    # 6. 结果绘图 (仿真结束后)
    # ------------------------------------------
    print("Simulation finished. Plotting results...")

    t_arr = np.array(history['t'])
    fig = plt.figure(figsize=(12, 9))

    # 轨迹跟踪
    plt.subplot(2, 2, 1)
    plt.plot(history['x'], history['y'], 'b', label='Actual')
    plt.plot(history['xd'], history['yd'], 'r--', label='Desired')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking (PID + Velocity Guidance)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 航向角跟踪
    plt.subplot(2, 2, 2)
    plt.plot(t_arr, history['phi'], 'b', label='Actual')
    plt.plot(t_arr, history['phid'], 'r--', label='Desired (u_G dir)')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.title('Heading')
    plt.legend()
    plt.grid(True)

    # 跟踪误差
    plt.subplot(2, 2, 3)
    plt.plot(t_arr, history['err_x'], 'r', label='err X')
    plt.plot(t_arr, history['err_y'], 'b', label='err Y')
    plt.plot(t_arr, history['err_norm'], 'k', label='norm')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Tracking Error')
    plt.legend()
    plt.grid(True)

    # 速度指令
    plt.subplot(2, 2, 4)
    plt.plot(t_arr, history['v_cmd'], 'g', label='v_cmd (m/s)')
    plt.plot(t_arr, history['w_cmd'], 'm', label='w_cmd (rad/s)')
    plt.xlabel('Time (s)')
    plt.title('Guidance Commands')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
