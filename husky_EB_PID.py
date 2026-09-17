import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import matplotlib.pyplot as plt


# ==========================================
# 辅助函数
# ==========================================
def get_state_from_mjdata(model, data, qpos_adr, dof_adr):
    """返回: 世界系位置, 世界系速度(6维), 航向角"""
    pos = data.qpos[qpos_adr:qpos_adr + 3].copy()
    vel = data.qvel[dof_adr:dof_adr + 6].copy()

    quat = data.qpos[qpos_adr + 3:qpos_adr + 7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    R = mat.reshape(3, 3)
    phi = np.arctan2(R[1, 0], R[0, 0])
    return pos, vel, phi


def add_line(scn, p1, p2, radius, rgba):
    """在 viewer 场景中添加一条圆柱线段 (绘制参考轨迹)"""
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
            [c + v[0] * v[0] * (1 - c),        v[0] * v[1] * (1 - c) - v[2] * s,  v[0] * v[2] * (1 - c) + v[1] * s],
            [v[1] * v[0] * (1 - c) + v[2] * s, c + v[1] * v[1] * (1 - c),         v[1] * v[2] * (1 - c) - v[0] * s],
            [v[2] * v[0] * (1 - c) - v[1] * s, v[2] * v[1] * (1 - c) + v[0] * s,  c + v[2] * v[2] * (1 - c)]
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

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    wheel_names = ['front_left_wheel_link', 'front_right_wheel_link',
                   'rear_left_wheel_link', 'rear_right_wheel_link']
    joint_names = ['front_left_wheel', 'front_right_wheel',
                   'rear_left_wheel', 'rear_right_wheel']

    wheel_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in wheel_names]
    wheel_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    wheel_dof_adrs = [model.jnt_dofadr[j] for j in wheel_joint_ids]

    pos_fl = model.body_pos[wheel_body_ids[0]]
    pos_fr = model.body_pos[wheel_body_ids[1]]
    L_wheel = abs(pos_fl[1] - pos_fr[1])
    d_car = L_wheel / 2.0
    fl_geom_id = model.body_geomadr[wheel_body_ids[0]]
    r = model.geom_size[fl_geom_id][0]

    base_jnt_id = model.body_jntadr[base_id]
    base_qpos_adr = model.jnt_qposadr[base_jnt_id]
    base_dof_adr = model.jnt_dofadr[base_jnt_id]

    # ------------------------------------------
    # 2. 执行器类型自动检测
    # ------------------------------------------
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
    # 3. 控制器参数 (全部在车体坐标系下)
    # ------------------------------------------
    # 外环位置 PID (车体系误差 -> 车体系制导向量)
    Kp_pos, Ki_pos, Kd_pos = 5.0, 0.1, 1.0
    I_limit = 10.0

    # 速度指令层
    Kv_fwd = 1.0      # 前向制导分量 -> 线速度增益
    v_max = 2.0
    w_max = 2.0

    # 航向环: 期望航向 = 车体系误差PID向量的方向 (机体系偏航角)
    Kp_att = 3.0      # P: 航向偏移
    Kd_att = 0.3      # D: 偏航速率阻尼 (作用于实测 wz)

    # motor 模式下的本地速度环 (上次遗漏, 已补上)
    kv_pd = 30.0
    tau_max = 100.0

    # 积分器状态 (车体系)
    I_exb = 0.0
    I_eyb = 0.0
    ephi_prev = 0.0   # 航向偏移保持值 (防 atan2 小向量抖动)

    # ------------------------------------------
    # 4. 仿真设置与参考轨迹
    # ------------------------------------------
    duration = 60.0
    dt = model.opt.timestep

    radius = 8.0
    omega_traj = 0.25
    center = np.array([-radius, 0.0])

    history = {k: [] for k in
               ['t', 'x', 'y', 'phi', 'phid_eff',
                'err_fwd', 'err_lat', 'err_norm', 'ephi', 'v_cmd', 'w_cmd']}

    traj_vis_points = []
    for th in np.linspace(0, 2 * np.pi, 73):
        traj_vis_points.append(np.array([
            center[0] + radius * np.cos(th),
            center[1] + radius * np.sin(th),
            0.2
        ]))

    # ------------------------------------------
    # 5. 启动可视化与控制循环
    # ------------------------------------------
    print("Launching MuJoCo Viewer (Body-Frame Error PID, bearing heading)...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_step = 0

        while viewer.is_running() and data.time < duration:
            t = data.time

            # ==================================================
            # 1. 目标轨迹 (世界系生成, 仅作为参考点来源)
            # ==================================================
            x_d = center[0] + radius * np.cos(omega_traj * t)
            y_d = center[1] + radius * np.sin(omega_traj * t)

            vx_d = -radius * omega_traj * np.sin(omega_traj * t)
            vy_d = radius * omega_traj * np.cos(omega_traj * t)

            # ==================================================
            # 2. 状态反馈
            # ==================================================
            pos, vel, phi = get_state_from_mjdata(model, data, base_qpos_adr, base_dof_adr)
            x, y = pos[0], pos[1]
            vx = vel[0]
            vy = vel[1]
            wz = vel[5]

            # ==================================================
            # 3. ★ 车体坐标系误差构造 ★
            # ==================================================
            ex = x_d - x
            ey = y_d - y

            c, s = math.cos(phi), math.sin(phi)

            exb =  ex * c + ey * s       # 前向误差 (车头方向)
            eyb = -ex * s + ey * c       # 横向误差 (车体左侧为正)

            # 参考/实际速度 -> 车体系 (用于误差导数)
            vx_d_b =  vx_d * c + vy_d * s
            vy_d_b = -vx_d * s + vy_d * c
            vx_b   =  vx * c + vy * s
            vy_b   = -vx * s + vy * c    # 差速车理想上 ≈ 0

            # 车体系误差的精确导数 (含旋转耦合项, 替代原 "参考速度-实际速度" 技巧)
            #   d(exb)/dt = (vx_d_b - vx_b) + eyb*wz
            #   d(eyb)/dt = (vy_d_b - vy_b) - exb*wz
            D_exb = (vx_d_b - vx_b) + eyb * wz
            D_eyb = (vy_d_b - vy_b) - exb * wz

            # ==================================================
            # 4. 外环 PID (车体系) -> 车体系制导向量
            # ==================================================
            I_exb = max(min(I_exb + exb * dt, I_limit), -I_limit)
            I_eyb = max(min(I_eyb + eyb * dt, I_limit), -I_limit)

            u_bx = Kp_pos * exb + Ki_pos * I_exb + Kd_pos * D_exb
            u_by = Kp_pos * eyb + Ki_pos * I_eyb + Kd_pos * D_eyb

            # ==================================================
            # 5. ★ 期望航向角: 由车体系误差PID向量方向构造 ★
            #    ephi = atan2(u_by, u_bx) 就是目标相对车头的偏航角,
            #    无需任何世界系 phi_d, 天然归一化在 [-pi, pi]
            # ==================================================
            if math.hypot(u_bx, u_by) > 1e-3:
                ephi = math.atan2(u_by, u_bx)
            else:
                ephi = ephi_prev     # 制导向量过小(已到目标), 保持上一帧
            ephi_prev = ephi

            # ==================================================
            # 6. 速度指令 (车体系)
            # ==================================================
            # 前向速度: 取制导向量的前向分量, 大航向偏差时先减速转向
            v_cmd = max(min(Kv_fwd * u_bx, v_max), -v_max)
            v_cmd *= max(math.cos(ephi), 0.0)

            # 角速度: P(航向偏移) + D(偏航速率阻尼) + 圆弧曲率前馈
            w_ff = v_cmd / radius
            w_cmd = Kp_att * ephi - Kd_att * wz + w_ff
            w_cmd = max(min(w_cmd, w_max), -w_max)

            # ==================================================
            # 7. 差速运动学逆解
            # ==================================================
            v_L = v_cmd - w_cmd * d_car
            v_R = v_cmd + w_cmd * d_car
            w_L = v_L / r
            w_R = v_R / r

            # ==================================================
            # 8. 执行器写入 (双模式)
            # ==================================================
            w_wheel_cmds = [w_L, w_R, w_L, w_R]
            if IS_VELOCITY_SERVO:
                for i, wc in enumerate(w_wheel_cmds):
                    data.ctrl[i] = max(min(wc, ctrl_hi[i]), ctrl_lo[i])
            else:
                for i, wc in enumerate(w_wheel_cmds):
                    w_meas = data.qvel[wheel_dof_adrs[i]]
                    tau = kv_pd * (wc - w_meas)
                    data.ctrl[i] = max(min(tau, tau_max), -tau_max)

            # ==================================================
            # 9. 步进仿真
            # ==================================================
            mujoco.mj_step(model, data)

            # ==================================================
            # 10. 数据记录
            # ==================================================
            history['t'].append(t)
            history['x'].append(x)
            history['y'].append(y)
            history['phi'].append(phi)
            history['phid_eff'].append(phi + ephi)   # 等效期望航向 = 当前航向 + 偏移
            history['err_fwd'].append(exb)
            history['err_lat'].append(eyb)
            history['err_norm'].append(math.hypot(exb, eyb))
            history['ephi'].append(ephi)
            history['v_cmd'].append(v_cmd)
            history['w_cmd'].append(w_cmd)

            # ==================================================
            # 11. 绘制参考轨迹和当前目标点
            # ==================================================
            viewer.user_scn.ngeom = 0

            for i in range(len(traj_vis_points) - 1):
                add_line(viewer.user_scn,
                         traj_vis_points[i],
                         traj_vis_points[i + 1],
                         0.02,
                         np.array([1.0, 0.0, 0.0, 1.0]))

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
            # 12. 同步可视化与实时控制
            # ==================================================
            viewer.sync()

            expected_time = start_time + simulation_step * dt
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            simulation_step += 1

    # ------------------------------------------
    # 6. 结果绘图
    # ------------------------------------------
    print("Simulation finished. Plotting results...")

    t_arr = np.array(history['t'])
    fig = plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plt.plot(history['x'], history['y'], 'b', label='Actual')
    plt.plot(center[0] + radius * np.cos(np.linspace(0, 2 * np.pi, 200)),
             center[1] + radius * np.sin(np.linspace(0, 2 * np.pi, 200)), 'r--', label='Desired')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking (Body-Frame PID)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t_arr, history['phi'], 'b', label='Actual yaw')
    plt.plot(t_arr, history['phid_eff'], 'r--', label='Desired yaw (phi + ephi)')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.title('Heading (bearing from body-frame PID)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t_arr, history['err_fwd'], 'r', label='err fwd (exb)')
    plt.plot(t_arr, history['err_lat'], 'b', label='err lat (eyb)')
    plt.plot(t_arr, history['err_norm'], 'k', label='norm')
    plt.xlabel('Time (s)')
    plt.ylabel('Body-frame Error (m)')
    plt.title('Body-Frame Tracking Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t_arr, history['ephi'], 'c', label='ephi (bearing)')
    plt.plot(t_arr, history['v_cmd'], 'g', label='v_cmd (m/s)')
    plt.plot(t_arr, history['w_cmd'], 'm', label='w_cmd (rad/s)')
    plt.xlabel('Time (s)')
    plt.title('Commands & Heading Offset')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
