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
    """返回: 世界系位置(3), 世界系速度(6), 航向角"""
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
    # 0. 测试模式开关
    # ------------------------------------------
    USE_STRAIGHT_TEST = False

    # ------------------------------------------
    # 1. 加载模型与参数提取
    # ------------------------------------------
    model = mujoco.MjModel.from_xml_path(r"robots\husky\husky.xml")
    data = mujoco.MjData(model)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_mass = model.body_mass[base_id]
    Izz = model.body_inertia[base_id][2]

    wheel_names = ['front_left_wheel_link', 'front_right_wheel_link',
                   'rear_left_wheel_link', 'rear_right_wheel_link']
    joint_names = ['front_left_wheel', 'front_right_wheel',
                   'rear_left_wheel', 'rear_right_wheel']

    wheel_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in wheel_names]
    wheel_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    wheel_dof_adrs = [model.jnt_dofadr[j] for j in wheel_joint_ids]

    pos_fl = model.body_pos[wheel_body_ids[0]]
    pos_fr = model.body_pos[wheel_body_ids[1]]
    pos_bl = model.body_pos[wheel_body_ids[2]]
    pos_br = model.body_pos[wheel_body_ids[3]]

    a = np.mean([np.abs(pos_fl[0]), np.abs(pos_fr[0]),
                 np.abs(pos_bl[0]), np.abs(pos_br[0])])
    b = np.mean([np.abs(pos_fl[1]), np.abs(pos_fr[1]),
                 np.abs(pos_bl[1]), np.abs(pos_br[1])])
    d_car = abs(pos_fl[1] - pos_fr[1]) / 2.0

    fl_geom_id = model.body_geomadr[wheel_body_ids[0]]
    r = model.geom_size[fl_geom_id][0]

    base_jnt_id = model.body_jntadr[base_id]
    base_qpos_adr = model.jnt_qposadr[base_jnt_id]
    base_dof_adr = model.jnt_dofadr[base_jnt_id]

    # ------------------------------------------
    # 2. 执行器类型检测
    # ------------------------------------------
    IS_VELOCITY_SERVO = (model.actuator_biastype[0] == int(mujoco.mjtBias.mjBIAS_AFFINE)
                         and model.actuator_biasprm[0][2] < 0)
    kv_servo = abs(model.actuator_biasprm[0][2]) if IS_VELOCITY_SERVO else 0.0

    ctrl_lo = model.actuator_ctrlrange[:, 0] if model.actuator_ctrllimited[0] else -np.inf
    ctrl_hi = model.actuator_ctrlrange[:, 1] if model.actuator_ctrllimited[0] else np.inf

    # 轮级速度伺服增益
    kv_pd = 30.0
    tau_max = 100.0

    print("=" * 60)
    if IS_VELOCITY_SERVO:
        print(f"执行器模式: VELOCITY伺服 (kv={kv_servo:.1f}), 运动学指令模式")
    else:
        print("执行器模式: MOTOR/力矩, 轮级速度伺服 + DSC动力学前馈")
    print(f"Params: Mass={base_mass:.1f}kg, Izz={Izz:.2f}, a={a:.2f}, b={b:.2f}")
    print(f"轮半径 r={r:.4f} m, 半轮距 d={d_car:.4f} m")
    print("=" * 60)

    # ------------------------------------------
    # 3. DSC 控制器参数
    # ------------------------------------------
    k1x = 2.0        # 第一面增益 (前向位置通道)
    k1i = 0.4
    k1d = 1.0
    k1_y = 3.0      # 第一面增益 (横向位置通道)
    I_limit = 8.0     # 积分限幅 (稳态需求 = 2.0/0.5 = 4, 留一倍余量)
    k1_phi = 5.0    # 第一面增益 (航向通道) — 转弯刚度, 可独立加大
    k2 = 50.0       # 第二面增益 (速度面)
    tau = 0.1      # 一阶低通滤波时间常数

    # 滤波器状态 (动态面)
    alpha_f_x = 0.0       # 前向期望速度 (本体系, 滤波后)
    alpha_f_y = 0.0       # 横向期望速度 (本体系, 仅用于构建航向)
    alpha_f_phi = 0.0     # 期望角速度 (滤波后)
    ephi_prev = 0.0
    I_zx = 0.0        # z1_x 的积分器


    # ------------------------------------------
    # 4. 等效惯量 (DSC 动力学前馈用)
    # ------------------------------------------
    wheel_mass_est = 1.0
    Iw = 0.5 * wheel_mass_est * r ** 2
    M11 = base_mass + 2 * Iw / r ** 2
    M33 = Izz + 2 * Iw * (a ** 2 + b ** 2) / r ** 2

    # ------------------------------------------
    # 5. 仿真设置与参考轨迹
    # ------------------------------------------
    duration = 60.0
    dt = model.opt.timestep

    radius = 8.0
    omega_traj = 0.25
    center = np.array([-radius, 0.0])

    history = {k: [] for k in
               ['t', 'x', 'y', 'phi', 'phid_eff',
                'err_fwd', 'err_lat', 'err_norm', 'ephi',
                'v_ref', 'w_cmd', 'wz', 'z2_x', 'z2_phi',
                'tau_L', 'tau_R']}

    if USE_STRAIGHT_TEST:
        traj_vis_points = [np.array([k, 0.0, 0.2]) for k in np.linspace(0, 30, 61)]
    else:
        traj_vis_points = []
        for th in np.linspace(0, 2 * np.pi, 73):
            traj_vis_points.append(np.array([
                center[0] + radius * np.cos(th),
                center[1] + radius * np.sin(th),
                0.2
            ]))

    # ------------------------------------------
    # 6. 启动可视化与控制循环
    # ------------------------------------------
    mode_str = "Straight-Line Test" if USE_STRAIGHT_TEST else "Circle Trajectory"
    print(f"Launching MuJoCo Viewer (DSC + Wheel-Servo, Husky) [{mode_str}]...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_step = 0

        while viewer.is_running() and data.time < duration:
            t = data.time

            # ==================================================
            # 1. 目标轨迹 (世界系)
            # ==================================================
            if USE_STRAIGHT_TEST:
                x_d = 10.0 * omega_traj * t
                y_d = 0.0
                vx_d = 10.0 * omega_traj
                vy_d = 0.0
            else:
                x_d = center[0] + radius * np.cos(omega_traj * t)
                y_d = center[1] + radius * np.sin(omega_traj * t)
                vx_d = -radius * omega_traj * np.sin(omega_traj * t)
                vy_d = radius * omega_traj * np.cos(omega_traj * t)

            # ==================================================
            # 2. 状态反馈
            # ==================================================
            pos, vel, phi = get_state_from_mjdata(model, data, base_qpos_adr, base_dof_adr)
            x, y = pos[0], pos[1]
            vx_world = vel[0]
            vy_world = vel[1]
            omega = vel[5]

            c, s = math.cos(phi), math.sin(phi)
            vx_body = vx_world * c + vy_world * s
            vy_body = -vx_world * s + vy_world * c

            # ==================================================
            # 3. 本体系误差与前馈
            # ==================================================
            ex = x_d - x
            ey = y_d - y
            exb = ex * c + ey * s
            eyb = -ex * s + ey * c

            vx_ff = vx_d * c + vy_d * s
            vy_ff = -vx_d * s + vy_d * c

            # ==================================================
            # 4. DSC 第一面 (x, y 通道)
            # ==================================================
            z1_x = -exb
            z1_y = -eyb

            D_exb = (vx_d * c + vy_d * s - vx_body) + eyb * omega

            I_zx = max(min(I_zx + exb * dt, I_limit), -I_limit)
            alpha_x = -k1x * z1_x - k1i * I_zx + k1d * D_exb + vx_ff   # = k1*exb + k1i*I + k1d*ė



            alpha_y = -k1_y * z1_y + vy_ff

            alpha_f_x_dot = -(alpha_f_x - alpha_x) / tau
            alpha_f_y_dot = -(alpha_f_y - alpha_y) / tau
            alpha_f_x += alpha_f_x_dot * dt
            alpha_f_y += alpha_f_y_dot * dt

            # ==================================================
            # 5. ★ 期望航向角: 由滤波虚拟控制量构建 ★
            # ==================================================
            if math.hypot(alpha_f_x, alpha_f_y) > 1e-3:
                ephi = math.atan2(alpha_f_y, alpha_f_x)
            else:
                ephi = ephi_prev
            ephi_prev = ephi

            # ==================================================
            # 6. ★ 前向速度调度 (旧版遗漏, 关键修正之一) ★
            #    大航向偏差时前向指令归零, 先转向后前进;
            #    调度作用于滤波面本身, 保证第二面/z2/前馈一致
            # ==================================================
            sched = max(math.cos(ephi), 0.0)
            v_ref = alpha_f_x * sched          # 调度后的前向期望速度

            # 曲率前馈 (滤波面 / 半径)
            omega_ff = 0.0 if USE_STRAIGHT_TEST else alpha_f_x / radius

            # ==================================================
            # 7. DSC 第一面 (phi 通道) + 第二面
            # ==================================================
            z1_phi = -ephi
            alpha_phi = -k1_phi * z1_phi + omega_ff   # = k1_phi*ephi + omega_ff

            alpha_f_phi_dot = -(alpha_f_phi - alpha_phi) / tau
            alpha_f_phi += alpha_f_phi_dot * dt

            # 第二面: 速度面误差 (前向参考用【调度后】的 v_ref, 保证一致性)
            z2_x = vx_body - v_ref
            z2_y = vy_body - alpha_f_y          # 差速车无横向执行器, 仅诊断
            z2_phi = omega - alpha_f_phi

            ax_d = -k2 * z2_x + alpha_f_x_dot - z1_x
            aphi_d = -k2 * z2_phi + alpha_f_phi_dot - z1_phi

            # ==================================================
            # 8. DSC 动力学前馈 -> 广义力
            # ==================================================
            C_x = -base_mass * omega * vy_body   # 差速车 vy≈0, 量值极小
            F_x = M11 * ax_d + C_x
            T_z = M33 * aphi_d

            tau_ff_common = F_x * r / 4.0          # 共模前馈 (每轮)
            tau_ff_diff = T_z * r / (4.0 * d_car)  # 差模前馈

            # ==================================================
            # 9. ★ 执行器写入: 轮级速度伺服 + DSC前馈 ★
            #    旧版失败根源: 纯 M33 折算力矩 (~N·m级) 沉没在
            #    滑移转向摩擦死区 (O(100 N·m)) 中 -> 永不转弯。
            #    现恢复 EB 版验证有效的高增益轮速环提供转弯权限,
            #    DSC 输出降级为前馈补偿。
            # ==================================================
            v_L = v_ref - alpha_f_phi * d_car
            v_R = v_ref + alpha_f_phi * d_car
            w_L_cmd = v_L / r
            w_R_cmd = v_R / r

            tau_L = tau_R = 0.0
            if IS_VELOCITY_SERVO:
                w_wheel_cmds = [w_L_cmd, w_R_cmd, w_L_cmd, w_R_cmd]
                for i, wc in enumerate(w_wheel_cmds):
                    data.ctrl[i] = max(min(wc, ctrl_hi[i]), ctrl_lo[i])
            else:
                w_meas_L = 0.5 * (data.qvel[wheel_dof_adrs[0]] + data.qvel[wheel_dof_adrs[2]])
                w_meas_R = 0.5 * (data.qvel[wheel_dof_adrs[1]] + data.qvel[wheel_dof_adrs[3]])

                # 轮级速度伺服 (高增益主通道) + DSC 动力学前馈
                tau_L = kv_pd * (w_L_cmd - w_meas_L) + tau_ff_common - tau_ff_diff
                tau_R = kv_pd * (w_R_cmd - w_meas_R) + tau_ff_common + tau_ff_diff

                w_wheel_cmds = [w_L_cmd, w_R_cmd, w_L_cmd, w_R_cmd]
                tau_wheel = [tau_L, tau_R, tau_L, tau_R]
                for i in range(4):
                    tq = max(min(tau_wheel[i], tau_max), -tau_max)
                    data.ctrl[i] = tq
                tau_L, tau_R = tau_wheel[0], tau_wheel[1]

            # ==================================================
            # 10. 步进仿真
            # ==================================================
            mujoco.mj_step(model, data)

            # ==================================================
            # 11. 数据记录
            # ==================================================
            history['t'].append(t)
            history['x'].append(x)
            history['y'].append(y)
            history['phi'].append(phi)
            history['phid_eff'].append(phi + ephi)
            history['err_fwd'].append(exb)
            history['err_lat'].append(eyb)
            history['err_norm'].append(math.hypot(exb, eyb))
            history['ephi'].append(ephi)
            history['v_ref'].append(v_ref)
            history['w_cmd'].append(alpha_f_phi)
            history['wz'].append(omega)
            history['z2_x'].append(z2_x)
            history['z2_phi'].append(z2_phi)
            history['tau_L'].append(tau_L)
            history['tau_R'].append(tau_R)

            # ==================================================
            # 12. 绘制参考轨迹和当前目标点
            # ==================================================
            viewer.user_scn.ngeom = 0

            for i in range(len(traj_vis_points) - 1):
                add_line(viewer.user_scn,
                         traj_vis_points[i],
                         traj_vis_points[i + 1],
                         0.02,
                         np.array([1.0, 0.0, 0.0, 0.5]))

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
            # 13. 同步可视化与实时控制
            # ==================================================
            viewer.sync()

            expected_time = start_time + simulation_step * dt
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            simulation_step += 1

    # ------------------------------------------
    # 7. 结果绘图与诊断输出
    # ------------------------------------------
    print("Simulation finished. Plotting results...")

    t_arr = np.array(history['t'])
    ephi_arr = np.array(history['ephi'])
    err_norm_arr = np.array(history['err_norm'])
    tauL_arr = np.abs(np.array(history['tau_L']))
    tauR_arr = np.abs(np.array(history['tau_R']))

    ss_mask = t_arr > (t_arr[-1] - 10.0)
    print("=" * 60)
    print(f"稳态航向偏移 |ephi| (后10s均值): "
          f"{np.degrees(np.abs(ephi_arr[ss_mask]).mean()):.2f} deg")
    print(f"稳态位置误差 norm (后10s均值): {err_norm_arr[ss_mask].mean():.4f} m")
    print(f"稳态偏航速率跟踪误差 |w_cmd - wz| (后10s均值): "
          f"{np.abs(np.array(history['w_cmd'])[ss_mask] - np.array(history['wz'])[ss_mask]).mean():.4f} rad/s")
    print(f"峰值轮矩: L={tauL_arr.max():.1f}, R={tauR_arr.max():.1f} Nm / 限幅 {tau_max}")
    if max(tauL_arr.max(), tauR_arr.max()) >= tau_max - 1.0:
        print("⚠ 力矩饱和! 降低轨迹速度或提高 tau_max")
    print("=" * 60)

    fig = plt.figure(figsize=(12, 9))

    # 轨迹跟踪
    plt.subplot(2, 3, 1)
    plt.plot(history['x'], history['y'], 'b', label='Actual')
    if USE_STRAIGHT_TEST:
        plt.plot([0, 30], [0, 0], 'r--', label='Desired')
    else:
        th = np.linspace(0, 2 * np.pi, 200)
        plt.plot(center[0] + radius * np.cos(th),
                 center[1] + radius * np.sin(th), 'r--', label='Desired')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking (DSC + Wheel Servo)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 航向角跟踪
    plt.subplot(2, 3, 2)
    plt.plot(t_arr, history['phi'], 'b', label='Actual yaw')
    plt.plot(t_arr, history['phid_eff'], 'r--', label='phi + ephi (from alpha_f)')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.title('Heading (bearing from filtered surface)')
    plt.legend()
    plt.grid(True)

    # 车体系误差
    plt.subplot(2, 3, 3)
    plt.plot(t_arr, history['err_fwd'], 'r', label='err fwd (exb)')
    plt.plot(t_arr, history['err_lat'], 'b', label='err lat (eyb)')
    plt.plot(t_arr, history['err_norm'], 'k', label='norm')
    plt.xlabel('Time (s)')
    plt.ylabel('Body-frame Error (m)')
    plt.title('Body-Frame Tracking Error')
    plt.legend()
    plt.grid(True)

    # 航向偏移 + 指令
    plt.subplot(2, 3, 4)
    plt.plot(t_arr, np.degrees(history['ephi']), 'c', label='ephi (deg)')
    plt.plot(t_arr, history['v_ref'], 'g', label='v_ref (m/s)')
    plt.xlabel('Time (s)')
    plt.title('Heading Offset & Scheduled Forward Cmd')
    plt.legend()
    plt.grid(True)

    # ★ 偏航速率跟踪: 转弯是否生效的核心指标 ★
    plt.subplot(2, 3, 5)
    plt.plot(t_arr, history['w_cmd'], 'm', label='alpha_f_phi (cmd)')
    plt.plot(t_arr, history['wz'], 'k--', label='omega (meas)')
    plt.xlabel('Time (s)')
    plt.ylabel('rad/s')
    plt.title('Yaw Rate: cmd vs meas (turn authority)')
    plt.legend()
    plt.grid(True)

    # 左右轮矩
    plt.subplot(2, 3, 6)
    plt.plot(t_arr, history['tau_L'], 'b', label='tau_L')
    plt.plot(t_arr, history['tau_R'], 'r', label='tau_R')
    plt.axhline(tau_max, color='k', linestyle='--')
    plt.axhline(-tau_max, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Wheel Torque (Nm)')
    plt.title('L/R Wheel Torque (differential = turning)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
