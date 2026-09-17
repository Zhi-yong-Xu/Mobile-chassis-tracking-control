import mujoco
import numpy as np
import mujoco.viewer
import matplotlib.pyplot as plt
import time


def get_state_from_mjdata(model, data):
    """
    从MuJoCo数据对象中提取当前状态
    返回: 世界坐标系下的位置, 速度, 航向角, 航向角速度
    """
    pos = data.qpos[:3].copy()
    vel = data.qvel[:6].copy()
    # 航向角 Phi (从四元数转换)
    quat = data.qpos[3:7]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    R = mat.reshape(3, 3)
    phi = np.arctan2(R[1, 0], R[0, 0])
    return pos, vel, phi


def add_line(scn, p1, p2, radius, rgba):
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
            [c + v[0] * v[0] * (1 - c),     v[0] * v[1] * (1 - c) - v[2] * s, v[0] * v[2] * (1 - c) + v[1] * s],
            [v[1] * v[0] * (1 - c) + v[2] * s, c + v[1] * v[1] * (1 - c),     v[1] * v[2] * (1 - c) - v[0] * s],
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
    model = mujoco.MjModel.from_xml_path(r"robots\summit_xl_description\McNam.xml")
    data = mujoco.MjData(model)

    # --- 参数提取 ---
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_mass = model.body_mass[base_id]
    Izz = model.body_inertia[base_id][2]

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "front_right_wheel_intermediate_link_geom_0")
    wheel_radius = model.geom_size[geom_id][0]

    def get_body_pos(body_name):
        _id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return model.body_pos[_id].copy()

    pos_fl = get_body_pos("front_left_wheel_intermediate_link")
    pos_fr = get_body_pos("front_right_wheel_intermediate_link")
    pos_bl = get_body_pos("back_left_wheel_intermediate_link")
    pos_br = get_body_pos("back_right_wheel_intermediate_link")

    a = np.mean([np.abs(pos_fl[0]), np.abs(pos_fr[0]),
                 np.abs(pos_bl[0]), np.abs(pos_br[0])])
    b = np.mean([np.abs(pos_fl[1]), np.abs(pos_fr[1]),
                 np.abs(pos_bl[1]), np.abs(pos_br[1])])

    wheel_mass_est = 1.0
    Iw = 0.5 * wheel_mass_est * wheel_radius ** 2
    print(f"Params: Mass={base_mass:.1f}kg, Izz={Izz:.2f}, a={a:.2f}, b={b:.2f}")

    # ------------------------------------------
    # 2. DSC控制器参数 (参照 DSC.m)
    # ------------------------------------------
    # DSC核心参数: k1=位置面增益, k2=速度面增益, tau=滤波器时间常数
    k1 = 5.0      # 第一面增益 (位置跟踪)
    k2 = 10.0     # 第二面增益 (速度跟踪)
    tau = 0.02    # 一阶低通滤波器时间常数

    # DSC滤波器状态变量: alpha_f (滤波后的虚拟控制量, 即期望速度)
    alpha_f_x = 0.0      # x轴滤波虚拟控制
    alpha_f_y = 0.0      # y轴滤波虚拟控制
    alpha_f_phi = 0.0    # phi轴滤波虚拟控制

    # ------------------------------------------
    # 3. 仿真设置
    # ------------------------------------------
    duration = 50.0
    dt = model.opt.timestep

    # 数据记录
    history_time = []
    history_x = []
    history_y = []
    history_phi = []
    history_xd = []
    history_yd = []
    history_phid = []

    radius = 5
    radius = 5
    omega_traj = 0.3

    # 预生成用于 Viewer 显示的参考轨迹点
    traj_vis_points = []
    for th in np.linspace(0, 2 * np.pi, 73):
        traj_vis_points.append(np.array([
            radius * np.cos(th) - radius,
            radius * np.sin(th),
            0.7          # 显示高度，可按机器人 base 高度调整
        ]))

    # ------------------------------------------
    # 4. 惯性矩阵 (用于力/力矩计算)
    # ------------------------------------------
    r = wheel_radius
    M_mat = np.diag([
        base_mass + 2 * Iw / r ** 2,
        base_mass + 2 * Iw / r ** 2,
        Izz + 2 * Iw * (a ** 2 + b ** 2) / r ** 2
    ])

    # ------------------------------------------
    # 5. 启动可视化与控制循环
    # ------------------------------------------
    print("Launching MuJoCo Viewer (DSC Control)...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        simulation_step = 0

        while data.time < duration:
            # --- 实时控制逻辑 ---
            t = data.time

            # ==================================================
            # 1. 目标轨迹 (世界坐标系)
            # ==================================================
            x_d = radius * np.cos(omega_traj * t) - radius
            y_d = radius * np.sin(omega_traj * t)
            phi_d = omega_traj * t + np.pi / 2
            phi_d = np.arctan2(np.sin(phi_d), np.cos(phi_d))  # 归一化

            # 目标轨迹一阶导数 (期望速度, 世界坐标系)
            vx_d_world = -radius * omega_traj * np.sin(omega_traj * t)
            vy_d_world = radius * omega_traj * np.cos(omega_traj * t)
            omega_d = omega_traj  # 期望角速度 (phi_d的导数)

            # 目标轨迹二阶导数 (期望加速度, 世界坐标系)
            ax_d_world = -radius * omega_traj ** 2 * np.cos(omega_traj * t)
            ay_d_world = -radius * omega_traj ** 2 * np.sin(omega_traj * t)

            # ==================================================
            # 2. 状态反馈
            # ==================================================
            pos, vel_world, phi = get_state_from_mjdata(model, data)
            x, y = pos[0], pos[1]
            vx_world = vel_world[0]
            vy_world = vel_world[1]
            omega = vel_world[5]

            # 世界速度 -> 本体速度
            c, s = np.cos(phi), np.sin(phi)
            R_world2body = np.array([[c, s], [-s, c]])
            v_body = R_world2body @ np.array([vx_world, vy_world])
            vx_body = v_body[0]
            vy_body = v_body[1]

            # ==================================================
            # 3. 位置误差计算
            # ==================================================
            ex = x_d - x
            ey = y_d - y

            # 位置误差转换到本体坐标系
            exb = ex * np.cos(phi) + ey * np.sin(phi)
            eyb = -ex * np.sin(phi) + ey * np.cos(phi)

            # 航向角误差 (归一化到[-pi, pi])
            ephi = phi_d - phi
            ephi = np.arctan2(np.sin(ephi), np.cos(ephi))

            # ==================================================
            # 4. 前馈量: 期望速度/加速度转换到本体坐标系
            # ==================================================
            vx_ff = vx_d_world * np.cos(phi) + vy_d_world * np.sin(phi)
            vy_ff = -vx_d_world * np.sin(phi) + vy_d_world * np.cos(phi)
            omega_ff = omega_d

            # ==================================================
            # 5. DSC 动态面控制 (核心算法, 参照 DSC.m)
            # ==================================================
            # --- 第一面: 位置跟踪 (对应DSC.m中的 z1 = x1 - xd) ---
            # z1 = 实际 - 期望 (DSC约定)
            z1_x = -exb       # 本体x方向: 实际位置 - 期望位置
            z1_y = -eyb       # 本体y方向
            z1_phi = -ephi    # 航向角方向

            # 虚拟控制律 alpha = -k1 * z1 + xd_dot (即期望速度)
            alpha_x = -k1 * z1_x + vx_ff
            alpha_y = -k1 * z1_y + vy_ff
            alpha_phi = -k1 * z1_phi + omega_ff

            # 一阶低通滤波器: tau * alpha_f_dot = -(alpha_f - alpha)
            # 滤波器导数直接由滤波方程给出 (避免微分爆炸问题)
            alpha_f_x_dot = -(alpha_f_x - alpha_x) / tau
            alpha_f_y_dot = -(alpha_f_y - alpha_y) / tau
            alpha_f_phi_dot = -(alpha_f_phi - alpha_phi) / tau

            # 欧拉积分更新滤波器状态
            alpha_f_x += alpha_f_x_dot * dt
            alpha_f_y += alpha_f_y_dot * dt
            alpha_f_phi += alpha_f_phi_dot * dt

            # --- 第二面: 速度跟踪 (对应DSC.m中的 z2 = x2 - alpha_f) ---
            # z2 = 实际速度 - 滤波后的期望速度
            z2_x = vx_body - alpha_f_x
            z2_y = vy_body - alpha_f_y
            z2_phi = omega - alpha_f_phi

            # 控制律 u = -k2 * z2 + alpha_f_dot - z1 (即加速度指令)
            # 注: alpha_f_dot 已包含期望轨迹的二阶导数信息 (前馈作用)
            #     -z1 项提供两个动态面之间的耦合阻尼 (Lyapunov稳定性保证)
            ax_d = -k2 * z2_x + alpha_f_x_dot - z1_x
            ay_d = -k2 * z2_y + alpha_f_y_dot - z1_y
            aphi_d = -k2 * z2_phi + alpha_f_phi_dot - z1_phi

            # ==================================================
            # 6. 力/力矩计算 (动力学补偿)
            # ==================================================
            # 科氏力/离心力补偿
            C_force = np.array([
                -base_mass * omega * vy_body,
                base_mass * omega * vx_body,
                0
            ])

            # F = M * a + C  (计算本体坐标系下的广义力)
            F_cmd = M_mat @ np.array([ax_d, ay_d, aphi_d]) + C_force

            # 雅可比伪逆: 将本体力转换为4个轮子的驱动力矩
            J_pinv = (1 / r) * np.array([
                [1,  1,  (a + b)],
                [1, -1, -(a + b)],
                [1, -1,  (a + b)],
                [1,  1, -(a + b)]
            ])

            tau_cmd = J_pinv @ F_cmd
            tau_cmd = np.clip(tau_cmd, -50, 50)
            data.ctrl[0] = tau_cmd[0]
            data.ctrl[1] = tau_cmd[1]
            data.ctrl[2] = tau_cmd[2]
            data.ctrl[3] = tau_cmd[3]

            # ==================================================
            # 7. 步进仿真
            # ==================================================
            mujoco.mj_step(model, data)

            # 轨迹标记点
            try:
                marker_site_id = model.site('trajectory_marker').id
                data.site_xpos[marker_site_id][:2] = np.array([x_d, y_d])
            except KeyError:
                pass

            # ==================================================
            # 8. 数据记录
            # ==================================================
            history_time.append(t)
            history_x.append(x)
            history_y.append(y)
            history_phi.append(phi)
            history_xd.append(x_d)
            history_yd.append(y_d)
            history_phid.append(phi_d)

            # ==================================================
            # 绘制参考轨迹和当前目标点
            # ==================================================
            viewer.user_scn.ngeom = 0

            # 画参考轨迹圆
            for i in range(len(traj_vis_points) - 1):
                add_line(
                    viewer.user_scn,
                    traj_vis_points[i],
                    traj_vis_points[i + 1],
                    0.02,                           # 线半径
                    np.array([1.0, 0.0, 0.0, 0.5])  # 半透明红色
                )

            # 画当前目标点
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.05, 0, 0]),          # 球半径
                    np.array([x_d, y_d, 0.7]),     # 当前位置目标点
                    np.eye(3).flatten(),
                    np.array([1.0, 1.0, 0.0, 1.0])  # 黄色
                )
                viewer.user_scn.ngeom += 1

            # ==================================================
            # 9. 同步可视化与实时控制
            # ==================================================
            viewer.sync()

            # 实时控制: 如果计算过快, 等待
            expected_time = start_time + simulation_step * dt
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            simulation_step += 1

    # ------------------------------------------
    # 6. 结果绘图 (仿真结束后)
    # ------------------------------------------
    print("Simulation finished. Plotting results...")

    plt.figure(figsize=(12, 5))

    # 轨迹跟踪
    plt.subplot(1, 2, 1)
    plt.plot(history_x, history_y, 'b', label='Actual')
    plt.plot(history_xd, history_yd, 'r--', label='Desired')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking (DSC)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 航向角跟踪
    plt.subplot(1, 2, 2)
    plt.plot(history_time, history_phi, 'b', label='Actual')
    plt.plot(history_time, history_phid, 'r--', label='Desired')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.title('Orientation Tracking (DSC)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
