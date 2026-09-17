import mujoco
import numpy as np
import mujoco.viewer
import matplotlib.pyplot as plt
import time  # 用于实时控制



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

    data = mujoco.MjData(model)

    # --- 参数提取 ---
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_mass = model.body_mass[base_id]
    
    Ixx = model.body_inertia[base_id][0]
    Iyy = model.body_inertia[base_id][1]
    Izz = model.body_inertia[base_id][2]

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_right_wheel_intermediate_link_geom_0")
    wheel_radius = model.geom_size[geom_id][0]

    def get_body_pos(body_name):
        _id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return model.body_pos[_id].copy()

    pos_fl = get_body_pos("front_left_wheel_intermediate_link")
    pos_fr = get_body_pos("front_right_wheel_intermediate_link")
    pos_bl = get_body_pos("back_left_wheel_intermediate_link")
    pos_br = get_body_pos("back_right_wheel_intermediate_link")

    a = np.mean([np.abs(pos_fl[0]), np.abs(pos_fr[0]), np.abs(pos_bl[0]), np.abs(pos_br[0])])
    b = np.mean([np.abs(pos_fl[1]), np.abs(pos_fr[1]), np.abs(pos_bl[1]), np.abs(pos_br[1])])

    wheel_mass_est = 1.0
    Iw = 0.5 * wheel_mass_est * wheel_radius**2
    
    print(f"Params: Mass={base_mass:.1f}kg, Izz={Izz:.2f}, a={a:.2f}, b={b:.2f}")

    # ------------------------------------------
    # 2. 控制器参数
    # ------------------------------------------
    Kp_pos = 5
    Ki_pos = 0.2
    Kd_pos = 0.1

    # 内环速度环增益
    Kp_vel = 3
    Ki_vel = 0
    Kd_vel = 0.1

    I_limit = 10.0
    
    # 状态变量初始化
    I_xb, I_yb, I_phi = 0, 0, 0
    I_x, I_y = 0, 0
    exb_last, eyb_last, ephi_last = 0, 0, 0
    ex_last, ey_last= 0, 0
    I_vx, I_vy, I_vphi = 0, 0, 0
    evx_last, evy_last, evphi_last = 0, 0, 0

    # ------------------------------------------
    # 3. 仿真设置
    # ------------------------------------------
    duration = 50.0
    dt = model.opt.timestep
    
    # 数据记录 (为了绘图性能，可以降低记录频率，但这里保持全记录)
    history_time = []
    history_x = []
    history_y = []
    history_phi = []
    history_xd = []
    history_yd = []
    history_phid = []

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
    # 4. 雅可比矩阵与惯性矩阵
    # ------------------------------------------
    r = wheel_radius
    J = (r/4.0) * np.array([
        [1, 1, 1, 1],
        [-1, 1, 1, -1],
        [-1/(a+b), 1/(a+b), -1/(a+b), 1/(a+b)]
    ])
    JJT_inv = np.linalg.inv(J @ J.T)
    J_pinv = J.T @ JJT_inv

    M_mat = np.diag([
        base_mass + 2 * Iw / r**2,
        base_mass + 2 * Iw / r**2,
        Izz + 2 * Iw * (a**2 + b**2) / r**2
    ])

    # ------------------------------------------
    # 5. 启动可视化
    # ------------------------------------------
    print("Launching MuJoCo Viewer...")

    # Try importing viewer as a submodule
    with mujoco.viewer.launch_passive(model, data) as viewer:
    
        # 使用 launch_passive 在后台开启窗口，主线程保持控制权
        # with mujoco.viewer.launch_passive(model, data) as viewer:

        # 实时同步变量
        start_time = time.time()
        simulation_step = 0

        # 循环直到仿真结束或窗口关闭
        # while viewer.is_running() and data.time < duration:
        while data.time < duration:
            
            # --- 实时控制逻辑 ---
            t = data.time # 使用仿真时间

            # 1. 目标轨迹

            x_d = radius * np.cos(omega_traj * t) - radius
            y_d = radius * np.sin(omega_traj * t)

            # phi_d = np.arctan2(-np.sin(omega_traj * t), np.cos(omega_traj * t))
            # phi_d = 0.0  # 始终保持朝向为0，或者你可以根据需要设置
            ref_pos = np.array([
                radius * np.cos(omega_traj * t) - radius,
                radius * np.sin(omega_traj * t)
            ])

            vx_d_world = -radius * omega_traj * np.sin(omega_traj * t)
            vy_d_world = radius * omega_traj * np.cos(omega_traj * t)
            
            ax_d_world = -radius * omega_traj**2 * np.cos(omega_traj * t)
            ay_d_world = -radius * omega_traj**2 * np.sin(omega_traj * t)

            # 2. 状态反馈
            pos, vel_world, phi = get_state_from_mjdata(model, data)
            x, y = pos[0], pos[1]
            vx_world = vel_world[0]
            vy_world = vel_world[1]
            omega = vel_world[5]

            c, s = np.cos(phi), np.sin(phi)
            R_world2body = np.array([[c, s], [-s, c]])
            
            v_body = R_world2body @ np.array([vx_world, vy_world])
            vx_body = v_body[0]
            vy_body = v_body[1]

            # 3. 外环位置PID
            ex = x_d - x
            ey = y_d - y

            
            # exb = ex * np.cos(phi) - ey * np.sin(phi)
            
            # eyb = ex * np.sin(phi) + ey *np.cos(phi)
            # 修正后（正确）：
            exb = ex * np.cos(phi) + ey * np.sin(phi)
            eyb = -ex * np.sin(phi) + ey * np.cos(phi)
            # ephi = np.arctan2(np.sin(ephi), np.cos(ephi))

            I_x += ex * dt; I_x = np.clip(I_x, -I_limit, I_limit)
            I_y += ey * dt; I_y = np.clip(I_y, -I_limit, I_limit)

            I_xb += exb * dt; I_xb = np.clip(I_xb, -I_limit, I_limit)
            I_yb += eyb * dt; I_yb = np.clip(I_yb, -I_limit, I_limit)
            

            Dex = (ex - ex_last) / dt; ex_last = ex
            Dey = (ey- ey_last) / dt; ey_last = ey
            Dexb = (exb - exb_last) / dt; exb_last = exb
            Deyb = (eyb- eyb_last) / dt; eyb_last = eyb
            

            vx_cmd_world = Kp_pos * ex + Ki_pos * I_x + Kd_pos * Dex
            vy_cmd_world = Kp_pos * ey + Ki_pos * I_y + Kd_pos * Dey

            vx_cmd_body = Kp_pos * exb + Ki_pos * I_xb + Kd_pos * Dexb
            vy_cmd_body = Kp_pos * eyb + Ki_pos * I_yb + Kd_pos * Deyb




            # 航向角沿轨迹切线方向
            phi_d = omega_traj * t + np.pi/2
            phi_d = np.arctan2(np.sin(phi_d), np.cos(phi_d))  # 归一化


            ephi = phi_d - phi
            ephi = np.arctan2(np.sin(ephi), np.cos(ephi))

            # 期望速度转换到本体坐标系
            vx_ff = vx_d_world * np.cos(phi) + vy_d_world * np.sin(phi)
            vy_ff = -vx_d_world * np.sin(phi) + vy_d_world * np.cos(phi)

                                    # 最终速度指令 = PID修正 + 前馈
            vx_cmd_body =  vx_cmd_body + vx_ff  # 【核心修改】
            vy_cmd_body = vy_cmd_body + vy_ff  # 【核心修改】

            Dephi = (ephi - ephi_last) / dt; ephi_last = ephi

            I_phi += ephi * dt; I_phi = np.clip(I_phi, -I_limit, I_limit)

   
            vphi_cmd = Kp_pos * ephi + Ki_pos * I_phi + Kd_pos * Dephi


            # 4. 内环动力学控制
            vx_cmd = vx_cmd_body 
            vy_cmd = vy_cmd_body 
            
            a_ff_world = np.array([ax_d_world, ay_d_world])
            a_ff_body = R_world2body @ a_ff_world
            ax_ff = a_ff_body[0]
            ay_ff = a_ff_body[1]

            evx = vx_cmd - vx_body
            evy = vy_cmd - vy_body
            evphi = vphi_cmd - omega

            I_vx += evx * dt; I_vx = np.clip(I_vx, -I_limit, I_limit)
            I_vy += evy * dt; I_vy = np.clip(I_vy, -I_limit, I_limit)
            I_vphi += evphi * dt; 

            devx = (evx - evx_last) / dt; evx_last = evx
            devy = (evy - evy_last) / dt; evy_last = evy
            devphi = (evphi - evphi_last) / dt; evphi_last = evphi

            ax_d = ax_ff + Kp_vel * evx + Ki_vel * I_vx + Kd_vel * devx
            ay_d = ay_ff + Kp_vel * evy + Ki_vel * I_vy + Kd_vel * devy
            aphi_d = 0.0 + Kp_vel * evphi + Ki_vel * I_vphi + Kd_vel * devphi

            # ax_d =  Kp_vel * evx + Ki_vel * I_vx + Kd_vel * devx
            # ay_d =  Kp_vel * evy + Ki_vel * I_vy + Kd_vel * devy
            # aphi_d = 0.0 + Kp_vel * evphi + Ki_vel * I_vphi + Kd_vel * devphi

            C_force = np.array([-base_mass * omega * vy_body, base_mass * omega * vx_body, 0])
            F_cmd = M_mat @ np.array([ax_d, ay_d, aphi_d]) + C_force

            J_pinv = (1/r) * np.array([
                    [1, 1,  (a+b)],
                    [1,  -1, -(a+b)],
                    [1,  -1, (a+b)],
                    [1,  1, -(a+b)]
                ])
            
            # tau_cmd = J_pinv @ np.array([vx_cmd_body, vy_cmd_body, vphi_cmd])

            tau_cmd = J_pinv @ F_cmd
            tau_cmd = np.clip(tau_cmd, -50, 50)

            data.ctrl[:] = tau_cmd


            # --- 步进仿真 ---
            # 这里的步进次数取决于你的控制频率和物理频率比例
            # 通常一次循环步进一次即可，MuJoCo内部会处理子步
            mujoco.mj_step(model, data)
            try:
                marker_site_id = model.site('trajectory_marker').id
                data.site_xpos[marker_site_id][:2] = ref_pos
            except KeyError:
                pass

            # --- 数据记录 ---
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

            # --- 同步可视化与实时控制 ---
            viewer.sync()
            
            # 简单的实时控制：如果计算过快，等待一会儿
            # 计算期望经过的时间
            expected_time = start_time + simulation_step * dt
            current_time = time.time()
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            
            # 更新计数器 (如果在一个循环里调用多次mj_step，需要相应增加)
            simulation_step += 1

    # ------------------------------------------
    # 6. 结果绘图 (仿真结束后)
    # ------------------------------------------
    print("Simulation finished. Plotting results...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_x, history_y, 'b', label='Actual')
    plt.plot(history_xd, history_yd, 'r--', label='Desired')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.title('Trajectory Tracking (Full PID)')
    plt.legend(); plt.axis('equal'); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_time, history_phi, 'b', label='Actual')
    plt.plot(history_time, history_phid, 'r--', label='Desired')
    plt.xlabel('Time (s)'); plt.ylabel('Yaw (rad)')
    plt.title('Orientation Tracking')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
