import os
import time
import sys
from configparser import ConfigParser
import jax.numpy as jnp
from robot import Robot
from controller import Controller
from common import check_psd, deg2rad
from se3 import SE3

N_LIM = 200
POSE_NUM = 4

if __name__ == "__main__":
    
    Kp = jnp.diag(jnp.array([175, 175, 200]))
    Kr = 0.3*jnp.eye(3)
    Kd = jnp.diag(jnp.array([1,1,1,5,5,1]))
    
    parser = ConfigParser()
    parser.read('./config/robot.conf')
    ip = parser.get('xArm', 'ip')
    
    robot = Robot(ip=ip)
    control = Controller(
        target_pose=robot.g_star,
        b_mat=robot.b_mat,
        joint_speed_limits=robot.joint_vel_limit,
        Kp=Kp,
        Kr=Kr,
        Kd=Kd
    )
    
    se3 = SE3()
    
    count = 0
    loop_times = []
    geodesic_values = []
    rot_errors = []
    pos_errors = []
    mass_vals = []
    jac_vals = []
    qdot_vals = []
    grav_vals = []
    pose_vals = []
    joint_vals = []
    
    try:
        
        q_dot = robot.joint_speeds
        qVals = robot.joint_vals
        pose = robot.get_pose
        jac = jac = robot.get_jacobian
        M = robot.get_mass_matrix
        
        # Warmup: compute the first input to the system to warm up
        control_gains = jnp.zeros((6,1))
        robot_input = []
        u = control.compute_gains(jac, q_dot, pose, robot)
        for i in range(0,6):
            control_gains = control_gains.at[i].set((u[i].item()))
        q_target = control_gains
        q_sat = jnp.round(jnp.where(se3.near_zero(q_target, 1e-3), 0, q_target), 2)
    
        while count < N_LIM:
            
            t0 = time.time()
            if count == 0:
                q_dot = q_sat
            else:
                q_dot = robot.joint_speeds
            
            qVals = robot.joint_vals
            pose = robot.get_pose
            jac = jac = robot.get_jacobian
            M = robot.get_mass_matrix
            
            control_gains = jnp.zeros((6,1))
            robot_input = []
            u = control.compute_gains(jac, q_dot, pose, robot)
            for i in range(0,6):
                control_gains = control_gains.at[i].set((u[i].item()))
            q_target = control_gains
            q_sat = jnp.round(jnp.where(se3.near_zero(q_target, 1e-3), 0, q_target), 2)
            # print(q_sat)
            print(f"Cartesian Set Velocity (Body frame): {q_sat.T}")
            for i in range(0,6):
                robot_input.append(q_sat[i].item())
                
            if count != 0:
                code = robot.arm.vc_set_cartesian_velocity(robot_input, is_radian=True, is_tool_coord=True)
            
            geo, rot_error, pos_error = control.geodesic(pose, M, jac)
            geodesic_values.append(geo)
            rot_errors.append(rot_error)
            pos_errors.append(pos_error)
            mass_vals.append(M)
            jac_vals.append(jac)
            qdot_vals.append(q_dot)
            grav_vals.append(robot.get_grav_vec)
            pose_vals.append(pose)
            joint_vals.append(qVals)
            print(f"Geodesic: {geo}, Rotation Error: {rot_error}, Position Error: {pos_error}")
            
            t1 = time.time()
            loop = t1-t0
            loop_times.append(loop)
            print(f"Loop time: {loop}")
            count += 1
            time.sleep(0.1)
    finally:
        print("Shutting down robot")
        final_geo, final_rot, final_pos = control.geodesic(pose, M, jac)
        print(f"Final Geodesic: {final_geo}, Number of inputs: {count}")
        jnp.savz(f'pose_{POSE_NUM}.npz',
                    joints=jnp.array(joint_vals),
                    geo_vals=jnp.array(geodesic_values),
                    rot_errors=jnp.array(rot_errors),
                    pos_errors=jnp.array(pos_errors),
                    mass_vals=jnp.array(mass_vals),
                    jac_vals=jnp.array(jac_vals),
                    qdot_vals=jnp.array(qdot_vals),
                    grav_vals=jnp.array(grav_vals),
                    pose_vals=jnp.array(pose_vals))
        robot.shutdown()
    
    ave_loop = sum(loop_times)/len(loop_times)
    print(f"Average loop time: {ave_loop} s")