"""
TODO:
- Update the mass and gravity computations to use the body Jacobian
- Test control law
"""
import os
import time
import sys
from configparser import ConfigParser
import jax.numpy as jnp
from robot import Robot
from controller import Controller
from common import check_psd, deg2rad
from se3 import SE3

N_LIM = 100

if __name__ == "__main__":
    
    Kp = 500*jnp.eye(3)
    Kr = jnp.eye(3)
    Kd = jnp.eye(6)
    
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
    try:
        while count < N_LIM:
            t0 = time.time()
            count += 1
            
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
            # print(f"Control gains = {control_gains.T}") # jnp.linalg.pinv(jac) @ 
            q_target = control_gains
            # print(f"Inverted = {jnp.linalg.pinv(jac) @ deg2rad(control_gains)}")
            q_sat = jnp.round(jnp.where(se3.near_zero(q_target, 1e-3), 0, q_target), 2) #control.saturate(
            # print(q_sat)
            print(f"Cartesian Set Velocity (Body frame): {q_sat.T}")
            for i in range(0,6):
                robot_input.append(q_sat[i].item())
            # ONLY ENABLE THIS COMMAND WHEN IT WORKS
            # print(robot_input)
            # robot.arm.vc_set_joint_velocity(robot_input, is_radian=True)
            code = robot.arm.vc_set_cartesian_velocity(robot_input, is_radian=True, is_tool_coord=True)
            
            # print(code)
            
            geo = control.geodesic(pose, M, jac)
            print(f"Geodesic: {geo}")
            
            t1 = time.time()
            loop = t1-t0
            loop_times.append(loop)
            print(f"Loop time: {loop}")
            # time.sleep(0.1)
    finally:
        print("Shutting down robot")
        final_geo = control.geodesic(pose, M, jac)
        print(f"Final Geodesic: {final_geo}")
        robot.shutdown()
    
    ave_loop = sum(loop_times)/len(loop_times)
    print(f"Average loop time: {ave_loop} s")