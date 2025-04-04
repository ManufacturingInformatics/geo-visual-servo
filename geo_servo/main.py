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
from common import check_psd
from se3 import SE3

N_LIM = 1

if __name__ == "__main__":
    
    Kp = 0.8*jnp.eye(3)
    Kr = 0.7*jnp.eye(3)
    Kd = 0.1*jnp.eye(6)
    
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
            
            print(f"q_dot = {q_dot}")
            print(f"q = {qVals}")
            print(f"Pose = {pose}")
            print(f"Target pose = {control.target_pose}")
            print(f"Mass matrix = {M}")
            
            control_gains = jnp.zeros((6,1))
            robot_input = []
            u = control.compute_gains(jac, q_dot, pose, robot)
            for i in range(0,6):
                control_gains = control_gains.at[i].set((u[i].item()))
            print(f"Control gains = {control_gains}")
            print(f"Non-saturated = {jnp.linalg.pinv(jac) @ control_gains}")
            q_set = control.saturate(jnp.linalg.pinv(jac) @ control_gains)
            
            for i in range(0,6):
                robot_input.append(q_set[i].item())
            # ONLY ENABLE THIS COMMAND WHEN IT WORKS
            print(robot_input)
            # robot.arm.vc_set_joint_velocity(robot_input, is_radian=True)
            
            t1 = time.time()
            loop = t1-t0
            loop_times.append(loop)
            print(f"Loop time: {loop}")
            # time.sleep(0.1)
    finally:
        print("Shutting down robot")
        robot.shutdown()
    
    ave_loop = sum(loop_times)/len(loop_times)
    print(f"Average loop time: {ave_loop} s")