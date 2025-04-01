import os
import time
import sys
from configparser import ConfigParser
import jax.numpy as jnp
from robot import Robot
from controller import Controller
from common import check_psd

if __name__ == "__main__":
    
    Kp = jnp.diag(jnp.array([5, 5, 2]))
    Kr = jnp.diag(jnp.array([5, 5, 2]))
    
    parser = ConfigParser()
    parser.read('./config/robot.conf')
    ip = parser.get('xArm', 'ip')
    
    robot = Robot(ip=ip)
    control = Controller(
        target_pose=robot.g_star,
        b_mat=robot.b_mat,
        joint_speed_limits=robot.joint_vel_limit,
        Kp=Kp,
        Kr=Kr
    )
    print(robot.dh_params)
    
    count = 0
    while count < 10:
        count += 1
        
        jac = robot.get_jacobian
        q_dot = robot.joint_speeds
        pose = robot.get_pose
        robot_input = []
        u = control.compute_gains(jac, q_dot, pose, robot)
        for i in range(0,6):
            robot_input.append(u[i].item())
        # print(robot_input)
        
        # ONLY ENABLE THIS COMMAND WHEN IT WORKS
        # robot.arm.vc_set_joint_velocity(robot_input, is_radian=True)
    
    robot.shutdown