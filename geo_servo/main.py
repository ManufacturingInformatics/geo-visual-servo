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
    
    Kp = 5*jnp.eye(3)
    Kr = 5*jnp.eye(3)
    Kd = 0.5*jnp.eye(6)
    
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
    while True:
        count += 1
        
        # q_dot = robot.joint_speeds
        
        qVals = robot.joint_vals
        pose = robot.get_pose
        jac = jac = robot.get_jacobian
        print(jnp.round(jnp.where(se3.near_zero(z=pose), 0, pose), 4))
        # print(jnp.round(jnp.where(se3.near_zero(z=jac), 0, jac), 4))
        ee_twist = jac @ robot.joint_speeds
        print(jnp.round(jnp.where(se3.near_zero(z=ee_twist), 0, ee_twist), 4))
        
        
        # robot_input = []
        # u = control.compute_gains(jac, q_dot, pose, robot)
        # for i in range(0,6):
        #     robot_input.append(u[i].item())
        # print(robot_input)
        
        # ONLY ENABLE THIS COMMAND WHEN IT WORKS
        # robot.arm.vc_set_joint_velocity(robot_input, is_radian=True)
    
    robot.shutdown