# TODO:
# - Create PD with GC control!

import os
import time
import sys
from configparser import ConfigParser
import jax.numpy as jnp
from robot import Robot
from controller import Controller
from common import check_psd

if __name__ == "__main__":
    
    parser = ConfigParser()
    parser.read('./config/robot.conf')
    ip = parser.get('xArm', 'ip')
    
    robot = Robot(ip=ip)
    control = Controller(
        target_pose=robot.g_star,
        b_mat=robot.b_mat,
        joint_speed_limits=robot.arm.joint_speed_limit
    )
    while True:
        
        jac = robot.get_jacobian
        q_dot = robot.joint_speeds
       # print(control.compute_gains(jac, q_dot))
        
        print(control.geodesic(robot.get_pose, robot.get_mass_matrix, robot.get_jacobian))
    