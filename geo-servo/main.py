import os
import time
import sys
from configparser import ConfigParser
from robot import Robot
from common import load_inertia_params

if __name__ == "__main__":
    
    parser = ConfigParser()
    parser.read('./config/robot.conf')
    ip = parser.get('xArm', 'ip')
    
    robot = Robot(ip=ip)
    pose = robot.get_pose(radians=False)
    # print(pose)
    print(robot.compute_rotation_matrix())
    print(robot.fk)
    