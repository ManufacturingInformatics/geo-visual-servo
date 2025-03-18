import os
import time
import sys
import jax.numpy as jnp
from math import sin, cos

from xarm.wrapper import XArmAPI
from common import load_inertia_params
from jaxlie import SE3


class Robot:
    
    def __init__(self, ip, motion_enable=True, set_mode=0):
        
        self.ip = ip
        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(enable=motion_enable)
        self.arm.set_mode(set_mode)
        self.mass, self.inertia = load_inertia_params()
        
        # Mechanical properties
        self.l1 = 0.267
        self.l2 = 0.289
        self.l3 = 0.078
        self.l4 = 0.343
        self.l5 = 0.076
        self.l6 = 0.097
        self._0theta2 = -1.385 # in radians
        self._0theta3 = 1.385 # in radians

        # Zero configurations        
        self.g_0 = jnp.array([[1.0, 0, 0,  2.07386743e-01],
                            [0, -1.0, 0, 0],
                            [0, 0, -1.0, 1.11026153e-01],
                            [0, 0, 0, 1.0]])
        
        # Target pose
        self.g_star = jnp.array([[-0.01110625, 0.01748104, -0.99978554, -0.29364628],
                                 [-0.00470503, 0.9998352, 0.01753419, 0.31577227],
                                 [0.9999273, 0.00489879, -0.01102217, 0.06060453],
                                 [0., 0., 0., 1.]])

        # DH parameters, in order: theta_i(offset), d, alpha, r
        self.dh_params = jnp.array(
            [[0, self.l1, -jnp.pi/2, 0],
             [self._0theta2, 0, 0, self.l2],
             [self._0theta3, 0, -jnp.pi/2, self.l3],
             [0, self.l4, jnp.pi/2, 0],
             [0, 0, -jnp.pi/2, self.l5],
             [0, self.l6, 0, 0]]
        )
        
        # Kinematics initialisation
        self.transforms = jnp.zeros((6,4,4))
        self.get_transforms()
        self.fk = jnp.zeros((4,4))
        self.forward_kinematics()
    
    def get_pose(self, radians=False) -> jnp.ndarray:
        return jnp.array(self.arm.get_position(is_radian=radians)[1])
    
    def compute_rotation_matrix(self) -> jnp.ndarray:
        """
        Computes the rotation matrix for the given pose of the end effector.

        Returns:
            jnp.ndarray: Rotation matrix in 3x3 format
        """
        pose = self.get_pose(radians=True)
        roll = pose[3] # x-axis, psi
        pitch = pose[4] # y-axis, theta
        yaw = pose[5] # z-axis, phi
        
        rot_mat = jnp.array(
            [[cos(pitch)*cos(yaw), sin(roll)*sin(pitch)*cos(yaw)-cos(roll)*sin(yaw), cos(roll)*sin(pitch)*cos(yaw)+sin(roll)*sin(yaw)],
             [cos(pitch)*sin(yaw), sin(roll)*sin(pitch)*sin(yaw)+cos(roll)*cos(yaw), cos(roll)*sin(pitch)*sin(yaw)-sin(roll)*cos(yaw)],
             [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]]
        )
        return rot_mat
    
    def get_transforms(self):
        jVals = self.arm.get_joint_states(is_radian=True)[1][0]
        for i in range(0, 6):
            self.transforms = self.transforms.at[i].set(
                [
                    [cos(jVals[i]+self.dh_params[i][0]), -sin(jVals[i]+self.dh_params[i][0])*cos(self.dh_params[i][2]), sin(jVals[i]+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*cos(jVals[i]+self.dh_params[i][0])],
                    [sin(jVals[i]+self.dh_params[i][0]), cos(jVals[i]+self.dh_params[i][0])*cos(self.dh_params[i][2]), -cos(jVals[i]+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*sin(jVals[i]+self.dh_params[i][0])],
                    [0, sin(self.dh_params[i][2]), cos(self.dh_params[i][2]), self.dh_params[i][1]],
                    [0, 0, 0, 1]
                ]
            )
        
    def forward_kinematics(self):
        temp_fk = self.transforms[0]
        for i in range(1,6):
            temp_fk = temp_fk @ self.transforms[i]
        self.fk = temp_fk