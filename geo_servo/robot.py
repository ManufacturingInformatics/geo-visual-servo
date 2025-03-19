import os
import time
import sys
import jax.numpy as jnp
import numpy as np # I don't want to use this, but it is necessary for now
from math import sin, cos
from sympy import Symbol, Matrix
import sympy as sy

from xarm.wrapper import XArmAPI
from common import load_inertia_params, load_tcp_params
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
        
        self.rot_axis = [2, 1, 1, 2, 1, 2]
        
        # Dynamics parameters
        self.tcp_params = load_tcp_params()
        self.arm.set_tcp_load(
            weight=self.tcp_params['weight'],
            center_of_gravity=[self.tcp_params['cx'], self.tcp_params['cy'], self.tcp_params['cz']]
        )
        self.mass_matrix = jnp.zeros((6,6))
        self.jacobian = jnp.zeros((6,6))

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
        
        # Symbolic functions for Jacobian
        self._q0 = Symbol('q0')
        self._q1 = Symbol('q1')
        self._q2 = Symbol('q2')
        self._q3 = Symbol('q3')
        self._q4 = Symbol('q4')
        self._q5 = Symbol('q5')
        self.q = Matrix([self._q0, self._q1, self._q2, self._q3, self._q4, self._q5])
        
        # Kinematics initialisation
        self.transforms = jnp.zeros((6,4,4))
        self.get_transforms()
        self.fk = jnp.zeros((4,4))
        self.forward_kinematics()
        
        # Initialise the robot to get it ready for velocity control mode
        self.arm.set_mode(mode=4)
        self.arm.set_state(0)
        
    def get_mass_matrix(self) -> jnp.ndarray:
        return self._compute_mass_matrix()
    
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
        """
        Computes the numerical joint transforms from the prior joint to the current joint (n-1)T(n)
        """
        qVals = self.arm.get_joint_states(is_radian=True)[1][0]
        for i in range(0, 6):
            self.transforms = self.transforms.at[i].set(
                [[cos(qVals[i]+self.dh_params[i][0]), -sin(qVals[i]+self.dh_params[i][0])*cos(self.dh_params[i][2]), sin(qVals[i]+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*cos(qVals[i]+self.dh_params[i][0])],
                 [sin(qVals[i]+self.dh_params[i][0]), cos(qVals[i]+self.dh_params[i][0])*cos(self.dh_params[i][2]), -cos(qVals[i]+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*sin(qVals[i]+self.dh_params[i][0])],
                 [0, sin(self.dh_params[i][2]), cos(self.dh_params[i][2]), self.dh_params[i][1]],
                 [0, 0, 0, 1]])
        
    def forward_kinematics(self):
        """
        Updates the forward kinematics from joint space to the task space. 
        """
        temp_fk = self.transforms[0]
        for i in range(1,6):
            temp_fk = temp_fk @ self.transforms[i]
        self.fk = temp_fk
        
    def _get_joints(self, is_radian=False) -> list:
        return self.arm.get_joint_states(is_radian=is_radian)[1][0]
                
    def _compute_mass_matrix(self) -> jnp.ndarray:
        """
        Computes the mass-inertia matrix for the manipulator at the current configuration. This function uses SymPy to compute the symbolic Jacobian and then the mass matrix for each joint. Note that this differs from the standard Jacobian function, which only calculates the full-body Jacobian. 

        Returns:
            jnp.ndarray: JAX NumPy array converted from the symbolic representation
        """
        qVals = self._get_joints()
        J = sy.zeros(6,6)
        M = sy.zeros(6,6)

        for i in range(0, 6):
            if i == 0:
                T = Matrix([
                    [sy.cos(self.q[i]+self.dh_params[i][0].item()), -sy.sin(self.q[i]+self.dh_params[i][0].item())*sy.cos(self.dh_params[i][2].item()), sy.sin(self.q[i]+self.dh_params[i][0].item())*sy.sin(self.dh_params[i][2].item()), self.dh_params[i][3].item()*sy.cos(self.q[i]+self.dh_params[i][0].item())],
                    [sy.sin(self.q[i]+self.dh_params[i][0].item()), sy.cos(self.q[i]+self.dh_params[i][0].item())*sy.cos(self.dh_params[i][2].item()), -sy.cos(self.q[i]+self.dh_params[i][0].item())*sy.sin(self.dh_params[i][2].item()), self.dh_params[i][3].item()*sy.sin(self.q[i]+self.dh_params[i][0].item())],
                    [0, sy.sin(self.dh_params[i][2].item()), sy.cos(self.dh_params[i][2].item()), self.dh_params[i][1].item()],
                    [0, 0, 0, 1]
                ])
                T_p = T[0:3,3]
                jV = T_p.jacobian(self.q)
                jW = Matrix([[T[0:3,self.rot_axis[i]]]])
                J[0:3,:] = jV
                J[3:6,i] = jW
                M = self.mass[i].item()*jV.T*jV + J[3:6,:].T*Matrix(self.inertia[i])*J[3:6,:]
            else:
                T_temp = Matrix([
                    [sy.cos(self.q[i]+self.dh_params[i][0].item()), -sy.sin(self.q[i]+self.dh_params[i][0].item())*sy.cos(self.dh_params[i][2].item()), sy.sin(self.q[i]+self.dh_params[i][0].item())*sy.sin(self.dh_params[i][2].item()), self.dh_params[i][3].item()*sy.cos(self.q[i]+self.dh_params[i][0].item())],
                    [sy.sin(self.q[i]+self.dh_params[i][0].item()), sy.cos(self.q[i]+self.dh_params[i][0].item())*sy.cos(self.dh_params[i][2].item()), -sy.cos(self.q[i]+self.dh_params[i][0].item())*sy.sin(self.dh_params[i][2].item()), self.dh_params[i][3].item()*sy.sin(self.q[i]+self.dh_params[i][0].item())],
                    [0, sy.sin(self.dh_params[i][2].item()), sy.cos(self.dh_params[i][2].item()), self.dh_params[i][1].item()],
                    [0, 0, 0, 1]
                ])
                T = T * T_temp
                T_p = T[0:3,3]
                jV = T_p.jacobian(self.q)
                jW = Matrix([[T[0:3,self.rot_axis[i]]]])
                J[0:3,:] = jV
                J[3:6,i] = jW
                M = self.mass[i].item()*jV.T*jV + J[3:6,:].T*Matrix(self.inertia[i])*J[3:6,:]
                
        eval_dict = {
            self.q[0]: qVals[0],
            self.q[1]: qVals[1],
            self.q[2]: qVals[2],
            self.q[3]: qVals[3],
            self.q[4]: qVals[4],
            self.q[5]: qVals[5]
        }
        M_eval = M.subs(eval_dict).evalf(n=3).applyfunc(lambda i: 0 if -1e-3<i<1e-3 else i)
        return jnp.array(
            np.array(M_eval).astype(np.float64)
        )
        
        