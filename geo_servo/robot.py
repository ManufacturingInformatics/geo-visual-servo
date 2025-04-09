import os
import time
import sys
import jax.numpy as jnp
import numpy as np # I don't want to use this, but it is necessary for now
from math import sin, cos
from se3 import SE3

from xarm.wrapper import XArmAPI
from common import load_inertia_params, load_tcp_params, check_psd, vee_map, deg2rad
from kinematics import Kinematics

GRAVITY = 9.81

class Robot:
    
    def __init__(self, ip, motion_enable=True, set_mode=0, is_radian=True):
        
        self.ip = ip
        self.is_radian = is_radian
        self.arm = XArmAPI(self.ip, is_radian=self.is_radian)
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
        
        # Dynamics parameters
        self.tcp_params = load_tcp_params()
        self.arm.set_tcp_load(
            weight=self.tcp_params['weight'],
            center_of_gravity=[self.tcp_params['cx'], self.tcp_params['cy'], self.tcp_params['cz']]
        )
        self.b_mat = jnp.ones((6,1))
        self.grav_vec = jnp.array([[0, -GRAVITY, 0]]).T
        
        # Target pose
        self.g_star = jnp.array([
            [0.0114, 0.0728, -0.99729997, -0.2703],
            [0.0139, 0.99719995, 0.0729, 0.3126],
            [0.99979997, -0.0147, 0.0104, 0.0386],
            [0, 0, 0, 1]])
        # [[-0.01110625, 0.01748104, -0.99978554, -0.29364628],
        # [-0.00470503, 0.9998352, 0.01753419, 0.31577227],
        # [0.9999273, 0.00489879, -0.01102217, 0.06060453],
        # [0., 0., 0., 1.]]
        
        # Joint speed limits
        self.joint_vel_limit = jnp.array([-(jnp.pi)/20, (jnp.pi)/20])

        # DH parameters, in order: theta_i(offset), d, alpha, r
        self.dh_params = jnp.array(
            [[0, self.l1, -jnp.pi/2, 0],
             [self._0theta2, 0, 0, self.l2], # First value self._0theta2
             [self._0theta3, 0, -jnp.pi/2, self.l3], # First value self._0theta3
             [0, self.l4, jnp.pi/2, 0],
             [0, 0, -jnp.pi/2, self.l5],
             [0, self.l6, 0, 0]]
        )
        
        # Alternative DH parameters in order: alpha, a, d, phi
        self.dh_params_alt = jnp.array( 
            [[-jnp.pi/2, 0, self.l1, 0],
             [0, self.l2, 0, self._0theta2],
             [-jnp.pi/2, self.l3, 0, self._0theta3],
             [jnp.pi/2, 0, self.l4, 0],
             [-jnp.pi/2, self.l5, 0, 0],
             [0, 0, self.l6, 0]]
        )
        
        # Kinematics initialisation
        self.kinematics = Kinematics()
        
        # Initialise the robot to get it ready for velocity control mode
        self.arm.set_mode(mode=5) # 4 for joint control, 5 for cartesian control
        self.arm.set_state(0)
        self.se3 = SE3()
        
    def shutdown(self):
        self.arm.vc_set_cartesian_velocity([0,0,0,0,0,0])
        self.arm.disconnect
        
    @property
    def joint_vals(self):
        """
        Property for the joint values, rather than a function. This also casts the joint values as a JAX NumPy array that allows for the useful computation of things. 

        Returns:
            jnp.ndarray : JAX NumPy array corresponding to the 6 joint positions
        """
        return jnp.asarray(self._get_joints())[0:6].reshape((6,1))
    
    @property
    def joint_speeds(self):
        """
        Property for the joint speeds, found in rad/s if self.is_radians=True. Converted from list to JAX NumPy array for convenience.
        
        Returns:
            jnp.ndarray: JAX NumPy array of the joint speeds for the robot, given in rad/s
        """
        return jnp.asarray(self._get_joint_speeds())[0:6].reshape((6,1))
    
    @property
    def get_mass_matrix(self) -> jnp.ndarray:
        return self._compute_mass_matrix()
    
    @property
    def get_pose(self) -> jnp.ndarray:
        qVals = self.joint_vals
        return self.kinematics.fk_body(qVals)
    
    @property
    def get_grav_vec(self) -> jnp.ndarray:
        return self._compute_gravity_matrix()
    
    @property
    def get_jacobian(self) -> jnp.ndarray:
        """
        Returns the 6x6 Jacobian matrix. This only applies to the xArm 6, which only has 6 joints. 

        Returns:
            jnp.ndarray: Jacobian array cast as a JAX NumPy array
        """
        qVals = self.joint_vals
        return self.kinematics.jacobian_body(qVals)
    
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
        
    def forward_kinematics(self):
        """
        Updates the forward kinematics from joint space to the task space. 
        """
        temp_fk = self.transforms[0]
        for i in range(1,6):
            temp_fk = temp_fk @ self.transforms[i]
        self.fk = temp_fk
        
    def _get_joints(self, is_radian=True) -> list:
        return self.arm.get_joint_states(is_radian=is_radian)[1][0]
    
    def _get_joint_speeds(self):
        return self.arm.realtime_joint_speeds
    
    def _compute_gravity_matrix(self) -> jnp.ndarray:
        """
        Computes the gravity vector at the current configuration

        Returns:
            jnp.ndarray: Gravity vector corresponding to the current configuration
        """
        qVals = self.joint_vals
        B_list = jnp.array(self.kinematics.B)
        J_b = jnp.array(self.kinematics.B.T).copy().astype(float)
        T = jnp.eye(4)
        G = jnp.zeros((6,1))
        for i in range(qVals.shape[0]-2, -1,-1):
            
            b = B_list[i+1,:]
            b_skew = self.se3.vec6_to_skew4(jnp.array(b*-qVals[i+1]).reshape((1,6)))
            mat_exp = self.se3.skew4_to_matrix_exp4(b_skew)
            T = T @ mat_exp
            
            adj_T = self.se3.adj(T)
            J_col = adj_T @ B_list[i,:]
            J_b = J_b.at[:,i].set(J_col)
            J_bv = J_b[0:3,:]
            G -= J_bv.T @ (self.mass[i].item() * self.grav_vec)
            
        return jnp.round(jnp.where(self.se3.near_zero(G), 0, G), 4)
                
    def _compute_mass_matrix(self) -> jnp.ndarray:
        """
        Computes the mass-inertia matrix for the manipulator at the current configuration.

        Returns:
            jnp.ndarray: JAX NumPy array of the mass matrix in Jacobian form
        """
        qVals = self.joint_vals
        B_list = jnp.array(self.kinematics.B)
        J_b = jnp.array(self.kinematics.B.T).copy().astype(float)
        T = jnp.eye(4)
        M = jnp.zeros((6,6))
        for i in range(qVals.shape[0]-2, -1,-1):
            
            b = B_list[i+1,:]
            b_skew = self.se3.vec6_to_skew4(jnp.array(b*-qVals[i+1]).reshape((1,6)))
            mat_exp = self.se3.skew4_to_matrix_exp4(b_skew)
            T = T @ mat_exp
            
            adj_T = self.se3.adj(T)
            J_col = adj_T @ B_list[i,:]
            J_b = J_b.at[:,i].set(J_col)
            J_bv = J_b[0:3,:]
            J_bw = J_b[3:,:]
            M = M + self.mass[i].item() * (J_bv.T @ J_bv) + (J_bw.T @ self.inertia[i] @ J_bw)
            
        return jnp.round(jnp.where(self.se3.near_zero(M), 0, M), 4)
        