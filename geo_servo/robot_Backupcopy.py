import os
import time
import sys
import jax.numpy as jnp
import numpy as np # I don't want to use this, but it is necessary for now
from math import sin, cos
from sympy import Symbol, Matrix
import sympy as sy

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
        
        self.rot_axis = [2, 1, 1, 2, 1, 2]
        
        # Dynamics parameters
        self.tcp_params = load_tcp_params()
        self.arm.set_tcp_load(
            weight=self.tcp_params['weight'],
            center_of_gravity=[self.tcp_params['cx'], self.tcp_params['cy'], self.tcp_params['cz']]
        )
        self.b_mat = jnp.ones((6,1))
        self.grav_vec = jnp.array([[0, -GRAVITY, 0]]).T
        
        # Target pose
        self.g_star = jnp.array([[-0.01110625, 0.01748104, -0.99978554, -0.29364628],
                                 [-0.00470503, 0.9998352, 0.01753419, 0.31577227],
                                 [0.9999273, 0.00489879, -0.01102217, 0.06060453],
                                 [0., 0., 0., 1.]])
        
        # Joint speed limits
        self.joint_vel_limit = jnp.array([-(jnp.pi)/50, (jnp.pi)/50])

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
        self.kinematics = Kinematics()
        
        # Initialise the robot to get it ready for velocity control mode
        self.arm.set_mode(mode=4)
        self.arm.set_state(0)
        
    @property
    def shutdown(self):
        self.arm.vc_set_joint_velocity([0,0,0,0,0,0])
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
        _ = self.get_transforms()
        self.forward_kinematics()
        return self.fk
    
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
        return self._jacobian()
    
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
    
    def get_transforms(self) -> jnp.ndarray:
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
        return self.transforms
        
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
    
    def _get_joint_speeds(self):
        return self.arm.realtime_joint_speeds
    
    def _jacobian(self) -> jnp.ndarray:
        """
        Computes the geometric Jacobian for the current state of the manipulator.

        Returns:
            jnp.ndarray: _description_
        """
        qVals = self._get_joints()
        J = sy.zeros(6,6)
        
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
                
        eval_dict = {
            self.q[0]: qVals[0],
            self.q[1]: qVals[1],
            self.q[2]: qVals[2],
            self.q[3]: qVals[3],
            self.q[4]: qVals[4],
            self.q[5]: qVals[5]
        }
        J_eval = J.subs(eval_dict).evalf(n=3)
        return jnp.array(
            np.array(J_eval).astype(np.float64)
        )
    
    def _compute_gravity_matrix(self) -> jnp.ndarray:
        """
        Computes the gravity vector at the current configuration

        Returns:
            jnp.ndarray: Gravity vector corresponding to the current configuration
        """
        qVals = self._get_joints()
        J = sy.zeros(6,6)
        G = sy.zeros(6,1)
        for i in range(0,6):
            if i == 0:
                T = Matrix([
                    [sy.cos(self.q[i]+self.dh_params[i][0].item()), -sy.sin(self.q[i]+self.dh_params[i][0].item())*sy.cos(self.dh_params[i][2].item()), sy.sin(self.q[i]+self.dh_params[i][0].item())*sy.sin(self.dh_params[i][2].item()), self.dh_params[i][3].item()*sy.cos(self.q[i]+self.dh_params[i][0].item())],
                    [sy.sin(self.q[i]+self.dh_params[i][0].item()), sy.cos(self.q[i]+self.dh_params[i][0].item())*sy.cos(self.dh_params[i][2].item()), -sy.cos(self.q[i]+self.dh_params[i][0].item())*sy.sin(self.dh_params[i][2].item()), self.dh_params[i][3].item()*sy.sin(self.q[i]+self.dh_params[i][0].item())],
                    [0, sy.sin(self.dh_params[i][2].item()), sy.cos(self.dh_params[i][2].item()), self.dh_params[i][1].item()],
                    [0, 0, 0, 1]
                ])
                T_p = T[0:3,3]
                jV = T_p.jacobian(self.q)
                G = - jV.T * self.mass[i].item() * self.grav_vec
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
                G -= jV.T * self.mass[i].item() * self.grav_vec
        eval_dict = {
            self.q[0]: qVals[0],
            self.q[1]: qVals[1],
            self.q[2]: qVals[2],
            self.q[3]: qVals[3],
            self.q[4]: qVals[4],
            self.q[5]: qVals[5]
        }
        G_eval = G.subs(eval_dict).evalf(n=3)
        return jnp.array(
            np.array(G_eval).astype(np.float64)
        )
                
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
        return jnp.diag(jnp.diagonal(jnp.array(
            np.array(M_eval).astype(np.float64)
        )))
        