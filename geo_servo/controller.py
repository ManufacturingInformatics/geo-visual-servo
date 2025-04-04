import os
import time
import sys
import jax.numpy as jnp
import jax
from common import pose_cross_map, momenta_cross_map, vee_map
from common import Saturation
from robot import Robot

class Controller:
    
    def __init__(self, 
                 target_pose, 
                 b_mat, 
                 joint_speed_limits,
                 Kp=jnp.eye(3), 
                 Kr=jnp.eye(3), 
                 Kd=jnp.eye(6)):
        
        # Control gains for Kp, Kr, Kd
        self.Kp = Kp
        self.Kr = Kr
        self.Kd = Kd
        
        # Matrix parameters from the system
        self.b_mat = b_mat
        self.joint_speed_limits = joint_speed_limits # in rad/s
        self.saturation = Saturation(self.joint_speed_limits[0], self.joint_speed_limits[1])
        
        # Target parameters
        assert target_pose.shape == (4,4)
        self.target_pose = target_pose
        
    def compute_gains(self, jacobian, joint_speeds, pose, robot):
        """
        This function computes the controller inputs in terms of the velocity controller for the manipulator. 

        Args:
            jacobian (jnp.ndarray): Geometric body Jacobian of the manipulator at the current configuration
            joint_speeds (jnp.ndarray): Joint speeds of the manipulator in rad/s
            pose (jnp.ndarray): End-effector pose of the manipulator in SE(3)
            robot (Robot): Robot class instatiated for getting the gravity vectors

        Returns:
            _type_: _description_
        """
        u_es = self._compute_energy_shaping(pose=pose, jacobian=jacobian, joint_speeds=joint_speeds, robot=robot)
        u_di = self._compute_damping_injection(jacobian=jacobian, joint_speeds=joint_speeds)
        u_dc = 0
        u = u_es + u_di + u_dc
        return u
    
    def geodesic(self, pose, mass_matrix, jacobian) -> jnp.float64:
        """
        Computes the approximate metric geodesic distance between the provided pose and the target pose. Uses the mass matrix and Jacobian to find the Hamiltonian metric tensor of the Lie-Poisson task space of the manipulator.

        Args:
            pose (jnp.ndarray): Current pose of the manipulator in SE(3)
            mass_matrix (jnp.ndarray): Mass matrix of the manipulator at the current configuration. Normally positive semi-definite.
            jacobian (jnp.ndarray): Body Jacobian of the manipulator at the current configuration. 

        Returns:
            jnp.float64: Returns the metric geodesic distance approximation. 
        """
        # Need to ensure that the provided pose is in the SE(3) form. 
        assert pose.shape == (4,4)
        
        # Extract rotation matrices
        R_0 = pose[0:3,0:3]
        R_1 = self.target_pose[0:3, 0:3]
        
        # Extract positions
        p_0 = pose[0:3, 3]
        p_1 = self.target_pose[0:3,-1]
        
        # Hamiltonian metric tensor
        j_inv = jnp.linalg.pinv(jacobian)
        m_bar = j_inv.T @ mass_matrix @ j_inv
        G_hamil =  jnp.linalg.pinv(m_bar)
        
        # Rotation and position differences
        delta_R = jnp.linalg.norm(G_hamil[3:6, 3:6], ord='fro') * jnp.acos(
             (jnp.trace((R_0.T @ R_1))-1)/2
        )
        delta_p = jnp.linalg.norm(
            G_hamil[0:3, 0:3] @ (p_0 - p_1), ord=None
        )
        return jnp.sqrt(delta_R + delta_p)
    
    def saturate(self, u) -> jnp.ndarray:
        """
        Saturates the control inputs to within the joint speed limits

        Args:
            u (jnp.nda]): Raw input provided by the controller algorithm

        Returns:
            jnp.ndarray: Saturated array of inputs
        """
        for i in range(0,6):
            u = u.at[i].set(self.saturation.saturate(u[i].item()))
        return u
    
    def _compute_energy_shaping(self, 
                                pose: jnp.ndarray, 
                                jacobian: jnp.ndarray, 
                                joint_speeds: jnp.ndarray,
                                robot: Robot) -> jnp.ndarray:
        """
        Compute the energy shaping gain for the manipulator

        Args:
            pose (jnp.ndarray): Pose in SE(3) of the end-effector
            jacobian (jnp.ndarray): Body Jacobian at the current configuration
            joint_speeds (jnp.ndarray): Joint speeds of the manipulator
            robot (Robot): Robot class for mass and gravity matrices

        Returns:
            jnp.ndarray: Energy shaping input for the controller
        """
        g_cross = pose_cross_map(pose)
        print(f"g_cross = {g_cross}")
        m_cross = momenta_cross_map(robot.get_mass_matrix, jacobian, joint_speeds)
        print(f"m_cross = {m_cross}")
        error = jnp.zeros((6,1))
        twists = jacobian @ joint_speeds
        R = pose[0:3,0:3]
        p = pose[0:3,-1].reshape(3,1)
        # print(R, p)
        e_temp = self.target_pose[0:3,0:3].T @ R - R.T @ self.target_pose[0:3,0:3]
        error = error.at[0:3].set(R.T @ self.Kp @ (p - self.target_pose[0:3,-1].reshape((3,1))))
        error = error.at[3:6].set(0.5 * self.Kr @ vee_map(e_temp))
        print(f"Error = {error}")
        G_vec = robot.get_grav_vec
        print(f"Gravity vector = {G_vec}")
        return g_cross.T @ G_vec - m_cross @ twists - error
    
    def _compute_damping_injection(self, jacobian, joint_speeds):
        """
        Computes the damping injection gain u_di for the system. This gain effectively acts as the derivative control on the joints to ensure that large speeds are penalised. 

        Args:
            jacobian (jnp.ndarray) : Robot jacobian at the current configuration
            joint_speeds (jnp.ndarray) : Joint speeds at the current configuration

        Returns:
            jnp.ndarray : Damping injection control gains for each joint. 
        """
        return -self.Kd @ jacobian @ joint_speeds 
    
    def _compute_disturb_comp(self):
        return -1