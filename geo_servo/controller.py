import os
import time
import sys
import jax.numpy as jnp
import jax

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
        self.min_joint_speed, self.max_joint_speed = joint_speed_limits # in rad/s
        
        # Target parameters
        assert target_pose.shape == (4,4)
        self.target_pose = target_pose
        
    def compute_gains(self, jacobian, joint_speeds):
        u_es = 0
        u_di = self._compute_damping_injection(jacobian=jacobian, joint_speeds=joint_speeds)
        u_dc = 0
        return u_es + u_di + u_dc
    
    def geodesic(self, pose, mass_matrix, jacobian) -> jnp.float64:
        """
        Computes the approximate metric geodesic distance between the provided pose and the target pose. Uses the mass matrix and Jacobian to find the Hamiltonian metric tensor of the Lie-Poisson task space of the manipulator.

        Args:
            pose (jnp.ndarray): Current pose of the manipulator in SE(3)
            mass_matrix (jnp.ndarray): Mass matrix of the manipulator at the current configuration. Normally positive semi-definite.
            jacobian (_type_): Body Jacobian of the manipulator at the current configuration. 

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
        p_1 = self.target_pose[0:3,3]
        
        # Hamiltonian metric tensor
        m_bar = jacobian.pinv().T @ mass_matrix @ jacobian.pinv()
        G_hamil =  m_bar.pinv()
        
        # Rotation and position differences
        delta_R = jnp.acos(
            (jnp.trace(G_hamil @ (R_0.T @ R_1 - jnp.eye(4))) + 2)/2
        )
        delta_p = jnp.linalg.norm(
            G_hamil @ (p_0 - p_1), ord='fro'
        )
        return jnp.sqrt(delta_R + delta_p)
    
    def _saturation(self):
        pass
    
    def _compute_energy_shaping(self):
        pass
    
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