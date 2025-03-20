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
        self.target_pose = target_pose
        
    def compute_gains(self, jacobian, joint_speeds):
        u_es = 0
        u_di = self._compute_damping_injection(jacobian=jacobian, joint_speeds=joint_speeds)
        u_dc = 0
        return u_es + u_di + u_dc
    
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