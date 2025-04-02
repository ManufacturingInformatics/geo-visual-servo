import jax.numpy as jnp

class Kinematics:
    
    def __init__(self):
        
        # Mechanical properties
        self.l1 = 0.267
        self.l2 = 0.289
        self.l3 = 0.078
        self.l4 = 0.343
        self.l5 = 0.076
        self.l6 = 0.097
        self._0theta2 = -1.385 # in radians
        self._0theta3 = 1.385 # in radians
        
        # DH parameters, in order: theta_i(offset), d, alpha, r
        self.dh_params = jnp.array( 
            [[0, self.l1, -jnp.pi/2, 0],
             [self._0theta2, 0, 0, self.l2],
             [self._0theta3, 0, -jnp.pi/2, self.l3],
             [0, self.l4, jnp.pi/2, 0],
             [0, 0, -jnp.pi/2, self.l5],
             [0, self.l6, 0, 0]]
        )
        
        # Transformation matrices
        self.transforms = jnp.zeros((6,4,4))
        
    def get_pose_dh(self, qVals: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the forward kinematics from the DH parameters.

        Args:
            qVals (jnp.ndarray): Joint values at the current configuration.

        Returns:
            jnp.ndarray: Pose in SE(3) based on the DH parameters and transformation matrices. 
        """
        for i in range(0, 6):
            self.transforms = self.transforms.at[i].set(
                [[cos(qVals[i]+self.dh_params[i][0]), -sin(qVals[i]+self.dh_params[i][0])*cos(self.dh_params[i][2]), sin(qVals[i]+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*cos(qVals[i]+self.dh_params[i][0])],
                 [sin(qVals[i]+self.dh_params[i][0]), cos(qVals[i]+self.dh_params[i][0])*cos(self.dh_params[i][2]), -cos(qVals[i]+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*sin(qVals[i]+self.dh_params[i][0])],
                 [0, sin(self.dh_params[i][2]), cos(self.dh_params[i][2]), self.dh_params[i][1]],
                 [0, 0, 0, 1]])
        temp_fk = self.transforms[0]
        for i in range(1,6):
            temp_fk = temp_fk @ self.transforms[i]    
        return temp_fk
        