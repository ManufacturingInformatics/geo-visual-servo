import jax.numpy as jnp
from jax.numpy import sin, cos
from se3 import SE3

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
        self.B = jnp.array(
            [[0, 0, 0, 0, 0, 0],
             [0, -1, -1, 0, -1, 0],
             [-1, 0, 0, 1, 0, 1],
             [0, -0.155, -0.4395, 0, -0.097, 0],
             [-0.207, 0, 0, 0.076, 0, 0],
             [0, 0.207, 0.1535, 0, 0.076, 0]]
        ).T
        
        # Need the zero configuration
        self.g_0 = jnp.array(
            [[1, 0, 0, 0.2074],
             [0, -1, 0, 0],
             [0, 0, -1, 0.112],
             [0, 0, 0, 1]]
        )
        
        self.se3 = SE3()
        
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
                [[cos(qVals[i].item()+self.dh_params[i][0]), -sin(qVals[i].item()+self.dh_params[i][0])*cos(self.dh_params[i][2]), sin(qVals[i].item()+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*cos(qVals[i].item()+self.dh_params[i][0])],
                 [sin(qVals[i].item()+self.dh_params[i][0]), cos(qVals[i].item()+self.dh_params[i][0])*cos(self.dh_params[i][2]), -cos(qVals[i].item()+self.dh_params[i][0])*sin(self.dh_params[i][2]), self.dh_params[i][3]*sin(qVals[i].item()+self.dh_params[i][0])],
                 [0, sin(self.dh_params[i][2]), cos(self.dh_params[i][2]), self.dh_params[i][1]],
                 [0, 0, 0, 1]])
        temp_fk = self.transforms[0]
        for i in range(1,6):
            temp_fk = temp_fk @ self.transforms[i]    
        return temp_fk
    
    def jacobian_body(self, qVals: jnp.ndarray) -> jnp.ndarray:
        """Computes the body jacobian given the list of screw axes in body form and the joint configuration
        :param b_list: 6xn matrix of the screw axes in body form (screw axes are the rows)
        :param theta_list: list of the joints configurations
        :return: 6xn jacobian matrix in body form
        """
        # we will compose J by columns
        B_list = jnp.array(self.B)
        J_b = jnp.array(B_list.T).copy().astype(float)

        T = jnp.eye(4)

        for i in range(qVals.shape[0] - 2, -1, -1):

            b = B_list[i+1, :]
            b_skew = self.se3.vec6_to_skew4(jnp.array(b * - qVals[i+1]).reshape((1,6)))
            mat_exp = self.se3.skew4_to_matrix_exp4(b_skew)
            T = T @ mat_exp

            adj_T = self.se3.adj(T)
            J_col = jnp.dot(adj_T, B_list[i, :])
            J_b = J_b.at[:,i].set(J_col)
        return J_b
    
    def fk_body(self, qVals: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the forward kinematics as a product of exponentials (twists) in the body (end-effector) frame

        Args:
            qVals (jnp.ndarray): Joint values at the current configuration

        Returns:
            jnp.ndarray: Forward kinematics as the product of exponentials
        """
        T = self.g_0
        for i, b in enumerate(self.B):
            b_line = jnp.array(b) * qVals[i].item()
            b_skew = self.se3.vec6_to_skew4(p=b_line.reshape((1,6)))
            mat_exp = self.se3.skew4_to_matrix_exp4(b_skew)
            # print(mat_exp)
            T = T @ mat_exp
            
        fk = jnp.round(jnp.where(self.se3.near_zero(T), 0, T), 4)
        return fk
        