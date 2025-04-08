import jax.numpy as jnp
from common import hat_map

class SE3:

    def vec6_to_skew4(self, p:jnp.ndarray) -> jnp.ndarray:
        """
        Function returns the skew symmetric representation of the twist

        Args:
            p (jnp.ndarray): S vector in the shape of (1,6) (w, v)

        Returns:
            jnp.ndarray: _description_
        """
        assert p.shape == (1,6)
        omega = p[:,0:3]
        v = p[:,3:]
        p_sk = self.hat_map(omega)
        twist_sk = jnp.r_[jnp.c_[p_sk, v.reshape(3,1)], [[0,0,0,0]]]
        return twist_sk
    
    def skew4_to_matrix_exp4(self, s_sk: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the matrix exponential of the skew symmetric representation in SE(3)

        Args:
            s_sk (jnp.ndarray): Skew-symmetric matrix for a pose in SE(3)

        Returns:
            jnp.ndarray: Matrix exponential
        """
        omegatheta_sk = s_sk[0:3, 0:3]
        omegatheta = self.vee_map(omegatheta_sk)  # 3D vector of exponential coordinates OmegaTheta
        vtheta = s_sk[0:3, 3]
        theta = self.axis_angle(omegatheta)[1]
        omega_sk = omegatheta_sk /theta
        matexp3 = self.skew3_to_matrix_exp3(omegatheta_sk)
        G = jnp.eye(3)*theta + (1 - jnp.cos(theta))*omega_sk + (theta - jnp.sin(theta)) * jnp.dot(omega_sk, omega_sk)
        v = jnp.dot(G, vtheta)/theta
        matexp4 = jnp.r_[jnp.c_[matexp3, v], [[0, 0, 0, 1]]]
        return matexp4

    def skew3_to_matrix_exp3(self, p_sk: jnp.ndarray) -> jnp.ndarray:

        ptheta = self.vee_map(p_sk) # exponential coordinates OmegaTheta
        theta = self.axis_angle(ptheta)[1]
        p_sk_pure = p_sk / theta
        mat_exp = jnp.array(jnp.eye(3) + jnp.sin(theta) * p_sk_pure + (1 - jnp.cos(theta)) * (p_sk_pure @ p_sk_pure))
        res = jnp.where(self.near_zero(mat_exp), 0, mat_exp)
        return res
        
    def vee_map(self, p_skew: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the inverse isomorphism that maps from SO(3) to R^3

        Args:
            p_skew (jnp.ndarray): Skew-symmetric rotation matrix in SO(3)

        Returns:
            jnp.ndarray: R^3 vector of the rotations
        """
        p = jnp.r_[[p_skew[2][1], p_skew[0][2], p_skew[1][0]]]
        return p
    
    def hat_map(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the skew-symmetric map from R^3 to SE(3). 

        Args:
            x (jnp.ndarray): Input vector in R^3

        Returns:
            jnp.ndarray: Returns hat map of the 
        """
        assert x.shape == (1,3)
        x_hat = jnp.array([
            [0, -x[0][2], x[0][1]],
            [x[0][2], 0, -x[0][0]],
            [-x[0][1], x[0][0], 0]
        ])
        return x_hat
    
    def near_zero(self, z: float, f:float =1e-6 ) -> bool:
        """
        Checks to see if the absolute value of 

        Args:
            z (float): Scalar value to be rounded. 

        Returns:
            bool: Boolean corresponding to whether the value is small enough to be rounded. 
        """
        return abs(z) < f
    
    def axis_angle(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the axis angle representation of a 3D vector of exponential coordinates

        Args:
            p (jnp.ndarray): _description_

        Returns:
            jnp.ndarray: _description_
        """
        return p / jnp.linalg.norm(p), jnp.linalg.norm(p)
    
    def adj(self, T: jnp.ndarray) -> jnp.ndarray:
        """ Computes the 6X6 skew symmetric adjoint representation of a 4x4 transformation matrix T
        :param T: 4x4 homogeneous transformation matrix in SE(3)
        :return adj_T : 6x6 adjoint representation of T
        """
        assert T.shape == (4,4)
        R = T[0:3, 0:3]
        p = T[0:3, 3].reshape((1,3))
        p_sk = self.hat_map(p)
        return jnp.r_[jnp.c_[R, jnp.zeros((3, 3))], jnp.c_[p_sk @ R, R]]

    def rot2RPY(self, R: jnp.ndarray) -> jnp.ndarray:
        #TODO: Add a function to convert from rotation matrix to RPY values. 
        pass
            