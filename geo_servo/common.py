import jax.numpy as jnp

class Saturation:
    
    def __init__(self, low, high):
        self.low = low
        self.high = high
        
    def saturate(self, val):
        if (val > self.low) & (val < self.high):
            return val
        else:
            if val < self.low:
                return self.low
            else:
                return self.high
            
def deg2rad(vals):
    """
    Degrees to radians converter

    Args:
        vals: Values in degrees

    Returns: Values in radians
    """
    return (vals * jnp.pi)/180

def check_psd(mat):
    """
    Checks to see if the matrix is positive definite. Used for the computation of the distance approximation in SE(3). 
    Values:
        1 :  Matrix is positive definite
        2 : Matrix is positive semi-definite
       -1 : Matrix is neither definite or semi-definite

    Args:
        mat (jnp.ndarray): JAX NumPy array that is the mass-inertia matrix of the manipulator at a given configuration

    Returns:
        int: Integer indicator as to whether the the matrix is positive definite or positive semi-definite.  
    """
    if jnp.all(jnp.linalg.eigvals(mat) > 0):
        return 1
    elif jnp.all(jnp.linalg.eigvals(mat) >= 0):
        return 2
    else: 
        return -1

def vee_map(R: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the vee map from a rotation matrix to R^3 rotations

    Args:
        R (jnp.ndarray): Rotation matrix in SE(3)

    Returns:
        jnp.ndarray: Rotation values in R^3
    """
    arr_out = jnp.zeros((3,1))
    arr_out = arr_out.at[0].set(-R[1,2])
    arr_out = arr_out.at[1].set(R[0,2])
    arr_out = arr_out.at[2].set(-R[0,1])
    return arr_out

def pose_cross_map(pose) -> jnp.ndarray:
    """
    Computes the cross map for the pose. Converts from the R^{4x4} to R^{6x6} notation for representing vectors. 

    Args:
        pose (jnp.ndarray): Pose of the end-effector in SE(3)

    Returns:
        jnp.ndarray: Cross-map of the pose of the end-effector in SE(3)
    """
    R = pose[0:3, 0:3]
    g_cross = jnp.zeros((6,6))
    g_cross = g_cross.at[0:3, 0:3].set(R)
    g_cross = g_cross.at[3:6, 3:6].set(R)
    return g_cross
    
def momenta_cross_map(mass_matrix: jnp.ndarray, jacobian: jnp.ndarray, joint_speeds: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross map of the conjugate momenta. 

    Args:
        mass_matrix (jnp.ndarray): Mass matrix of the manipulator at the current configuration
        jacobian (jnp.ndarray): Body Jacobian of the manipulator at the current configuration
        joint_speeds (jnp.ndarray): Joint velocities of the manipulator

    Returns:
        jnp.ndarray: Returns the cross-map conjugate momenta. 
    """
    if joint_speeds.shape != (6,1):
        joint_speeds = joint_speeds.T
    j_inv = jnp.linalg.pinv(jacobian)
    m_bar = j_inv.T @ mass_matrix @ j_inv
    twist = jacobian @ joint_speeds
    momenta = m_bar @ twist
    m_cross = jnp.zeros((6,6))
    m_cross = m_cross.at[0:3,3:6].set(hat_map(momenta[0:3].reshape((1,3))))
    m_cross = m_cross.at[3:6,0:3].set(hat_map(momenta[0:3].reshape((1,3))))
    m_cross = m_cross.at[3:6,3:6].set(hat_map(momenta[3:6].reshape((1,3))))
    return m_cross

def hat_map(x: jnp.ndarray) -> jnp.ndarray:
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
            

