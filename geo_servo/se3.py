import jax.numpy as jnp
from common import hat_map

def vec6_to_skew3(p:jnp.ndarray) -> jnp.ndarray:
    """
    Function returns the skew symmetric representation of the twist

    Args:
        p (jnp.ndarray): S vector in the shape of (1,6) (w, v)

    Returns:
        jnp.ndarray: _description_
    """
    assert p.shape == (1,6)
    omega = s[0:3]
    v = s[3:]
    p_sk = hat_map(omega)
    twist_sk = jnp.r_[jnp.c_[p_sk, v.reshape(3,1)], [[0,0,0,0]]]
    return twist_sk
    