import jax.numpy as jnp
import yaml
from configparser import ConfigParser

def load_inertia_params():
    """
    Loads the mass and inertial characteristics from the YAML description file, and returns an array of masses for each link and the inertia matrices

    Returns:
        mass_vals (jnp.ndarray): JAX NumPy array of the mass values
        inertia_vals (list): List of JAX NumPy arrays for the inertia params for each joint
    """
    mass_vals = jnp.zeros((6,))
    inertia_vals = []
    with open('./config/xarm6_inertia.yaml', 'r') as file:
        inertia_params = yaml.safe_load(file)
        for i in range(0,6):
            link = 'link' + str(i+1)
            inertia = inertia_params[link]['inertia']
            inertia_vals.append(
                jnp.array(
                    [[inertia['ixx'], -inertia['ixy'], -inertia['ixz']],
                     [0, inertia['iyy'], -inertia['iyz']],
                     [0, 0, inertia['izz']]]
                )
            )
            mass_vals = mass_vals.at[i].set(inertia_params[link]['mass'])
        return mass_vals, inertia_vals
    
def load_tcp_params() -> dict:
    """
    Loads the tool offset and weight parameters to allow for gravity compensation

    Returns:
        dict: Dictionary of weight and TCP offset values to be used in the gravity compensation of the robot arm
    """
    parser = ConfigParser()
    parser.read('./config/robot.conf')
    tcp_weight = parser.get('TCP', 'weight')
    tcp_cx = parser.get('TCP', 'cx')
    tcp_cy = parser.get('TCP', 'cy')
    tcp_cz = parser.get('TCP', 'cz')
    tcp_params = {
        'weight': float(tcp_weight),
        'cx': float(tcp_cx),
        'cy': float(tcp_cy),
        'cz': float(tcp_cz)
    }
    return tcp_params
            

