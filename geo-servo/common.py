import numpy as np
import yaml

def load_inertia_params():
    """
    Loads the mass and inertial characteristics from the YAML description file, and returns an array of masses for each link and the inertia matrices

    Returns:
        np.ndarray: NumPy array of the masses for each link in order
        List: List of NumPy arrays that correspond to the inertia arrays for each link
    """
    mass_vals = np.zeros((6,))
    inertia_vals = []
    with open('./config/xarm6_inertia.yaml', 'r') as file:
        inertia_params = yaml.safe_load(file)
        for i in range(0,6):
            link = 'link' + str(i+1)
            inertia = inertia_params[link]['inertia']
            inertia_vals.append(
                np.array(
                    [[inertia['ixx'], -inertia['ixy'], -inertia['ixz']],
                     [0, inertia['iyy'], -inertia['iyz']],
                     [0, 0, inertia['izz']]]
                )
            )
            mass_vals[i] = inertia_params[link]['mass']

        return mass_vals, inertia_vals
            

