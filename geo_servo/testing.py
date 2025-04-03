import jax.numpy as np
from robot import Robot
from configparser import ConfigParser

def deg2rad(vec):
    return (vec * np.pi)/180

def dh_to_htm(alpha=None, a=None, d=None, phi=None):
    """
    Produces the Homogeneous Transformation Matrix corresponding to the Denavit-Hartenberg parameters (alpha, a, d, phi)
    :param alpha: rad
    :param a: float
    :param d: float
    :param phi: rad
    :return: 4x4 homogeneous transformation matrix
    """

    T = np.array([[np.cos(phi), -np.sin(phi)*np.cos(alpha), np.sin(phi)*np.sin(alpha), a*np.cos(phi)],
                  [np.sin(phi), np.cos(phi)*np.cos(alpha), -np.cos(phi) * np.sin(alpha), a * np.sin(phi)],
                  [0, np.sin(alpha), np.cos(alpha), d ],
                  [0, 0, 0, 1]])
    return T


def near_zero(z):
    """
    Determines whether a scalar is zero
    :param z: scalar
    :return: bool
    """
    return abs(z) < 1e-6


def forward_kinematics_dh(dh_table, qVals=None):
    """
    Computes the Forward Kinematics given the DH Table of Denavit Hartenberg Parameters
    :param dh_table: n x 4 np.ndarray
    :return: 4x4 homogeneous transformation matrix
    """
    if qVals is not None:
        for i in range(0,6):
            dh_table = dh_table.at[i,-1].add(qVals[i].item())
    print(dh_table)

    fk = np.eye(4)

    for joint in dh_table:
        alpha, a, d, phi = joint
        tj = dh_to_htm(alpha, a, d, phi)
        fk = fk @ tj

    res = np.where(near_zero(fk), 0, np.round(fk, 3))
    return res
    

if __name__ == "__main__":
    
    parser = ConfigParser()
    parser.read('./config/robot.conf')
    ip = parser.get('xArm', 'ip')
    
    robot = Robot(ip=ip, is_radian=True)
    
    qVals = robot.joint_vals
    print(qVals.shape)

    # DH Parameters Table
    dh_table = robot.dh_params_alt # Needs to be in the format alpha, a, d, phi

    # FORWARD KINEMATICS applying DH
    fk_dh = forward_kinematics_dh(dh_table, deg2rad(qVals))

    print(f"\nDh Parameters: \n{dh_table}")
    print(f"\nForward Kinematics T0{dh_table.shape[0]} applying DH for the configuration {qVals.T}: \n{fk_dh}")