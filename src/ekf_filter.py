from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from measurement_data import MeasurementData


np.set_printoptions(formatter={'float_kind': "{: .3f}".format})
# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
print(f"Parent directory of the script: {parent_dir_script}")

class EkfFilter:
    def __init__(self, measurement_data):
        self.measurement_data = measurement_data
        self.R = None
        self.Gq = np.matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.Gw = np.matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    def G_Matrix(self, data):
        rpy = data['rpy']
        roll= rpy[0]
        pitch = rpy[1]
        return np.matrix([
            [np.cos(pitch), 0, -np.sin(roll)*np.cos(pitch)],
            [0, 1, np.sin(roll)],
            [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)],
            
        ])
    
    def R_matrix(self, data):
        """
        Calculate the R matrix based on the measurement data.
        :param measurement_data: Measurement data.
        :return: R matrix.
        """
        rpy = data['rpy']
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        # self.R = rotation_z @ rotation_x @ rotation_y
        self.R = rotation_y @ rotation_x @ rotation_z
        check = R.from_matrix(self.R).as_euler('zxy', degrees=False)
        
        return self.R
        
        # self.x = None
        # self.P = None
        # self.Q = None
        # self.R = None
        # self.F = None
        # self.H = None
        # self.dt = 0.1
        # self.x0 = np.array([0, 0, 0, 0, 0, 0])
        # self.P0 = np.eye(6) * 1e-3
        # self.Q0 = np.eye(6) * 1e-3
        # self.R0 = np.eye(3) * 1e-3
        # self.F0 = np.eye(6)
        # self.H0 = np.eye(3, 6)
        # self.F0[0, 3] = self.dt
        # self.F0[1, 4] = self.dt
        # self.F0[2, 5] = self.dt
        # self.H0[0, 0] = 1
        # self.H0[1, 1] = 1
        # self.H0[2, 2] = 1
        # self.F = self.F0
        # self.H = self.H0    
