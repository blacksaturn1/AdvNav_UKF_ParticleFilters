from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from measurement_data import MeasurementData
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints

np.set_printoptions(formatter={'float_kind': "{: .3f}".format})
# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
print(f"Parent directory of the script: {parent_dir_script}")

class EkfFilter:
    def __init__(self, measurement_data):
        self.measurement_data = measurement_data
        self.R = None

        sigmas = JulierSigmaPoints(n=15, kappa=0)
        
        # self.Q = Q_discrete_white_noise(dim=2, dt=1., var=2.35)
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        std_x, std_y = .3, .3
        # self.R = np.diag([std_x**2, std_y**2, std_y**2, std_y**2, std_y**2, std_y**2])
        self.R = self.P_Matrix()
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        self.ukf = UnscentedKalmanFilter(dim_x=15, dim_z=6, dt=1., hx=self.hx, fx=self.fx, points=sigmas)
        self.ukf.R = self.R
        self.ukf.Q = self.Q

        
    def predict(self, dt, data):
        """
        Predict the next state based on the current state and measurement data.
        :param x: Current state.
        :param dt: Time step.
        :param data: Measurement data.
        :return: Predicted state.
        """
        self.ukf.predict(dt,fx=self.fx,data=data)
        return self.ukf.x
    
    def update(self, z):
        """
        Update the state based on the measurement data.
        :param x: Current state.
        :param data: Measurement data.
        :return: Updated state.
        """
        self.ukf.update(z)
        
        return self.ukf.x
        
    
    def G_Matrix(self, rpy):
        # rpy = data['rpy']
        roll= rpy[0]
        pitch = rpy[1]
        c_pitch = np.cos(pitch)
        sc_02 = -np.sin(roll)*np.cos(pitch)
        s_roll = np.sin(roll)
        s_pitch = np.sin(pitch)
        c_roll_pitch = np.cos(roll)*np.cos(pitch)
        return np.array([
            [c_pitch, 0, sc_02],
            [0, 1, s_roll],
            [s_pitch, 0, c_roll_pitch],
            
        ])
    

    def fx(self,x, dt,data):
        xout = np.empty_like(x)
        # P vector
        xout[0] = x[6] * dt + x[0]
        xout[1] = x[7] * dt + x[1]
        xout[2] = x[8] * dt + x[2]
        # q vector
        G_matrix = self.G_Matrix(x[3:6])
        U_w = np.array([data['omg']]).T
        q_dot = np.linalg.inv(G_matrix) @ U_w
        xout[3:6] = q_dot.squeeze()
        U_a = np.array([data['acc']]).T
        Rq_matrix = self.Rq_matrix(data)
        g = np.array([[0, 0, -9.81]]).T
        xout[6:9] = (Rq_matrix @ U_a + g).squeeze()
        return xout
        # F = np.array([xout])
        # return F.T
    
    def hx(self, x):
        """
        Measurement function.
        :param x: State vector.
        :return: Measurement vector.
        """
        hx=self.H @ x
        return hx.T
        
    
    def P_Matrix(self):
        P = np.array([
            [ 0.012,  0.002,  0.012,  0.008,  0.009, -0.001],
            [ 0.002,  0.005,  0.002,  0.005, -0.002, -0.000],
            [ 0.012,  0.002,  0.020,  0.006,  0.008, -0.001],
            [ 0.008,  0.005,  0.006,  0.010,  0.003, -0.000],
            [ 0.009, -0.002,  0.008,  0.003,  0.008, -0.001],
            [-0.001, -0.000, -0.001, -0.000, -0.001,  0.000]])
        return P
    
    def Q_Matrix(self, data):
        pass

    def Rq_matrix(self, data):
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
        check = R.from_matrix(self.R).as_euler('xyz', degrees=False)
        
        return self.R
        
       
