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
from filterpy.common import Q_continuous_white_noise
from filterpy.kalman import JulierSigmaPoints

# np.set_printoptions(formatter={'float_kind': "{: .3f}".format})
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
        self.Q = np.diag([0]*15)
        self.Q = np.zeros((15,15))
        self.Q = np.diag([0]*15)
        self.Q1 = Q_discrete_white_noise(dim=4, dt=1., var=2.35)
        self.Q2 = Q_discrete_white_noise(dim=2, dt=1., var=2.35)
        self.Q[0:4,0:4] = self.Q1
        
        self.Q = np.diag([1]*15)
        self.Q[0:4,0:4] = self.Q1
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.Q = np.zeros((15,15))
        self.Q[14,14] = 0.1
        self.Q = np.zeros((15,15))
        self.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.Q[4:6, 4:6] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.Q[6:8, 6:8] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.Q[8:10, 8:10] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.Q[10:12, 10:12] = Q_discrete_white_noise(2, dt=1, var=0.02)
        self.Q[12:14, 12:14] = Q_discrete_white_noise(2, dt=1, var=0.02)
        
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.Q = np.diag([2.]*9 + [1.1]*6)
        self.Q = np.diag([.175]*15)
        self.Q = np.eye(15)*10.0001
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.Q = np.eye(15)*0.001
        self.Q = np.eye(15)*0.2
        self.Q[14,14] = 0.6
        self.check_covariance_matrix(self.Q)
        # self.Q[2:4,2:4] = self.Q2
        # Check if symmetric matrix
        


        # self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        std_x, std_y = .3, .3
        # self.R = np.diag([std_x**2, std_y**2, std_y**2, std_y**2, std_y**2, std_y**2])
        
        
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        self.ukf = UnscentedKalmanFilter(dim_x=15, dim_z=6, dt=1., hx=self.hx, fx=self.fx, points=sigmas)
        #  the measurement mean and noise covariance
        self.R = self.P_Matrix()
        self.ukf.R = self.R
        self.check_covariance_matrix(self.R)
        
        self.ukf.Q = self.Q

    def check_covariance_matrix(self, matrix):
        if np.allclose(matrix, matrix.T):
            print("Covariance matrix is symmetric.")
        else:
            print("Covariance matrix is not symmetric.")

        # Check if positive definite
        eigenvalues = np.linalg.eigvals(matrix)
        if np.all(eigenvalues > 0):
            print("Covariance matrix is positive definite.")
        else:
            print("Covariance matrix is not positive definite.")
    
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
        self.ukf.update(z.squeeze())
        
        return self.ukf.x
        
    
    def G_Matrix(self, rpy):
        # rpy = data['rpy']
        roll= rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
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
        g = np.array([[0, 0, 9.81]]).T
        xout[6:9] = (Rq_matrix @ U_a + g).squeeze()
        # Define the covariance matrices for gyroscope and accelerometer bias noise
        sigma_bg_x = 0.2
        sigma_bg_y = 0.2
        sigma_bg_z = 5.5
        sigma_ba_x = 0.2
        sigma_ba_y = 0.2
        sigma_ba_z = 5.5
        # Define the covariance matrices for gyroscope and accelerometer bias noise
        Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
 
        # Generate random noise for biases (Nbg and Nba)

        Nbg = np.random.multivariate_normal(mean=np.zeros(3), cov=Qg)

        Nba = np.random.multivariate_normal(mean=np.zeros(3), cov=Qa)
        xout[9:12] = x[9:12] + Nbg
        xout[9:12] = Nbg
        xout[12:15] = x[12:15] + Nba
        xout[12:15] = Nba
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
        P = np.array(
[[ 0.01248985 , 0.00179274 , 0.01191035 , 0.00812441,  0.00853663, -0.00074059],
 [ 0.00179274 , 0.00494662 , 0.00222319 , 0.00453181, -0.00188542, -0.00014287],
 [ 0.01191035 , 0.00222319 , 0.01989463,  0.00623472,  0.00840728, -0.00132054],
 [ 0.00812441 , 0.00453181 , 0.00623472 , 0.00973619,  0.00250991, -0.00037419],
 [ 0.00853663 ,-0.00188542 , 0.00840728 , 0.00250991,  0.00830289, -0.00050637],
 [-0.00074059 ,-0.00014287 ,-0.00132054, -0.00037419, -0.00050637,  0.00012994]]


        )
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
        
       
