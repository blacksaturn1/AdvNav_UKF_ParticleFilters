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
from filterpy.kalman import MerweScaledSigmaPoints

# np.set_printoptions(formatter={'float_kind': "{: .3f}".format})
# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
# print(f"Parent directory of the script: {parent_dir_script}")

class UkfFilter:
    def __init__(self, measurement_data):
        self.debug = False
        self.measurement_data = measurement_data
        self.R = None

        points = JulierSigmaPoints(n=15, kappa=0.1)
        # points = MerweScaledSigmaPoints(n=15, alpha=.1, beta=2., kappa=0)
        
        self.Q = np.eye(15)*0.0015
        self.Q[0,0]=0.01
        self.Q[1,1]=0.01
        self.Q[2,2]=0.015
        self.Q[3,3]=0.001
        self.Q[4,4]=0.001
        self.Q[5,5]=1
        self.check_covariance_matrix(self.Q)
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        self.ukf = UnscentedKalmanFilter(dim_x=15, dim_z=6, dt=0.1, hx=self.hx, fx=self.fx, points=points)
        #  the measurement mean and noise covariance
        self.R = self.P_Matrix()
        self.ukf.R = self.R
        self.check_covariance_matrix(self.R)
        
        self.ukf.Q = self.Q

        # # Define the covariance matrices for gyroscope and accelerometer bias noise
        sigma_bg_x = 0.0025
        sigma_bg_y = 0.01
        sigma_bg_z = 0.0015
        
        
        sigma_ba_x = 0.015
        sigma_ba_y = 0.015
        sigma_ba_z = 0.015
        

        self.Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        self.Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
        # Generate random noise for biases (Nbg and Nba)

        # self.Nbg = np.random.multivariate_normal(mean=np.zeros(3), cov=Qg)
        # self.Nba = np.random.multivariate_normal(mean=np.zeros(3), cov=Qa)

    def check_covariance_matrix(self, matrix):
        if np.allclose(matrix, matrix.T):
            if self.debug:
                print("Covariance matrix is symmetric.")
        else:
            print("Covariance matrix is not symmetric.")

        # Check if positive definite
        eigenvalues = np.linalg.eigvals(matrix)
        if np.all(eigenvalues > 0):
            if self.debug:
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
        roll= rpy[0]   # phi
        pitch = rpy[1] # theta
        yaw = rpy[2]   # psi
        c_pitch = np.cos(pitch)
        sc_02 = -np.sin(roll)*np.cos(pitch)
        s_roll = np.sin(roll)
        s_pitch = np.sin(pitch)
        c_roll_pitch = np.cos(roll)*np.cos(pitch)
        return np.array([
            [c_pitch, 0, sc_02],
            [0,       1, s_roll],
            [s_pitch, 0, c_roll_pitch],
            
        ])
    

    def fx(self,x, dt,data):
        xout = x.copy()
        if data.get('omg') is None or data.get('acc') is None:
            return xout
        
        gyro_bias_prev = x[9:12]
        accel_bias_prev = x[12:15]
        # gyro_bias_next = gyro_bias_prev + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)*dt
        gyro_bias_next = (np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg))*dt
        accel_bias_next = (np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa))*dt
        xout[9:12] = gyro_bias_next
        xout[12:15] = accel_bias_next

        G = self.G_Matrix(x[3:6])
        U_w = (np.array([data['omg']]) + gyro_bias_prev).T
        q_dot = np.linalg.inv(G) @ U_w
        xout[3:6] = q_dot.squeeze()
        
        U_a = (np.array([data['acc']]) + accel_bias_prev ).T
        Rq_matrix = self.Rq_matrix(x[3:6])
        g = np.array([[0, 0, 9.81]]).T
        xout[6:9] = (Rq_matrix.T @ U_a - g).squeeze()
        
        xout[0] = (x[6] * dt + x[0]) + x[9]
        xout[1] = (x[7] * dt + x[1]) + x[10]
        xout[2] = (x[8] * dt + x[2]) + x[11]
        
        return xout
    
    
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
        P2 = np.diag([.008]*6)
        P2[0,0] = 0.008
        P2[1,1] = 0.008
        P2[2,2] = 0.008
        P2[3,3] = 0.08
        P2[4,4] = 0.008
        P2[5,5] = 0.001
        # return P2
        return P2
        
        
        
    
    def Q_Matrix(self, data):
        pass

    def Rq_matrix(self, rpy):
        """
        Calculate the R matrix based on the measurement data.
        :param measurement_data: Measurement data.
        :return: R matrix.
        """
        # rpy = data['rpy']
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        # self.R = rotation_z @ rotation_x @ rotation_y
        # self.R = rotation_y @ rotation_x @ rotation_z
        # self.R = rotation_z @ rotation_y @ rotation_x
        # r= rotation_x @ rotation_y @ rotation_z
        r= rotation_z @ rotation_y @ rotation_x
        # r = R.from_euler('zyx', rpy, degrees=False).as_matrix()
        # check = R.from_matrix(self.R).as_euler('xyz', degrees=False)
        
        return r
        
       
