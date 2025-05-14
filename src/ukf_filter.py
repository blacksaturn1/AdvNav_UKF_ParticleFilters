from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import sqrtm

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
        self.n_states = 15
        self.n_measurements = 6
        
        self.R = np.diag([
            .0015, .0015, .0015,           # position (large if starting with vague initial pose)
            np.deg2rad(15.0)**2,np.deg2rad(15.0)**2, np.deg2rad(15.0)**2           # orientation (low confidence)
        ])
        self.check_covariance_matrix(self.R)
        self.P = np.diag([
            .5, .5, .5,           # position (large if starting with vague initial pose)
            np.deg2rad(75.0)**2, np.deg2rad(75.0)**2, np.deg2rad(75.0)**2,           # orientation (in radians; small if initial orientation is known)
            0.2, 0.2, 0.2,           # velocity (moderate confidence)
            
            0.01, 0.01, 0.01,      # gyro bias (smaller drift uncertainty if high-quality IMU)
            0.1, 0.1, 0.1        # accel bias (usually start near zero bias with modest uncertainty)
        ])
        self.check_covariance_matrix(self.P)
        self.Q = np.diag([
            .001, .001, .001,           # position (large if starting with vague initial pose)
            np.deg2rad(15.0)**2, np.deg2rad(15.0)**2, np.deg2rad(15.0)**2,           # orientation (in radians; small if initial orientation is known)
            .02, .02, .02,           # velocity (moderate confidence)
            
            0.1, 0.1, 0.1,      # gyro bias (smaller drift uncertainty if high-quality IMU)
            0.1, 0.1, 0.1        # accel bias (usually start near zero bias with modest uncertainty)
        ])
        self.check_covariance_matrix(self.Q)
        

        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        
        # Define the covariance matrices for gyroscope and accelerometer bias noise
        sigma_bg_x = 0.05
        sigma_bg_y = 0.05
        sigma_bg_z = 0.0015
        sigma_ba_x = 0.015
        sigma_ba_y = 0.015
        sigma_ba_z = 0.015
        self.Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        self.Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
        # Generate random noise for biases (Nbg and Nba)
        alpha = 0.1
        beta = 2
        kappa = 0
        self.n = self.n_states
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n

        # Weights
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        # State
        self.x = np.zeros(self.n)
        




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
    
    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        self.x = self.x.reshape(-1,1)
        sigma_points[0] = self.x.squeeze()
        # Calculate square root of the scaled covariance matrix
        try:
            sqrt_P = np.linalg.cholesky((self.n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            sqrt_P = sqrtm((self.n + self.lambda_) * self.P)
            if np.iscomplexobj(sqrt_P):
                sqrt_P = np.real(sqrt_P)
        for i in range(self.n):
            sigma_points[i + 1] = (self.x + sqrt_P[:, i].reshape(-1,1)).squeeze()
            sigma_points[i + 1 + self.n] = (self.x - sqrt_P[:, i].reshape(-1,1)).squeeze()
            
        return sigma_points
    
    def predict(self, dt, data):
        """
        Predict the next state based on the current state and measurement data.
        :param x: Current state.
        :param dt: Time step.
        :param data: Measurement data.
        :return: Predicted state.
        """
        sigma_pts = self.generate_sigma_points()
        propagated = np.array([self.fx(pt, dt,data) for pt in sigma_pts])
        self.x = np.sum(self.Wm[:, None] * propagated, axis=0)
        # Calculate the predicted covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            dx = propagated[i] - self.x
            dx = dx.reshape(-1, 1)
            P_pred += self.Wc[i] * dx @ dx.T

        # Add process noise
        P_pred += self.Q
        
        # Symmetrize and jitter
        P_pred = 0.5 * (P_pred + P_pred.T)
        P_pred += 1e-9 * np.eye(self.n)

        self.P = P_pred
        self._sigma_pts_pred = propagated
        # self.x[3:6] = (self.x[3:6] + np.pi) % (2 * np.pi) - np.pi
        return self.x.squeeze()
        
    def update(self, z):
        """
        Update the state based on the measurement data.
        :param x: Current state.
        :param data: Measurement data.
        :return: Updated state.
        """

        Z_sigma = np.array([self.hx(pt) for pt in self._sigma_pts_pred])
        z_pred = np.sum(self.Wm[:, None] * Z_sigma, axis=0)

        S = self.R.copy()
        for i in range(2 * self.n + 1):
            dz = (Z_sigma[i] - z_pred).reshape(-1, 1)
            S += self.Wc[i] * dz @ dz.T

        Pxz = np.zeros((self.n, self.n_measurements))
        for i in range(2 * self.n + 1):
            dx = (self._sigma_pts_pred[i] - self.x).reshape(-1, 1)
            dz = (Z_sigma[i] - z_pred).reshape(-1, 1)
            Pxz += self.Wc[i] * dx @ dz.T

        K = Pxz @ np.linalg.inv(S)
        # self.P = self.P - K @ S @ K.T
        # Joseph form for covariance update
        I = np.eye(self.P.shape[0])
        P_updated = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)  # Symmetrize
        P_updated += 1e-9 * np.eye(P_updated.shape[0])  # Add jitter
        self.P = P_updated
        self.x = self.x.reshape(-1, 1) + K @ (z - z_pred.reshape(-1, 1))
        return self.x.squeeze()
        
    
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
        
        xout[9:12] += np.random.multivariate_normal(np.zeros(3), self.Qg) * dt
        xout[12:15] += np.random.multivariate_normal(np.zeros(3), self.Qa) * dt

        rpy = x[3:6]
        G = self.G_Matrix(rpy)
        U_w = (np.array([data['omg']]) ).T
        q_dot = np.linalg.inv(G) @ U_w
        xout[3:6] += q_dot.squeeze()*dt
        
        U_a = (np.array([data['acc']]) ).T
        Rq_matrix = self.Rq_matrix(x[3:6])
        g = np.array([[0, 0, 9.81]]).T
        xout[6:9] = (Rq_matrix.T @ U_a - g).squeeze()*dt
        # xout[6:9] *= 0.97  # damp velocity slightly
        xout[0] = (x[6] * dt + x[0])
        xout[1] = (x[7] * dt + x[1])
        xout[2] = (x[8] * dt + x[2])
        xout[3:6] = (xout[3:6] + np.pi) % (2 * np.pi) - np.pi
        return xout
    
    
    def hx(self, x):
        """
        Measurement function.
        :param x: State vector.
        :return: Measurement vector.
        """
        hx=self.H @ x
        return hx.T
        
    
    def Rq_matrix(self, rpy):
        """
        Calculate the R matrix based on the measurement data.
        :param measurement_data: Measurement data.
        :return: R matrix.
        """
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        # r= rotation_x @ rotation_y @ rotation_z
        r= rotation_z @ rotation_y @ rotation_x
        # r = R.from_euler('zyx', rpy, degrees=False).as_matrix()
        # check = R.from_matrix(self.R).as_euler('xyz', degrees=False)
        
        return r
        
       
