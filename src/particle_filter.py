import numpy as np
# import cupy as cp
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm, multivariate_normal


class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.mean_init = np.zeros(15)
        
        self.cov_init = np.eye(15)*0.0015
        
        self.particles = self.init_state_sampler(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        self.process_model = self.fx

        # Define the covariance matrices for gyroscope and accelerometer bias noise
        # sigma_bg_x = 1
        # sigma_bg_y = 1
        # sigma_bg_z = 1
        # sigma_ba_x = 2
        # sigma_ba_y = 2
        # sigma_ba_z = 2
        sigma_bg_x = 0.5
        sigma_bg_y = 0.5
        sigma_bg_z = 0.5
        sigma_ba_x = 0.5
        sigma_ba_y = 0.5
        sigma_ba_z = 0.5

        self.Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        self.Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
        # Generate random noise for biases (Nbg and Nba)
        self.Nbg = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)
        self.Nba = np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa)
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        self.measurement_cov = np.eye(6)*0.001
        self.measurement_cov[0, 0] = 0.001
        self.measurement_cov[1, 1] = 0.001
        self.measurement_cov[2, 2] = 0.001
        self.measurement_cov[3, 3] = 0.01
        self.measurement_cov[4, 4] = 0.01
        self.measurement_cov[5, 5] = 0.01
        
        # self.measurement_cov[0, 0] = 0.015
        # self.measurement_cov[1, 1] = 0.015
        # self.measurement_cov[2, 2] = 0.015
        # self.measurement_cov[3, 3] = 0.001
        # self.measurement_cov[4, 4] = 0.001
        # self.measurement_cov[5, 5] = 0.001
        


    def predict(self, dt=1, control_input=None):
        for i in range(self.num_particles):
            self.particles[i] = self.process_model(self.particles[i], dt, control_input)

    def update(self, measurement):
        mvn = multivariate_normal( cov=self.measurement_cov )
        pos_and_orientation_diff = self.particles[:, 0:6] - measurement[0:6].T # position/orientation error for each particle 
        self.weights = mvn.pdf(pos_and_orientation_diff)
        self.weights += 1e-300  # prevent divide by zero
        self.weights /= np.sum(self.weights)
        # mvn = multivariate_normal( cov=self.measurement_cov )
        # for i in range(self.num_particles):
        #     residual = measurement.flatten() - self.particles[i, 0:6].flatten()  # position error for each particle    
        #     self.weights[i] = mvn.pdf(residual)
        # self.weights += 1e-300  # prevent divide by zero
        # self.weights /= np.sum(self.weights)
        # return self.weights
        

        # # Compute error between particle state and measurement
        # meas_pos = measurement[0:3].T      # shape (3,)
        # meas_orient = measurement[3:6].T
        # pos_diff = self.particles[:, 0:3] - meas_pos  # position error for each particle
        # orient_diff = self.particles[:, 3:6] - meas_orient  # orientation error for each particle
        # # orient_diff = normalize_angle(orient_diff)     # account for angle wrapping
        
        # # Measurement noise standard deviations
        # pos_std = np.array([0.001, 0.001, 0.001])
        # orient_std = np.array([0.001, 0.001, 0.001])
        
        # # Compute normalized squared errors
        # pos_norm_err = (pos_diff / pos_std) ** 2       # element-wise squared error / variance
        # orient_norm_err = (orient_diff / orient_std) ** 2
        
        # # Sum of squared errors across all dimensions (position + orientation)
        # total_error = np.sum(pos_norm_err, axis=1) + np.sum(orient_norm_err, axis=1)
        # # total_error = np.sum(pos_diff, axis=1) + np.sum(orient_diff, axis=1)
        # #total_error = np.sum(pos_diff, axis=1) + orient_diff
        
        # # Compute likelihood for each particle: exp(-0.5 * total_error)
        # likelihood = np.exp(-0.5 * total_error)
        
        # # Multiply by prior weight (Bayesian update)
        # # new_weights = likelihood * self.weights
        # # new_weights = likelihood * self.weights.reshape(1000,1)
        # new_weights = likelihood * self.weights
        # # Avoid all weights becoming zero (in case of extreme outlier measurement)
        # if np.all(new_weights == 0):
        #     new_weights = np.ones_like(new_weights) * 1e-12
        
        # # Normalize weights to sum to 1
        # new_weights += 1e-300 
        # new_weights = new_weights / np.sum(new_weights)
        
        # self.weights = new_weights

    def resample(self):
        indices = np.random.choice(
            self.num_particles, size=self.num_particles, p=self.weights
        )
        self.particles = self.particles[indices]
        # self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        all = np.average(self.particles, weights=self.weights, axis=0)
        
        # # Orientation: average on a circle by using sin/cos
        # sin_roll  = np.average(np.sin(self.particles[:, 3]), weights=self.weights)
        # cos_roll  = np.average(np.cos(self.particles[:, 3]), weights=self.weights)
        # sin_pitch = np.average(np.sin(self.particles[:, 4]), weights=self.weights)
        # cos_pitch = np.average(np.cos(self.particles[:, 4]), weights=self.weights)
        # sin_yaw   = np.average(np.sin(self.particles[:, 5]), weights=self.weights)
        # cos_yaw   = np.average(np.cos(self.particles[:, 5]), weights=self.weights)
        # avg_roll  = np.arctan2(sin_roll, cos_roll)
        # avg_pitch = np.arctan2(sin_pitch, cos_pitch)
        # avg_yaw   = np.arctan2(sin_yaw, cos_yaw)
        # all[3:6] = np.array([avg_roll, avg_pitch, avg_yaw])

        return all
    
    # self.measurement_model = lambda x, measurement: np.exp(-np.linalg.norm(x - measurement) ** 2 / (2 * 0.1 ** 2))
    
    def init_state_sampler(self, num_particles, state_dim):
        # Implement the initialization of particles here
        # For example, you can sample from a uniform distribution
        # return np.random.uniform(-1, 1, (num_particles, state_dim))
        particles = np.random.multivariate_normal(self.mean_init, self.cov_init, size=num_particles)
        return particles
    
    def fx2(self, x, dt, data):
        xout = x.copy()
        if data.get('omg') is None or data.get('acc') is None:
            return xout
        angular_velocity = np.array([data['omg']])
        linear_acceleration = np.array([data['acc']])
        position_prev = x[0:3]
        orientation_prev = x[3:6]
        velocity_prev = x[6:9]
        gyro_bias_prev = x[9:12]
        accel_bias_prev = x[12:15]

        # 1. Update Orientation
        corrected_angular_velocity = angular_velocity - gyro_bias_prev
        rotation_vector = corrected_angular_velocity * dt
        rotation = R.from_rotvec(rotation_vector)
        # quaternion_prev = R.from_rotvec(orientation_prev).as_quat()
        quaternion_prev = R.from_euler('xyz', orientation_prev, degrees=False).as_quat()
        quaternion_next = rotation.as_quat() * quaternion_prev  # Quaternion multiplication: q_next = q_rot * q_prev
        # quaternion_next = R.from_quat(quaternion_next).as_quat() # Ensure unit quaternion (numerical stability)
        quaternion_next = quaternion_next / np.linalg.norm(quaternion_next)  # Ensure unit quaternion (numerical stability)
        xout[3:6] = R.from_quat(quaternion_next).as_euler('xyz', degrees=False)
        
        # 2. Update Velocity
        # Get rotation matrix from previous quaternion
        rotation_matrix_prev = R.from_quat(quaternion_prev).as_matrix()
        # Corrected acceleration in body frame
        corrected_acceleration_body = linear_acceleration - accel_bias_prev
        # Corrected linear acceleration in world framedata['rpy']
        gravity = np.array([0, 0, 9.81]).T  # Assuming Z-axis is upwards
        acceleration_nav = (rotation_matrix_prev @ corrected_acceleration_body.T).T + gravity
        # Update velocity
        velocity_next = velocity_prev + acceleration_nav * dt
        process_noise = np.diag([0.01, 0.01, 0.01])  # Process noise covariance for position
        velocity_next += np.random.multivariate_normal(
                            np.zeros(3), process_noise) * dt
        xout[6:9] = velocity_next
        
        # 3. Update Position
        position_next = position_prev + (velocity_next * dt) #+ (0.5 * (acceleration_nav * dt**2))
        process_noise_cov = np.diag([0.01, 0.01, 0.01])  # Process noise covariance for position
        position_next += np.random.multivariate_normal(
                            np.zeros(3), process_noise_cov) * dt
        xout[0:3] = position_next

        # 4. Update Biases
        # gyro_bias_next = gyro_bias_prev + np.random.multivariate_normal(
        #     np.zeros(3), self.Qg    ) * dt
        # accel_bias_next = accel_bias_prev + np.random.multivariate_normal(
        #     np.zeros(3), self.Qa    ) * dt
        gyro_bias_next = gyro_bias_prev + self.Nbg*dt
        accel_bias_next = accel_bias_prev + self.Nba*dt
        
        xout[9:12] = gyro_bias_next
        xout[12:15] = accel_bias_next
        return xout



    def fx(self, x, dt, data):
        # xout = np.empty_like(x)data['rpy']
        xout = x.copy()
        if data.get('omg') is None or data.get('acc') is None:
            return xout
         # P vector

        gyro_bias_prev = x[9:12]
        accel_bias_prev = x[12:15]

        gyro_bias_next = gyro_bias_prev + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)*dt
        accel_bias_next = accel_bias_prev + np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa)*dt
        
        xout[9:12] = gyro_bias_next
        xout[12:15] = accel_bias_next

        G = self.G_Matrix(x[3:6])
        U_w = (np.array([data['omg']]) + gyro_bias_prev).T
        q_dot = np.linalg.inv(G) @ U_w
        xout[3:6] = q_dot.squeeze()
        
        U_a = (np.array([data['acc']]) + accel_bias_prev ).T
        # Rq_matrix = self.Rq_matrix(data['rpy'])
        Rq_matrix = self.Rq_matrix(x[3:6])
        # Rq_matrix = self.pose_to_rotation_matrix(x[3:6])
        g = np.array([[0, 0, 9.81]]).T
        # xout[6:9] = (Rq_matrix.T @ (U_a - g)).squeeze()
        # xout[6:9] = (Rq_matrix.T @ (U_a - g)).squeeze()
        xout[6:9] = ((Rq_matrix.T @ U_a) - g).squeeze()
        
        # xout[0] = (x[6] * dt + x[0]) + (0.5 * (x[6] * dt**2)) + xout[9]
        # xout[1] = (x[7] * dt + x[1]) + (0.5 * (x[7] * dt**2)) + xout[10]
        # xout[2] = (x[8] * dt + x[2]) + (0.5 * (x[8] * dt**2)) + xout[11]
        
        xout[0] = (x[6] * dt + x[0]) + x[9]
        xout[1] = (x[7] * dt + x[1]) + x[10]
        xout[2] = (x[8] * dt + x[2]) + x[11]
        # xout[0] = (x[6] * dt + x[0])
        # xout[1] = (x[7] * dt + x[1])
        # xout[2] = (x[8] * dt + x[2])
        


        return xout

    def Rq_matrix(self, rpy):
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        # self.R = rotation_z @ rotation_x @ rotation_y
        # self.R = rotation_y @ rotation_x @ rotation_z
        self.r = rotation_z @ rotation_y @ rotation_x
        
        # check = R.from_matrix(self.R).as_euler('xyz', degrees=False)
        return self.r

    def pose_to_rotation_matrix(self,rpy):
        roll, pitch, yaw = rpy
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx  # ZYX convention


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