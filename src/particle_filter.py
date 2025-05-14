import numpy as np
# import cupy as cp
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm, multivariate_normal


class ParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.mean_init = np.zeros(15)
        
        
        self.weights = np.ones(num_particles) / num_particles
        self.process_model = self.fx

        # Define the covariance matrices for gyroscope and accelerometer bias noise
        sigma_bg_x = .7
        sigma_bg_y = .7
        sigma_bg_z = .7
        sigma_ba_x = 0.75
        sigma_ba_y = 0.75
        sigma_ba_z = 0.75
        self.Qg = np.diag([sigma_bg_x**2, sigma_bg_y**2, sigma_bg_z**2])  # Gyroscope bias noise covariance
        self.Qa = np.diag([sigma_ba_x**2, sigma_ba_y**2, sigma_ba_z**2])  # Accelerometer bias noise covariance
        self.H = np.array([[1, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           ])
        
        self.R = np.diag([
            .001, .001, .001,           # position (large if starting with vague initial pose)
            np.deg2rad(5.0)**2,np.deg2rad(5.0)**2, np.deg2rad(5.0)**2           # orientation (low confidence)
        ])
        self.P = np.diag([
            .1, .1, .1,           # position (large if starting with vague initial pose)
            np.deg2rad(10.0)**2, np.deg2rad(10.0)**2, np.deg2rad(10.0)**2,           # orientation (in radians; small if initial orientation is known)
            0.01, 0.01, 0.01,           # velocity
            
            0.01, 0.01, 0.01,      # gyro bias (smaller drift uncertainty if high-quality IMU)
            0.1, 0.1, 0.1        # accel bias (usually start near zero bias with modest uncertainty)
        ])
        self.particles = self.init_state_sampler(num_particles, state_dim)

    def predict(self, dt=1, control_input=None):
        for i in range(self.num_particles):
            self.particles[i] = self.process_model(self.particles[i], dt, control_input)

    def update(self, measurement):
        mvn = multivariate_normal(mean=np.zeros(6), cov=self.R )
        # Clip the angles
        self.particles[:,3:6]=np.arctan2( np.sin(self.particles[:,3:6]),np.cos(self.particles[:,3:6]) )
        measurement[3:6] = np.arctan2( np.sin(measurement[3:6]),np.cos(measurement[3:6]) )
        pos_and_orientation_diff = self.particles[:, 0:6] - measurement[0:6].T # position/orientation error for each particle 
        self.weights = mvn.pdf(pos_and_orientation_diff)
        self.weights += 1e-300  # prevent divide by zero
        self.weights /= np.sum(self.weights)
        

    def resample(self):
        N_eff = 1. / np.sum(self.weights**2)
        if N_eff < self.num_particles / 2:
            indices = np.random.choice(
                self.num_particles, size=self.num_particles, p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)
        

    def estimate(self):
        all = np.average(self.particles, weights=self.weights, axis=0)
        
        # Orientation: average on a circle by using sin/cos
        sin_roll  = np.average(np.sin(self.particles[:, 3]), weights=self.weights)
        cos_roll  = np.average(np.cos(self.particles[:, 3]), weights=self.weights)
        sin_pitch = np.average(np.sin(self.particles[:, 4]), weights=self.weights)
        cos_pitch = np.average(np.cos(self.particles[:, 4]), weights=self.weights)
        sin_yaw   = np.average(np.sin(self.particles[:, 5]), weights=self.weights)
        cos_yaw   = np.average(np.cos(self.particles[:, 5]), weights=self.weights)
        avg_roll  = np.arctan2(sin_roll, cos_roll)
        avg_pitch = np.arctan2(sin_pitch, cos_pitch)
        avg_yaw   = np.arctan2(sin_yaw, cos_yaw)
        all[3:6] = np.array([avg_roll, avg_pitch, avg_yaw])

        return all
    
    def init_state_sampler(self, num_particles, state_dim):
        # Implement the initialization of particles here
        # For example, you can sample from a uniform distribution
        # return np.random.uniform(-1, 1, (num_particles, state_dim))
        particles = np.random.multivariate_normal(self.mean_init, self.P, size=num_particles)
        return particles
    
    # def fx2(self, x, dt, data):
    #     xout = x.copy()
    #     if data.get('omg') is None or data.get('acc') is None:
    #         return xout
    #     angular_velocity = np.array([data['omg']])
    #     linear_acceleration = np.array([data['acc']])
    #     position_prev = x[0:3]
    #     orientation_prev = x[3:6]
    #     velocity_prev = x[6:9]
    #     gyro_bias_prev = x[9:12]
    #     accel_bias_prev = x[12:15]

    #     # 1. Update Orientation
    #     corrected_angular_velocity = angular_velocity - gyro_bias_prev
    #     rotation_vector = corrected_angular_velocity * dt
    #     rotation = R.from_rotvec(rotation_vector)
    #     # quaternion_prev = R.from_rotvec(orientation_prev).as_quat()
    #     quaternion_prev = R.from_euler('xyz', orientation_prev, degrees=False).as_quat()
    #     quaternion_next = rotation.as_quat() * quaternion_prev  # Quaternion multiplication: q_next = q_rot * q_prev
    #     # quaternion_next = R.from_quat(quaternion_next).as_quat() # Ensure unit quaternion (numerical stability)
    #     quaternion_next = quaternion_next / np.linalg.norm(quaternion_next)  # Ensure unit quaternion (numerical stability)
    #     xout[3:6] = R.from_quat(quaternion_next).as_euler('xyz', degrees=False)
        
    #     # 2. Update Velocity
    #     # Get rotation matrix from previous quaternion
    #     rotation_matrix_prev = R.from_quat(quaternion_prev).as_matrix()
    #     # Corrected acceleration in body frame
    #     corrected_acceleration_body = linear_acceleration - accel_bias_prev
    #     # Corrected linear acceleration in world framedata['rpy']
    #     gravity = np.array([0, 0, 9.81]).T  # Assuming Z-axis is upwards
    #     acceleration_nav = (rotation_matrix_prev @ corrected_acceleration_body.T).T + gravity
    #     # Update velocity
    #     velocity_next = velocity_prev + acceleration_nav * dt
    #     process_noise = np.diag([0.01, 0.01, 0.01])  # Process noise covariance for position
    #     velocity_next += np.random.multivariate_normal(
    #                         np.zeros(3), process_noise) * dt
    #     xout[6:9] = velocity_next
        
    #     # 3. Update Position
    #     position_next = position_prev + (velocity_next * dt) #+ (0.5 * (acceleration_nav * dt**2))
    #     process_noise_cov = np.diag([0.01, 0.01, 0.01])  # Process noise covariance for position
    #     position_next += np.random.multivariate_normal(
    #                         np.zeros(3), process_noise_cov) * dt
    #     xout[0:3] = position_next

    #     # 4. Update Biases
    #     # gyro_bias_next = gyro_bias_prev + np.random.multivariate_normal(
    #     #     np.zeros(3), self.Qg    ) * dt
    #     # accel_bias_next = accel_bias_prev + np.random.multivariate_normal(
    #     #     np.zeros(3), self.Qa    ) * dt
    #     gyro_bias_next = gyro_bias_prev + self.Nbg*dt
    #     accel_bias_next = accel_bias_prev + self.Nba*dt
        
    #     xout[9:12] = gyro_bias_next
    #     xout[12:15] = accel_bias_next
    #     return xout



    def fx(self, x, dt, data):
        xout = x.copy()
        if data.get('omg') is None or data.get('acc') is None:
            return xout
        
        gyro_bias_prev = x[9:12]
        accel_bias_prev = x[12:15]
        xout[9:12] += np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qg)*dt
        xout[12:15] += np.random.multivariate_normal(mean=np.zeros(3), cov=self.Qa)*dt
        
        G = self.G_Matrix(x[3:6])
        U_w = (np.array(data['omg']) + gyro_bias_prev).T
        q_dot = np.linalg.inv(G) @ U_w
        xout[3:6] += q_dot.squeeze()*dt 
        # handle angle wrapping
        xout[3:6] = np.arctan2(np.sin(xout[3:6]), np.cos(xout[3:6]))  # Normalize angles
        
        U_a = (np.array([data['acc']]) + accel_bias_prev ).T
        Rq_matrix = self.Rq_matrix(x[3:6])
        g = np.array([[0, 0, 9.81]]).T
        process_noise_velocity = np.random.multivariate_normal(mean=np.zeros(3), cov=np.diag([.01,.01,.01]))*dt
        xout[6:9] = (Rq_matrix.T @ U_a - g).squeeze() * dt
        xout[6:9] += process_noise_velocity
        xout[0] = (x[6] * dt + x[0])
        xout[1] = (x[7] * dt + x[1])
        xout[2] = (x[8] * dt + x[2])
        process_noise_position = np.random.multivariate_normal(mean=np.zeros(3), cov=np.diag([.75,.75,.75]))*dt
        xout[0:3] += process_noise_position
        return xout

    def Rq_matrix(self, rpy):
        rotation_x = R.from_euler('x', rpy[0], degrees=False).as_matrix()
        rotation_y = R.from_euler('y', rpy[1], degrees=False).as_matrix()
        rotation_z = R.from_euler('z', rpy[2], degrees=False).as_matrix()
        r = rotation_z @ rotation_y @ rotation_x
        # r = rotation_x @ rotation_y @ rotation_z
        return r

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