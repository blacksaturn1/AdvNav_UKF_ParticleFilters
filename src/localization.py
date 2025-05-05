from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from measurement_data import MeasurementData
from ukf_filter import UkfFilter
from particle_filter import ParticleFilter
# np.set_printoptions(formatter={'float_kind': "{: .3f}".format})
# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
# print(f"Parent directory of the script: {parent_dir_script}")

class Localization:
    def __init__(self, file):

        """
        Initialize the Localization class.
        :param file: Name of the MATLAB file to load.
        """
        self.data_dir = pjoin(parent_dir_script, 'data', 'data')
        self.mat_contents = None
        self.actual_vicon_np = None
        self.results_np = None
        self.actual_vicon_aligned_np = None
        self.diff_matrix = None
        self.cov_matrix = None
        self.loadMatlabData(file)
        self.pf = ParticleFilter(
            num_particles=1000,
            state_dim=15
        )
    
    # def process_model(self,x, u):
    #     # simple linear motion with noise
    #     return x + 1.0 + np.random.normal(0, 1)

    # def measurement_model(self,x, z):
    #     # Gaussian likelihood
    #     return np.exp(-0.5 * ((z - x) ** 2) / 2.0)

    def init_sampler(self,N, dim):
        return np.random.uniform(-10, 10, size=(N, dim))

    def process_particle_filter(self):
        
        measurement_data = MeasurementData()
        position = None
        self.time = []
        self.results_np = None
        self.results_filtered_np = None
        self.x = np.zeros((15,1))
        
        dt = 0.
        time_last = 0.
        for data in self.mat_contents['data']:
            if isinstance(data['id'],np.ndarray):
                # This has no April tags found in the image
                if len(data['id']) == 0:
                    continue
            # Estimate the pose for each item in the data
            position,orientation = measurement_data.estimate_pose(data)  # Estimate the pose for each item in the data   
            # if not is_initialized:
            #     self.x[0:3] = position
            #     self.x[3:6] = orientation.T
            #     self.x[9:12] = np.array([[0.01,0.01,0.01]]).T
            #     self.x[12:15] = np.array([[0.01,0.01,0.01]]).T
            #     is_initialized = True
            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue  # Skip this item if pose estimation failed
            dt = data['t'] - time_last
            time_last = data['t']

            self.pf.predict(dt,data)
            z = np.hstack((np.array(position).T,orientation))
            self.pf.update(z.T)
            self.pf.resample()
            filtered_state_x = self.pf.estimate()
            result = np.hstack((np.array(position).T,orientation))
            result = np.hstack((result, np.array([[data['t']]])))
            filtered_state_x = np.hstack((filtered_state_x, np.array([data['t']])))
            # result= np.hstack((np.array(position).squeeze(),orientation,data['t']))
            self.results_np = result if self.results_np is None else np.vstack((self.results_np, result))
            self.results_filtered_np= filtered_state_x if self.results_filtered_np is None else np.vstack((self.results_filtered_np, filtered_state_x))
        
    def loadMatlabData(self,file_name):
        """
        Load MATLAB data file.
        :param file_name: Name of the MATLAB file to load.
        :return: Loaded data.
        """
        mat_fname = pjoin(self.data_dir, file_name)
        self.mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        self.actual_vicon_np = np.vstack((self.mat_contents['vicon'], np.array([self.mat_contents['time']])))
        return self.mat_contents
    
    def process_data(self):
        """
        Process the measurement data to extract relevant information.
        :return: Processed data.
        """
        measurement_data = MeasurementData()
        position = None
        ekfFilter = UkfFilter(self.mat_contents)
        self.time = []
        self.results_np = None
        self.results_filtered_np = None
        self.x = np.zeros((15,1))
        is_initialized = False
        dt = 0.
        time_last = 0.
        for data in self.mat_contents['data']:
            if isinstance(data['id'],np.ndarray):
                # This has no April tags found in the image
                if len(data['id']) == 0:
                    continue
            # Estimate the pose for each item in the data
            position,orientation = measurement_data.estimate_pose(data)  # Estimate the pose for each item in the data   
            # if not is_initialized:
            #     self.x[0:3] = position
            #     self.x[3:6] = orientation.T
            #     self.x[9:12] = np.array([[0.01,0.01,0.01]]).T
            #     self.x[12:15] = np.array([[0.01,0.01,0.01]]).T
            #     is_initialized = True
            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue  # Skip this item if pose estimation failed
            dt = data['t'] - time_last
            time_last = data['t']
            filtered_state_x = ekfFilter.predict(dt,data)
            
            z = np.hstack((np.array(position).T,orientation))
            filtered_state_x = ekfFilter.update(z.T)
            result = np.hstack((np.array(position).T,orientation))
            result = np.hstack((result, np.array([[data['t']]])))
            filtered_state_x = np.hstack((filtered_state_x, np.array([data['t']])))
            # result= np.hstack((np.array(position).squeeze(),orientation,data['t']))
            self.results_np = result if self.results_np is None else np.vstack((self.results_np, result))
            self.results_filtered_np= filtered_state_x if self.results_filtered_np is None else np.vstack((self.results_filtered_np, filtered_state_x))
        return self.results_np
    
    def plot_trajectory(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        self.__plot_trajectory_vicon__()
        self.__plot_trajectory_estimated__()
        self.__plot_trajectory_estimated_filtered__()




    def __plot_trajectory_vicon__(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        # Define the trajectory data (example)
        
        x = self.actual_vicon_np[0, :]
        y = self.actual_vicon_np[1, :] 
        z = self.actual_vicon_np[2, :]
        
        # Plot the trajectory
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(x, y, z, label='Actual', color='b', linewidth=2)  # Set color and linewidth for better visibility

        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Trajectory Plot')

        # Add a legend
        ax.legend()

        self.ax = ax
        self.fig = fig

         
    def __plot_trajectory_estimated__(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        # data = self.position_data
        # self.measurement_position_data_np = np.array(data).squeeze().T

        # Define the trajectory data (example)
        # self.results_np.T.squeeze()[6,:]
        x = self.results_np.T.squeeze()[0,:]
        y = self.results_np.T.squeeze()[1,:]
        z = self.results_np.T.squeeze()[2,:]
        
        # Plot the trajectory
        self.ax.plot(x, y, z, label='Estimated', color='r', linewidth=1, linestyle='-' )  # Set color and linewidth for better visibility

        # Set labels and title
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('3D Trajectory Plot')

        # Add a legend
        self.ax.legend()

        # Show the plot
        # plt.show()

    def __plot_trajectory_estimated_filtered__(self):
        """results_filtered_np
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        # data = self.position_data
        # self.measurement_position_data_np = np.array(data).squeeze().T

        # Define the trajectory data (example)
        # self.results_np.T.squeeze()[6,:]
        x = self.results_filtered_np.T.squeeze()[0,:]
        y = self.results_filtered_np.T.squeeze()[1,:]
        z = self.results_filtered_np.T.squeeze()[2,:]
        
        # Plot the trajectory
        self.ax.plot(x, y, z, label='Filtered Estimate', color='green', linewidth=1, linestyle='-' )  # Set color and linewidth for better visibility

        # Set labels and title
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Z-axis')
        self.ax.set_title('3D Trajectory Plot')

        # Add a legend
        self.ax.legend()

        # Show the plot
        plt.show()


    


    def plot_orientation(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        
        roll  = self.actual_vicon_np[3, :]
        pitch = self.actual_vicon_np[4, :]
        yaw   = self.actual_vicon_np[5, :]
        x = self.actual_vicon_np[-1, :]
        
        
        # Plot the trajectory
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        fig.suptitle('Roll / Pitch / Yaw Plot')
        # x = self.results_filtered_np.T.squeeze()[0,:]
        # y = self.results_filtered_np.T.squeeze()[1,:]
        # z = self.results_filtered_np.T.squeeze()[2,:]
        
        # Plot the trajectory
        axs[0].plot(x, roll, label='Actual', color='b', linewidth=1)  # Set color and linewidth for better visibility
        axs[0].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[3,:], label='Estimated', color='r', linewidth=1)  # Set color and linewidth for better visibility
        axs[0].plot(self.results_filtered_np.T.squeeze()[15,:], self.results_filtered_np.T.squeeze()[3,:], label='Filtered', color='g', linewidth=1)  # Set color and linewidth for better visibility

        axs[1].plot(x, pitch, label='Actual', color='b', linewidth=1)  # Set color and linewidth for better visibility
        axs[1].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[4,:], label='Estimated', color='r', linewidth=1)  # Set color and linewidth for better visibility
        axs[1].plot(self.results_filtered_np.T.squeeze()[15,:], self.results_filtered_np.T.squeeze()[4,:], label='Filtered', color='g', linewidth=1)  # Set color and linewidth for better visibility

        axs[2].plot(x, yaw, label='Actresults_filtered_npual', color='b', linewidth=1)  # Set color and linewidth for better visibility
        axs[2].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[5,:], label='Estimated', color='r', linewidth=1)  # Set color and linewidth for better visibility
        axs[2].plot(self.results_filtered_np.T.squeeze()[15,:], self.results_filtered_np.T.squeeze()[5,:], label='Filtered', color='g', linewidth=1)  # Set color and linewidth for better visibility
        
        # Set labels and title
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Roll (rad)')
        axs[0].set_title('Roll Plot')
        axs[0].legend()
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Pitch (rad)')
        axs[1].set_title('Pitch Plot')
        axs[1].legend()
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Yaw (rad)')
        axs[2].set_title('Yaw Plot')    
        axs[2].legend()
        plt.subplots_adjust(wspace=0.4, hspace=0.6) # Adjust values as needed
        
        # Show the plot
        plt.show()
        return fig,axs
    
    def interpolate(self,time_target,t1, t2,y1, y2):
        """
        Interpolate the data to match the target time.
        :param x_target: Target time values.
        :param y_target: Target data values.
        :param x_source: Source time values.
        :param y_source: Source data values.
        :return: Interpolated data.
        """
        interpolated_data = y1 + ((time_target - t1) * (y2 - y1) / (t2 - t1))
        return interpolated_data

    def calculate_covariance(self):
        """
        Calculate the covariance of the estimated trajectory.
        :return: Covariance matrix.
        """
        if self.results_np is None:
            print("No results available to calculate covariance.")
            return None
        
        self.actual_vicon_aligned_np = None
        for idx,x_measurement_model in enumerate(self.results_np[:, -1]):
            x = float(x_measurement_model)
            min_idx = np.argmin(self.actual_vicon_np[-1,:] < x)
            if min_idx == 0:
                continue
            if min_idx == self.actual_vicon_np[-1,:].shape[0]-1:
                min_idx = min_idx-1
            x_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[0,min_idx],
                    self.actual_vicon_np[0,min_idx+1])
            y_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[1,min_idx],
                self.actual_vicon_np[1,min_idx+1])
            z_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[2,min_idx],
                self.actual_vicon_np[2,min_idx+1])
            roll_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[3,min_idx],
                self.actual_vicon_np[3,min_idx+1])
            pitch_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[4,min_idx],
                self.actual_vicon_np[4,min_idx+1])
            yaw_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[5,min_idx],
                self.actual_vicon_np[5,min_idx+1])
            new_row = [x_interpolated,y_interpolated,z_interpolated,
                roll_interpolated,
                pitch_interpolated,
                yaw_interpolated,x]
            
            self.actual_vicon_aligned_np = new_row if self.actual_vicon_aligned_np is None \
                else np.vstack((self.actual_vicon_aligned_np,new_row))
        max_idx = min(self.actual_vicon_aligned_np.shape[0], self.results_np.shape[0])
        self.diff_matrix = self.actual_vicon_aligned_np.T[0:6,:max_idx] - self.results_np.T.squeeze()[0:6,:max_idx] 
        temp_matrix = None
        for idx, row in enumerate(self.diff_matrix.T):
            v = np.matrix(row).T @ np.matrix(row)
            if temp_matrix is None:
                temp_matrix = v
            else:    
                temp_matrix += v
        self.cov_matrix = temp_matrix / (self.diff_matrix.shape[1]-1)
        print("Covariance Matrix:")
        print(self.cov_matrix)

        # Check if symmetric matrix
        if np.allclose(self.cov_matrix, self.cov_matrix.T):
            print("Covariance matrix is symmetric.")
        else:
            print("Covariance matrix is not symmetric.")

        # Check if positive definite
        eigenvalues = np.linalg.eigvals(self.cov_matrix)
        if np.all(eigenvalues > 0):
            print("Covariance matrix is positive definite.")
        else:
            print("Covariance matrix is not positive definite.")

        return self.cov_matrix

    def calculate_rmse(self):
        """
        Calculate the covariance of the estimated trajectory.
        :return: Covariance matrix.
        """
        if self.results_filtered_np is None:
            print("No results available to calculate RSME.")
            return None
        
        self.actual_vicon_aligned_np = None
        for idx,x_measurement_model in enumerate(self.results_filtered_np[:, -1]):
            x = float(x_measurement_model)
            min_idx = np.argmin(self.actual_vicon_np[-1,:] < x)
            if min_idx == 0:
                continue
            if min_idx == self.actual_vicon_np[-1,:].shape[0]-1:
                min_idx = min_idx-1
            x_interpolated = self.interpolate(x,
                    self.actual_vicon_np[-1,min_idx],
                    self.actual_vicon_np[-1,min_idx+1],
                    self.actual_vicon_np[0,min_idx],
                    self.actual_vicon_np[0,min_idx+1])
            y_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[1,min_idx],
                self.actual_vicon_np[1,min_idx+1])
            z_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[2,min_idx],
                self.actual_vicon_np[2,min_idx+1])
            roll_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[3,min_idx],
                self.actual_vicon_np[3,min_idx+1])
            pitch_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[4,min_idx],
                self.actual_vicon_np[4,min_idx+1])
            yaw_interpolated = self.interpolate(x,
                self.actual_vicon_np[-1,min_idx],
                self.actual_vicon_np[-1,min_idx+1],
                self.actual_vicon_np[5,min_idx],
                self.actual_vicon_np[5,min_idx+1])
            new_row = [x_interpolated,y_interpolated,z_interpolated,
                roll_interpolated,
                pitch_interpolated,
                yaw_interpolated,x]
            
            self.actual_vicon_aligned_np = new_row if self.actual_vicon_aligned_np is None \
                else np.vstack((self.actual_vicon_aligned_np,new_row))
        max_idx = min(self.actual_vicon_aligned_np.shape[0], self.results_np.shape[0])
        
        self.diff_matrix_estimated = self.actual_vicon_aligned_np.T[0:3,:max_idx] - self.results_np.T.squeeze()[0:3,:max_idx] 
        distance_sum = 0
        for x in range(self.diff_matrix_estimated.T.shape[0]):
            distance_sum += (self.diff_matrix_estimated.T[x,0]**2 + 
                             self.diff_matrix_estimated.T[x,1]**2 + 
                             self.diff_matrix_estimated.T[x,2]**2)**0.5
        
        rmse_measurement_model = np.sqrt(distance_sum/self.diff_matrix_estimated.T.shape[0])
        # print("RMSE of measurement model: ", rmse_measurement_model)


        self.diff_matrix_estimated = self.actual_vicon_aligned_np.T[0:3,:max_idx] - self.results_filtered_np.T.squeeze()[0:3,:max_idx] 
        distance_sum = 0
        for x in range(self.diff_matrix_estimated.T.shape[0]):
            distance_sum += (self.diff_matrix_estimated.T[x,0]**2 + 
                             self.diff_matrix_estimated.T[x,1]**2 + 
                             self.diff_matrix_estimated.T[x,2]**2)**0.5
        
        rmse_filtered = np.sqrt(distance_sum/self.diff_matrix_estimated.T.shape[0])
        # print("RMSE of Filtered: ", rmse_filtered)
        rmse_difference = rmse_measurement_model - rmse_filtered
        print("RMSE difference: ", rmse_difference)
        return rmse_difference
    

        
        


