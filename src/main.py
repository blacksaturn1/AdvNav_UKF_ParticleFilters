from os.path import dirname, join as pjoin
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from measurement_data import MeasurementData
from localization import Localization
# np.set_printoptions(formatter={'float_kind': "{: .3f}".format})

# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
# print(f"Parent directory of the script: {parent_dir_script}")



##########################################################################
    
def loadMatlabData(file_name):
    measurement_data = MeasurementData()
    mat_contents = measurement_data.loadMatlabData(file_name)
    return mat_contents,measurement_data

def checkMatlabData(file_name):
    measurement_data = MeasurementData()
    mat_contents = measurement_data.loadMatlabData(file_name)
    measurement_data.checkMatlabData()

def check_data():
    checkMatlabData('studentdata0.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata1.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata2.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata3.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata4.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata5.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata6.mat')
    print("-----------------------------------------------------")
    checkMatlabData('studentdata7.mat')

def plot_trajectory_test():
    """
    Plot the trajectory of the measurement data.
    :param data: Measurement data.
    """
    # Define the trajectory data (example)
    t = np.linspace(0, 10, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = t
    # Plot the trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the trajectory
    ax.plot(x, y, z, label='Trajectory')

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Trajectory Plot')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

def get_world_corners_test():
    data,m = loadMatlabData('studentdata0.mat')  # Load the MATLAB data file
    p = m.get_corners_world_frame(0)
    print("Corners for AprilTag index 0 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(1)
    print("Corners for AprilTag index 1 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(12)
    print("Corners for AprilTag index 12 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(24)
    print("Corners for AprilTag index 24 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(36)
    print("Corners for AprilTag index 36 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(48)
    print("Corners for AprilTag index 48 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(60)
    print("Corners for AprilTag index 60 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(72)
    print("Corners for AprilTag index 72 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(84)
    print("Corners for AprilTag index 84 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(96)
    print("Corners for AprilTag index 96 in world frame:")
    print(p)  # Print the corners in the world frame
    
    
    p = m.get_corners_world_frame(61)
    print("Corners for AprilTag index 61 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(73)
    print("Corners for AprilTag index 73 in world frame:")
    print(p)  # Print the corners in the world frame
  
def tests():
    plot_trajectory_test()
    get_world_corners_test()
    check_data()
    get_world_corners_test()
    
    # process_measurement_data('studentdata0.mat')
    # process('studentdata0.mat')
    
    # tag_corners_world = generate_tag_corners()
    # mat_contents['data'][6]['id']

def process_particle_filter(file_name, particle_count=250):
    """
    Process the measurement data from the specified MATLAB file.
    :param file_name: Name of the MATLAB file to process.
    :return: Processed data.
    """
    rmse = .0    
    localization = Localization(file_name)
    localization.process_particle_filter(particle_count=particle_count)
    # localization.process_data()
    # # localization.calculate_covariance()
    rmse,_ = localization.calculate_rmse()
    localization.plot_trajectory()  # Plot the trajectory
    localization.plot_orientation()  # Plot the roll trajectory

    return rmse
    
def process_ukf_filter(file_name):
    """
    Process the measurement data from the specified MATLAB file.
    :param file_name: Name of the MATLAB file to process.
    :return: Processed data.
    """
    rmse = .0    
    localization = Localization(file_name)
    localization.process_ukf_filter()
    # localization.calculate_covariance()
    rmse,_ = localization.calculate_rmse()
    localization.plot_trajectory()  # Plot the trajectory
    localization.plot_orientation()  # Plot the roll trajectory

    return rmse
    
def run_particle_filter_experiment(particle_count):
    results = []
    # DONT RUN FOR ALL FILES BECAUSE OF studentdata0 doesnt have 'omg' or 'acc' data
    # results.append(process_particle_filter('studentdata0.mat', particle_count))

    results.append(process_particle_filter('studentdata1.mat', particle_count))
    results.append(process_particle_filter('studentdata2.mat', particle_count))
    results.append(process_particle_filter('studentdata3.mat', particle_count))
    # results.append(process_particle_filter('studentdata4.mat', particle_count))
    results.append(process_particle_filter('studentdata5.mat', particle_count))
    results.append(process_particle_filter('studentdata6.mat', particle_count))
    results.append(process_particle_filter('studentdata7.mat', particle_count))
    print(f"RMSE results for all datasets with Particle Filter [{particle_count}]:", sum(results)/len(results))

def run_ukf_filter_experiment():
    results = []
    # results.append(process_ukf_filter('studentdata0.mat'))
    results.append(process_ukf_filter('studentdata1.mat'))
    results.append(process_ukf_filter('studentdata2.mat'))
    results.append(process_ukf_filter('studentdata3.mat'))
    # results.append(process_ukf_filter('studentdata4.mat')) # problem with this file
    results.append(process_ukf_filter('studentdata5.mat'))
    results.append(process_ukf_filter('studentdata6.mat'))
    results.append(process_ukf_filter('studentdata7.mat'))
    print("RMSE results for all datasets with UKF Filter:", sum(results)/len(results))

if __name__ == "__main__":
    
    # Ukf Filter
    # run_ukf_filter_experiment()
    # Particle Filter
    run_particle_filter_experiment(250)
    # run_particle_filter_experiment(500)
    # run_particle_filter_experiment(750)
    # run_particle_filter_experiment(1000)
    # run_particle_filter_experiment(2000)
    # run_particle_filter_experiment(3000)
    # run_particle_filter_experiment(4000)
    # run_particle_filter_experiment(5000)