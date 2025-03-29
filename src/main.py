from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
print(f"Parent directory of the script: {parent_dir_script}")

class MeasurementData:
    def __init__(self):
        self.data_dir = pjoin(parent_dir_script, 'data', 'data')

    def loadMatlabData(self,file_name):
        """
        Load MATLAB data file.
        :param file_name: Name of the MATLAB file to load.
        :return: Loaded data.
        """
        mat_fname = pjoin(self.data_dir, file_name)
        self.mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        return self.mat_contents
    
    def checkMatlabData(self):
        """
        Check if the MATLAB data file exists.
        :return: True if the file exists, False otherwise.
        """
        
        for item in self.mat_contents['data']:
            image1 = np.array(item['img'])
            #image = cv2.cvtColor(image1, cv2.IMREAD_GRAYSCALE)
            # plt.imshow(image1,cmap='gray') # Use 'gray' for grayscale images, or other colormaps for different visualizations
            # plt.colorbar() # Optional: Add a colorbar to show the mapping of values to colors
            # plt.show()
            
            if isinstance(item['id'], int):
                cv2.imshow('Image', image1)
                cv2.waitKey(3*1000 )
                cv2.destroyAllWindows()
            # elif isinstance(item['id'], list):
            else:
                if len(item['id'])>0:
                    print(f"List: {item['id']}")
                    cv2.imshow('Image', image1)
                    cv2.waitKey(3*1000 )
                    cv2.destroyAllWindows()

    
def loadMatlabData(file_name):
    measurement_data = MeasurementData()
    mat_contents = measurement_data.loadMatlabData(file_name)
    return mat_contents

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
def main():
    data = loadMatlabData('studentdata0.mat')
    plot_trajectory(data)

def plot_trajectory(data):
    """
    Plot the trajectory of the measurement data.
    :param data: Measurement data.
    """
    # Define the trajectory data (example)
    t = np.linspace(0, 1, data['vicon'].shape[1])
    x = data['vicon'][0, :]
    y = data['vicon'][1, :] 
    z = data['vicon'][2, :]
    
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
    
if __name__ == "__main__":  
    
    main()
    
def tests():
    plot_trajectory_test()
    # mat_contents['data'][6]['id']