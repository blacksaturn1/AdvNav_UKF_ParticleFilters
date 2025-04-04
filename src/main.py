from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float_kind': "{: .3f}".format})

# Get parent directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_script = os.path.dirname(script_dir)
print(f"Parent directory of the script: {parent_dir_script}")

class MeasurementData:
    def __init__(self,ax=None, fig=None):
        self.ax = ax
        self.fig = fig
        self.data_dir = pjoin(parent_dir_script, 'data', 'data')
        self.camera_matrix = np.array([
            [314.1779, 0.0, 199.4848],
            [0.0, 314.2218, 113.7838],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911], dtype=np.float64)
        self.camera_to_world_transform = np.array([
            [1.0, 0.0, 0.0, -0.04],  # Rotation matrix (identity)
            [0.0, 1.0, 0.0, 0.0],  # Rotation matrix (identity)
            [0.0, 0.0, 1.0, -0.03],  # Translation along Z-axis (example value)
            [0.0, 0.0, 0.0, 1.0]   # Homogeneous coordinate
        ], dtype=np.float64)
        # = np.linalg.inv(self.camera_to_world_transform)  # Inverse of the camera to world transform
        self.transform_from_camera_to_drone = self.homogeneous_matrix(-0.04, 0.0, -0.03, np.pi, 0, np.pi/4)  # Example values for translation and rotation
        print("Camera to World Transform Matrix:")
        print(self.transform_from_camera_to_drone)

    
    def apply_transform(self,transform, x):
        '''
        https://alexsm.com/homogeneous-transforms/
        '''
        x_h = np.ones((x.shape[0] + 1, x.shape[1]), dtype=x.dtype)
        x_h[:-1, :] = x
        x_t = np.dot(transform, x_h)
        # return x_t[:3] / x_t[-1]
        return x_t[:3]
    
    def loadMatlabData(self,file_name):
        """
        Load MATLAB data file.
        :param file_name: Name of the MATLAB file to load.
        :return: Loaded data.
        """
        mat_fname = pjoin(self.data_dir, file_name)
        self.mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        return self.mat_contents
    
    def get_corners_world_frame(self,april_tag_index):
        """
        Get the corners of the AprilTag in the world frame.
        :param
        april_tag_index: Index of the AprilTag in the data.
        :return: 3D points of the corners in the world frame as a 3x4 matrix.
        """
        # Get the index of the AprilTag from the data
        col = april_tag_index//12
        row = april_tag_index % 12
        square_size = 0.152  
        space_size = 0.152
        columns_3_4_and_6_7 = 0.178
        difference = columns_3_4_and_6_7 - square_size # Difference between the square size and the space size for columns 6 and 7
        
        # Calculate x
        p1_x =((row) * (square_size + space_size))+square_size  # X coordinate based on the row index
        p2_x = p1_x
        p3_x = p1_x-square_size
        p4_x = p3_x
        
        # Calculate y
        p1_y =((col) * (square_size + space_size))  # Y coordinate based on the row index
        # Adjust for columns 6 and 7
        if col >= 3:  # For columns 6 and 7
            p1_y+=difference  # Adjust y coordinate for columns 6 and 7
        if col >= 5:  # For columns 6 and 7
            p1_y+=difference  # Adjust y coordinate for columns 6 and 7
        p2_y = p1_y+square_size  # Y coordinate for the second point (bottom right)
        p3_y = p2_y  # Y coordinate for the  point (upper right)
        p4_y = p1_y  # Y coordinate for the  point (upper left)

        # Create the 3D points in the world frame
        p1 = np.array([p1_x, p1_y, 0.0])  # Top left corner
        p2 = np.array([p2_x, p2_y, 0.0])  # Bottom left corner
        p3 = np.array([p3_x, p3_y, 0.0])  # Bottom right corner
        p4 = np.array([p4_x, p4_y, 0.0])  # Top right corner
        object_points = np.array([p1, p2, p3, p4], dtype=np.float64)  # Create an array of the corners
        return object_points
        

    def process_measurement_data(self):
        """
        Process the measurement data to extract relevant information.
        :return: Processed data.
        """
        position_data = []
        orientation_data = []
        position = None
        last_position = None
        for data in self.mat_contents['data']:
            if isinstance(data['id'],np.ndarray):
                if len(data['id']) == 0:
                    if len(position_data) == 0:
                        continue
                    else:
                        continue
                        # If there are no more AprilTags, return the processed data
                        position_data.append(last_position)  # Append the last estimate pose result to processed data
                        orientation_data.append(last_position)  
                        continue  # Skip to the next iteration if there are no AprilTags detected
                    
            # Estimate the pose for each item in the data
            position,orientation = self.estimate_pose(data)  # Estimate the pose for each item in the data   
            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue  # Skip this item if pose estimation failed
            last_position = position  # Store the last valid position
            position_data.append(position)  # Append the last estimate pose result to processed data
            orientation_data.append(orientation)  
        
        return position_data , orientation_data


    def estimate_pose(self, data):
        """
        Estimate the pose of an object using the given image points and object points.
        :param
        data: single element of the input data
        :param object_points: 3D points in the object coordinate system.
        :return: Position and estimate of the robot in the world frame.
        """
        image1 = np.array(data['img'])
        processed_data = {}
        if isinstance(data['id'], int):
            # Process single AprilTag
            objectPoints = self.get_corners_world_frame(data['id'])
            imagePoints = np.array([data['p1'], data['p2'], data['p3'], data['p4']], dtype=np.float64)  # Create an array of the corners
            if np.any(imagePoints < 0):
                print(f"Warning: Invalid image points for AprilTag ID {data['id']}. Skipping this tag.")
                return None,None
            # # Ensure imagePoints is a 2D array of shape (4, 2)
            # if imagePoints.shape[0] != 4 or imagePoints.shape[1] != 2:
            #     print(f"Error: Invalid image points shape for AprilTag ID {data['id']}. Expected (4, 2), got {imagePoints.shape}.")
            #     return None,None
            
            retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.camera_matrix, self.dist_coeffs)
            if retval is None:
                print("Error: solvePnP failed to estimate pose.")
                return None,None
            
            # If solvePnP is successful, it returns the rotation vector (rvec) and translation vector (tvec)``
            rotM, _ = cv2.Rodrigues(rvec)
            cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)

            # orientation_vec = self.apply_transform(self.transform_from_camera_to_drone, tvec)  # Apply the camera to world transform
            cameraPosition = np.clip(cameraPosition, 0, None)  # Clip the camera position to avoid negative values
            # if np.any(cameraPosition<-0.05):
            #     # Check if the translation vector has any negative values
            #     print(f"Warning: Invalid translation vector for AprilTag ID {data['id']}. Skipping this tag.")
            #     return None, None  # Skip this tag if the translation vector is invalid
            processed_data={
                'image': image1,
                'id': data['id'],
                'objectPoints': objectPoints,
                'rvec': rvec,  # Rotation vector
                'tvec': tvec,   # Translation vector
                'rotM': rotM,
                'cameraPosition': cameraPosition  # This is the position of the object in the world frame
            }
        else:
            # Process multiple AprilTags
            
            for i,tag_id in enumerate(data['id']):
                objectPoints = self.get_corners_world_frame(tag_id)
                p1=np.array([data['p1'][0][i],data['p1'][1][i]])  
                p2=np.array([data['p2'][0][i],data['p2'][1][i]])  
                p3=np.array([data['p3'][0][i],data['p3'][1][i]])  
                p4=np.array([data['p4'][0][i],data['p4'][1][i]])
                if p1[0]<0 or p1[1]<0 or p2[0]<0 or p2[1]<0 or p3[0]<0 or p3[1]<0 or p4[0]<0 or p4[1]<0:
                    print(f"Warning: Invalid image points for AprilTag ID {tag_id}. Skipping this tag.")
                    continue  # Skip this tag if any of the image points are invalid  
                imagePoints = np.array([p1,p2,p3,p4], dtype=np.float64)  # Create an array of the corners
                retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.camera_matrix, self.dist_coeffs)
                if retval is None or retval == False:
                    print(f"Error: solvePnP failed for AprilTag ID {tag_id}. Skipping this tag.")
                    continue
                
                rotM, _ = cv2.Rodrigues(rvec)

                cameraPosition = -np.matrix(rotM).T * np.matrix(tvec)
                cameraPosition = np.clip(cameraPosition, 0, None)  # Clip the camera position to avoid negative values
                # orientation_vec = self.apply_transform(self.transform_from_camera_to_drone, tvec)  # Apply the camera to world transform
                # if np.any(cameraPosition < 0):
                #     # Check if the cameraPosition vector has any negative values
                #     print(f"Warning: Invalid cameraPosition vector for AprilTag ID {tag_id}. Skipping this tag.")
                #     continue
                processed_data = {
                    'image': image1,
                    'id': data['id'],
                    'objectPoints': objectPoints,
                    'rvec': rvec,  # Rotation vector
                    'tvec': tvec,   # Translation vector
                    'rotM': rotM,
                    'cameraPosition': cameraPosition  # This is the position of the object in the world frame
                }
                break # Only process the first tag in the list for demonstration purposes
        # return processed_data['orientation_vec'],processed_data['rotM']
        if 'cameraPosition' not in processed_data or processed_data['cameraPosition'] is None:
            print("****************************Error: No valid orientation vector found.")
            return None, None
        return processed_data['cameraPosition'],processed_data['rotM']
    
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

    def translation_matrix(self,tx, ty, tz):
        """Creates a translation matrix."""
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_x(self,angle):
        """Creates a rotation matrix around the x-axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_y(self,angle):
        """Creates a rotation matrix around the y-axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

    def rotation_matrix_z(self,angle):
        """Creates a rotation matrix around the z-axis."""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def homogeneous_matrix(self,tx, ty, tz, rx, ry, rz):
        """
        Combines translation and rotation into a single homogeneous matrix.
        Google Search: homogeneous transformation matrix python in 3D
        """
        translation = self.translation_matrix(tx, ty, tz)
        rotation_x = self.rotation_matrix_x(rx)
        rotation_y = self.rotation_matrix_y(ry)
        rotation_z = self.rotation_matrix_z(rz)
        
        # Apply rotations in ZYX order (common convention)
        rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        
        # Combine rotation and translation
        transform = np.dot(translation, rotation)
        
        return transform
    
    def plot_trajectory_estimated(self,data):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        data_np = np.array(data).squeeze().T

        # data2 = data.T
        # data2 = data2.squeeze()
        # Define the trajectory data (example)
        x = data_np[0,:]
        y = data_np[1,:]
        z = data_np[2,:]
        
        # Plot the trajectory
        # fig = plt.figure(figsize=(8, 6))
        # ax = self.fig.add_subplot(222, projection='3d')  # Use the existing ax if provided
        # Plot the trajectory
        ax.plot(x, y, z, label='Trajectory', color='r', linewidth=1  )  # Set color and linewidth for better visibility

        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Trajectory Plot')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()

    
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

def plot_trajectory_0():
    data,_ = loadMatlabData('studentdata0.mat')
    ax, fig = plot_trajectory_vicon(data)
    return ax, fig


# def plot_trajectory_estimated(data):
#     """
#     Plot the trajectory of the measurement data.
#     :param data: Measurement data.
#     """
#     data_np = np.array(data).squeeze().T

#     # data2 = data.T
#     # data2 = data2.squeeze()
#     # Define the trajectory data (example)
#     x = data_np[0,:]
#     y = data_np[1,:]
#     z = data_np[2,:]
    
#     # Plot the trajectory
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     # Plot the trajectory
#     ax.plot(x, y, z, label='Trajectory')

#     # Set labels and title
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     ax.set_title('3D Trajectory Plot')

#     # Add a legend
#     ax.legend()

#     # Show the plot
#     plt.show()


def plot_trajectory_vicon(data):
    """
    Plot the trajectory of the measurement data.
    :param data: Measurement data.
    """
    # Define the trajectory data (example)
    x = data['vicon'][0, :]
    y = data['vicon'][1, :] 
    z = data['vicon'][2, :]
    
    # Plot the trajectory
    fig = plt.figure(figsize=(8, 6))
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

    # Show the plot
    # plt.show()
    return ax,fig
    

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
    p = m.get_corners_world_frame(12)
    print("Corners for AprilTag index 12 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(24)
    print("Corners for AprilTag index 24 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(36)
    print("Corners for AprilTag index 36 in world frame:")
    print(p)  # Print the corners in the world frame
    
    p = m.get_corners_world_frame(49)
    print("Corners for AprilTag index 49 in world frame:")
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
    plot_trajectory_0()  # Plot trajectory for studentdata0.mat
    check_data()
    # mat_contents['data'][6]['id']

def process_measurement_data(file_name,ax,fig):
    """
    Process the measurement data from the specified MATLAB file.
    :param file_name: Name of the MATLAB file to process.
    :return: Processed data.
    """
    measurement_data = MeasurementData(ax,fig)
    mat_contents = measurement_data.loadMatlabData(file_name)
    position_data, orientation_data = measurement_data.process_measurement_data()

    measurement_data.plot_trajectory_estimated(position_data)  # Plot the estimated trajectory
    # plot_trajectory_estimated(position_data)  # Plot the trajectory


    return position_data

if __name__ == "__main__":
    # get_world_corners_test()
    ax,fig= plot_trajectory_0()   
    process_measurement_data('studentdata0.mat',ax,fig)  # Process measurement data from studentdata0.mat
    #main()
    #tests()  # Run all tests
