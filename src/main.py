from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
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
        # self.camera_to_world_transform = np.array([
        #     [1.0, 0.0, 0.0, -0.04],  # Rotation matrix (identity)
        #     [0.0, 1.0, 0.0, 0.0],  # Rotation matrix (identity)
        #     [0.0, 0.0, 1.0, -0.03],  # Translation along Z-axis (example value)
        #     [0.0, 0.0, 0.0, 1.0]   # Homogeneous coordinate
        # ], dtype=np.float64)
        # = np.linalg.inv(self.camera_to_world_transform)  # Inverse of the camera to world transform
        # self.transform_from_camera_to_drone = self.homogeneous_matrix(-0.04, 0.0, -0.03, np.pi, 0, np.pi/4)  # Example values for translation and rotation
        # self.transform_from_camera_to_drone = self.homogeneous_matrix(-0.04, 0.0, -0.03, 0, 0, 0)  # Example values for translation and rotation
        
        # print("Camera to World Transform Matrix:")
        # print(self.transform_from_camera_to_drone)

        self.position_data = []
        self.orientation_data = []
    
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
        
        position = None
        last_position = None
        self.position_data = []
        self.orientation_data = []
        self.time = []
        self.results_np = None
        for data in self.mat_contents['data']:
            if isinstance(data['id'],np.ndarray):
                if len(data['id']) == 0:
                    if len(self.position_data) == 0:
                        continue
                    else:
                        continue
                        # If there are no more AprilTags, return the processed data
                        self.position_data.append(last_position)  # Append the last estimate pose result to processed data
                        self.orientation_data.append(last_position)  
                        continue  # Skip to the next iteration if there are no AprilTags detected
                    
            # Estimate the pose for each item in the data
            position,orientation = self.estimate_pose(data)  # Estimate the pose for each item in the data   
            if position is None or orientation is None:
                print("Warning: Pose estimation failed for the current data item. Skipping this item.")
                continue  # Skip this item if pose estimation failed
            last_position = position  # Store the last valid position
            self.position_data.append(position)  # Append the last estimate pose result to processed data
            self.orientation_data.append(orientation)
            self.time.append(data['t'])
            result= np.hstack((np.array(position).squeeze(),orientation,data['t']))
            

            self.results_np = result if self.results_np is None else np.vstack((self.results_np, result))
        return self.results_np
    ###########################################################################################

    def calculate_drone_position(self, id, objectPoints,rvec, tvec,image1,vicon=None):
        
            # If solvePnP is successful, it returns the rotation vector (rvec) and translation vector (tvec)``
            r_world_to_camera, _ = cv2.Rodrigues(rvec)
            rx, rz = np.pi, np.pi/4
            # r_world_to_camera = np.linalg.inv(r_world_to_camera)
            rotation_x = self.rotation_matrix_x(rx)[0:3,0:3]
            rotation_z = self.rotation_matrix_z(rz)[0:3,0:3]
            r_camera_to_robot = rotation_x @ rotation_z
            t_camera_to_robot = np.array([[-0.04], [0.0], [-0.03]])
            
            cameraPosition =   (-np.matrix(r_world_to_camera).T @  np.matrix(tvec))+t_camera_to_robot
            r_world_to_robot =  r_world_to_camera @ r_camera_to_robot
            
            euler_angles = R.from_matrix(r_world_to_robot).as_euler('xyz', degrees=False)

            processed_data={
                'image': image1,
                'id': id,
                'objectPoints': objectPoints,
                'rvec': rvec,  # Rotation vector
                'tvec': tvec,   # Translation vector
                'orientation': euler_angles,
                'cameraPosition': cameraPosition  # This is the position of the object in the world frame
            }         

            return processed_data

    def estimate_pose(self, data, ignore_invalid=False):
        """
        Estimate the pose of an object using the given image points and object points.
        :param
        data: single element of the input data
        :return: Position and estimate of the robot in the world frame.
        """
        image1 = np.array(data['img'])
        processed_data = {}
        if isinstance(data['id'], int):
            # Process single AprilTag
            objectPoints = self.get_corners_world_frame(data['id'])
            imagePoints = np.array([data['p1'], data['p2'], data['p3'], data['p4']], dtype=np.float64)  # Create an array of the corners
            if ignore_invalid and np.any(imagePoints < 0):
                print(f"Warning: Invalid image points for AprilTag ID {data['id']}. Skipping this tag.")
                return None,None
            
            retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.camera_matrix, self.dist_coeffs)
            if retval is None:
                print("Error: solvePnP failed to estimate pose.")
                return None,None
                    
            processed_data= self.calculate_drone_position(data['id'],objectPoints,rvec, tvec,image1)
        else:
            # Process multiple AprilTags
            imagePointsCollection = None
            objectPointsCollection = None
            for i,tag_id in enumerate(data['id']):
                objectPoints = self.get_corners_world_frame(tag_id)
                p1=np.array([data['p1'][0][i],data['p1'][1][i]])  
                p2=np.array([data['p2'][0][i],data['p2'][1][i]])  
                p3=np.array([data['p3'][0][i],data['p3'][1][i]])  
                p4=np.array([data['p4'][0][i],data['p4'][1][i]])
                if ignore_invalid and (p1[0]<0 or p1[1]<0 or p2[0]<0 or p2[1]<0 or p3[0]<0 or p3[1]<0 or p4[0]<0 or p4[1]<0):
                    print(f"Warning: Invalid image points for AprilTag ID {tag_id}. Skipping this tag.")
                    continue  # Skip this tag if any of the image points are invalid  
                imagePoints = np.array([p1,p2,p3,p4], dtype=np.float64)  # Create an array of the corners
                imagePointsCollection = imagePoints if imagePointsCollection is None else np.vstack((imagePointsCollection, imagePoints))
                objectPointsCollection = objectPoints if objectPointsCollection is None else np.vstack((objectPointsCollection, objectPoints))
            
            retval, rvec, tvec = cv2.solvePnP(objectPointsCollection, imagePointsCollection, self.camera_matrix, self.dist_coeffs)
            if retval is None or retval == False:
                print(f"Error: solvePnP failed for AprilTag ID {tag_id}. Skipping this tag.")
                return None, None
            
            processed_data= self.calculate_drone_position(tag_id,objectPointsCollection,rvec, tvec,image1)

            
        if 'cameraPosition' not in processed_data or processed_data['cameraPosition'] is None:
            print("****************************Error: No valid orientation vector found.")
            return None, None
        return processed_data['cameraPosition'],processed_data['orientation']
    
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
    def apply_transform(self,transform, x):
        '''
        https://alexsm.com/homogeneous-transforms/
        '''
        x_h = np.ones((x.shape[0] + 1, x.shape[1]), dtype=x.dtype)
        x_h[:-1, :] = x
        x_t = np.dot(transform, x_h)
        # return x_t[:3] / x_t[-1]
        return x_t[:3]

    '''
    def translation_matrix(self,tx, ty, tz):
        """Creates a translation matrix."""
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
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
            # rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
            # Apply rotations in XYZ order (common convention)
            rotation = np.dot(rotation_x, np.dot(rotation_y, rotation_z))
            
            # Combine rotation and translation
            transform = np.dot(translation, rotation)
            
            return transform
    '''    
    def plot_trajectory_vicon(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        # Define the trajectory data (example)
        data = self.mat_contents
        x = data['vicon'][0, :]
        y = data['vicon'][1, :] 
        z = data['vicon'][2, :]
        
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
               
    def plot_trajectory_estimated(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        data = self.position_data
        data_np = np.array(data).squeeze().T

        # Define the trajectory data (example)
        x = data_np[0,:]
        y = data_np[1,:]
        z = data_np[2,:]
        
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
        plt.show()


    def plot_orientation(self):
        """
        Plot the trajectory of the measurement data.
        :param data: Measurement data.
        """
        x = self.mat_contents['time']
        data = self.mat_contents
        roll  = data['vicon'][3, :]
        pitch = data['vicon'][4, :]
        yaw   = data['vicon'][5, :]
        
        # Plot the trajectory
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        fig.suptitle('Roll / Pitch / Yaw Plot')
        
        # Plot the trajectory
        axs[0].plot(x, roll, label='Actual', color='b', linewidth=1)  # Set color and linewidth for better visibility
        axs[0].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[3,:], label='Estimated', color='r', linewidth=1)  # Set color and linewidth for better visibility

        axs[1].plot(x, pitch, label='Actual', color='b', linewidth=1)  # Set color and linewidth for better visibility
        axs[1].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[4,:], label='Estimated', color='r', linewidth=1)  # Set color and linewidth for better visibility

        axs[2].plot(x, yaw, label='Actual', color='b', linewidth=1)  # Set color and linewidth for better visibility
        axs[2].plot(self.results_np.T.squeeze()[6,:], self.results_np.T.squeeze()[5,:], label='Estimated', color='r', linewidth=1)  # Set color and linewidth for better visibility
        
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

# def plot_trajectory_0():
#     data,_ = loadMatlabData('studentdata0.mat')
#     ax, fig = plot_trajectory_vicon(data)
#     return ax, fig



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
    
def generate_tag_corners():
    """Generates the world coordinates for AprilTag corners in a 12x9 grid."""
    tag_size = 0.152  # Size of each AprilTag
    spacing = 0.152   # Default spacing between tags
 
    # Extra spacing applies between columns 3-4 and 6-7 (0-indexed)
    extra_spacing_cols = {3: 0.178 - spacing, 6: 0.178 - spacing}
 
    tag_corners_world = {}
 
    for row in range(12):  # Rows go down (x-direction)
        x = row * (tag_size + spacing)
 
        for col in range(9):  # Columns go right (y-direction)
            y = 0
            for c in range(col):
                y += tag_size
                if c in extra_spacing_cols:
                    y += extra_spacing_cols[c]
                else:
                    y += spacing
 
            # Define the four corners in world frame: P1 to P4
            P1 = np.array([x + tag_size, y, 0])            # Bottom-left
            P2 = np.array([x + tag_size, y + tag_size, 0]) # Bottom-right
            P3 = np.array([x, y + tag_size, 0])            # Top-right
            P4 = np.array([x, y, 0])                       # Top-left
 
            tag_id = col * 12 + row  # Row-major order
            tag_corners_world[tag_id] = np.array([P1, P2, P3, P4])
 
    return tag_corners_world

def tests():
    plot_trajectory_test()
    get_world_corners_test()
    check_data()
    tag_corners_world = generate_tag_corners()
    # mat_contents['data'][6]['id']

def process_measurement_data(file_name):
    """
    Process the measurement data from the specified MATLAB file.
    :param file_name: Name of the MATLAB file to process.
    :return: Processed data.
    """
    measurement_data = MeasurementData()
    measurement_data.loadMatlabData(file_name)
    measurement_data.process_measurement_data()
    measurement_data.plot_trajectory_vicon()  # Plot the actual trajectory
    measurement_data.plot_trajectory_estimated()  # Plot the estimated trajectory
    measurement_data.plot_orientation()  # Plot the roll trajectory
    

    

if __name__ == "__main__":
    
    # process_measurement_data('studentdata0.mat')
    process_measurement_data('studentdata1.mat')
    # process_measurement_data('studentdata5.mat')
    # process_measurement_data('studentdata2.mat')
    # process_measurement_data('studentdata3.mat')
    # process_measurement_data('studentdata4.mat')
    # process_measurement_data('studentdata5.mat')
    # process_measurement_data('studentdata6.mat')
    # process_measurement_data('studentdata7.mat')

    #main()
    #tests()  # Run all tests
