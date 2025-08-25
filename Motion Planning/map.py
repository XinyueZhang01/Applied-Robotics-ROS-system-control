import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.morphology import binary_dilation
import rospy
import sys

rect_list = np.array([[0, -9, 2.5, 2],
                    [8.75, -9.25, 2.75, 1.25],
                    [4.5, -5.5, 4, 2.75],
                    [-0.25, -1.0, 3.75, 2.0],
                    [-4.5, -5.5, 4, 2.75],
                    [-8.75, -9.25, 2.75, 1.25],
                    [-9.5, 1.5, 1.25, 2.75],
                    [-5, 1, 1.25, 3.5],
                    [-0.25, 1, 2.75, 3.5],
                    [5, 1, 1.25, 3.5],
                    [9.25, 1.5, 1.25, 2.75],
                    [5.5, 6.0, 3.75, 1.75],
                    [0.0, 6.0, 1.75, 3.0],
                    [-5.5, 6, 3.75, 1.75],
                    [-10, 0, 0.1, 20],
                    [10, 0, 0.1, 20],
                    [0, -10, 20, 0.1],
                    [0, 10, 20, 0.1]], dtype=float)

xmin = -10.1
xmax = 10.1
ymin = -10.1
ymax = 10.1

DENIRO_width = 1

scale = 16

def generate_map():
    # Number of pixels in each direction
    N_x = int((xmax - xmin) * scale)
    N_y = int((ymax - ymin) * scale)
    
    # Initialise the map to be an empty 2D array
    img = np.zeros([N_x, N_y], dtype = np.float)

    for x1, y1, w, h in rect_list:
        x0 = int((x1 - w/2 - xmin) * scale)
        y0 = int((y1 - h/2 - ymin) * scale)
        x3 = int((x1 + w/2 - xmin) * scale)
        y3 = int((y1 + h/2 - ymin) * scale)
        
        # Fill the obstacles with 1s
        img[y0:y3, x0:x3] = 1
    return img, scale, scale
    

def expand_map(img, robot_width): 
    """
    Expands obstacles in the configuration space (C-space) by applying a circular mask to the binary occupancy map.
    This accounts for the robot's footprint, ensuring that obstacles are inflated to prevent collisions.

    Parameters:
    img (numpy array): The binary occupancy map where obstacles are marked.
    robot_width (float): The physical width of the robot in meters.

    Returns:
    expanded_img (numpy array): The binary map after applying dilation, representing expanded obstacles.
    """
    # Convert the robot's width from meters to pixels using the scaling factor
    robot_px = int(robot_width * scale)   #size of the robot in pixels x axis
    
    # Initialize a square matrix filled with zeros to create the structuring element for dilation
    # In the previous square mask implementation, the matrix was initialized with ones (np.ones), 
    # assuming a square footprint for the robot. Here, a circular mask is used, requiring a zero-initialized matrix.
    robot_mask = np.zeros((robot_px, robot_px)) #create a matrix of zeros
    
    # iterate the matrix
    for i in range(robot_px):
        for j in range(robot_px):
            # since Python arrays are zero-indexed
            # Adjust indices to center the coordinate system around the mask's midpoint
            x = i - (robot_px - 1) / 2  
            y = j - (robot_px - 1) / 2  
            # Compute the Euclidean distance from the center of the mask using the Pythagorean theorem
            distance = np.sqrt(x**2 + y**2)  
            # If the distance falls within the robot's radius, mark the pixel as part of the mask (value = 1)
            if distance <= robot_px / 2:
                robot_mask[i, j] = 1
    
    #below is the code to show the mask
    #plt.imshow(robot_mask, vmin=0, vmax=1, origin='lower')
    #plt.show() #show the square mask plot
    expanded_img = binary_dilation(img, robot_mask) # image is expanded using the mask we created
    
    # Return the processed map with expanded obstacles
    return expanded_img
                        
            
    
    

def main(task):
    if task == 'view':
        print("============================================================")
        print("Generating the map")
        print("------------------------------------------------------------")
        img, xscale, yscale = generate_map()
        plt.imshow(img, vmin=0, vmax=1, origin='lower')
        plt.show()
    
    elif task == 'expand':
        print("============================================================")
        print("Generating the C-space map")
        print("------------------------------------------------------------")
        img, xscale, yscale = generate_map()
        c_img = expand_map(img, DENIRO_width)
        plt.imshow(c_img, vmin=0, vmax=1, origin='lower')
        plt.show()
        
    

if __name__ == "__main__":
    tasks = ['view', 'expand']
    if len(sys.argv) <= 1:
        print('Please include a task to run from the following options:\n', tasks)
    else:
        task = str(sys.argv[1])
        if task in tasks:
            print("Running Coursework 2 -", task)
            main(task)
        else:
            print('Please include a task to run from the following options:\n', tasks)



