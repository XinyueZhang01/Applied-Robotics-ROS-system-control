from math import cos, sin, acos, asin, atan2, sqrt, pi
from numpy import genfromtxt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from map import generate_map, expand_map, DENIRO_width
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rospy
import sys

deniro_position = np.array([0, -6.0])
deniro_heading = 0.0
deniro_linear_vel = 0.0
deniro_angular_vel = 0.0

map = generate_map()

initial_position = np.array([0.0, -6.0])
goal = np.array([8.0, 8.0])


def deniro_odom_callback(msg):
    global deniro_position, deniro_heading, deniro_linear_vel, deniro_angular_vel
    deniro_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
    r = R.from_quat([msg.pose.pose.orientation.x,
                     msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z,
                     msg.pose.pose.orientation.w])
    deniro_heading = r.as_euler('xyz')[2]
    deniro_linear_vel = np.sqrt(msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2)
    deniro_angular_vel = msg.twist.twist.angular.z


def set_vref_publisher():
    rospy.init_node("motion_planning_node")

    wait = True
    while(wait):
        now = rospy.Time.now()
        if now.to_sec() > 0:
            wait = False

    vref_topic_name = "/robot/diff_drive/command"
    #rostopic pub /robot/diff_drive/command geometry_msgs/Twist -r 10 -- '[0.0, 0.0, 0.0]' '[0.0, 0.0, -0.5]'
    pub = rospy.Publisher(vref_topic_name, Twist, queue_size=1000)
    
    odom_topic_name = "odom"
    sub = rospy.Subscriber(odom_topic_name, Odometry, deniro_odom_callback)
    return pub


def cmd_vel_2_twist(v_forward, omega):
    twist_msg = Twist()
    twist_msg.linear.x = v_forward
    twist_msg.linear.y = 0
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = omega
    return twist_msg

import numpy as np
import rospy
from math import atan2, sqrt

import numpy as np
import rospy
from math import atan2, sqrt

class MotionPlanner():
    """
    A class to handle motion planning for a robot (e g , DE NIRO) using a given planning algorithm
    """

    def __init__(self, map, scale, goal):
        """
        Initializes the motion planner

        Args:
            map (numpy array): The occupancy grid or pixel map of the environment
            scale (tuple): Scaling factors (xscale, yscale) for converting between world and map coordinates
            goal (tuple): The goal position in world coordinates
        """
        self.vref_publisher = set_vref_publisher()  # Initializes a publisher to send velocity commands
        self.pixel_map = map  # The map used for navigation
        self.xscale, self.yscale = scale  # Scaling factors for coordinate transformations
        self.goal = goal  # Goal position in world coordinates

    def send_velocity(self, vref):
        """
        Converts Cartesian velocity commands (v_x, v_y) into robot commands and publishes them

        Args:
            vref (tuple): The velocity reference (v_x, v_y) in Cartesian coordinates
        """
        # vref is given in Cartesian coordinates (v_x, v_y)
        # DE NIRO uses linear and angular velocity (v_forward, omega)

        print("vx:\t", vref[0], ",\tvy:\t", vref[1])  # Print velocity components

        # Compute desired heading direction from velocity vector
        v_heading = atan2(vref[1], vref[0])  

        # Compute heading error (difference between robot's current heading and desired heading)
        heading_error = deniro_heading - v_heading  

        # Compute angular velocity (omega) to correct heading
        omega = 1 * heading_error  

        # Only drive forward if the robot is facing (approximately) the correct direction
        if abs(heading_error) < 0.1:
            v_forward = min(max(sqrt(vref[0]**2 + vref[1]**2), 0.1), 0.2)  # Limit speed between 0.1 and 0.2
        else:
            v_forward = 0  # Stop moving if the heading error is too large

        # Convert (v_forward, omega) to a Twist message and publish it
        twist_msg = cmd_vel_2_twist(v_forward, omega)
        print("v_fwd:\t", v_forward, ",\tw:\t", omega)  # Print computed velocities

        self.vref_publisher.publish(twist_msg)  # Send command to the robot

    def map_position(self, world_position):
        """
        Converts world coordinates to map (pixel) coordinates

        Args:
            world_position (numpy array): Position(s) in world coordinates (x, y)
        
        Returns:
            numpy array: Corresponding position(s) in map coordinates
        """
        world_position = world_position.reshape((-1, 2))  # Ensure input is in (N,2) format
        map_x = np.rint(world_position[:, 0] * self.xscale + self.pixel_map.shape[0] / 2)
        map_y = np.rint(world_position[:, 1] * self.yscale + self.pixel_map.shape[1] / 2)
        map_position = np.vstack((map_x, map_y)).T  # Stack x and y into an (N,2) array
        return map_position

    def world_position(self, map_position):
        """
        Converts map (pixel) coordinates back to world coordinates

        Args:
            map_position (numpy array): Position(s) in map coordinates (x, y)
        
        Returns:
            numpy array: Corresponding position(s) in world coordinates
        """
        map_position = map_position.reshape((-1, 2))  # Ensure input is in (N,2) format
        world_x = (map_position[:, 0] - self.pixel_map.shape[0] / 2) / self.xscale
        world_y = (map_position[:, 1] - self.pixel_map.shape[1] / 2) / self.yscale
        world_position = np.vstack((world_x, world_y)).T  # Stack x and y into an (N,2) array
        return world_position

    def run_planner(self, planning_algorithm):
        """
        Runs the motion planning loop

        Args:
            planning_algorithm (function): A function that computes the next velocity reference
        
        The loop runs at 25 Hz, continuously sending velocity commands until the goal is reached
        """
        rate = rospy.Rate(25)  # Set loop rate to 25 Hz
        
        while not rospy.is_shutdown():
            vref, complete = planning_algorithm()  # Get velocity reference and completion status
            self.send_velocity(vref)  # Send velocity command

            if complete:
                print("Completed motion")  # Stop if the goal is reached
                break

            rate.sleep()  # Maintain loop rate

    
    def setup_waypoints(self):
        ############################################################### TASK B
        # Create an array of waypoints for the robot to navigate via to reach the goal
        # For all navigation tasks, DE NIRO will be navigated 
        # from a starting position of (0.0, -6.0) m to (8.0, 8.0) m
        waypoints = np.array([[0, -6],                              
                              [2,-3.6], # Intermediate waypoint to refine the path
                              [6,-1.2], # Ensures smoother transitions around obstacles
                              [7.85,4.9], # Close to the final approach
                              [8, 8]])  # Final goal location  
        # These five waypoints represent the nodes forming the shortest path based on 
        # visibility graph analysis, ensuring minimal detours.                          

        # Append the waypoints to include the start and goal positions
        waypoints = np.vstack([initial_position, waypoints, self.goal])
        
        # Convert world coordinates to pixel coordinates for visualization and computation
        pixel_goal = self.map_position(self.goal)  
        pixel_waypoints = self.map_position(waypoints)
        
        # calculating the length of each individual path between 2 points
        # since the selected 2 potential optimal routes are all consist of 5 sublinear segment
        # So there are 5 small length and add them up to be the total distance
        path_length_1 = np.linalg.norm((waypoints[1] - waypoints[0]))
        path_length_2 = np.linalg.norm((waypoints[2] - waypoints[1]))
        path_length_3 = np.linalg.norm((waypoints[3] - waypoints[2]))
        path_length_4 = np.linalg.norm((waypoints[4] - waypoints[3]))
        path_length_5 = np.linalg.norm((waypoints[5] - waypoints[4]))        
        # Calculating and printing the total path length by adding the up 5 lengths up
        path_length_total = path_length_1 + path_length_2 + path_length_3 + path_length_4 + path_length_5
        print("Total path length = ", path_length_total)

        # Display the computed waypoints in both world and pixel coordinates
        print('Waypoints:\n', waypoints)
        print('Waypoints in pixel coordinates:\n', pixel_waypoints)
        
        
        # Plotting
        # Visualization of the waypoints on the C-space map
        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        plt.scatter(pixel_waypoints[:, 0], pixel_waypoints[:, 1])
        plt.plot(pixel_waypoints[:, 0], pixel_waypoints[:, 1])
        plt.show()
        
        # Store waypoints for future reference in navigation tasks
        self.waypoints = waypoints
        self.waypoint_index = 0
    
    def waypoint_navigation(self):
        complete = False
        
        # get the current waypoint
        current_waypoint = self.waypoints[self.waypoint_index, :]
        # calculate the vector from DE NIRO to waypoint
        waypoint_vector = current_waypoint - deniro_position
        # calculate the distance from DE NIRO to waypoint
        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        # calculate the unit direction vector from DE NIRO to waypoint
        waypoint_direction = waypoint_vector / distance_to_waypoint
        
        # Calculate a reference velocity based on the direction of the waypoint
        vref = waypoint_direction * 0.5
        
        # If we have reached the waypoint, start moving to the next waypoint
        if distance_to_waypoint < 0.05:
            self.waypoint_index += 1    # increase waypoint index
            
        # If we have reached the last waypoint, stop
        if self.waypoint_index > self.waypoints.shape[0]:
            vref = np.array([0, 0])
            complete = True
        return vref, complete

    def potential_field(self):
        ############################################################### TASK C
        complete = False
        
        # compute the positive force attracting the robot towards the goal
        # vector to goal position from DE NIRO
        goal_vector = goal - deniro_position
        # distance to goal position from DE NIRO
        distance_to_goal = np.linalg.norm(goal_vector)
        # unit vector in direction of goal from DE NIRO
        pos_force_direction = goal_vector / distance_to_goal
        
        # potential function
        pos_force_magnitude =  1  # Attractive force parameter is chosen to be constant
        # tuning parameter
        K_att = 1     # will tune the value to be 1, 10, 100
        
        # normalised positive force
        positive_force = K_att * pos_force_direction * pos_force_magnitude 
        
        # positive force
        positive_force = K_att * pos_force_direction * pos_force_magnitude  # normalised positive force
        
        # compute the negative force repelling the robot away from the obstacles
        obstacle_pixel_locations = np.argwhere(self.pixel_map == 1)
        # coordinates of every obstacle pixel
        obstacle_pixel_coordinates = np.array([obstacle_pixel_locations[:, 1], obstacle_pixel_locations[:, 0]]).T
        # coordinates of every obstacle pixel converted to world coordinates
        obstacle_positions = self.world_position(obstacle_pixel_coordinates)
        
        # vector to each obstacle from DE NIRO
        obstacle_vector = obstacle_positions - deniro_position   # vector from DE NIRO to obstacle
        # distance to obstacle from DE NIRO
        distance_to_obstacle = np.linalg.norm(obstacle_vector, axis=1).reshape((-1, 1))  # magnitude of vector
        # unit vector in direction of obstacle from DE NIRO
        force_direction = obstacle_vector / distance_to_obstacle   # normalised vector (for direction)
        
        
        # The repulsive force from each obstacle is **inversely proportional** to the square of its distance.
        # This follows the **Inverse-Square Law**, meaning the force increases significantly when close to an obstacle,
        # Strong Avoidance near obstacles
        force_magnitude = -1.0 / (distance_to_obstacle**2)  # Repulsive force magnitude
        # Tuning parameter for repulsive force strength
        # A higher value of K_rep increases the effect of obstacles on DE NIRO's movement.
        K_rep = 18    
        
        # Compute the force exerted by an individual obstacle pixel
        # The repulsive force is **directional**, pointing away from the obstacle.
        # It is computed as the unit force direction vector scaled by the force magnitude.
        obstacle_force = force_direction * force_magnitude

        # total repulsive force acting on DE NIRO due to all detected obstacles
        # The sum of all obstacle forces is averaged over the number of obstacle pixels.
        # This ensures that repulsive forces are scaled appropriately, preventing excessive influence from a single pixel.
        negative_force = K_rep * np.sum(obstacle_force, axis=0) / obstacle_pixel_locations.shape[0]
        
        # Uncomment these lines to visualise the repulsive force from each obstacle pixel
        # Make sure to comment it out again when you run the motion planner fully
        # plotskip = 10   # only plots every 10 pixels (looks cleaner on the plot)
        # plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        # plt.quiver(obstacle_pixel_coordinates[::plotskip, 0], obstacle_pixel_coordinates[::plotskip, 1],
        # obstacle_force[::plotskip, 0] * self.xscale, obstacle_force[::plotskip, 1] * self.yscale)
        # plt.show()

        print("positive_force:", positive_force)
        print("negative_force:", negative_force)
        
        # Reference velocity is the resultant force
        vref = positive_force + negative_force
        
        # If the goal has been reached, stop
        if distance_to_goal < 0.05:
            vref = np.array([0, 0])
            complete = True
        return vref, complete
    def generate_random_points(self, N_points):
        ############################################################### TASK D
        """
        Generates random points in a bounded 2D space while avoiding obstacles.
        The function ensures that generated points are within an obstacle-free region.
        """
        # Initialize variables to store accepted and rejected points
        N_accepted = 0  # number of accepted samples
        accepted_points = np.empty((1, 2))  # empty array to store accepted samples
        rejected_points = np.empty((1, 2))  # empty array to store rejected samples
        
        while N_accepted < N_points:    # keep generating points until N_points have been accepted
        
            points = np.random.uniform(-10, 10, (N_points - N_accepted, 2))  # generate random coordinates
            pixel_points = self.map_position(points)    # get the point locations on our map
            rejected = np.zeros(N_points - N_accepted)   # create an empty array of rejected flags
            
            # Your code here!
            # Loop through the generated points and check if their pixel location corresponds to an obstacle in self.pixel_map
            # Remember that indexing a 2D array is [row, column], which is [y, x]!
            # You might have to make sure the pixel location is an integer so it can be used to index self.pixel_map
            
            # Iterate over generated points to check for obstacles in pixel_map
            for i in range(len(pixel_points)):
                px_y = int(pixel_points[i, 1])# Extract y-coordinate (row index)
                px_x = int(pixel_points[i, 0]) # Extract x-coordinate (column index)
            # self.pixel_map[px_y, px_x] = 1 when an obstacle is present
                if self.pixel_map[px_y,px_x] == 1:
                    rejected[i] = 1# Mark the point as rejected

            # Separate accepted and rejected points based on rejection flags
            new_accepted_points = pixel_points[np.argwhere(rejected == 0)].reshape((-1, 2))
            new_rejected_points = pixel_points[np.argwhere(rejected == 1)].reshape((-1, 2))
           
            # keep an array of generated points that are accepted
            accepted_points = np.vstack((accepted_points, new_accepted_points))
            
            # keep an array of generated points that are rejected (for visualisation)
            rejected_points = np.vstack((rejected_points, new_rejected_points))
            
            # Update the number of accepted points (subtract the initial placeholder)
            N_accepted = accepted_points.shape[0] - 1     
        
        # throw away that first 'empty' point we added for initialisation
        accepted_points = accepted_points[1:, :]
        rejected_points = rejected_points[1:, :]   
        
        # visualise the accepted and rejected points
        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')  # setup a plot of the map
        
        plt.scatter(accepted_points[:, 0], accepted_points[:, 1], c='b')    # plot accepted points in blue
        plt.scatter(rejected_points[:, 0], rejected_points[:, 1], c='r')    # plot rejected points in red
        
        deniro_pixel = self.map_position(initial_position)
        goal_pixel = self.map_position(goal)
        plt.scatter(deniro_pixel[0, 0], deniro_pixel[0, 1], c='w')  # plot DE NIRO as a white point
        plt.scatter(goal_pixel[0, 0], goal_pixel[0, 1], c='g')  # plot the goal as a green point
        
        plt.show()
        
        world_points = self.world_position(accepted_points) # calculate the position of the accepted points in world coordinates
        world_points = np.vstack((initial_position, world_points, goal))    # add DE NIRO's position to the beginning of these points, and the goal to the end
        
        return world_points
    
    def create_graph(self, points):
        ############################################################### TASK E i
        """
        Creates a graph where nodes (points) are connected based on distance constraints and obstacle avoidance.
        """
        # Choose your minimum and maximum distances to produce a suitable graph
        mindist = 0.5
        maxdist = 5.0
        
        # Calculate a distance matrix between every node to every other node
        distances = cdist(points, points)
        
        # Create two dictionaries
        graph = {}  # dictionary of each node, and the nodes it connects to
        distances_graph = {}    # dictionary of each node, and the distance to each node it connects to
        
        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')  # setup a plot of the map
        
        for i in range(points.shape[0]):    # loop through each node
        
            points_in_range = points[(distances[i] >= mindist) & (distances[i] <= maxdist)] # get nodes an acceptable distance of the current node
            distances_in_range = distances[i, (distances[i] >= mindist) & (distances[i] <= maxdist)] # get the corresponding distances to each of these nodes
            
            if points_in_range.shape[0] > 0:    # if there are any nodes in an acceptable range
            
                # set up arrays of nodes with edges that don't collide with obstacles, and their corresponding distances
                collision_free_points = np.empty((1, 2))
                collision_free_distances = np.empty((1, 1))
                
                for j in range(points_in_range.shape[0]):   # loop through the nodes an acceptable distance of the current node
                
                    pxA = self.map_position(points[i])  # get the current node position on the map
                    pxB = self.map_position(points_in_range[j])     # get the node in range position on the map

                    collision = self.check_collisions(points[i], points_in_range[j])    # check if there is a collision on the edge between two points
                    
                    if collision:
                        # if there is a collision, plot the edge in red
                        plt.plot([pxA[0, 0], pxB[0, 0]], [pxA[0, 1], pxB[0, 1]], c='r')
                        pass
                        
                    else:
                        # if there is no collision, add the node in range to the array of nodes that have no collisions
                        collision_free_points = np.append(collision_free_points, points_in_range[j].reshape((1, 2)), axis=0)
                        # add the corresponding distance to the array of distances
                        collision_free_distances = np.append(collision_free_distances, distances_in_range[j].reshape((1, 1)))
                        
                        # plot the edge in blue
                        plt.plot([pxA[0, 0], pxB[0, 0]], [pxA[0, 1], pxB[0, 1]], c='b')
                        
                # after we've looped through every point, update the two dictionaries
                graph[str(points[i])] = collision_free_points[1:]
                distances_graph[str(points[i])] = collision_free_distances[1:]
                
        # Plotting
        deniro_pixel = self.map_position(initial_position)
        goal_pixel = self.map_position(goal)
        
        plt.scatter(deniro_pixel[0, 0], deniro_pixel[0, 1], c='w') # Mark DE NIRO (white)
        plt.scatter(goal_pixel[0, 0], goal_pixel[0, 1], c='g') # Mark goal (green)
        
        plt.show()
        
        return graph, distances_graph
    
    def check_collisions(self, pointA, pointB):
    
        ############################################################### TASK E ii    
        """
        Checks if the straight-line path between two points collides with an obstacle.
        """
         
        # Calculate the distance between the two point
        vector = np.subtract(pointB, pointA)  # vector of point A to point B
        distance = np.linalg.norm(vector)  # distance = mangitude of vector from A to B
        
        # Calculate the UNIT direction vector pointing from pointA to pointB
        direction = vector / distance # unit vector = vector / magnitude
        # Choose a resolution for collision checking
        resolution = 0.05  # resolution to check collision to in m
        
        # Create an array of points to check collisions at
        edge_points = pointA.reshape((1, 2)) + np.arange(0, distance, resolution).reshape((-1, 1)) * direction.reshape((1, 2))
        # Convert the points to pixels
        edge_pixels = self.map_position(edge_points)
        
        for pixel in edge_pixels:   # loop through each pixel between pointA and pointB
            collision = self.pixel_map[int(pixel[1]), int(pixel[0])]    # if the pixel collides with an obstacle, the value of the pixel map is 1
            if collision == 1:
                return True     # if there's a collision, immediately return True
        return False    # if it's got through every pixel as hasn't returned yet, return False
    
    def dijkstra(self, graph, edges):
    
        ############################################################### TASK F
        
        goal_node = goal  # Define the goal node
        
         # Step 1: Create a dataframe of unvisited nodes
        nodes = list(graph.keys())  # List all nodes in the graph
        
        # Initialise each cost to a very high number
        initial_cost = 1e6  # Large number to represent infinity
        
        # Create a dataframe for unvisited nodes with three columns: 'Node', 'Cost', and 'Previous'
        unvisited = pd.DataFrame(
            {'Node': nodes,  # List of all nodes in the graph
            'Cost': [initial_cost for node in nodes],  # Initialize all node costs to a very high value (infinity)
            'Previous': ['' for node in nodes]})# Track the previous node in the optimal path
        # Set 'Node' as the index to allow direct lookups and updates using node names
        unvisited.set_index('Node', inplace=True)
        
        # Set the first node's cost to zero
        unvisited.loc[[str(initial_position)], ['Cost']] = 0.0
        
        # Create a dataframe of visited nodes (initially empty)
        visited = pd.DataFrame({'Node':[''], 'Cost':[0.0], 'Previous':['']})
        visited.set_index('Node', inplace=True)
        
        # Display initial state of dataframes
        print('--------------------------------')
        print('Unvisited nodes')
        print(unvisited.head())
        print('--------------------------------')
        print('Visited nodes')
        print(visited.head())
        print('--------------------------------')
        print('Running Dijkstra')
        
        # Dijkstra's algorithm!
        # Step 2: Run Dijkstra’s algorithm until the goal node is reached
        while str(goal_node) not in visited.index.values:
            
            # Select the node with the minimum cost from the unvisited nodes
            current_node = unvisited[unvisited['Cost']==unvisited['Cost'].min()]
            current_node_name = current_node.index.values[0]    # the node's name (string)
            current_cost = current_node['Cost'].values[0]       # the distance from the starting node to this node (float)
            current_tree = current_node['Previous'].values[0]   # a list of the nodes visited on the way to this one (string)
            
            # Get all connected nodes and their edge costs
            connected_nodes = graph[current_node.index.values[0]]   # get all of the connected nodes to the current node (array)
            connected_edges = edges[current_node.index.values[0]]   # get the distance from each connected node to the current node   
             
             # Step 3: Loop through connected nodes to update costs if a shorter path is found
            for next_node_name, edge_cost in zip(connected_nodes, connected_edges):
                next_node_name = str(next_node_name)    # the next node's name (string)
                
                if next_node_name not in visited.index.values:  # if we haven't visited this node before
                    
                    # update this to calculate the cost of going from the initial node to the next node via the current node
                    next_cost_trial = current_cost + edge_cost # set this to calculate the cost of going from the initial node to the next node via the current node
                    next_cost = unvisited.loc[[next_node_name], ['Cost']].values[0] # the previous best cost we've seen going to the next node
                   
                    # if it costs less to go the next node from the current node, update then next node's cost and the path to get there
                    if next_cost_trial < next_cost:
                        unvisited.loc[[next_node_name], ['Cost']] = next_cost_trial
                        unvisited.loc[[next_node_name], ['Previous']] = current_tree + current_node_name    # update the path to get to that node
            
            # Step 4: Move current node from unvisited to visited
            unvisited.drop(current_node_name, axis=0, inplace=True)     # remove current node from the unvisited list
            visited.loc[current_node_name] = [current_cost, current_tree]   # add current node to the visited list
        
        # Step 5: Display final results
        print('--------------------------------')
        print('Unvisited nodes')
        print(unvisited.head())
        print('--------------------------------')
        print('Visited nodes')
        print(visited.head())
        print('--------------------------------')
        
        # Step 6: Retrieve the optimal path and cost
        optimal_cost = visited.loc[[str(goal_node)], ['Cost']].values[0][0]  # Optimal cost (float)
        optimal_path = visited.loc[[str(goal_node)], ['Previous']].values[0][0]  # Optimal path (string)
        
        # Step 7: Convert the optimal path from a string to an actual array of waypoints to travel to
        string_waypoints = optimal_path[1:-1].split('][')# Split string path into list
        optimal_waypoints = np.array([np.fromstring(waypoint, sep=' ') for waypoint in string_waypoints])
        optimal_waypoints = np.vstack((optimal_waypoints, goal))    # add the goal as the final waypoint
        
        # Step 8: Display results
        print('Results')
        print('Goal node: ', str(goal_node))
        print('Optimal cost: ', optimal_cost)
        print('Optimal path:\n', optimal_waypoints)
        print('--------------------------------')
        
        # Step 9: Plot the optimal path on the map
        optimal_pixels = self.map_position(optimal_waypoints)
        plt.plot(optimal_pixels[:, 0], optimal_pixels[:, 1], c='b')# Plot the path
        
        deniro_pixel = self.map_position(initial_position)
        goal_pixel = self.map_position(goal)
        
        plt.imshow(self.pixel_map, vmin=0, vmax=1, origin='lower')
        plt.scatter(deniro_pixel[0, 0], deniro_pixel[0, 1], c='w')# Start position
        plt.scatter(goal_pixel[0, 0], goal_pixel[0, 1], c='g')# Goal position
        
        plt.show()
        
        # Step 10: Setup the waypoints for normal waypoint navigation
        self.waypoints = optimal_waypoints
        self.waypoint_index = 0
        
        
def main(task):
    """
    Main function to execute different motion planning tasks.
    """
    
    # load the map and expand it
    img, xscale, yscale = generate_map()
    c_img = expand_map(img, DENIRO_width)
    
    # load the motion planner
    planner = MotionPlanner(c_img, (xscale, yscale), goal=goal)
    
    # Execute the waypoints navigationn in task B 
    if task == 'waypoints':
        print("============================================================")
        print("Running Waypoint Navigation")
        print("------------------------------------------------------------")
        planner.setup_waypoints()
        planner.run_planner(planner.waypoint_navigation)
    
    # Execute the waypoints navigationn in task C 
    elif task == 'potential':
        print("============================================================")
        print("Running Potential Field Algorithm")
        print("------------------------------------------------------------")
        planner.run_planner(planner.potential_field)
    
    # Execute the waypoints navigationn in task D-F
    elif task == 'prm':
        print("============================================================")
        print("Running Probabilistic Road Map")
        print("------------------------------------------------------------")
        points = planner.generate_random_points(N_points=100)# N_points set the number of sample points in prm mapping
        graph, edges = planner.create_graph(points) # creating the graph from points plotted
        planner.dijkstra(graph, edges) # running the Dijkstra's algorithn on the mapped prm graph 
        planner.run_planner(planner.waypoint_navigation) # connect the optimal points by Dijkstra's algorithm 
    

if __name__ == "__main__":
    """
    Entry point of the script. Parses command-line arguments and executes the corresponding task.
    """
    
    tasks = ['waypoints', 'potential', 'prm']# List of available tasks
    
    # Check if a task argument has been provided
    if len(sys.argv) <= 1:
        print('Please include a task to run from the following options:\n', tasks)
    else:
        task = str(sys.argv[1])# Get the task argument
        if task in tasks:
            print("Running Coursework 2 -", task)# Indicate the selected task
            main(task)# Execute the main function with the specified task
        else:
            print('Please include a task to run from the following options:\n', tasks)
            
