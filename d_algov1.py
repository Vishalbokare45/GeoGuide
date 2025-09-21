
#  Filename: direction_algo.py
#  Functions: dijkstra_shortest_path, direction, angle, find_nearest_index, turning_str
#  Global Variables: coordinates, cord, bot_id

import heapq
import math

#coordinate for index wise nodes 
coordinates = [{'x': 35, 'y': 540}, {'x': 59, 'y': 497}, {'x': 55, 'y': 406}, {'x': 60, 'y': 296}, {'x': 63, 'y': 225}, {'x': 79, 'y': 157}, {'x': 123, 'y': 155}, {'x': 374, 'y': 174}, {'x': 422, 'y': 252}, {'x': 420, 'y': 320}, {'x': 419, 'y': 403}, {'x': 341, 'y': 399}, {'x': 243, 'y': 399}, {'x': 240, 'y': 490}, {'x': 137, 'y': 494}, {'x': 134, 'y': 299}, {'x': 243, 'y': 305}, {'x': 247, 'y': 241},{'x':341,'y':312}]
cord=[{'x': 35, 'y': 540}, {'x': 59, 'y': 497}, {'x': 55, 'y': 406}, {'x': 60, 'y': 296}, {'x': 63, 'y': 225}, {'x': 79, 'y': 157}, {'x': 123, 'y': 155}, {'x': 374, 'y': 174}, {'x': 422, 'y': 252}, {'x': 420, 'y': 320}, {'x': 419, 'y': 403}, {'x': 341, 'y': 399}, {'x': 243, 'y': 399}, {'x': 240, 'y': 490}, {'x': 137, 'y': 494}, {'x': 134, 'y': 299}, {'x': 243, 'y': 305}, {'x': 247, 'y': 241},{'x':341,'y':312}]  
#aruko id of bot
bot_id=100




def dijkstra_shortest_path(graph, start, end):
    # """
    # * Function Name: dijkstra_shortest_path
    #  * Input:
    #     - graph (dict): A dictionary representing the graph where keys are nodes and values are lists of tuples.
    #       Each tuple contains a neighbor node and the distance to that neighbor.
    #     - start: The starting node for finding the shortest path.
    #     - end: The destination node for finding the shortest path.
    #  * Output: 
    #     - path (list or None): A list representing the shortest path from the start node to the end node.
    #       If no path is found, returns None.
    #  * Logic: 
    #     This function implements Dijkstra's algorithm to find the shortest path in a weighted graph.
    #     It takes a graph represented as a dictionary, the starting node, and the destination node as input.
    #     The algorithm maintains a priority queue to explore nodes in order of increasing distance from the start node.
    #     It initializes the distance to each node as infinity and the predecessor of each node as None.
    #     The algorithm iteratively explores nodes and updates their distances and predecessors as it finds shorter paths.
    #     Once the destination node is reached, it reconstructs the shortest path from start to end.
    #  * Example Call: 
    #     graph = {'A': [('B', 1), ('C', 4)],
    #              'B': [('A', 1), ('C', 2), ('D', 5)],
    #              'C': [('A', 4), ('B', 2), ('D', 1)],
    #              'D': [('B', 5), ('C', 1)]}
    #     shortest_path = dijkstra_shortest_path(graph, 'A', 'D')
    #     print(shortest_path)
    

    # Priority queue to store (distance, node). Starts with the start node at distance 0.
    queue = [(0, start)]
    # Dictionary to keep track of the minimum distance to reach each node.
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    # Dictionary to track the predecessor of each node along the shortest path.
    predecessors = {node: None for node in graph}

    while queue:
        # Get the node with the smallest distance so far.
        current_distance, current_node = heapq.heappop(queue)

        if current_node == end:
            break  # Stop if the end node has been reached.

        # Visit each neighbor of the current node.
        for neighbor, distance in graph[current_node]:
            new_distance = current_distance + distance

            # If a shorter path to the neighbor is found,
            # update the neighbor's distance, set the predecessor, and enqueue it.
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (new_distance, neighbor))

    # Reconstruct the shortest path from start to end.
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.reverse()  # Reverse the path to start from the beginning.

    # Return the shortest path
    return path if path[0] == start else None



def direction(x1,y1,x2,y2,bot_angle):
    # """
    #  * Function Name: direction
    #  * Input:
    #     - x1 (float): x-coordinate of the starting point.
    #     - y1 (float): y-coordinate of the starting point.
    #     - x2 (float): x-coordinate of the target point.
    #     - y2 (float): y-coordinate of the target point.
    #     - bot_angle (float): Angle of the bot's orientation in degrees (-180 to 180).
    #  * Output: 
    #     - dir (str): A direction towards which the bot should move ('f' for forward, 'b' for backward, 
    #       'l' for left, 'r' for right).
    #  * Logic: 
    #     This function calculates the direction in which a bot should move from its current position 
    #     to a target position based on the Cartesian coordinates of both positions and the angle of 
    #     the bot's orientation. It determines the direction by comparing the differences in x and y 
    #     coordinates between the current and target positions. Depending on the greater difference 
    #     (x or y), and the orientation angle of the bot, it assigns the appropriate direction.
    #  * Example Call: 
    #     direction(0, 0, 1, 1, 45)
        
    
    x_dis=abs(x1-x2)
    y_dis=abs(y1-y2)
    if(max(x_dis,y_dis)==x_dis):                   # Logic to determine direction based on x distance
        if(x2>x1):
            if(bot_angle>=-45 and bot_angle<=45):
                return "r"
            elif(bot_angle>=45 and bot_angle<=135):
                return "f"
            elif(bot_angle>=135 or bot_angle<=-135):
                return "l"
            else:
                return "u"
        
        else:
            if(bot_angle>=-45 and bot_angle<=45):
                return "l"
            elif(bot_angle>=45 and bot_angle<=135):
                return "u"
            elif(bot_angle>=135 or bot_angle<=-135):
                return "r"
            else:
                return "f"
    else:                                            # Logic to determine direction based on y distance 
        if(y1>y2):
            if(bot_angle>=-45 and bot_angle<=45):
                return "f"
            elif(bot_angle>=45 and bot_angle<=135):
                return "l"
            elif(bot_angle>=135 or bot_angle<=-135):
                return "u"
            else:
                return "r"
        
        else:
            if(bot_angle>=-45 and bot_angle<=45):
                return "u"
            elif(bot_angle>=45 and bot_angle<=135):
                return "r"
            elif(bot_angle>=135 or bot_angle<=-135):
                return "f"
            else:
                return "l"  
    

def angle(x1,y1,x2,y2):
    # """
    #  * Function Name: angle
    #  * Input:
    #     - x1 (float): x-coordinate of the first point.
    #     - y1 (float): y-coordinate of the first point.
    #     - x2 (float): x-coordinate of the second point.
    #     - y2 (float): y-coordinate of the second point.
    #  * Output: 
    #     - angle (float): Angle in degrees between the line connecting the two points and the x-axis.
    #       The angle is measured counterclockwise from the positive x-axis.
    #  * Logic: 
    #     This function calculates the angle between two points (x1, y1) and (x2, y2).
    #     It first calculates the differences in x and y coordinates between the two points.
    #     Then, it determines whether the x or y distance is greater, which helps determine the orientation.
    #     If the x distance is greater, it returns 90 degrees if x2 is greater than x1 (right direction),
    #     otherwise -90 degrees (left direction).
    #     If the y distance is greater, it returns 0 degrees if y2 is less than y1 (downward direction),
    #     otherwise 180 degrees (upward direction).
    #  * Example Call: 
    #     angle(0, 0, 1, 1)

    x_dis=abs(x1-x2)
    y_dis=abs(y1-y2)
    if(max(x_dis,y_dis)==x_dis):
        if(x2>x1):
            return 90
        else:
            return -90
    else:
        if(y2<y1):
            return 0
        else:
            return 180

 
def find_nearest_index(target):
    # """
    #  * Function Name: find_nearest_index
    #  * Input:
    #     - target (dict): A dictionary containing the coordinates of the target point.
    #       It should have keys 'x' and 'y' representing the x-coordinate and y-coordinate respectively.
    #  * Output: 
    #     - nearest_index (int or None): The index of the nearest coordinate in the 'coordinates' list.
    #       Returns None if the 'coordinates' list is empty.
    #  * Logic: 
    #     This function finds the index of the nearest coordinate to a target point in a list of coordinates.
    #     It iterates through the 'coordinates' list and calculates the Euclidean distance between each coordinate
    #     and the target point. The index of the coordinate with the minimum distance is stored as the nearest_index.
    #     The function utilizes the math.sqrt function to calculate the square root and the ** operator for exponentiation.
    #  * Example Call: 
    #     coordinates = [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}, {'x': 5, 'y': 6}]
    #     target = {'x': 4, 'y': 5}
    #     nearest_index = find_nearest_index(target)
    #     print(nearest_index)

    
    min_distance = float('inf')
    nearest_index = None
    
    for i, coord in enumerate(coordinates):                                                    # Loop through coordinates
        distance = math.sqrt((coord['x'] - target['x'])**2 + (coord['y'] - target['y'])**2)    # Calculate distance between target and current coord
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    return nearest_index



def turning_str(l,bot_angle,ArUco_details_dict):
    # """
    #  * Function Name: turning_str
    #  * Input:
    #     - l (list): A list of integers representing the sequence of ArUco markers.
    #       Each integer corresponds to a specific ArUco marker.
    #     - bot_angle (float): Angle of the bot's orientation in degrees (-180 to 180).
    #     - ArUco_details_dict (dict): A dictionary containing details of ArUco markers.
    #       Keys are ArUco marker IDs, and values are dictionaries containing 'x' and 'y' coordinates.
    #  * Output: 
    #     - turn (str): A string representing the sequence of directions the bot should take to traverse the given path.
    #       Directions include 'f' for forward, 'b' for backward, 'l' for left, and 'r' for right.
    #  * Logic: 
    #     This function generates a string of directions that instruct the bot on how to navigate through a sequence of ArUco markers.
    #     It iterates through the list 'l', which represents the sequence of ArUco markers.
    #     For each marker, it calculates the angle between the previous and current marker using the 'angle' function,
    #     then determines the direction the bot should turn using the 'direction' function.
    #     The calculated directions are appended to the 'turn' string.
    #  * Example Call: 
    #     path = [7, 12, 18, 9, 17, 8]
    #     bot_angle = 45
    #     ArUco_details_dict = {7: {'x': 10, 'y': 20}, 12: {'x': 15, 'y': 25}, ...}
    #     directions = turning_str(path, bot_angle, ArUco_details_dict)
    #     print(directions)

    turn=""
    flag=0
    if(l[0]=='7'):                                                                                 # Special case logic
        flag=1
    turn+=direction(cord[l[0]]['x'],cord[l[0]]['y'],cord[l[1]]['x'],cord[l[1]]['y'],bot_angle)     # Call direction function for first nodes
    for i in range(1,len(l)-1):
        if(l[i] in [17,18]):
            continue
        ang=angle(cord[l[i-1]]['x'],cord[l[i-1]]['y'],cord[l[i]]['x'],cord[l[i]]['y'])             # Calculate angle between nodes
        turn+=direction(cord[l[i]]['x'],cord[l[i]]['y'],cord[l[i+1]]['x'],cord[l[i+1]]['y'],ang)   # Call direction function
    return turn
