# ''' 
# * Author: Rajas Bhosale
# * Filename: task_6.py
# * Functions: rem_garbage, cropping_v3, cropping_v5, cropping_v1, detect_ArUco_details, mark_ArUco_image, send_instruction, receive_from_usp32, signal_handler, cleanup
# * Global Variables: ArUco_details_dict, bot_id, ArUco_corners, priority_order, key_id_mapping, min_max_mapping, graph
# '''


from d_algov1 import dijkstra_shortest_path,turning_str,find_nearest_index
from Detection import detect
import socket
import sys	
import cv2
from cv2 import aruco
from live_tracking_QGIS import read_csv,write_csv,tracker
import time
import numpy as np
import math

#To save the details of aruco
ArUco_details_dict={}
#Aruko id of bot
bot_id=100
#to save aruco corners
ArUco_corners={}
#Required order of preority
priority_order=['fire','destroyed_buildings', 'humanitarian_aid', 'military_vehicles','combat']
key_id_mapping={"A":"14","B":"11","C":"18","D":"15","E":"6","0":"0"} #event key mapping to assumed nodes
min_max_mapping={"A":(137,494),"B":(341,399),"C":(341,312),"D":(134,299),"E":(123,155),"0":(35,540)} # stopping coordinates near the events 

#adjacency matrix of the nodes to create a graph
graph = {
    '0': [('1', 49.24)],
    '1': [('0', 49.24), ('2', 91.09), ('14', 78.06)],
    '2': [('1', 91.09), ('12', 188.13), ('3', 110.11)],
    '3': [('2', 110.11), ('15',74.06 ), ('4', 71.06)],
    '4': [('3', 71.06), ('17', 184.69), ('5', 69.86)],
    '5': [('4', 69.86), ('6',44.05)],
    '6': [('5', 44.05), ('7', 251.72)],
    '7': [('6', 251.72), ('8', 91.59)],
    '8': [('7', 91.59), ('17', 175.35),('9', 68.03)],
    '9': [('8', 68.03), ('18', 79.40), ('10', 83.01)],
    '10': [('9', 83.01), ('11', 78.10)],
    '11': [('10', 78.10), ('12', 98.00)],
    '12': [('11', 98.00), ('2', 188.13),('16', 94.00),('13', 91.05)],
    '13': [('12', 91.05), ('14', 103.08)],
    '14': [('13', 103.08), ('1', 78.06)],
    '15': [('3', 74.06), ('16', 109.17)],
    '16': [('15', 109.17), ('12', 94.00), ('18', 98.25),('17', 64.12)],
    '17': [('16', 64.12), ('4', 184.69),('8', 175.35)],
    '18': [('9', 79.40), ('16', 98.25)],
}

def rem_garbage(lis):
    # """
    # * Function Name: rem_garbage
    #  * Input:
    #     - lis (list): A list of elements.
    #  * Output: 
    #     - updated_list (list): The input list with specified "garbage" elements removed.
    #  * Logic:
    #     This function removes specified "garbage" elements from the input list.
    #     It first defines a list named 'gar' containing the identifiers of garbage elements to be removed.
    #     Then, it iterates through the input list 'lis' (excluding the first and last elements) to check for garbage elements.
    #     If an element in 'lis' matches any element in the 'gar' list, it is added to the 'to_remove' list.
    #     After iterating through 'lis', the function removes all elements listed in 'to_remove' from 'lis'.
    #     Finally, it returns the updated list without the garbage elements.
    #  * Example Call: 
    #     cleaned_list = rem_garbage(input_list)
    # """
    
    gar=["14","11","18","15","6"]
    to_remove=[]
    for i in range(1,len(lis)-1):
        if(lis[i] in gar):
            to_remove.append(lis[i])
    for i in to_remove:
        lis.remove(i)
    return lis

def cropping_v3(frame):
    # """
    # * Function Name: cropping_v3
    # * Input:
    #     - frame (numpy.ndarray): An input frame/image represented as a NumPy array.
    # * Output: 
    #     - cropped_image (numpy.ndarray): The cropped region of the input frame/image.
    # * Logic: 
    #     This function performs cropping on an input frame/image using manually defined corner points.
    #     It defines four corner points (x1, y1), (x2, y2), (x3, y3), and (x4, y4) to form a rectangle.
    #     The bounding rectangle around these points is calculated, which defines the region of the arena to be cropped.
    #     Finally,the cropped arena is returned.
    # * Example Call: 
    #     cropped = cropping_v3(input_frame)
    # """
    print("h2")
    x1, y1 = 85,38  # top-left
    x2, y2 = 531,35  # top-right
    x3, y3 = 531,552  # bottom-right
    x4, y4 = 84,550  # bottom-lefts

    # Define the rectangle to crop
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    rect = cv2.boundingRect(np.array(pts))

    # Get the coordinates for cropping
    x, y, w, h = rect

    # Crop the image
    cropped_image = frame[y:y + h, x:x + w]
    return cropped_image

def cropping_v5(frame):
    # """
    # * Function Name: cropping_v5
    # * Input:
    #     - frame (numpy.ndarray): An input frame/image represented as a NumPy array.
    # * Output: 
    #     - cropped_image (numpy.ndarray): The cropped region of the input frame/image.
    # * Logic: 
    #     This function performs cropping on an input frame/image using manually defined corner points.
    #     It defines four corner points (x1, y1), (x2, y2), (x3, y3), and (x4, y4) to form a rectangle.
    #     The bounding rectangle around these points is calculated, which defines the region of the arena to be cropped.
    #     Finally, the cropped arena is returned.
    #     It is used to show the e-Yantra logo along with the theme name.
    # * Example Call: 
    #     cropped = cropping_v5(input_frame)
    # ""
    print("h1")
    x1, y1 = 66,39  # top-left
    x2, y2 = 531,35  # top-right
    x3, y3 = 531,552  # bottom-right
    x4, y4 = 66,550  # bottom-left

    # Define the rectangle to crop
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    rect = cv2.boundingRect(np.array(pts))

    # Get the coordinates for cropping
    x, y, w, h = rect
    
    # Crop the image
    cropped_image = frame[y:y + h, x:x + w]
    return cropped_image

def cropping_v1(frame):
    # """
    # * Function Name: cropping_v1
    # * Input:
    #     - frame (numpy.ndarray): An input frame/image represented as a NumPy array.
    # * Output: 
    #     - cropped_image (numpy.ndarray): The cropped region of the input frame/image.
    # * Logic: 
    #     This function performs cropping on an input frame/image using manually defined corner points.
    #     It defines four corner points (x1, y1), (x2, y2), (x3, y3), and (x4, y4) to form a rectangle.
    #     The bounding rectangle around these points is calculated, which defines the region of the arena to be cropped.
    #     Finally, the cropped arena is returned.
    # * Example Call: 
    #     cropped = cropping_v1(input_frame)
    # """
    
    x1, y1 = 400,40  # top-left
    x2, y2 = 1361,40 # top-right
    x3, y3 = 1369,1030  # bottom-right
    x4, y4 = 400,1030  # bottom-left
    # Define the rectangle to crop
    pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    rect = cv2.boundingRect(np.array(pts))

    # Get the coordinates for cropping
    x, y, w, h = rect
    
    # Crop the image
    cropped_image = frame[y:y + h, x:x + w]
    return cropped_image



# Function to detect ArUco marker details
def detect_ArUco_details(image):
    # """
    # * Function Name: detect_ArUco_details
    # * Input:
    #     - image (numpy.ndarray): An input image containing ArUco markers, represented as a NumPy array.
    # * Output: None
    # * Logic: 
    #     This function detects ArUco markers in the input image and extracts their details.
    #     It uses the ArUco marker detection capabilities provided by OpenCV.
    #     First, it initializes an ArUco dictionary and detector parameters.
    #     Then, it detects markers' corners and IDs in the image using the detector.
    #     For each detected marker, it calculates its center coordinates and orientation angle.
    #     The marker's ID, center coordinates, and angle are stored in the ArUco_details_dict dictionary.
    #     Additionally, the corner coordinates of each marker are stored in the ArUco_corners dictionary.
    #     If no markers are detected in the image, the function does not perform any operations.
    # * Example Call: 
    #     detect_ArUco_details(input_image)
    # """
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    aruco_parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_parameters)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is not None:                                            # Check if any markers are detected.
        for i in range(len(ids)):                                  # Iterate through each detected marker.
            marker_id = int(ids[i][0])                             # Extract the marker id and corners of the detected marker.
            marker_corners = corners[i][0]

            center_x = int(np.mean(marker_corners[:, 0]))          # Calculate the center coordinates of the detected marker.
            center_y = int(np.mean(marker_corners[:, 1]))
            
            dx = marker_corners[1, 0] - marker_corners[0, 0]       # Calculate the angle of the detected marker.
            dy = marker_corners[1, 1] - marker_corners[0, 1]
            angle = np.degrees(np.arctan2(dy, dx))

            ArUco_details_dict[marker_id] = [[center_x, center_y], int(angle)]      # Store the marker details (center coordinates and angle) in ArUco_details_dict.
            ArUco_corners[marker_id] = marker_corners.astype(float)                 # Store the marker corners in ArUco_corners.

    
    
    return ArUco_details_dict, ArUco_corners 

def mark_ArUco_image(image,ArUco_details_dict, ArUco_corners):
    # """
    # * Function Name: detect_ArUco_details
    # * Input:
    #     - image (numpy.ndarray): An input image containing ArUco markers, represented as a NumPy array.
    # * Output: None
    # * Logic: 
    #     This function detects ArUco markers in the input image and extracts their details.
    #     It uses the ArUco marker detection capabilities provided by OpenCV.
    #     First, it initializes an ArUco dictionary and detector parameters.
    #     Then, it detects markers' corners and IDs in the image using the detector.
    #     For each detected marker, it calculates its center coordinates and orientation angle.
    #     The marker's ID, center coordinates, and angle are stored in the ArUco_details_dict dictionary.
    #     Additionally, the corner coordinates of each marker are stored in the ArUco_corners dictionary.
    #     If no markers are detected in the image, the function does not perform any operations.
    # * Example Call: 
    #     detect_ArUco_details(input_image)
    # """
    
    mi=2260.31          # Initialize a variable mi with the value 2260.31.
    present_id=0        # Initialize a variable mi with the value 2260.31.
    try:
        bot_x, bot_y= ArUco_details_dict[bot_id][0][0],ArUco_details_dict[bot_id][0][1]
    except:
        bot_x,bot_y= 32,842
    for ids, details in ArUco_details_dict.items():                     # Iterate through each key-value pair in ArUco_details_dict.
        center = details[0]                                             # Extract the center coordinates of the ArUco marker.
        dis=math.sqrt((center[0] - bot_x)**2 + (center[1] - bot_y)**2)  # Calculate the distance between the current ArUco marker and the bot using Euclidean distance formula
        if ids!=100:                                                    # Check if the current ArUco marker id is not equal to 100 (assuming 100 is some special marker)
            if mi>dis:
                mi=dis
                present_id=ids
        
        # Drawing markers on image
        cv2.circle(image, center, 5, (0,0,255), -1)

        corner = ArUco_corners[int(ids)]                                                # Extract the corner coordinates of the ArUco marker using its id.
        
        # Draw filled circles at each corner of the ArUco marker.
        # Each corner is represented by a different color.
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2) 

        cv2.line(image,center,(tl_tr_center_x, tl_tr_center_y),(255,0,0),5)
        display_offset = int(math.sqrt((tl_tr_center_x - center[0])**2+(tl_tr_center_y - center[1])**2))
        cv2.putText(image,str(ids),(center[0]+int(display_offset/2),center[1]),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)   # Draw text showing the ArUco marker id next to its center.
        angle = details[1]
        cv2.putText(image,str(angle),(center[0]-display_offset,center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)          # Draw text showing the angle next to the center of the ArUco marker.
    return image,present_id

def send_instruction(instruction, usp32_socket):
    # """
    # * Function Name: send_instruction
    # * Input:
    #     - instruction (str): The instruction to be sent over the socket.
    #     - usp32_socket: The socket object used for communication.
    # * Output: None
    # * Logic: 
    #     This function sends an instruction over the specified socket connection.
    #     It encodes the instruction string into bytes using UTF-8 encoding and sends it over the socket.
    #     If the communication is successful, the instruction is transmitted to the recipient.
    # * Example Call: 
    #     send_instruction("Turn left", usp32_socket)
    # """
    print(instruction)
    usp32_socket.sendall(str.encode(instruction))

def receive_from_usp32(conn):
    # """
    # * Function Name: receive_from_usp32
    # * Input:
    #     - conn: The socket connection object from which data will be received.
    # * Output: 
    #     - str: A string indicating the received message type ('ready', 'last_node', 'finish', or 'no').
    #     - bool: False if a timeout occurs during socket receive.
    # * Logic: 
    #     This function attempts to receive data from the specified socket connection.
    #     It sets a timeout for the receive operation to prevent blocking.
    #     If data is received, it checks for specific substrings in the received message to determine the message type.
    #     If the message contains "red", it indicates that the USP32 is ready.
    #     If the message contains "ln", it indicates that it's the last node.
    #     If the message contains "final", it indicates that it's the final message.
    #     Otherwise, it returns "no" to indicate a regular message.
    #     If a timeout occurs during socket receive, the function returns False.
    # * Example Call: 
    #     message_type = receive_from_usp32(connection)
    # """
    
    conn.settimeout(0.01)
    try:
        data = conn.recv(1024)
        if "red" in  str(data):
            return "ready"  
        elif "ln" in str(data):
            return "last_node"
        elif "final" in str(data):
            return "finish"
        else:
            return "no"
    except socket.timeout:
        return False
    
    
if __name__ == "__main__":
    lat_lon = read_csv('mud.csv')
    def signal_handler(sig, frame):
    #  """
    # * Function Name: signal_handler
    # * Input:
    #     - sig: The signal number received by the handler.
    #     - frame: The current stack frame at the time the signal was received.
    # * Output: None
    # * Logic: 
    #     This function serves as a signal handler for specific signals.
    #     When a signal is received, it prints a message indicating clean-up actions are being performed.
    #     It then calls a cleanup function to perform any necessary clean-up operations.
    #     After cleanup, the function exits the program with a status code of 0.
    # * Example Call: signal.signal(signal.SIGINT, signal_handler)
    # """
        
        print('Clean-up !')
        cleanup()
        sys.exit(0)

    def cleanup():
    #  """
    # * Function Name: cleanup
    # * Input: None
    # * Output: None
    # * Logic: 
    #     This function performs clean-up operations before exiting the program.
    #     It closes the socket connection 's' and prints a message indicating that clean-up is done.
    # * Example Call: cleanup()
    # """
        
        s.close()
        print("cleanup done")

    
    ip = "192.168.43.17"     #Enter IP address of laptop after connecting it to WIFI hotspot
    nodearrived= False
    print("server listening on " + ip)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, 8002))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")          # Print a message indicating that a connection is established along with the client's address.
            n=0                                    # Initialize a counter variable n.
            end_node=""                            # Initialize an empty string to hold the end node.
            identified_labels=detect()
            sorted_keys = sorted(identified_labels.keys(), key=lambda x: priority_order.index(identified_labels[x])) 
            print(sorted_keys) # Sort the keys of the identified_labels dictionary based on a custom priority order.
            sorted_keys.append("0")          # Append "0" to the sorted_keys list.
            lis_ind=0 
            cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
            print(f"Connected by {addr}")
            last=0                                 # Initialize a variable last to keep track of the last node.
            fake=10                                # Initialize a variable fake with a value of 10.
            curr_time=time.time()
            while True:
                _,_img=cap.read()  
                # img=cropping_v1(_img)                        # Crop the image using the cropping_v3 function.
                img = _img
                img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)  # Crop the image using the cropping_v3 function.
                # img=cv2.resize(img, (1920, 1080))              # Resize the rotated image to a specified width and height
                show_img=_img
                show_img=cv2.rotate(show_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                # img=cv2.resize(img, (1920, 1080)) 
                # show_img=cv2.resize(show_img, (893, 900))
                if(time.time()-curr_time>8):                 # Check if 8 seconds have passed since the last iteration
                    nodearrived = receive_from_usp32(conn)   # Receive data from USP32  
                    ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
                    # print(ArUco_details_dict) # Detect ArUco marker details and corners in the image.
                    img, present_id = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)  # Mark ArUco markers on the image and retrieve the present marker ID.
                    try:
                        t_point = tracker(present_id, lat_lon)
                    except:
                        pass
                    # print({'x': ArUco_details_dict[100][0][0], 'y': ArUco_details_dict[100][0][1]})    
                    start_node=str(find_nearest_index({'x': ArUco_details_dict[100][0][0], 'y': ArUco_details_dict[100][0][1]}))
                    # print( "start node :"+ start_node )# Determine the start node based on the position of ArUco marker with ID 100.
                    if(nodearrived=="finish"):           # Check if the nodearrived signal indicates task completion.
                        send_instruction("zmt\n", conn)  # Send the "zmt" instruction to ESP32
                        print("Task Completed")
                        time.sleep(5)
                        break
                    if(nodearrived=="last_node") or (last==1):  # Check if the nodearrived signal indicates arrival at the last node or if last flag is set.
                        last=1                                  # Set last to 1 if it's not already set.
                        if(end_node=="0"):                      # Check if the end_node is "0" and the start_node is within certain coordinates.
                            if(start_node=="0"):
                                if(ArUco_details_dict[100][0][0]>min_max_mapping["0"][0] and ArUco_details_dict[100][0][0]<min_max_mapping["0"][1]):
                                    send_instruction("end\n", conn)  # Send the "end" instruction to ESP32
                        elif(ArUco_details_dict[100][0][0]>min_max_mapping[sorted_keys[lis_ind]][0] and ArUco_details_dict[100][0][0]<min_max_mapping[sorted_keys[lis_ind]][1]):   # Check if the present marker's x-coordinate falls within specified range for the current end node.
                            
                            send_instruction("stop\n", conn)  # Send the "stop" instruction to ESP32
                            print("stop sent")
                            last=0                            # Reset the last flag and increment the lis_ind index.
                            lis_ind+=1
                    if(nodearrived=="ready"):                    # Check if the nodearrived signal indicates readiness for the next node.

                        n+=1                                           # Increment the node counter.
                        end_node=key_id_mapping[sorted_keys[lis_ind]]  # Determine the end node based on the sorted_keys list and lis_ind index.
                        print(end_node)
                        shortest_path = dijkstra_shortest_path(graph, start_node, end_node)
                        print(shortest_path)# Find the shortest path using Dijkstra's algorithm.
                        shortest_path=rem_garbage(shortest_path) 
                        int_list = [int(x) for x in shortest_path]         # Convert the shortest path to a list of integers.
                        bot_angle=0                                        # Calculate the bot's angle and generate the route string.
                        route=turning_str(int_list, ArUco_details_dict[100][1],ArUco_details_dict)
                        route=route+"\n"
                        print(route)
                        send_instruction(route, conn)           # Send the route instruction to ESP32
                        nodearrived=""
                cv2.imshow("Marked Image", show_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):    # Break the loop if the 'q' key is pressed.
                    break
            print("task completed in:", time.time()-curr_time-8)     # Print the time taken to complete the task.
            cap.release()                                            # Release the camera capture object.
            cv2.destroyAllWindows()                                  # Destroy all OpenCV windows.
