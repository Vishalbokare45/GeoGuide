import cv2
import numpy as np

# Initialize graph and node variables
graph = {}
node_positions = {}
current_node = 0

# Mouse click event handler
def click_event(event, x, y, flags, param):
    global current_node

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust coordinates for the 90-degree counterclockwise rotation
        rotated_x = y  # y becomes the new x in the rotated frame
        rotated_y = frame.shape[1] - 1 - x  # Invert x to get the correct y

        # Store the clicked point as a new node
        node_name = str(current_node)
        node_positions[node_name] = (rotated_x, rotated_y)
        graph[node_name] = []  # Initialize adjacency list

        # Draw the red circle and node label on the transformed coordinates
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dot where clicked
        cv2.putText(frame, node_name, (x - 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"Node {node_name} added at ({rotated_x}, {rotated_y})")
        current_node += 1

    elif event == cv2.EVENT_RBUTTONDOWN and len(node_positions) > 1:
        # Connect the last two nodes with an edge
        nodes = list(node_positions.keys())
        node1, node2 = nodes[-2], nodes[-1]  # Last two nodes

        pos1, pos2 = node_positions[node1], node_positions[node2]
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))  # Euclidean distance

        # Add edges to the graph
        graph[node1].append((node2, round(distance, 2)))
        graph[node2].append((node1, round(distance, 2)))

        # Draw the edge between the two nodes on the frame
        cv2.line(frame, (pos1[1], frame.shape[1] - 1 - pos1[0]), 
                 (pos2[1], frame.shape[1] - 1 - pos2[0]), (255, 0, 0), 2)

        print(f"Edge added between {node1} and {node2} with distance {round(distance, 2)}")

# Open the camera feed
cap = cv2.VideoCapture(0)  # 0 is the default camera index

# Set up window and mouse callback
cv2.namedWindow("Rotated Camera POV")
cv2.setMouseCallback("Rotated Camera POV", click_event)

print("Left-click to add nodes, right-click to connect the last two nodes.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Rotate the frame by 90 degrees counterclockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Draw previously marked nodes and edges
    for node, (x, y) in node_positions.items():
        # Draw circles at transformed positions
        cv2.circle(frame, (y, frame.shape[1] - 1 - x), 5, (0, 0, 255), -1)
        cv2.putText(frame, node, (y - 10, frame.shape[1] - 1 - x - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for node, edges in graph.items():
        for neighbor, _ in edges:
            pos1 = node_positions[node]
            pos2 = node_positions[neighbor]
            # Draw edges between connected nodes
            cv2.line(frame, (pos1[1], frame.shape[1] - 1 - pos1[0]), 
                     (pos2[1], frame.shape[1] - 1 - pos2[0]), (255, 0, 0), 2)

    # Show the rotated frame
    cv2.imshow("Rotated Camera POV", frame)

    # Press 'q' to quit and print the final graph
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nFinal Graph Dictionary:")
        print(graph)
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
