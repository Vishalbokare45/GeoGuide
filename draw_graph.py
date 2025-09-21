import cv2

graph = {
    '0': [('1', 49.24)],
    '1': [('0', 49.24), ('2', 91.09), ('14', 78.06)],
    '2': [('1', 91.09), ('12', 188.13), ('3', 110.11)],
    '3': [('2', 110.11), ('15', 74.06), ('4', 71.06)],
    '4': [('3', 71.06), ('17', 184.69), ('5', 69.86)],
    '5': [('4', 69.86), ('6', 44.05)],
    '6': [('5', 44.05), ('7', 251.72)],
    '7': [('6', 251.72), ('8', 91.59)],
    '8': [('7', 91.59), ('17', 175.35), ('9', 68.03)],
    '9': [('8', 68.03), ('18', 79.40), ('10', 83.01)],
    '10': [('9', 83.01), ('11', 78.10)],
    '11': [('10', 78.10), ('12', 98.00)],
    '12': [('11', 98.00), ('2', 188.13), ('16', 94.00), ('13', 91.05)],
    '13': [('12', 91.05), ('14', 103.08)],
    '14': [('13', 103.08), ('1', 78.06)],
    '15': [('3', 74.06), ('16', 109.17)],
    '16': [('15', 109.17), ('12', 94.00), ('18', 98.25), ('17', 64.12)],
    '17': [('16', 64.12), ('4', 184.69), ('8', 175.35)],
    '18': [('9', 79.40), ('16', 98.25)],
}

coordinates = [
    {'x': 35, 'y': 540}, {'x': 59, 'y': 497}, {'x': 55, 'y': 406},
    {'x': 60, 'y': 296}, {'x': 63, 'y': 225}, {'x': 79, 'y': 157},
    {'x': 123, 'y': 155}, {'x': 374, 'y': 174}, {'x': 422, 'y': 252},
    {'x': 420, 'y': 320}, {'x': 419, 'y': 403}, {'x': 341, 'y': 399},
    {'x': 243, 'y': 399}, {'x': 240, 'y': 490}, {'x': 137, 'y': 494},
    {'x': 134, 'y': 299}, {'x': 243, 'y': 305}, {'x': 247, 'y': 241},
    {'x': 341, 'y': 312}
]

def draw_graph_on_frame(frame, graph, coordinates):
    # Draw nodes
    for i, coord in enumerate(coordinates):
        x, y = coord['x'], coord['y']
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circle
        cv2.putText(frame, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)  # Label with node number

    # Draw edges
    for node, edges in graph.items():
        for neighbor, _ in edges:
            x1, y1 = coordinates[int(node)]['x'], coordinates[int(node)]['y']
            x2, y2 = coordinates[int(neighbor)]['x'], coordinates[int(neighbor)]['y']
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green line

# Open the camera feed
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Rotate frame 90 degrees counterclockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Draw the graph on the frame
    draw_graph_on_frame(frame, graph, coordinates)

    # Display the frame
    cv2.imshow('Graph on Camera Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
