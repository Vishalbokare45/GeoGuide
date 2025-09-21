import cv2
import numpy as np
from cv2 import aruco
# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
# Open the camera feed
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert frame to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the frame
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # Draw detected markers and their IDs
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            # Get the corner points of the detected marker
            corner = corners[i][0]

            # Calculate the angle of rotation using the corner points
            # Find the center of the marker
            center_x = int((corner[0][0] + corner[2][0]) / 2)
            center_y = int((corner[0][1] + corner[2][1]) / 2)

            # Calculate the angle of orientation
            dx = corner[1][0] - corner[0][0]
            dy = corner[1][1] - corner[0][1]
            angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert radians to degrees
            angle = angle % 360  # Normalize angle to be between 0 and 360

            # Display the angle on the frame
            cv2.putText(frame, f'Angle: {angle:.2f} degrees', (center_x - 40, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Aruco Marker Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
