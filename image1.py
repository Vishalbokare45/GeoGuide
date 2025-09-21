import cv2

# Function to handle mouse click events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Display coordinates on the rotated frame
        print(f"Left Click: ({x}, {y})")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circle at click
        cv2.putText(frame, f"{x},{y}", (x, y), font, 0.5, (255, 0, 0), 1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Display color values at the clicked point
        print(f"Right Click: ({x}, {y})")
        b, g, r = frame[y, x]  # Get the color at the clicked pixel
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{b},{g},{r}", (x, y), font, 0.5, (255, 255, 0), 1)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Create window and set the mouse callback
cv2.namedWindow('Rotated Camera Feed')
cv2.setMouseCallback('Rotated Camera Feed', click_event)

while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Rotate the frame 90 degrees counterclockwise
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Display the rotated frame
    cv2.imshow('Rotated Camera Feed', rotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
