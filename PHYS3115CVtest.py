import cv2  # Import OpenCV library for video capture, display, and image processing

# 0 = first camera detected by OS (your iPhone via OBS Virtual Camera / Continuity)
cap = cv2.VideoCapture(0)  
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)

# Start infinite loop to grab frames from camera
while True:
    ret, frame = cap.read()  # Grab a single frameq
    # ret = True if frame was successfully captured, False otherwise
    # frame = NumPy array (height x width x channels) representing imageq

    if not ret:  # Check if the frame was captured
        print("No frame received")  # Warn user
        break  # Exit loop if camera feed fails

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Frame differencing: highlights new streaks or motion
    diff = cv2.absdiff(prev_gray, gray)

    _, thresh = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("iPhone Feed via OBS", frame)  # Display current frame in a window

    # Display results
    cv2.imshow("Cloud Chamber", frame)
    cv2.imshow("Detected Trails", thresh)

    prev_gray = gray.copy()  # update background

    # cv2.waitKey(1) waits 1 ms for a key press
    # & 0xFF masks only the lower 8 bits (ASCII compatibility)
    # ord('q') converts 'q' → 113
    # So this checks: "Did the user press 'q'?"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

# Clean up
cap.release()             # Release the camera so other programs can use it
cv2.destroyAllWindows()   # Close all OpenCV windows