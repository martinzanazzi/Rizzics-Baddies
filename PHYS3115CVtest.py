import cv2
import numpy as np

cap = cv2.VideoCapture(0)  
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (1, 1), 0)

alpha = 0.2  # background update speed (smaller = slower background)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = gray.astype("float")

trail = None          # <--- NEW: stores fading trails
trail_decay = 0.067   # <--- roughly 1/15 for 0.5s persistence at 30 fps

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    cv2.accumulateWeighted(gray, background, alpha)
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # --- NEW: accumulate fading trail ---
    if trail is None:
        trail = thresh.copy().astype("float")
    cv2.accumulateWeighted(thresh, trail, trail_decay)
    trail_display = cv2.convertScaleAbs(trail)

    # Display results
    cv2.imshow("iPhone Feed via OBS", frame)
    cv2.imshow("Detected Trails (Persistent)", trail_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()