import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)  
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (1, 1), 0)

alpha = 0.2  # background update speed (smaller = slower background)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = gray.astype("float")

trail = None
trail_decay = 0.067

# warmup: ignore detections while background model stabilizes
warmup_seconds = 5.0
start_time = time.time()

# --- NEW persistent counters ---
total_count = 0
prev_centroids = []
min_dist = 30  # distance threshold (pixels) for "same" blob

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    cv2.accumulateWeighted(gray, background, alpha)
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background))
    _, thresh = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)

    # --- accumulate fading trail ---
    if trail is None:
        trail = thresh.copy().astype("float")
    cv2.accumulateWeighted(thresh, trail, trail_decay)
    trail_display = cv2.convertScaleAbs(trail)

    # --- find moving blobs ---
    contours, _ = cv2.findContours(trail_display, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    curr_centroids = []

    # compute warmup state
    elapsed = time.time() - start_time
    warming_up = elapsed < warmup_seconds

    for c in contours:
        area = cv2.contourArea(c)
        if area > 300:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            curr_centroids.append((cx, cy))
            # only draw bounding boxes after warmup
            if not warming_up:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- detect new blobs (not near old ones) ---
    new_objects = 0
    for (cx, cy) in curr_centroids:
        if all(np.hypot(cx - px, cy - py) > min_dist for (px, py) in prev_centroids):
            new_objects += 1

    # only update totals after warmup
    if not warming_up:
        total_count += new_objects
    prev_centroids = curr_centroids

    # --- Display results ---
    if warming_up:
        remaining = max(0.0, warmup_seconds - elapsed)
        cv2.putText(frame, f"Warming up: {remaining:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    else:
        cv2.putText(frame, f"Total Tracks: {total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("iPhone Feed via OBS", frame)
    cv2.imshow("Detected Trails (Persistent)", trail_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()