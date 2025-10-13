import cv2
import numpy as np

video_path = "/Users/martinzanazzi/Desktop/IMG_4858.MOV"
cap = cv2.VideoCapture(video_path)

ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Couldn't read video.")

# Initialize accumulator
accumulated = prev_frame

frame_count = 0
frame_skip = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        # Compute absolute difference (motion)
        diff = cv2.absdiff(frame, prev_frame)

        # Convert to grayscale and threshold
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 9, 255, cv2.THRESH_BINARY)

        # Use mask to accumulate moving areas
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        accumulated = np.clip(accumulated + (mask_3ch // 255) * 10, 0, 255).astype(np.uint8)

    prev_frame = frame.copy()
    frame_count += 1

cap.release()

cv2.imwrite("/Users/martinzanazzi/Desktop/laser_motion_composite.png", accumulated)
print("Saved motion composite ✅")