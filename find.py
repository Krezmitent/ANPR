# find_line_positions.py  — run this once, then delete it
import cv2
import config

cap = cv2.VideoCapture(config.VIDEO_PATH)
ret, frame = cap.read()
if ret:
    cv2.imwrite("output/sample_frame.jpg", frame)
    print(f"Frame saved. Resolution: {frame.shape[1]}x{frame.shape[0]}")
cap.release()