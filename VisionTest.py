import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarksConnections

# Path to the model downloaded from Google's MediaPipe model garden.
# Resolved relative to this file so the script works from any working directory.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (device 0). Check that a camera is connected and camera permissions are granted.")

with vision.PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB because MediaPipe uses RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode needs a monotonically increasing timestamp in milliseconds
        timestamp_ms = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            for landmarks in result.pose_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                # Draw the skeleton connections
                for conn in PoseLandmarksConnections.POSE_LANDMARKS:
                    cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2)

                # Draw the landmark points
                for x, y in points:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("MediaPipe Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
