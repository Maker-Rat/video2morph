import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

def get_gripper_value(hand_landmarks):
    landmarks = hand_landmarks
    fingers = [
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17),
    ]
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    scores = []
    for tip_idx, mcp_idx in fingers:
        tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
        mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])
        tip_dist = np.linalg.norm(tip - wrist)
        mcp_dist = np.linalg.norm(mcp - wrist)
        scores.append(tip_dist / (mcp_dist + 1e-6))
    raw = np.mean(scores)
    return np.clip((raw - 1.0) / 1.5, 0.0, 1.0)

options = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
gripper_smooth = 0.5
alpha = 0.2

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            gripper_raw = get_gripper_value(landmarks)
            gripper_smooth = alpha * gripper_raw + (1 - alpha) * gripper_smooth

            # Draw landmarks manually
            h, w = frame.shape[:2]
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # === SEND TO ROBOT HERE ===
            print(f"Gripper: {gripper_smooth:.2f}")

        # Visual feedback bar
        h, w = frame.shape[:2]
        bar_len = int(gripper_smooth * 200)
        cv2.rectangle(frame, (10, 10), (210, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 10), (10 + bar_len, 40), (0, 200, 100), -1)
        cv2.putText(frame, f"Gripper: {gripper_smooth:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 2)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()