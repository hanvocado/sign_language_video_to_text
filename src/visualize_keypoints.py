"""Lightweight visualization of face+hands landmarks on a video file or webcam."""
import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
import cv2, mediapipe as mp
mp_holistic = mp.solutions.holistic

def visualize_video(path):
    cap = cv2.VideoCapture(path)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            annotated = frame.copy()
            if results.face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(annotated, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)
            if results.left_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(annotated, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(annotated, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            cv2.imshow('Keypoints', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release(); cv2.destroyAllWindows(); holistic.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video', default=None, help='Path to video. If omitted, webcam will be used.')
    args = p.parse_args()
    if args.video:
        visualize_video(args.video)
    else:
        # use webcam
        visualize_video(0)
