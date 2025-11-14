import os, sys, argparse, time
from collections import deque
import cv2, numpy as np, torch
from src.utils.utils import load_label_map, load_checkpoint
from src.model.model import build_model
from src.config.config import DEVICE
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose, lh, rh = [], [], []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z])
    else:
        pose = [0.0] * 33 * 3

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh = [0.0] * 21 * 3

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh = [0.0] * 21 * 3

    return np.array(pose + lh + rh, dtype=np.float32)


def realtime(args):
    label_list = load_label_map(args.label_map)
    model = build_model(num_classes=len(label_list), input_dim=args.input_dim).to(DEVICE)
    ck = load_checkpoint(args.ckpt, device=DEVICE)
    model.load_state_dict(ck['model_state'])
    model.eval()

    seq_len = args.seq_len
    cap = cv2.VideoCapture(args.camera_id)
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    buffer = deque(maxlen=seq_len)

    prev_gray = None
    MOTION_THRESHOLD = 12   # tune: 8–20 works well

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Motion detection ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                cv2.putText(frame, "Initializing...", (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imshow("Realtime Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            motion = cv2.absdiff(prev_gray, gray)
            _, th = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
            motion_level = th.mean()
            prev_gray = gray

            movement_detected = motion_level > MOTION_THRESHOLD

            # --- Process only when movement appears ---
            if movement_detected:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                vec = extract_keypoints(results)
                buffer.append(vec)

                if len(buffer) == seq_len:
                    arr = np.stack(list(buffer), axis=0).astype(np.float32)
                    X = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits = model(X)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    pred = int(probs.argmax())
                    label = label_list[pred]
                    conf = float(probs[pred])

                    cv2.putText(frame, f"{label} ({conf:.2f})", (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                else:
                    cv2.putText(frame, f"Capturing... {len(buffer)}/{seq_len}",
                                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)
            else:
                # reset buffer to avoid stale sequences
                buffer.clear()
                cv2.putText(frame, "Waiting for motion...", (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

            cv2.imshow('Realtime Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--label_map', required=True)
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--input_dim', type=int, default=225)
    p.add_argument('--camera_id', type=int, default=0)
    args = p.parse_args()
    realtime(args)
