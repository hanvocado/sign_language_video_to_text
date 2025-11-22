import os, sys, argparse, time
from collections import deque
import cv2, numpy as np, torch
from src.utils.utils import load_label_map, load_checkpoint
from src.model.model import build_model
from src.config.config import DEVICE
import mediapipe as mp
from src.utils.common_functions import *

mp_holistic = mp.solutions.holistic

def realtime(args):
    DISPLAY_DURATION = 1.5
    last_pred_label = None
    last_pred_conf = None
    last_pred_time = 0

    # Load label map + model
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

    prev_gray = None
    MOTION_THRESHOLD = 3
    STILL_FRAMES_REQUIRED = 8

    state = "waiting"
    segment = []
    still_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---------------------------------------------
            # Motion detection (FSM trigger)
            # ---------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                continue

            motion = cv2.absdiff(prev_gray, gray)
            _, th = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
            motion_level = th.mean()
            prev_gray = gray

            movement = motion_level > MOTION_THRESHOLD

            # ---------------------------------------------
            # STATE MACHINE
            # ---------------------------------------------
            if state == "waiting":
                cv2.putText(frame, "Waiting for sign...", (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

                if movement:
                    state = "recording"
                    segment = []
                    still_count = 0

            elif state == "recording":
                cv2.putText(frame, f"Recording ({len(segment)} frames)", (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

                # ---- Extract keypoints for this frame ----
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                vec = extract_keypoints(results)  # from utils
                segment.append(vec)

                if movement:
                    still_count = 0
                else:
                    still_count += 1

                # ---------------------------------------------------
                # SIGN COMPLETE — STOPPED MOVING
                # ---------------------------------------------------
                if still_count >= STILL_FRAMES_REQUIRED:

                    if len(segment) > 0:

                        raw_arr = np.array(segment, dtype=np.float32)
                        total_frames = len(raw_arr)

                        # -------------------------------
                        # FRAME SAMPLING
                        # -------------------------------
                        indices = sample_frames(total_frames, seq_len, mode="2")
                        sampled = raw_arr[indices]

                        # -------------------------------
                        # KEYPOINT NORMALIZATION
                        # -------------------------------
                        sampled = normalize_keypoints(sampled)

                        # convert to tensor
                        X = torch.from_numpy(sampled).unsqueeze(0).to(DEVICE)

                        # run model
                        with torch.no_grad():
                            logits = model(X)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                        pred = int(probs.argmax())
                        label = label_list[pred]
                        conf = float(probs[pred])

                        last_pred_label = label
                        last_pred_conf = conf
                        last_pred_time = time.time()

                        cv2.putText(frame, f"{label} ({conf:.2f})", (10,80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                    # reset FSM
                    state = "waiting"
                    segment = []
                    still_count = 0

            # keep last prediction on screen
            if last_pred_label is not None:
                if time.time() - last_pred_time <= DISPLAY_DURATION:
                    cv2.putText(frame, f"{last_pred_label} ({last_pred_conf:.2f})",
                                (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3)
                else:
                    last_pred_label = None

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
