import numpy as np

# =====================================================
# Normalization
# =====================================================

def normalize_keypoints(seq, left_wrist_idx=15, right_wrist_idx=16):
    """
    Normalize keypoints:
    - Center at midpoint between wrists
    - Scale by bounding box diagonal
    """
    num_landmarks = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)

    # Get reference point (center of wrists)
    lw = seq3d[:, left_wrist_idx, :2]
    rw = seq3d[:, right_wrist_idx, :2]
    
    # Check if both wrists are missing (all zeros)
    lw_missing = np.all(lw == 0, axis=1)
    rw_missing = np.all(rw == 0, axis=1)
    both_missing = lw_missing & rw_missing
    
    # Calculate reference point
    ref = (lw + rw) / 2
    
    # For frames with missing wrists, use mean of all keypoints
    if np.any(both_missing):
        mean_all = np.mean(seq3d[:, :, :2], axis=1)
        ref[both_missing] = mean_all[both_missing]
    
    # Center
    seq3d[:, :, 0] -= ref[:, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref[:, 1].reshape(-1, 1)

    # Scale by bounding box diagonal
    min_c = np.min(seq3d[:, :, :2], axis=1)
    max_c = np.max(seq3d[:, :, :2], axis=1)
    scale = np.linalg.norm(max_c - min_c, axis=1)
    scale[scale == 0] = 1  # Avoid division by zero
    seq3d[:, :, :2] /= scale.reshape(-1, 1, 1)

    return seq3d.reshape(seq.shape[0], -1)


# =====================================================
# Keypoint Extraction
# =====================================================

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

# =====================================================
# Frame Sampling
# =====================================================

def get_chunks(l, n):
    """
    Divide list `l` into `n` chunks as evenly as possible.
    Guarantees: never returns empty chunks.
    """
    if len(l) == 0:
        return [[] for _ in range(n)]

    if len(l) < n:
        # pad by repeating the last element
        l = l + [l[-1]] * (n - len(l))

    k, m = divmod(len(l), n)
    chunks = [
        l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n)
    ]

    # Final safety: ensure no empty chunk
    for i in range(n):
        if len(chunks[i]) == 0:
            chunks[i] = [l[-1]]

    return chunks


def safe_pick(chunk, pick_index):
    """Pick element safely from chunk."""
    if len(chunk) == 0:
        return 0
    pick_index = min(max(pick_index, 0), len(chunk) - 1)
    return chunk[pick_index]


def sampling_mode_1(chunks):
    """
    Your original logic but safe for short videos.
    """
    sampling = []
    L = len(chunks)

    for i, chunk in enumerate(chunks):
        if i == 0 or i == 1:
            sampling.append(safe_pick(chunk, -1))
        elif i == L - 1 or i == L - 2:
            sampling.append(safe_pick(chunk, 0))
        else:
            sampling.append(safe_pick(chunk, len(chunk) // 2))

    return sampling


def sampling_mode_2(frames, n_sequence):
    """
    Remove idle frames (first+last chunk's worth), then sample n_sequence frames.
    Works even if frames < 12.
    """

    L = len(frames)

    # If the video is too short for the full pre-sampling logic
    if L < 12:
        # fallback to uniform sampling
        chunks = get_chunks(frames, n_sequence)
        return sampling_mode_1(chunks)

    # Normal mode:
    chunks_12 = get_chunks(frames, 12)

    # drop first + last chunk
    middle = chunks_12[1:-1]

    # flatten
    sub_frame_list = [x for c in middle for x in c]

    # If sub_frame_list becomes too small
    if len(sub_frame_list) < n_sequence:
        chunks = get_chunks(sub_frame_list, n_sequence)
    else:
        chunks = get_chunks(sub_frame_list, n_sequence)

    return sampling_mode_1(chunks)


def sample_frames(total_frames, seq_len, mode="2"):
    frames = list(range(total_frames))
    if mode == "1":
        return sampling_mode_1(get_chunks(frames, seq_len))
    elif mode == "2":
        return sampling_mode_2(frames, seq_len)
    else:
        raise ValueError("Invalid sampling mode")
