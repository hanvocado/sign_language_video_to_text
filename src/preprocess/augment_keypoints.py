import numpy as np
import os, argparse, glob
from tqdm import tqdm

# -------------------------------------------------------------
# Rotation: randomly rotate (x, y) coordinates of all landmarks
# -------------------------------------------------------------
def rotate_points(seq, max_angle=10):
    """Rotate all (x, y) coordinates by a random small angle (in degrees)."""
    angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    
    # seq shape: (seq_len, num_features)
    # Each frame has triplets (x, y, z, x, y, z, ...)
    coords = seq.reshape(seq.shape[0], -1, 3)  # (T, N, 3)
    xy = coords[:, :, :2]  # take (x, y)
    rotated_xy = np.dot(xy, R.T)  # apply rotation
    coords[:, :, :2] = rotated_xy
    return coords.reshape(seq.shape)

# -------------------------------------------------------------
# Add small Gaussian noise to simulate landmark jitter
# -------------------------------------------------------------
def add_noise(seq, sigma=0.01):
    """Add Gaussian noise to all coordinates."""
    return seq + np.random.normal(0, sigma, seq.shape)

# -------------------------------------------------------------
# Temporal jitter: randomly drop frames and pad/truncate to fixed length
# -------------------------------------------------------------
def temporal_jitter(seq, seq_len, drop_prob=0.05):
    """Randomly drop a few frames and pad/truncate to fixed seq_len."""
    keep = np.random.rand(seq.shape[0]) > drop_prob
    seq = seq[keep]

    if len(seq) == 0:  # avoid empty sequence
        return np.tile(seq[-1], (seq_len, 1))

    # Pad or truncate back to fixed length
    if len(seq) < seq_len:
        pad = np.tile(seq[-1], (seq_len - len(seq), 1))
        seq = np.vstack([seq, pad])
    elif len(seq) > seq_len:
        idx = np.linspace(0, len(seq) - 1, seq_len).astype(int)
        seq = seq[idx]
    return seq

# -------------------------------------------------------------
# Combine augmentations randomly
# -------------------------------------------------------------
def augment_once(seq, seq_len):
    """Apply a random combination of augmentations."""
    if np.random.rand() < 0.3:
        seq = rotate_points(seq)
    if np.random.rand() < 0.5:
        seq = add_noise(seq)
    if np.random.rand() < 0.9:
        seq = temporal_jitter(seq, seq_len)
    return seq

# -------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment .npy landmark sequences")
    parser.add_argument("--input_dir", default="data/npy_raw", help="Input directory of .npy sequences")
    parser.add_argument("--output_dir", default="data/npy_aug", help="Output directory for augmented data")
    parser.add_argument("--seq_len", type=int, default=64, help="Fixed sequence length")
    parser.add_argument("--num_aug", type=int, default=2, help="Number of augmentations per file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_files = 0
    total_aug = 0

    label_dirs = glob.glob(os.path.join(args.input_dir, "*"))
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        out_label_dir = os.path.join(args.output_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)

        npy_files = glob.glob(os.path.join(label_dir, "*.npy"))
        for npy_path in tqdm(npy_files, desc=f"Augmenting {label}"):
            seq = np.load(npy_path)
            base = os.path.splitext(os.path.basename(npy_path))[0]
            total_files += 1

            for i in range(args.num_aug):
                aug_seq = augment_once(seq.copy(), args.seq_len)
                out_path = os.path.join(out_label_dir, f"{base}_aug{i+1}.npy")
                np.save(out_path, aug_seq)
                total_aug += 1

    print("\nData augmentation completed!")
    print(f"Original sequences: {total_files}")
    print(f"Augmented sequences generated: {total_aug}")
    print(f"Total after augmentation: {total_files + total_aug}")
    print(f"Output directory: {args.output_dir}")
