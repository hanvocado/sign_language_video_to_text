# 📊 PHÂN TÍCH CHI TIẾT SOURCE CODE - sign_language_video_to_text

---

## I. OVERVIEW

Dự án **Sign Language Recognition (Vietnamese)** là một pipeline hoàn chỉnh từ video → keypoints → LSTM training → realtime inference.

**Core Technology Stack:**
- **PyTorch**: Deep learning framework
- **MediaPipe Holistic**: Pose + Hand landmark extraction
- **OpenCV**: Video processing
- **NumPy/Pandas**: Data manipulation
- **scikit-learn**: Scaling, metrics

**Main Pipeline:**
```
Raw Videos (data/raw_unprocessed)
    ↓ preprocess_video.py
Preprocessed Videos (data/raw) - 30fps, 1280x720, motion-detected segments
    ↓ video2npy.py
Keypoint Sequences (data/npy) - (seq_len=64, 225 features)
    ↓ split_dataset.py
Split Train/Val/Test (data/splits/*.csv)
    ↓ train.py
Trained Model (models/checkpoints/best.pth)
    ↓ infer_realtime.py or app.py
Real-time Predictions
```

---

## II. DETAILED COMPONENT ANALYSIS

### 2.1 **Configuration: `src/config/config.py`**

```python
# Directory Structure
BASE_DIR = ...
DATA_DIR, RAW_DIR, NPY_DIR, SPLIT_DIR, MODEL_DIR, CKPT_DIR

# Model Configuration
FEATURE_DIM = 225  # 99 (pose) + 63 (left_hand) + 63 (right_hand)
SEQ_LEN = 64       # Fixed sequence length
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 40
DEVICE = 'cuda' or 'cpu'
```

**Keypoint Breakdown:**
- **Pose**: 33 landmarks × 3 coords = 99
- **Left Hand**: 21 landmarks × 3 = 63
- **Right Hand**: 21 landmarks × 3 = 63
- **Total**: 225 features/frame

---

### 2.2 **Video Preprocessing: `src/preprocess/preprocess_video.py`**

#### Purpose:
Chuẩn hóa video và tự động cắt những phần chỉ có chuyển động.

#### Key Functions:

**1. `detect_motion(frame1, frame2, threshold=25, min_area=500)`**
- So sánh 2 frame liên tiếp bằng `cv2.absdiff()`
- Threshold pixel difference
- Dilate để fill gaps
- Tìm contours và check diện tích

```python
diff = cv2.absdiff(frame1, frame2)           # Tính sự khác biệt
_, thresh = cv2.threshold(diff, 25, 255)    # Ngưỡng hóa
dilated = cv2.dilate(thresh, kernel, 2)     # Mở rộng vùng
contours = cv2.findContours(dilated)        # Tìm contours
# Nếu contour_area > min_area → có chuyển động
```

**2. `preprocess_video(input_path, output_path, ...)`**
- **Input**: Video gốc
- **Output**: Video (hoặc nhiều segments) đã chuẩn hóa

**Processing Steps:**
```
1. Open video dengan cv2.VideoCapture()
2. For each frame:
   - Resize: 1280×720 (16:9)
   - Motion detection (compare grayscale frames)
   - Normalize pixel values: [0, 255] → [0, 1]
   - Write to output nếu đang "recording"
3. FSM (Finite State Machine):
   - "waiting" → chờ chuyển động
   - "recording" → ghi frames khi có chuyển động
   - Quay về "waiting" khi chuyển động dừng lâu
```

**3. `normalize_frame(frame)`**
```python
# Min-max normalization
frame_min = frame.min()
frame_max = frame.max()
normalized = (frame - frame_min) / (frame_max - frame_min)  # [0, 1]
return (normalized * 255).astype(np.uint8)  # Convert back to [0, 255]
```

**Parameters:**
- `fps`: 30 (chuẩn hóa tần số)
- `width, height`: 1280×720
- `max_duration`: 5-10 giây/clip
- `motion_threshold`: 25 (pixel diff threshold)
- `min_motion_area`: 500 pixels
- `min_motion_frames`: 5 frames liên tiếp
- `motion_buffer`: 10 frames trước/sau chuyển động

---

### 2.3 **Video to Keypoints: `src/preprocess/video2npy.py`**

#### Purpose:
Chuyển video thành `.npy` sequences chứa keypoints.

#### Key Functions:

**1. `sample_frames(total_frames, seq_len, mode="2")`**

**Mode 1 - Uniform Sampling:**
```python
def sampling_mode_1(chunks):
    for i, chunk in enumerate(chunks):
        if i == 0 or i == 1:
            pick chunk[-1]      # Last frame (start)
        elif i == len-1 or i == len-2:
            pick chunk[0]       # First frame (end)
        else:
            pick chunk[len//2]  # Middle frame
```

**Mode 2 - Smart Sampling (DEFAULT):**
```python
def sampling_mode_2(frames, n_sequence):
    if len(frames) < 12:
        fallback_to_mode_1()
    
    chunks_12 = split_into_12_chunks(frames)
    middle = chunks_12[1:-1]  # Bỏ chunk đầu + cuối (thường tĩnh)
    sub_frame_list = flatten(middle)
    chunks = split_into_n_sequence_chunks(sub_frame_list)
    return sampling_mode_1(chunks)
```

**Purpose**: Loại bỏ phần "tĩnh" ở đầu/cuối video, chỉ lấy phần chuyển động.

**2. `extract_keypoints(results)`**
```python
# Extract từ MediaPipe results
pose = []    # 33 landmarks × 3 = 99
left_hand = [] # 21 landmarks × 3 = 63
right_hand = []# 21 landmarks × 3 = 63

return np.array([pose + left_hand + right_hand])  # 225-dim vector
```

**3. `convert_video_to_npy(video_path, output_path, seq_len=64, ...)`**
```python
cap = cv2.VideoCapture(video_path)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Sample frame indices
indices = sample_frames(total_frames, seq_len)

seq = []
with mediapipe.Holistic() as holistic:
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        seq.append(keypoints)

# Output: (64, 225) numpy array
arr = np.stack(seq)
np.save(output_path, arr)
```

---

### 2.4 **Dataset Splitting: `src/preprocess/split_dataset.py`**

#### Purpose:
Chia dataset thành train/val/test với balanced class distribution.

#### Key Functions:

**1. `collect_nested(data_dir)`**
```python
# Input: data_dir/GLOSS1/*.mp4, data_dir/GLOSS2/*.mp4
# Output: [{"path": "...", "label": "GLOSS1"}, ...]
```

**2. `split_dataset(..., train_ratio=0.7, val_ratio=0.15, seed=42)`**
```python
# 1) Collect all videos
df = pd.DataFrame(rows)

# 2) Stratified split (maintain class distribution)
train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.15/(0.7+0.15), stratify=...)

# 3) Move actual files
move_files(train_df, "train", output_dir)
move_files(val_df, "val", output_dir)
move_files(test_df, "test", output_dir)

# Output structure:
# output_dir/
#   train/<gloss>/*.mp4 (or .npy)
#   val/<gloss>/*.mp4
#   test/<gloss>/*.mp4
```

---

### 2.5 **Keypoint Normalization: `src/preprocess/normalize_keypoints.py`**

#### Purpose:
Chuẩn hóa keypoints để model học được pose-invariant features.

#### Normalization Formula:
```
L̂_t = (L_t - L_ref) / ||L_max - L_min||
```

#### Implementation:
```python
def normalize_keypoints(seq, wrist_left_idx=15, wrist_right_idx=16):
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)  # (64, 75, 3)
    
    # 1. Reference point (midpoint of wrists)
    wrist_left = seq3d[:, 15, :2]   # Pose landmark 15
    wrist_right = seq3d[:, 16, :2]  # Pose landmark 16
    ref_point = (wrist_left + wrist_right) / 2
    
    # 2. Translation: Center at midpoint
    seq3d[:, :, 0] -= ref_point[:, 0]
    seq3d[:, :, 1] -= ref_point[:, 1]
    
    # 3. Scale: Normalize by bounding box diagonal
    min_coords = np.min(seq3d[:, :, :2], axis=1)
    max_coords = np.max(seq3d[:, :, :2], axis=1)
    scale = np.linalg.norm(max_coords - min_coords, axis=1)
    scale[scale == 0] = 1.0
    
    seq3d[:, :, :2] /= scale.reshape(-1, 1, 1)
    
    return seq3d.reshape(seq.shape[0], -1)
```

---

### 2.6 **Data Loader: `src/model/data_loader.py`**

#### Purpose:
PyTorch Dataset class để load và augment data cho training/inference.

#### 2.6.1 **SignLanguageDataset**

```python
class SignLanguageDataset(Dataset):
    def __init__(
        self,
        data_dir,           # Root dir: train/, val/, test/
        seq_len=30,
        source='npy',       # 'npy' hoặc 'video'
        split='train',
        normalize=True,     # Normalize keypoints?
        augment=False,      # Augmentation?
        label_map=None,     # Predefined label mapping
        scaler_path=None,   # Optional sklearn scaler
    ):
        # Scan directories
        split_dir = os.path.join(data_dir, split)
        self.samples = []  # [(file_path, label), ...]
        
        # Build label mapping
        if label_map is None:
            labels = sorted(os.listdir(split_dir))
            self.label_to_idx = {l: i for i, l in enumerate(labels)}
        else:
            self.label_to_idx = label_map
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # 1. Load keypoints
        if self.source == 'npy':
            arr = np.load(file_path)  # (seq_len, 225)
        else:
            arr = extract_keypoints_from_video(file_path, self.seq_len)
        
        # 2. Pad/Truncate to seq_len
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], 225))
            arr = np.vstack([arr, pad])
        else:
            arr = arr[:self.seq_len]
        
        # 3. Augment BEFORE normalize (important!)
        if self.augment:
            arr = augment_keypoints(arr)
        
        # 4. Normalize
        if self.normalize:
            arr = normalize_keypoints(arr)
        
        # 5. Apply scaler
        if self.scaler is not None:
            frames = [self.scaler.transform(f.reshape(1, -1)) for f in arr]
            arr = np.stack(frames)
        
        # 6. Convert to torch tensor
        label_idx = self.label_to_idx[label]
        return (
            torch.from_numpy(arr).float(),      # (64, 225)
            torch.tensor(label_idx)             # scalar
        )
```

#### 2.6.2 **Data Augmentation: `augment_keypoints(seq, config)`**

```python
def augment_keypoints(seq, config):
    seq3d = seq.reshape(seq.shape[0], -1, 3)
    
    # 1. Random Rotation
    angle = np.random.uniform(-15, 15) * π / 180
    R = [[cos(angle), -sin(angle)],
         [sin(angle),  cos(angle)]]
    seq3d[:, :, :2] = seq3d[:, :, :2] @ R.T
    
    # 2. Random Scaling
    scale = np.random.uniform(0.85, 1.15)
    seq3d[:, :, :2] *= scale
    
    # 3. Random Translation
    shift = np.random.uniform(-0.08, 0.08, 2)
    seq3d[:, :, 0] += shift[0]
    seq3d[:, :, 1] += shift[1]
    
    # 4. Horizontal Flip + Hand Swap
    if np.random.random() < 0.5:
        seq3d[:, :, 0] = -seq3d[:, :, 0]  # Flip x
        # Swap left hand (33:54) with right hand (54:75)
        seq3d[:, 33:54, :], seq3d[:, 54:75, :] = \
            seq3d[:, 54:75, :].copy(), seq3d[:, 33:54, :].copy()
    
    # 5. Time Masking
    if np.random.random() < 0.2:
        mask_len = np.random.randint(1, 4)
        start = np.random.randint(0, seq.shape[0] - mask_len)
        seq3d[start:start+mask_len, :, :] = 0
    
    return seq3d.reshape(seq.shape)
```

#### 2.6.3 **SignLanguageDatasetFromCSV**

Alternative dataset class để load từ CSV index:
```
path,label
/path/to/file.npy,người
/path/to/file.npy,tôi
...
```

---

### 2.7 **Model Architecture: `src/model/model.py`**

#### Purpose:
Define các mô hình LSTM/BiLSTM/GRU cho temporal sequence modeling.

#### Models:

**1. SimpleLSTM**
```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,      # 225 (keypoints/frame)
            hidden_dim,     # 128 (hidden state)
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):  # x: (batch, seq_len=64, 225)
        out, (hn, cn) = self.lstm(x)
        # out: (batch, 64, hidden_dim)
        return self.classifier(out[:, -1, :])  # Use last hidden state
```

**2. BiLSTM (Bidirectional)**
```python
class BiLSTM(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=128, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True  # ← Bidirectional!
        )
        # Output: (batch, 64, hidden_dim*2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.classifier(out[:, -1, :])
```

**3. SimpleGRU**
```python
class SimpleGRU(nn.Module):
    # Similar to SimpleLSTM but uses GRU (fewer parameters)
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=10):
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
```

---

### 2.8 **Training: `src/model/train.py`**

#### Purpose:
Main training loop với early stopping, learning rate scheduling, logging.

#### 2.8.1 **EarlyStopping**
```python
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
    
    def __call__(self, val_acc):
        if val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        else:
            self.best_score = val_acc
            self.counter = 0
        return False
```

#### 2.8.2 **Training Loop**

```python
def train(args):
    # 1. Load data
    train_ds = SignLanguageDataset(..., split='train', augment=True)
    val_ds = SignLanguageDataset(..., split='val', augment=False, label_map=train_ds.get_label_map())
    
    label_map = train_ds.get_label_map()
    
    # 2. Build model
    model = build_model(
        model_type='lstm',
        input_dim=225,
        hidden_dim=128,
        num_classes=len(label_map),
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # 3. Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        # Train step
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Val step
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                logits = model(X)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(y.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Save best + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, "best.pth")
        
        scheduler.step(val_acc)
        
        if early_stopping(val_acc):
            break
```

#### 2.8.3 **CLI Arguments**

```bash
python -m src.model.train \
    --data_dir data/wlasl/wlasl100 \
    --source npy \                   # Load from .npy files
    --seq_len 64 \
    --model_type lstm \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.3 \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 100 \
    --patience 20 \
    --ckpt_dir models/checkpoints
```

---

### 2.9 **Realtime Inference: `src/infer_realtime.py`**

#### Purpose:
Real-time sign language recognition từ webcam.

#### Key Features:

**1. Motion Detection FSM**
```python
state = "waiting"
segment = []
still_count = 0

while True:
    ret, frame = cap.read()
    
    # Detect motion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(prev_gray, gray)
    motion_level = motion.mean()
    
    # FSM
    if state == "waiting":
        if motion_level > MOTION_THRESHOLD:
            state = "recording"
            segment = []
    
    elif state == "recording":
        # Extract keypoints
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        vec = extract_keypoints(results)
        segment.append(vec)
        
        if motion_level < MOTION_THRESHOLD:
            still_count += 1
        else:
            still_count = 0
        
        # Sign complete
        if still_count >= STILL_FRAMES_REQUIRED:
            # Process segment
            arr = np.array(segment)  # (N, 225)
            
            # Pad to seq_len
            if len(arr) < seq_len:
                arr = np.pad(arr, ((0, seq_len-len(arr)), (0, 0)))
            else:
                arr = arr[:seq_len]
            
            # Predict
            X = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(X)
                pred = logits.argmax(dim=1)
                conf = logits.softmax(dim=1)[0, pred].item()
            
            label = label_list[pred]
            
            # Display on frame
            cv2.putText(frame, f"{label} ({conf:.2f})", ...)
            
            # Reset
            state = "waiting"
            segment = []
            still_count = 0
```

**2. Parameters**
- `--ckpt`: Path to best.pth
- `--label_map`: Path to label_map.json
- `--seq_len`: 64 (phải match với training)
- `--input_dim`: 225
- `--camera_id`: 0 (webcam index)

---

### 2.10 **Evaluation: `src/model/eval.py`**

```python
def evaluate(args):
    # Load model
    model = build_model(...).to(device)
    ck = load_checkpoint(args.ckpt, device=device)
    model.load_state_dict(ck['model_state'])
    
    # Load test data
    label_list = load_label_map(args.label_map)
    ds = SignLanguageDataset(..., label_map=label_list)
    loader = DataLoader(ds, batch_size=32)
    
    # Evaluate
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            ys.extend(y.numpy())
    
    # Print metrics
    print(classification_report(ys, preds, target_names=label_list))
    print(confusion_matrix(ys, preds))
```

---

### 2.11 **Utilities: `src/utils/utils.py`**

```python
def ensure_dir(path):
    """Create directory if not exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, path, extra=None):
    """Save model + optimizer + metadata"""
    payload = {
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'epoch': epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_checkpoint(path, device='cpu'):
    """Load checkpoint"""
    return torch.load(path, map_location=device)

def save_label_map(label_list, path):
    """Save ordered list of labels"""
    with open(path, 'w') as f:
        json.dump(label_list, f)

def load_label_map(path):
    """Load label map from JSON"""
    with open(path, 'r') as f:
        return json.load(f)
```

---

### 2.12 **Logger: `src/utils/logger.py`**

```python
class ProjectLogger:
    @classmethod
    def get_logger(
        cls,
        name,
        log_dir="logs",
        log_file=None,
        level=logging.INFO,
        console_output=True
    ):
        """Get logger with file + console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # File handler (detailed)
        file_handler = logging.FileHandler(...)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler (simplified)
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
```

---

## III. DATA FLOW EXAMPLE

### Scenario: Training trên 3 từ (người, tôi, Việt Nam)

```
Step 1: Preprocessing
├─ data/raw_unprocessed/
│  ├─ người/*.mp4 (10 videos)
│  ├─ tôi/*.mp4    (10 videos)
│  └─ Việt Nam/*.mp4 (10 videos)
└─ Output: data/raw/
   ├─ người/*.mp4 (motion-detected segments)
   ├─ tôi/*.mp4
   └─ Việt Nam/*.mp4

Step 2: Video to Keypoints
├─ Input: data/raw/ (30 videos)
└─ Output: data/npy/
   ├─ người/ (30 files, each 64×225)
   ├─ tôi/ (30 files)
   └─ Việt Nam/ (30 files)

Step 3: Split Dataset
├─ Input: data/npy/ (90 files total)
├─ Stratified split: 70/15/15
└─ Output: data/splits/
   ├─ train.csv (63 files, 70%)
   ├─ val.csv (13 files, 15%)
   └─ test.csv (14 files, 15%)

Step 4: Training
├─ Load train/val data
├─ Build LSTM model (input_dim=225, hidden_dim=128, num_classes=3)
├─ Optimize for 50 epochs
├─ Save best checkpoint (e.g., 75.56% accuracy)
└─ Output: models/checkpoints/best.pth + label_map.json

Step 5: Evaluation
├─ Load best model + test data
├─ Evaluate on 14 test samples
└─ Print: Accuracy, Precision, Recall, F1-score

Step 6: Realtime Inference
├─ Load model + label_map
├─ Capture from webcam
├─ Motion detection FSM
├─ Extract keypoints → Model → Prediction
└─ Display on screen
```

---

## IV. KEY DESIGN PATTERNS

### 1. **Lazy Initialization**
- Dataset labels được tạo từ folder structure
- Nếu cung cấp `label_map`, sử dụng predefined mapping (consistent across train/val/test)

### 2. **Dual-Mode Data Loading**
- `source='npy'`: Load từ .npy files (nhanh)
- `source='video'`: Extract from videos (on-the-fly)

### 3. **Augmentation BEFORE Normalization**
```python
# Correct order:
arr = augment_keypoints(arr)        # Random transform
arr = normalize_keypoints(arr)      # Center + scale
arr = scaler.transform(arr)         # Feature scaling
```

### 4. **FSM for Realtime Processing**
```
waiting → (motion detected) → recording → (still_frames_count) → waiting
```

### 5. **Checkpoint System**
```python
checkpoint = {
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': epoch,
    'val_acc': val_acc,
    'label_map': label_map,
    'args': args_dict
}
```

---

## V. CURRENT MODEL PERFORMANCE

Based on checkpoints:

| Checkpoint | Accuracy | Epoch |
|-----------|----------|-------|
| best_epoch46_acc0.7556.pth | **75.56%** | 46 |
| best_epoch44_acc0.6889.pth | 68.89% | 44 |
| best_epoch41_acc0.6667.pth | 66.67% | 41 |

**Best Model**: `best_epoch46_acc0.7556.pth` (75.56% accuracy on Vietnamese signs)

---

## VI. CONFIGURATION SUMMARY

| Parameter | Default | Notes |
|-----------|---------|-------|
| seq_len | 64 | Fixed sequence length |
| feature_dim | 225 | Pose(99) + Hands(126) |
| batch_size | 32 | Training batch size |
| learning_rate | 1e-3 | Initial LR |
| epochs | 100 | Max training epochs |
| hidden_dim | 256 | LSTM hidden state |
| num_layers | 2 | Stacked LSTM layers |
| dropout | 0.3 | Regularization |
| early_stopping_patience | 20 | Epochs to wait |
| label_smoothing | 0.1 | Loss regularization |

---

## VII. COMMON ISSUES & SOLUTIONS

| Issue | Cause | Solution |
|-------|-------|----------|
| Model predicts "người" poorly | Imbalanced training data | Collect more data or use class weights |
| NaN loss during training | Exploding gradients | Gradient clipping (max_norm=1.0) |
| Poor accuracy | Keypoints not normalized | Check normalization step |
| Realtime lag | Processing speed | Reduce seq_len or use simpler model |
| Video extraction fails | Corrupted video | Preprocess step handles this |

---

## VIII. EXAMPLE COMMANDS

### Preprocessing
```bash
python -m src.preprocess.preprocess_video \
    --input_dir data/raw_unprocessed \
    --output_dir data/raw \
    --fps 30 --width 1280 --height 720 \
    --skip_existing
```

### Video to Keypoints
```bash
python -m src.preprocess.video2npy \
    --input_dir data/raw \
    --output_dir data/npy \
    --seq_len 64 \
    --skip_existing
```

### Split Dataset
```bash
python -m src.preprocess.split_dataset \
    --data_dir data/npy \
    --output_dir data/splits \
    --train_ratio 0.7 --val_ratio 0.15
```

### Training
```bash
python -m src.model.train \
    --data_dir data/splits \
    --source npy \
    --seq_len 64 \
    --model_type lstm \
    --batch_size 32 \
    --lr 1e-3 \
    --epochs 100 \
    --ckpt_dir models/checkpoints
```

### Evaluation
```bash
python -m src.model.eval \
    --index_csv data/splits/test.csv \
    --ckpt models/checkpoints/best.pth \
    --label_map models/checkpoints/label_map.json
```

### Realtime Inference
```bash
python -m src.infer_realtime \
    --ckpt models/checkpoints/best.pth \
    --label_map models/checkpoints/label_map.json \
    --seq_len 64
```

---

## IX. DEPENDENCIES

```
torch>=1.12
torchvision
opencv-python>=4.5
mediapipe>=0.10.0
numpy
pandas
scikit-learn
joblib
tqdm
```

---

## X. PROJECT STRUCTURE

```
sign_language_video_to_text/
├── app.py                      # Streamlit UI (if available)
├── requirements.txt
├── README.md
│
├── src/
│   ├── config/
│   │   └── config.py           # Global configuration
│   │
│   ├── preprocess/
│   │   ├── preprocess_video.py        # Video normalization
│   │   ├── video2npy.py               # Keypoint extraction
│   │   ├── split_dataset.py           # Train/val/test split
│   │   └── normalize_keypoints.py     # Keypoint normalization
│   │
│   ├── model/
│   │   ├── model.py            # LSTM/BiLSTM/GRU architectures
│   │   ├── data_loader.py      # PyTorch Dataset classes
│   │   ├── train.py            # Training loop
│   │   └── eval.py             # Evaluation script
│   │
│   ├── infer_realtime.py       # Realtime webcam inference
│   │
│   └── utils/
│       ├── logger.py           # Logging utility
│       └── utils.py            # Helper functions
│
├── data/
│   ├── raw_unprocessed/        # Original videos (user input)
│   ├── raw/                    # Preprocessed videos
│   ├── npy/                    # Keypoint sequences
│   └── splits/                 # train.csv, val.csv, test.csv
│
├── models/
│   └── checkpoints/            # Saved models + label_map.json
│
└── logs/                       # Training logs
```

---

**End of Analysis**

