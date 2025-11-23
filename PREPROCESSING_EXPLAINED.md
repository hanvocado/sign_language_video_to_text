# 📚 Preprocessing Deep Dive

## 🔍 Vấn Đề Chi Tiết

### Tại sao không có dự đoán?

```python
# Training (được huấn luyện trên dữ liệu này):
keypoints = normalize(extract(video_frame))
model.train(keypoints)

# Real-time (CŨNG CẬP NHẬT):
keypoints = extract(webcam_frame)  # ❌ THIẾU normalize!
prediction = model(keypoints)       # ❌ KHÔNG KHỚP!
```

---

## 🧮 Phép Toán Normalize

### Input
```
Pose landmarks:  33 landmarks × 3 (x,y,z) = 99 dims
Left hand:       21 landmarks × 3 (x,y,z) = 63 dims  
Right hand:      21 landmarks × 3 (x,y,z) = 63 dims
────────────────────────────────────────────────────
TOTAL:           75 landmarks × 3 = 225 dims
```

### Bước 1: Extract Raw Keypoints
```python
# MediaPipe output (normalized to 0-1 by MediaPipe itself)
keypoints = [
    0.5,  0.3,  0.2,   # Left eye
    0.55, 0.32, 0.21,  # Right eye
    0.45, 0.6,  0.15,  # Left wrist (idx 15)
    0.52, 0.61, 0.16,  # Right wrist (idx 16)
    ...
]
Shape: (225,)  Range: 0-1
```

### Bước 2: Reshape để Xử Lý
```python
# Reshape thành (75, 3) để dễ làm việc
seq3d = keypoints.reshape(75, 3)

seq3d = [
    [0.5,  0.3,  0.2 ],    # Landmark 0
    [0.55, 0.32, 0.21],    # Landmark 1
    ...
    [0.45, 0.6,  0.15],    # Landmark 15 (LEFT WRIST)
    [0.52, 0.61, 0.16],    # Landmark 16 (RIGHT WRIST)
    ...
]
```

### Bước 3: Tính Reference Point
```python
# Reference = center giữa 2 cổ tay
left_wrist  = seq3d[15, :2]   = [0.45, 0.6]
right_wrist = seq3d[16, :2]   = [0.52, 0.61]
ref = (left_wrist + right_wrist) / 2 = [0.485, 0.605]
```

### Bước 4: Center (Dịch)
```python
# Trừ reference từ tất cả keypoints
seq3d[:, 0] -= ref[0]   # Trừ x
seq3d[:, 1] -= ref[1]   # Trừ y

# Sau:
seq3d = [
    [0.5-0.485,   0.3-0.605,    0.2   ],   # [-0.015, -0.305, 0.2]
    [0.55-0.485,  0.32-0.605,   0.21  ],   # [0.065, -0.285, 0.21]
    ...
    [0.45-0.485,  0.6-0.605,    0.15  ],   # [-0.035, -0.005, 0.15]
    [0.52-0.485,  0.61-0.605,   0.16  ],   # [0.035, 0.005, 0.16]
    ...
]
# Giờ 2 cổ tay ở gần origin (0, 0) ✓
```

### Bước 5: Scale (Chuẩn Hóa Kích Thước)
```python
# Tính bounding box
min_x, min_y = -0.5, -0.7   # Điểm min
max_x, max_y = 0.4, 0.3    # Điểm max

# Diagonal = √[(max-min)² + (max-min)²]
diagonal = sqrt((0.4-(-0.5))² + (0.3-(-0.7))²)
         = sqrt(0.9² + 1.0²)
         = sqrt(0.81 + 1.0)
         = sqrt(1.81)
         = 1.345

# Chia tất cả keypoints cho diagonal
seq3d[:, 0] /= 1.345
seq3d[:, 1] /= 1.345

# Kết quả: tất cả keypoints trong [-1, 1] range ✓
```

### Output
```python
# Sau normalize:
normalized = [
    [-0.011,  -0.227,   0.2  ],    # X, Y trong [-1, 1]
    [0.048,   -0.212,   0.21 ],
    ...
    [-0.026,  -0.004,   0.15 ],    # 2 cổ tay ở gần (0, 0)
    [0.026,   0.004,    0.16 ],
    ...
]

Shape: (225,)  Range: -1 to 1  ✓ NORMALIZED
```

---

## 🎯 Tại Sao Cần Normalize?

### 1. Invariant to Scale (Bất Biến về Kích Thước)
```
Người nhỏ thực hiện gesture:
  Left wrist: (0.3, 0.5) → normalized: (-0.1, 0.1)

Người lớn thực hiện gesture giống hệt:
  Left wrist: (0.2, 0.3) → normalized: (-0.1, 0.1)  

✓ Cùng kết quả sau normalize!
```

### 2. Invariant to Position (Bất Biến về Vị Trí)
```
Gesture gần phải:
  Wrist: (0.7, 0.5) → normalized: (-0.2, 0.05)

Gesture gần trái:
  Wrist: (0.3, 0.5) → normalized: (-0.2, 0.05)

✓ Cùng kết quả sau normalize!
```

### 3. Fixed Range for Neural Network (Phạm Vi Cố Định cho NN)
```
Raw input: [0.2, 0.8, 0.1, 0.5, ...]    Range: 0-1
  ↓ Neural network phải học trên phạm vi này
  ❌ Model học được trên phạm vi 0-1

Normalized: [-0.5, 0.3, -0.8, 0.1, ...]  Range: -1 to 1
  ↓ Neural network được huấn luyện
  ✓ Model khớp phạm vi này
```

---

## 💡 Ví Dụ Thực Tế

### Scenario: Người thực hiện kí hiệu "người"

#### Training Phase
```python
# Frame từ video training
frame = Video.read(...)
keypoints = MediaPipe.extract(frame)
    # = [0.5, 0.3, ..., 0.45, 0.6, ..., 0.52, 0.61, ...]

normalized = normalize_keypoints(keypoints)
    # = [-0.01, -0.30, ..., -0.035, -0.005, ..., 0.035, 0.005, ...]

model.train(normalized, label="người")
```

#### Real-time Phase (TRƯỚC FIX)
```python
# Frame từ webcam
frame = Webcam.read(...)
keypoints = MediaPipe.extract(frame)
    # = [0.5, 0.3, ..., 0.45, 0.6, ..., 0.52, 0.61, ...]
    # (Giống với training!)

# ❌ NHƯNG KHÔNG NORMALIZE!
prediction = model(keypoints)
    # Model: "Đây không phải dữ liệu tôi nhận dạng!"
    # Output: Random, confidence thấp
```

#### Real-time Phase (SAU FIX) ✅
```python
# Frame từ webcam
frame = Webcam.read(...)
keypoints = MediaPipe.extract(frame)
    # = [0.5, 0.3, ..., 0.45, 0.6, ..., 0.52, 0.61, ...]

# ✅ NORMALIZE!
normalized = normalize_keypoints(keypoints)
    # = [-0.01, -0.30, ..., -0.035, -0.005, ..., 0.035, 0.005, ...]
    # (Giống training!)

prediction = model(normalized)
    # Model: "Đây là 'người'!"
    # Output: người, confidence: 0.92 ✓
```

---

## 📊 Visualization

### Before Normalize (❌ SAI)
```
Input Space:                Model Expected Space:
┌─────────────┐             ┌──────────────────┐
│ 0 ... 1     │             │ -2 ... 2         │
│ │           │             │ ││               │
│ │ • • • •   │             │ ││               │
│ └─────────────┘            │ └──────────────────┘
       ❌                           ❌
    MISMATCH!                  Input khác phạm vi
```

### After Normalize (✅ ĐÚNG)
```
Input Space:                Model Expected Space:
┌─────────────┐             ┌──────────────────┐
│ -1 ... 1    │             │ -2 ... 2         │
│ │           │             │ ││               │
│ │ • • • •   │             │ ││ • • • •       │
│ └─────────────┘            │ └──────────────────┘
       ✅                          ✅
    MATCH!                    Input khớp phạm vi
```

---

## 🔬 Code Flow

```python
# Raw keypoints
keypoints = [0.5, 0.3, 0.2, 0.55, 0.32, 0.21, ...]  # 225 dims

# Reshape
seq3d = keypoints.reshape(75, 3)

# Extract wrist positions
lw = seq3d[15, :2]  # [0.45, 0.6]
rw = seq3d[16, :2]  # [0.52, 0.61]

# Reference (center)
ref = (lw + rw) / 2  # [0.485, 0.605]

# Center
seq3d[:, :2] -= ref  # Trừ reference

# Calculate scale
bbox_diag = |max - min|
scale = sqrt(bbox_diag_x² + bbox_diag_y²)

# Scale
seq3d[:, :2] /= scale

# Reshape back
normalized = seq3d.reshape(225,)

# Result: [-0.01, -0.30, ..., 0.035, 0.005, ...]  Range: -1 to 1 ✓
```

---

## ✅ Verification

### Cách Kiểm Tra Normalize Đúng

```python
# Sau normalize:
arr = normalize_keypoints(keypoints_array)

print(f"Min: {arr.min()}")     # Should be ≈ -2 to -5
print(f"Max: {arr.max()}")     # Should be ≈ 2 to 5
print(f"Mean: {arr.mean()}")   # Should be ≈ 0
print(f"Std: {arr.std()}")     # Should be ≈ 0.5-1.0

# ✓ Nếu thấy: [-3.2, 2.8, 0.1, 0.7] → Normalize đúng!
# ❌ Nếu thấy: [0.2, 0.8, 0.3, 0.5] → Chưa normalize!
```

---

## 🎓 Kết Luận

| Concept | Ý Nghĩa |
|---------|---------|
| **Raw Keypoints** | Tọa độ trực tiếp từ MediaPipe (0-1 range) |
| **Normalize** | Transform sang fixed range (-1 to 1) |
| **Reference Point** | Center giữa 2 cổ tay |
| **Scaling Factor** | Diagonal của bounding box |
| **Invariance** | Bất biến về vị trí & kích thước |
| **Why Important** | Model được huấn luyện trên dữ liệu normalized |

**Sự thiếu normalize → Không khớp dữ liệu training → Dự đoán sai**

---

**Status:** ✅ Fixed in `web_app/server.py`  
**Impact:** Predictions now work correctly!
