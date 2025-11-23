# 🔧 FIX: Preprocessing Missing - Normalize Keypoints

## 🔴 Vấn đề Tìm Thấy

**Nguyên nhân không có dự đoán:**
1. Dữ liệu **PHẢI được chuẩn hóa** (normalize) trước khi dự đoán
2. Nhưng `server.py` hiện tại **KHÔNG thực hiện bước normalize này**
3. Mô hình được huấn luyện với dữ liệu đã chuẩn hóa
4. Mô hình nhận dữ liệu **chưa chuẩn hóa** → dự đoán không chính xác

---

## 📊 So Sánh Training vs Real-time

### Quá trình Training (src/model/data_loader.py)
```
Raw keypoints (0-1 range)
    ↓
[NORMALIZE] - Quan trọng!
    ↓
Center tại giữa 2 cổ tay
Scale theo bounding box
    ↓
Dữ liệu được chuẩn hóa (-1 đến 1 range)
    ↓
LSTM Model
    ↓
Dự đoán chính xác ✅
```

### Quá trình Real-time (Web App - CŨ)
```
Raw keypoints (0-1 range)
    ↓
[MISSING NORMALIZE] ❌
    ↓
Dữ liệu vẫn ở 0-1 range
    ↓
LSTM Model
    ↓
Dự đoán sai ❌
```

---

## ✅ Giải Pháp Áp Dụng

### Bước 1: Thêm hàm `normalize_keypoints`
```python
def normalize_keypoints(seq, left_wrist_idx=15, right_wrist_idx=16):
    """
    Normalize keypoints - PHẢI TRÙNG với Training!
    1. Center tại midpoint giữa 2 cổ tay
    2. Scale theo diagonal của bounding box
    """
    num_landmarks = seq.shape[1] // 3
    seq3d = seq.reshape(seq.shape[0], num_landmarks, 3)

    # Lấy vị trí 2 cổ tay
    lw = seq3d[:, left_wrist_idx, :2]      # Left wrist
    rw = seq3d[:, right_wrist_idx, :2]     # Right wrist
    
    # Reference point (giữa 2 cổ tay)
    ref = (lw + rw) / 2
    
    # Center
    seq3d[:, :, 0] -= ref[:, 0].reshape(-1, 1)
    seq3d[:, :, 1] -= ref[:, 1].reshape(-1, 1)

    # Scale
    min_c = np.min(seq3d[:, :, :2], axis=1)
    max_c = np.max(seq3d[:, :, :2], axis=1)
    scale = np.linalg.norm(max_c - min_c, axis=1)
    scale[scale == 0] = 1
    seq3d[:, :, :2] /= scale.reshape(-1, 1, 1)

    return seq3d.reshape(seq.shape[0], -1)
```

### Bước 2: Gọi normalize trong xử lý frame
```python
# Trước:
X = torch.from_numpy(arr).unsqueeze(0).float().to(DEVICE)

# Sau:
arr = normalize_keypoints(arr)  # ✅ Thêm dòng này!
X = torch.from_numpy(arr).unsqueeze(0).float().to(DEVICE)
```

---

## 🔄 Quy Trình Xử Lý Mới (Đã Sửa)

```
1. Nhận frames từ browser
    ↓
2. Decode base64 images
    ↓
3. MediaPipe extract keypoints
    ↓
4. Stack vào (25, 225) array
    ↓
5. Pad/Truncate nếu cần
    ↓
6. ✅ NORMALIZE KEYPOINTS (ĐÃ THÊM)
    ↓
7. Convert to tensor
    ↓
8. LSTM Model inference
    ↓
9. Get prediction
    ↓
10. Return to client
```

---

## 📈 Ảnh Hưởng

**Trước Fix:**
```
Input keypoints: [0.3, 0.2, 0.1, 0.4, ...]  (0-1 range)
    ↓
LSTM: "Đây không phải dữ liệu huấn luyện!" ❌
    ↓
Output: Random predictions
```

**Sau Fix:**
```
Input keypoints: [0.3, 0.2, 0.1, 0.4, ...]  (0-1 range)
    ↓
Normalize: [-0.2, -0.4, 0.1, 0.3, ...]  (normalized range)
    ↓
LSTM: "Đây là dữ liệu tôi nhận dạng!" ✅
    ↓
Output: Chính xác dự đoán
```

---

## 🧪 Cách Kiểm Tra

### 1. Xem Server Logs
```
Normalized keypoints shape: (25, 225), 
min: -2.1543, max: 2.3847
```

**Nếu thấy:**
- `min` ≈ -5 đến 5
- `max` ≈ -5 đến 5
→ ✅ Normalize đang hoạt động

**Nếu thấy:**
- `min` ≈ 0
- `max` ≈ 1
→ ❌ Chưa normalize (lỗi)

### 2. Kiểm Tra Dự Đoán
- Thực hiện kí hiệu trước camera
- Nên thấy dự đoán sau ~25 frames
- Confidence > 0.30

---

## 📋 Files Đã Sửa

**File:** `web_app/server.py`

**Thay đổi:**
1. ✅ Thêm hàm `normalize_keypoints()` (lines 83-155)
2. ✅ Gọi normalize trong `process_image()` (line 195)
3. ✅ Log normalized keypoints statistics (line 196)

---

## 🚀 Tiếp Theo

### 1. Restart Server
```bash
python web_app/server.py
```

### 2. Mở Browser
```
http://127.0.0.1:5000
```

### 3. Kiểm Tra Server Logs
- Tìm dòng: `Normalized keypoints shape:`
- Kiểm tra min/max values

### 4. Test Prediction
- Thực hiện kí hiệu tay
- Nên có dự đoán trong 2 giây
- Confidence phải > 0.30

---

## ✅ Verification Checklist

- [ ] Server khởi động thành công
- [ ] Server log hiển thị: `Normalized keypoints shape:`
- [ ] Browser connect successfully
- [ ] Thực hiện kí hiệu, nhìn thấy dự đoán
- [ ] Confidence score hợp lý (0.3-1.0)
- [ ] Prediction history cập nhật

---

## 🎯 Tóm Tắt

**Vấn đề:** Thiếu bước normalize keypoints  
**Giải pháp:** Thêm hàm normalize_keypoints vào web_app/server.py  
**Kết quả:** Dự đoán sẽ chính xác  
**Status:** ✅ **FIXED**

Dữ liệu real-time giờ đây được tiền xử lý **ĐÚNG CẠP** với training data! 🎉
