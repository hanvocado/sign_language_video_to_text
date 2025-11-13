# Sign Language Recognition (MediaPipe Holistic + LSTM)

**Mục tiêu:** xây dựng pipeline chuyển video -> landmark (.npy) -> huấn luyện LSTM nhận diện ký hiệu (face + hands) -> inference realtime.

## Cấu trúc project

```
sign-language-recognition/
├── data/
│   ├── raw/            # video gốc: data/raw/<label>/*.mp4
│   ├── npy/            # output .npy sequences after conversion: data/npy/<label>/*.npy
│   └── splits/         # train.csv / val.csv / test.csv (indexes)
├── models/
│   └── checkpoints/    # saved model checkpoints + label map + scaler
├── notebooks/          # optional EDA notebook
├── src/                # scripts & modules (below files)
├── requirements.txt
└── README.md
```

## Các script chính (với mô tả nhanh)

- `src/preprocess/video2npy.py`: chuyển video (RGB) thành `.npy` sequences; mỗi frame -> vector keypoints (pose + left hand + right hand). Mặc định mỗi .npy có `seq_len` frames (pad/truncate). Có thể dùng `--skip_existing` để bỏ qua các file `.npy` đã sinh trước đó.
- `src/preprocess/split_dataset.py`: duyệt `data/npy/*/*.npy` và sinh `train.csv`, `val.csv`, `test.csv` theo tỷ lệ (mặc định 70/15/15).
- `src/preprocess/preprocessing.py`: gộp index, tạo scaler (StandardScaler) cho features (tùy chọn).
- `src/model/data_loader.py`: `SignLanguageDataset` là PyTorch Dataset đọc `.npy` theo index csv.
- `src/model/model.py`: kiến trúc `LSTMClassifier` (stacked LSTM -> linear).
- `src/model/train.py`: huấn luyện mô hình, lưu checkpoint tốt nhất, lưu `label_map.json` (list labels in order) và (nếu có) scaler.joblib.
- `src/model/eval.py`: đánh giá trên `test.csv` và in classification report + confusion matrix.
- `src/infer_realtime.py`: chạy webcam realtime, trích xuất keypoints, giữ sliding window `seq_len` và dự đoán nhãn, hiển thị trên màn hình.
- `src/utils/visualize_keypoints.py`: vẽ face+hands landmarks lên video (dùng để kiểm tra extraction).

## Quick start (example)

1. Cài dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Thu thập video

- Đặt video gốc vào `data/raw_unprocessed/<label>/`.
- Mỗi `<label>` là một class (ví dụ: `hello`, `thanks`, ...).
- Mỗi clip chỉ nên chứa 1 ký hiệu, thời lượng 2–5 giây.

3. Chuẩn hóa video
   Dùng script `preprocess_videos.py` để:

- Đưa video về **30fps, 1280×720 (16:9)**.
- Tự động phát hiện chuyển động và cắt video.
- **Pixel value normalization** về range [0,1] sử dụng min-max normalization.
- Tùy chọn `--skip_existing` để bỏ qua các video đã được cắt (đã tồn tại file output trong `data/raw`).

  ```bash
  python -m src.preprocess.preprocess_video --input_dir data/raw_unprocessed --output_dir data/raw --motion_threshold 90 --skip_existing
  ```

4. Convert tất cả video sang npy (pose+hands, seq_len=64 mặc định):
   ```bash
   python -m src.preprocess.video2npy --input_dir data/raw --output_dir data/npy --seq_len 64 --skip_existing
   ```
5. Sinh split index:
   ```bash
   python -m src.preprocess.split_dataset --data_dir data/npy --output_dir data/splits --train_ratio 0.7 --val_ratio 0.15
   ```
6. (Tùy chọn) tạo scaler (fit StandardScaler trên các frame trung bình):
   ```bash
   python -m src.preprocess.preprocessing --index_csv data/splits/train.csv --scaler_path models/checkpoints/scaler.joblib
   ```
7. Huấn luyện:
   ```bash
   python -m src.model.train --train_csv data/splits/train.csv --val_csv data/splits/val.csv --epochs 50
   ```
8. Đánh giá:
   ```bash
   python -m src.model.eval --index_csv data/splits/test.csv --ckpt models/checkpoints/best.pth --label_map models/checkpoints/label_map.json
   ```
9. Realtime inference:
   ```bash
   python -m src.infer_realtime --ckpt models/checkpoints/best.pth --label_map models/checkpoints/label_map.json
   ```

## Ghi chú kỹ thuật

- Feature vector hiện tại: **pose (33*3) + left hand (21*3) + right hand (21\*3) = 225** chiều.
- Tất cả file `.npy` được lưu dạng `(seq_len, 225)`.
- Coordinate normalization sử dụng wrist joints làm reference point theo công thức: L̂_t = (L_t - L_ref) / ||L_max - L_min||
- `train.py` lưu `label_map.json` (list labels theo thứ tự index) trong folder checkpoint để inference tải lại mapping.
