# Sign Language Recognition (MediaPipe Holistic + LSTM)

**Mục tiêu:** xây dựng pipeline chuyển video -> landmark (.npy) -> huấn luyện LSTM nhận diện ký hiệu (face + hands) -> inference realtime.

## Cấu trúc project

```
sign-language-recognition/
├── data/
├── models/
│   └── checkpoints/    # saved model checkpoints + label map
├── src/                # scripts & modules
├── requirements.txt
└── README.md
```

## Quick start (example)

1. Cài dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Thu thập video

- Đặt video gốc vào `data/wlvsl/raw_unprocessed/<label>/`.
- Mỗi `<label>` là một class (ví dụ: `hello`, `thanks`, ...).
- Mỗi clip chỉ nên chứa 1 ký hiệu, thời lượng 2–5 giây.

3. Chuẩn hóa video
   Dùng script `preprocess_videos.py` để:

- Đưa video về **30fps, 1280×720 (16:9)**.
- Tự động phát hiện chuyển động và cắt video.
- **Pixel value normalization** về range [0,1] sử dụng min-max normalization.
- Tùy chọn `--skip_existing` để bỏ qua các video đã được cắt (đã tồn tại file output trong `data/raw`).

  ```bash
  python -m src.preprocess.preprocess_video --input_dir data/wlvsl/raw_unprocessed --output_dir data/wlvsl/raw
  ```

4. Split dataset into train/val/test:
   ```bash
   python -m src.preprocess.split_dataset --data_dir data/wlvsl/raw --output_dir data/wlvsl/splits
   ```

5. Convert tất cả video sang npy (pose+hands):
   ```bash
   python -m src.preprocess.video2npy --input_dir data/wlvsl/splits/train --output_dir data/wlvsl/npy/train --seq_len 30 --sampling_mode 2

   python -m src.preprocess.video2npy --input_dir data/wlvsl/splits/val --output_dir data/wlvsl/npy/val --seq_len 30 --sampling_mode 2

   python -m src.preprocess.video2npy --input_dir data/wlvsl/splits/test --output_dir data/wlvsl/npy/test --seq_len 30 --sampling_mode 2
   ```

6. Huấn luyện và đánh giá:
   ```bash
   python -m src.model.train --data_dir data/wlvsl/npy --source npy --seq_len 30 --ckpt_dir models/checkpoints/vsl_v1
   ```

7. Realtime inference:
   ```bash
   python -m src.infer_realtime --ckpt models/checkpoints/vsl_v1/best.pth --label_map models/checkpoints/vsl_v1/label_map.json --seq_len 30
   ```

8. **Web Application (Webapp):**

   Ứng dụng web real-time nhận diện ngôn ngữ kí hiệu thông qua webcam.

   **Tính năng:**
   - Real-time video capture từ webcam 
   - Phát hiện chuyển động (Motion Detection FSM)
   - Nhận diện ký hiệu tự động
   - Hiển thị kết quả với độ tin cậy (confidence)
   - Lịch sử dự đoán

   **Cách chạy:**

   a. Cài đặt dependencies cho webapp (nếu chưa có):
   ```bash
   pip install -r requirements.txt
   ```

   b. Khởi động Flask server:
   ```bash
   python .\src\webapp\server.py
   ```

   c. Mở browser và truy cập: `http://localhost:5000`

   d. Cho phép truy cập webcam khi popup yêu cầu.

   **Cấu hình Webapp:**

   File `webapp/config.py` chứa các tham số:
   - `MODEL_PATH`: Đường dẫn đến model checkpoint (mặc định: `models/checkpoints/vsl_v1/best.pth`)
   - `LABEL_MAP_PATH`: Đường dẫn đến label map (mặc định: `models/checkpoints/vsl_v1/label_map.json`)
   - `MIN_PREDICTION_CONFIDENCE`: Ngưỡng độ tin cậy tối thiểu (mặc định: 0.35)
   - `STILL_FRAMES_REQUIRED`: Số frame không chuyển động để kết thúc ghi nhận (mặc định: 8)

   **Cấu trúc Webapp:**
   ```
   webapp/
   ├── server.py              # Flask server chính
   ├── config.py              # Cấu hình
   ├── static/
   │   ├── app.js            # Logic capture & Socket.IO
   │   └── style.css         # CSS
   └── templates/
       └── index.html        # HTML giao diện
   ```

## Ghi chú kỹ thuật

- Feature vector hiện tại: **pose (33*3) + left hand (21*3) + right hand (21\*3) = 225** chiều.
- Tất cả file `.npy` được lưu dạng `(seq_len, 225)`.
- Coordinate normalization sử dụng wrist joints làm reference point theo công thức: L̂_t = (L_t - L_ref) / ||L_max - L_min||
- `train.py` lưu `label_map.json` (list labels theo thứ tự index) trong folder checkpoint để inference tải lại mapping.
