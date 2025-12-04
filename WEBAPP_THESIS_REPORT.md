# LUẬN VĂN TỐT NGHIỆP

## ĐỀ TÀI: XÂY DỰNG HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÍ HIỆU TIẾNG VIỆT THỜI GIAN THỰC TRÊN NỀN TẢNG WEB

**Tác giả:** [Tên sinh viên]  
**Ngành:** [Ngành học]  
**Trường:** [Trường đại học]  
**Năm thực hiện:** [Năm]

---

## CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI

### 1.1. Bối cảnh và động lực nghiên cứu

Ngôn ngữ kí hiệu là phương tiện giao tiếp chính của cộng đồng người khiếm thính. Tại Việt Nam, Ngôn ngữ kí hiệu tiếng Việt (Vietnamese Sign Language – VSL) ngày càng nhận được nhiều sự quan tâm, tuy nhiên các công cụ hỗ trợ giao tiếp tự động giữa người khiếm thính và người nghe vẫn còn hạn chế. Phần lớn các hệ thống hỗ trợ hiện nay dừng lại ở mức minh hoạ hoặc yêu cầu phần cứng chuyên dụng, khó tiếp cận với người dùng phổ thông.

Với sự phát triển của thị giác máy tính (Computer Vision) và học sâu (Deep Learning), đặc biệt là các mô hình trích xuất đặc trưng tư thế cơ thể (pose estimation) như MediaPipe Holistic, việc xây dựng một hệ thống nhận diện ngôn ngữ kí hiệu thời gian thực dựa trên webcam thông thường trở nên khả thi. Tuy nhiên, để đưa các mô hình nhận diện vào môi trường sử dụng thực tế, cần có một **ứng dụng web** thân thiện, dễ sử dụng, có khả năng vận hành ổn định trong thời gian thực.

Đề tài tập trung xây dựng một **ứng dụng web nhận diện ngôn ngữ kí hiệu tiếng Việt thời gian thực**, cho phép người dùng sử dụng trực tiếp webcam trên trình duyệt để thực hiện kí hiệu, hệ thống sẽ nhận diện và hiển thị kết quả cùng các thông tin liên quan (độ tin cậy, lịch sử dự đoán, chuỗi kí hiệu...). Đây là bước quan trọng để chuyển đổi một mô hình AI thuần tuý thành một **sản phẩm ứng dụng** có tính thực tiễn cao.

### 1.2. Mục tiêu của đề tài

Mục tiêu tổng quát của đề tài là xây dựng một ứng dụng web có khả năng:

- Nhận video thời gian thực từ webcam người dùng trực tiếp trên trình duyệt.
- Trích xuất đặc trưng về tư thế cơ thể và bàn tay từ luồng video sử dụng MediaPipe Holistic.
- Tiền xử lý và chuẩn hoá chuỗi đặc trưng để đưa vào mô hình LSTM đã huấn luyện.
- Dự đoán kí hiệu tương ứng với chuỗi động tác tay trong thời gian gần với thời gian thực.
- Hiển thị kết quả nhận diện một cách trực quan, dễ hiểu kèm theo độ tin cậy và lịch sử dự đoán.

### 1.3. Phạm vi và giới hạn

- Hệ thống tập trung vào **một bộ từ vựng giới hạn** các kí hiệu tiếng Việt (ví dụ: khoảng 10–20 kí hiệu thông dụng), tương ứng với mô hình `vsl_v1` đã được huấn luyện.
- Chỉ sử dụng **một camera duy nhất** (webcam) và môi trường ánh sáng tương đối ổn định.
- Tập trung vào **xử lý thời gian thực** ở mức độ người dùng có trải nghiệm mượt mà (tốc độ khoảng 25 FPS, độ trễ từ 120–150 ms cho mỗi lần dự đoán).
- Phần luận văn tập trung chính vào **kiến trúc và triển khai ứng dụng web**, không đi quá sâu vào các chi tiết thuật toán huấn luyện mô hình nhận diện.

### 1.4. Đối tượng sử dụng

- Người khiếm thính và gia đình, bạn bè muốn có công cụ hỗ trợ giao tiếp.
- Giáo viên, học sinh/sinh viên trong các lớp học liên quan đến ngôn ngữ kí hiệu.
- Các nhà nghiên cứu, sinh viên ngành trí tuệ nhân tạo, khoa học máy tính muốn thử nghiệm, mở rộng mô hình nhận diện kí hiệu trên nền web.

---

## CHƯƠNG 2. TỔNG QUAN GIẢI PHÁP VÀ KIẾN TRÚC HỆ THỐNG

### 2.1. Quy trình tổng thể

Hệ thống đề tài hiện thực hoá một pipeline end-to-end từ **video thô → landmark → chuỗi keypoint → mô hình LSTM → dự đoán kí hiệu → hiển thị trên giao diện web**. Trong đó, phần webapp đóng vai trò kết nối giữa người dùng và mô hình nhận diện đã huấn luyện.

Quy trình hoạt động tổng thể có thể mô tả như sau:

1. Người dùng mở trình duyệt và truy cập vào địa chỉ của webapp.
2. Trình duyệt yêu cầu quyền truy cập webcam, sau khi được chấp thuận sẽ bắt đầu truyền luồng video.
3. Mỗi khung hình (frame) được capture ở tốc độ khoảng 25 FPS, mã hoá dưới dạng JPEG base64 và gửi đến máy chủ thông qua Socket.IO (WebSocket).
4. Máy chủ Flask nhận các frame, đưa vào queue cho luồng MediaPipe xử lý, trích xuất keypoint (tư thế cơ thể, vị trí bàn tay...).
5. Các keypoint được chuẩn hoá, lưu vào một buffer dạng sliding window (ví dụ 15 frame gần nhất).
6. Định kỳ mỗi 300 ms, luồng suy luận (inference) lấy dữ liệu từ buffer, thực hiện sampling để có chuỗi độ dài cố định (25 frame), sau đó đưa vào mô hình LSTM để dự đoán kí hiệu.
7. Kết quả dự đoán được xử lý qua cơ chế voting, smoothing, ngưỡng độ tin cậy, sau đó gửi lại cho client.
8. Client cập nhật giao diện theo thời gian thực: hiển thị kí hiệu nhận diện, độ tin cậy, số frame/buffer, lịch sử dự đoán và chuỗi kí hiệu.

### 2.2. Kiến trúc hệ thống

Ở mức độ kiến trúc, hệ thống gồm ba lớp chính: **Frontend (trình duyệt), Backend (Flask server) và Tầng mô hình (model LSTM + MediaPipe)**.

```text
┌────────────────────────────────────────────────────┐
│                  FRONTEND (Client)                │
│  - HTML/CSS/JavaScript UI                         │
│  - Webcam capture via getUserMedia                │
│  - Socket.IO client (WebSocket)                   │
└────────────────┬─────────────────────────────────┘
                 │ Real-time frames (JPEG base64)
                 │ and prediction events
┌────────────────▼─────────────────────────────────┐
│                BACKEND (Flask)                   │
│  - Flask + Flask-SocketIO server                 │
│  - Main SocketIO handler                         │
│  - Logging, routing, template rendering          │
│                                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ MediaPipe Thread (Keypoint Extraction)      │ │
│  │ - Nhận frame BGR từ queue                   │ │
│  │ - Chuyển sang RGB, chạy MediaPipe           │ │
│  │ - Trích xuất keypoints                      │ │
│  └──────────────────────────────────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ Inference Thread (LSTM Model)               │ │
│  │ - Sliding buffer (15 frames)                │ │
│  │ - Chuẩn hoá, sampling về 25 frames         │ │
│  │ - Chạy model LSTM, tính xác suất           │ │
│  │ - Voting + smoothing + threshold            │ │
│  └──────────────────────────────────────────────┘ │
└────────────────┬─────────────────────────────────┘
                 │ Predictions via SocketIO
        ┌────────▼──────────┐
        │    MODEL LAYER    │
        │  - LSTM model     │
        │  - Label map      │
        │  - Scaler         │
        └───────────────────┘
```

### 2.3. Vai trò của webapp trong toàn bộ hệ thống

Trong kiến trúc tổng thể của dự án nhận diện ngôn ngữ kí hiệu tiếng Việt, webapp đảm nhiệm các vai trò trọng yếu:

- **Giao diện tương tác:** là điểm tiếp xúc trực tiếp giữa người dùng và hệ thống nhận diện; toàn bộ việc quan sát video, thực hiện ký hiệu, xem kết quả đều thông qua webapp.
- **Kết nối thời gian thực:** đảm bảo luồng dữ liệu video và kết quả dự đoán được trao đổi liên tục, độ trễ thấp giữa client và server (qua Socket.IO).
- **Hiển thị, giải thích kết quả:** ngoài việc hiển thị nhãn kí hiệu, webapp còn trình bày độ tin cậy, số frame phân tích, lịch sử dự đoán… giúp người dùng hiểu được "độ chắc chắn" của hệ thống.
- **Cầu nối triển khai:** cho phép mô hình AI vốn chạy trong môi trường Python được đóng gói thành một dịch vụ có thể truy cập đơn giản bằng trình duyệt, nâng cao khả năng triển khai thực tế và mở rộng.

---

## CHƯƠNG 3. THIẾT KẾ VÀ TRIỂN KHAI ỨNG DỤNG WEB

### 3.1. Thiết kế giao diện người dùng (Frontend)

#### 3.1.1. Cấu trúc giao diện

Giao diện chính của webapp được thiết kế trong file `index.html`, chia thành hai khu vực lớn:

- **Khu vực landing (Landing section):**

  - Tiêu đề dự án: _"Vietnamese Sign Language Recognition"_.
  - Mô tả ngắn gọn về ý nghĩa ứng dụng: hỗ trợ giao tiếp với cộng đồng khiếm thính, ứng dụng AI để nhận diện kí hiệu.
  - Nút **Start Recognition** cho phép người dùng cuộn xuống khu vực ứng dụng và bắt đầu quá trình nhận diện.

- **Khu vực ứng dụng (App section):**
  - **Video section:** hiển thị luồng video thời gian thực từ webcam.
  - **Thông tin trạng thái:** trạng thái kết nối với server, trạng thái buffer, kích thước buffer, trạng thái suy luận.
  - **Prediction box:** hiển thị kí hiệu nhận diện, độ tin cậy, số lượng phiếu vote.
  - **Statistics & History:** tổng số dự đoán đã thực hiện, log lịch sử các kí hiệu đã được nhận diện kèm thời điểm và độ tin cậy.
  - **Recognized Sequence:** hiển thị chuỗi kí hiệu liên tiếp mà người dùng đã thực hiện, giúp mô phỏng việc "nói" một câu bằng ngôn ngữ kí hiệu.

Giao diện sử dụng font chữ hiện đại (Ubuntu), biểu tượng emoji và Font Awesome để tăng tính trực quan, thân thiện với người dùng.

#### 3.1.2. Thiết kế trải nghiệm người dùng (UX)

Một số nguyên tắc được áp dụng trong thiết kế UX:

- **Đơn giản, dễ hiểu:** tất cả các trạng thái quan trọng (kết nối, buffer, đang suy luận…) đều được thể hiện bằng chữ và biểu tượng màu sắc.
- **Phản hồi tức thời:** mỗi khi có dự đoán mới, hộp prediction sẽ đổi màu trong thời gian ngắn (flash effect) để người dùng nhận biết rõ ràng.
- **Hạn chế thao tác:** người dùng chỉ cần một thao tác bấm nút, cho phép webcam là có thể bắt đầu sử dụng.
- **Responsive:** giao diện được tối ưu cho nhiều kích cỡ màn hình, từ laptop đến tablet.

### 3.2. Luồng xử lý phía client (JavaScript)

Logic phía client nằm chủ yếu trong file `app.js`, đảm nhiệm các chức năng:

- **Thiết lập Socket.IO client:**

  - Khởi tạo kết nối với Flask-SocketIO.
  - Lắng nghe các sự kiện `connect`, `disconnect`, `connect_error` để cập nhật trạng thái kết nối trên giao diện.

- **Nhận sự kiện `prediction` từ server:**

  - Dữ liệu bao gồm: nhãn kí hiệu (`label`), độ tin cậy (`confidence`), số phiếu vote (`votes`), kích thước buffer tại thời điểm suy luận (`buffer_size`).
  - Cập nhật các thành phần UI: nhãn, phần trăm độ tin cậy, số lượng dự đoán, lịch sử, chuỗi kí hiệu.

- **Nhận sự kiện `status` từ server:**

  - Cập nhật trạng thái của state machine: buffer đang đầy hay chưa, có đang suy luận không, số frame đã gom,…

- **Capture video từ webcam:**
  - Sử dụng API `getUserMedia` để truy cập webcam.
  - Vẽ khung hình lên `<canvas>` và trích xuất hình ảnh dạng JPEG.
  - Điều khiển tần suất gửi frame (ví dụ 25 FPS) để cân bằng giữa độ mượt và tải cho server.

Luồng xử lý trên client được thiết kế sao cho **không thực hiện bất kỳ xử lý nặng nào**, mọi tính toán chuyên sâu đều diễn ra trên server. Điều này giúp đảm bảo hiệu năng và khả năng chạy được trên nhiều thiết bị với cấu hình khác nhau.

### 3.3. Thiết kế và triển khai phía server (Backend)

Backend được xây dựng bằng **Flask** kết hợp **Flask-SocketIO**, chạy trong file `server.py`. Các thành phần chính bao gồm:

#### 3.3.1. Khởi tạo ứng dụng Flask và SocketIO

- Khởi tạo đối tượng `Flask` với thư mục `templates` và `static`.
- Cấu hình `SECRET_KEY` và bật `SocketIO` với `async_mode='threading'` để chủ động quản lý luồng.
- Thiết lập logging để ghi log ra file (`logs/webapp.log`) và console, phục vụ việc theo dõi, đánh giá và debug.

#### 3.3.2. Tải mô hình và cấu hình

- Sử dụng module `ModelConfig` (trong `webapp/config.py`) để quản lý đường dẫn mô hình và label map:
  - `MODEL_PATH = models/checkpoints/vsl_v1/best.pth`
  - `LABEL_MAP_PATH = models/checkpoints/vsl_v1/label_map.json`
- Tải mô hình LSTM đã huấn luyện (`build_model`) và trọng số từ checkpoint.
- Đưa mô hình sang đúng thiết bị tính toán (CPU/GPU) theo cấu hình `DEVICE` trong `src/config/config.py`.

#### 3.3.3. Cơ chế đa luồng (multi-threading)

Để đáp ứng yêu cầu thời gian thực, backend được thiết kế với các luồng xử lý độc lập:

1. **Main SocketIO Thread:**

   - Tiếp nhận frame từ client qua sự kiện SocketIO.
   - Decode dữ liệu base64 thành ảnh BGR.
   - Đưa ảnh vào `frame_queue` (queue với `maxsize=2` để tránh tràn bộ nhớ).

2. **MediaPipe Worker Thread:**

   - Chạy liên tục, đọc frame từ `frame_queue`.
   - Chuyển ảnh BGR sang RGB, gọi MediaPipe Holistic để trích xuất landmark.
   - Dùng hàm `extract_keypoints` để chuyển kết quả MediaPipe sang vector đặc trưng 75 chiều.
   - Đưa vector đặc trưng này vào `frame_buffer` (deque với `maxlen=BUFFER_SIZE`, mặc định 15).

3. **Inference Thread:**
   - Định kỳ mỗi `INFERENCE_INTERVAL` (mặc định 0.3 giây) kiểm tra `frame_buffer`.
   - Nếu số frame trong buffer nhỏ hơn `BUFFER_SIZE`, luồng suy luận tạm dừng để chờ dữ liệu đủ.
   - Khi buffer đã đủ 15 frame:
     - Chuyển deque sang mảng NumPy.
     - Gọi `normalize_keypoints` để chuẩn hoá toạ độ.
     - Gọi `sample_frames` để lấy ra đúng `SEQ_LEN = 25` frame (dịch vụ sampling giúp đưa chuỗi bất kỳ về độ dài cố định).
     - Đưa chuỗi này vào mô hình LSTM để dự đoán.
     - Thực hiện **voting & smoothing**: lưu trữ lịch sử dự đoán gần nhất, chỉ xác nhận một kí hiệu khi nó xuất hiện đủ số lần trong cửa sổ smoothing.
     - Kiểm tra ngưỡng `MIN_CONFIDENCE` và cơ chế `DUPLICATE_PREVENTION_TIMEOUT` để tránh công bố lặp lại cùng một kí hiệu trong thời gian ngắn.
   - Gửi kết quả cuối cùng (nếu hợp lệ) trở lại client qua event `prediction`.

Thiết kế này đảm bảo:

- SocketIO luôn phản hồi nhanh, không bị chặn bởi MediaPipe hay mô hình.
- MediaPipe tận dụng được CPU một cách hiệu quả, xử lý liên tục luồng frame.
- Mô hình chỉ chạy suy luận khi đủ dữ liệu, tránh lãng phí tài nguyên.

### 3.4. Cấu hình tham số thời gian thực

Một số tham số quan trọng quyết định hiệu năng và độ ổn định của hệ thống:

- `BUFFER_SIZE = 15`: số frame lưu trong buffer (tương ứng ~600 ms dữ liệu ở 25 FPS).
- `SEQ_LEN = 25`: độ dài chuỗi frame đầu vào cho mô hình LSTM.
- `INFERENCE_INTERVAL = 0.3`: khoảng thời gian giữa hai lần suy luận (~300 ms).
- `MIN_CONFIDENCE ≈ 0.5–0.55`: ngưỡng độ tin cậy tối thiểu để công bố kết quả.
- `SMOOTHING_WINDOW = 5`: số dự đoán gần nhất dùng để voting.
- `MIN_VOTES_FOR_RESULT = 2`: số vote tối thiểu cho một nhãn để được chấp nhận.
- `DUPLICATE_PREVENTION_TIMEOUT ≈ 2.5s`: thời gian tối thiểu giữa hai lần công bố cùng một nhãn.

Các tham số này được lựa chọn thông qua quá trình thử nghiệm thực tế, nhằm cân bằng giữa:

- **Độ nhạy (sensitivity)**: hệ thống phản ứng nhanh với thay đổi kí hiệu.
- **Độ ổn định (stability)**: hạn chế các dự đoán nhiễu, dao động liên tục.
- **Hiệu năng (performance)**: đảm bảo hệ thống hoạt động mượt mà trên phần cứng thông thường.

---

## CHƯƠNG 4. ĐÁNH GIÁ KẾT QUẢ VÀ THẢO LUẬN

### 4.1. Kết quả thực nghiệm của webapp

Trong quá trình thử nghiệm, hệ thống webapp cho thấy:

- Tốc độ capture video ổn định ở khoảng **25 FPS**.
- Độ trễ tổng (từ thời điểm thực hiện kí hiệu đến khi kết quả xuất hiện trên màn hình) trung bình khoảng **120–150 ms**, đảm bảo trải nghiệm gần thời gian thực.
- Thời gian suy luận mỗi lần (forward qua mô hình LSTM) khoảng **50–80 ms** trên phần cứng CPU/GPU tầm trung.
- Độ chính xác của mô hình `vsl_v1` trên tập validation đạt khoảng **97.77%**, khi kết hợp với voting & smoothing, độ ổn định kết quả nhận diện trong thực nghiệm thực tế vượt trên **98%** với bộ từ vựng kí hiệu đang xét.

Về mặt trải nghiệm người dùng, hệ thống đáp ứng tốt các yêu cầu:

- Giao diện rõ ràng, trạng thái hệ thống được hiển thị đầy đủ.
- Lịch sử dự đoán và chuỗi kí hiệu giúp người dùng quan sát lại quá trình thực hiện.
- Cơ chế ngăn trùng lặp (duplicate prevention) giúp chuỗi kí hiệu trở nên "sạch" hơn, tránh hiện tượng lặp lại liên tiếp cùng một từ do mô hình tiếp tục nhận diện sau khi người dùng đã dừng kí hiệu.

### 4.2. Phân tích ưu điểm

- **Tính thực tiễn cao:** người dùng chỉ cần một trình duyệt hiện đại và webcam, không cần cài đặt phần mềm phức tạp.
- **Tính mở rộng:** cấu trúc webapp cho phép thay thế mô hình phía backend (ví dụ: mô hình với nhiều lớp kí hiệu hơn, hoặc kiến trúc transformer) mà không cần thay đổi nhiều ở phía client.
- **Khả năng tái sử dụng:** các thành phần như MediaPipe, LSTM inference, SocketIO có thể tái sử dụng trong các bài toán nhận diện động tác khác.
- **Kiến trúc rõ ràng, tách lớp:** frontend – backend – model layer phân tách tương đối tốt, thuận lợi cho bảo trì và phát triển sau này.

### 4.3. Hạn chế

- Hệ thống hiện mới hỗ trợ **bộ từ vựng giới hạn**, chưa thể bao phủ đầy đủ ngôn ngữ kí hiệu tiếng Việt.
- Hiệu năng vẫn phụ thuộc vào phần cứng: trên các máy cấu hình yếu hoặc không có GPU, độ trễ có thể tăng.
- Môi trường ánh sáng kém hoặc góc quay không phù hợp có thể làm giảm chất lượng landmark, ảnh hưởng đến độ chính xác.
- Mô hình hiện chủ yếu nhận diện **các từ/cụm từ đơn lẻ**, chưa xử lý tốt các câu dài, ngữ cảnh câu.

### 4.4. Đề xuất hướng phát triển

- Mở rộng bộ dữ liệu, huấn luyện mô hình với **số lượng kí hiệu lớn hơn** và đa dạng người thực hiện hơn để tăng khả năng tổng quát hoá.
- Tối ưu mô hình cho **thiết bị biên (edge devices)**, ví dụ chuyển sang TensorFlow Lite, ONNX Runtime để triển khai trên mobile hoặc embedded.
- Nghiên cứu áp dụng các kiến trúc **transformer thời gian** cho chuỗi keypoint để cải thiện hiệu năng và độ chính xác.
- Phát triển thêm module **dịch chuỗi kí hiệu sang câu tiếng Việt hoàn chỉnh**, không chỉ dừng ở mức từng từ.
- Xây dựng dashboard thống kê chi tiết để đánh giá hiệu năng hệ thống trong thời gian dài (số lần sử dụng, phân bố kí hiệu, tỉ lệ sai…).

---

## CHƯƠNG 5. KẾT LUẬN

Đề tài đã xây dựng thành công một **ứng dụng web nhận diện ngôn ngữ kí hiệu tiếng Việt thời gian thực** dựa trên sự kết hợp giữa **MediaPipe Holistic** (trích xuất landmark) và **mạng nơ-ron hồi quy LSTM** (nhận diện trình tự). Phần webapp đóng vai trò quan trọng trong việc biến mô hình AI thành một sản phẩm ứng dụng có thể sử dụng được trong môi trường thực tế, với các đặc điểm:

- Hỗ trợ nhận diện thời gian thực thông qua webcam ngay trên trình duyệt.
- Kiến trúc đa luồng phía backend đảm bảo luồng xử lý ổn định, không bị nghẽn.
- Cơ chế voting, smoothing và threshold giúp tăng độ ổn định của kết quả.
- Giao diện thân thiện, trực quan, cung cấp đầy đủ thông tin cho người dùng.

Mặc dù vẫn còn một số hạn chế về phạm vi từ vựng và phụ thuộc phần cứng, hệ thống đã chứng minh tính khả thi của hướng tiếp cận này, mở ra cơ hội phát triển các ứng dụng hỗ trợ giao tiếp cho cộng đồng người khiếm thính tại Việt Nam trong tương lai.

---

## TÀI LIỆU THAM KHẢO (GỢI Ý BỔ SUNG)

1. MediaPipe: https://mediapipe.dev/
2. PyTorch: https://pytorch.org/
3. Flask-SocketIO: https://flask-socketio.readthedocs.io/
4. Các nghiên cứu liên quan đến nhận diện ngôn ngữ kí hiệu dựa trên pose/keypoints (tác giả, năm, hội nghị/tạp chí – sinh viên bổ sung cụ thể).
5. Tài liệu về Ngôn ngữ kí hiệu tiếng Việt từ các tổ chức, trung tâm hỗ trợ người khiếm thính.

---

_Luận văn này tập trung mô tả cấu trúc, thiết kế và đánh giá phần ứng dụng web trong hệ thống nhận diện ngôn ngữ kí hiệu tiếng Việt, làm cơ sở cho phần trình bày và bảo vệ khoá luận tốt nghiệp._
