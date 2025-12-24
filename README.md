# MALLORN Astronomical Classification Challenge

Dự án phân loại các sự kiện thiên văn (TDE - Tidal Disruption Events) dựa trên dữ liệu đường cong ánh sáng (light curves).

## Phương pháp (Methodology)

Chúng tôi sử dụng phương pháp tiếp cận dựa trên **Feature Engineering** kết hợp với mô hình **LightGBM**.

### 1. Trích xuất đặc trưng (Feature Extraction)
Từ dữ liệu chuỗi thời gian (time-series) của các bộ lọc màu (filters), chúng tôi trích xuất các nhóm đặc trưng quan trọng:
- **Thống kê cơ bản:** Min, Max, Mean, Std, Skew, Kurtosis, Amplitude.
- **Biến thiên theo thời gian:** Thời gian đạt đỉnh (Time to peak), độ dốc (Slope), độ rộng xung (Duration).
- **Phân tích phổ (FFT):** Các hệ số Fourier để bắt thông tin tần số.
- **Khớp mẫu vật lý (Model Fitting):** Khớp đường cong ánh sáng với mô hình TDE lý thuyết ($f \propto t^{-5/3}$) để tính sai số (Chi-squared).
- **Đặc trưng màu sắc:** Tương quan và độ trễ giữa các dải sóng (u, g, r, i, z, y).

### 2. Mô hình hóa (Modeling)
- **Model:** LightGBM (Gradient Boosting Decision Tree).
- **Chiến lược huấn luyện:**
  - Sử dụng **Stratified K-Fold** (5 folds) để đảm bảo tính ổn định.
  - Xử lý mất cân bằng dữ liệu bằng tham số `scale_pos_weight`.
  - Tối ưu hóa ngưỡng phân loại (Threshold Tuning) dựa trên điểm F1 của tập OOF (Out-of-Fold).

## Cấu trúc thư mục

- `models/`: Chứa mã nguồn Python đã được làm sạch.
  - `feature_extraction.py`: Script trích xuất đặc trưng.
  - `train.py`: Script huấn luyện mô hình LightGBM và dự đoán.
- `notebooks/`:
  - `final-and-image-gen.ipynb`: Notebook tổng hợp, bao gồm cả quá trình sinh biểu đồ báo cáo.
- `submissions/`:
  - `submission.csv`: Kết quả dự đoán cuối cùng (F1 Score: ~0.62).
- `dataset/`: Chứa dữ liệu đầu vào (nếu có).
- `images/`: Hình ảnh và biểu đồ minh họa.

## Hướng dẫn sử dụng

1. Đảm bảo dữ liệu `train_full_lightcurves.csv` và `train_log.csv` nằm trong thư mục gốc hoặc `dataset/`.
2. Chạy `notebooks/final-and-image-gen.ipynb` để tái hiện toàn bộ kết quả và biểu đồ.
3. Hoặc chạy các script trong `models/` để thực hiện từng bước riêng biệt (Lưu ý cần điều chỉnh đường dẫn dữ liệu nếu chạy script rời).

---
*Dự án thực hiện bởi nhóm sinh viên UET-INT3405E.*
