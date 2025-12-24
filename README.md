# UET-INT3405E-FinalProject-MALLORN
MALLORN Astronomical Classification Challenge - Final Project

## Cấu trúc thư mục

- **models/**: Chứa mã nguồn các mô hình chính (LightGBM ~0.6033/0.62) và các phiên bản chọn lọc.
  - `Feature_Extract_*.py`: Các script trích xuất đặc trưng (trước đây là file1).
  - `Train_model*.py`: Các script huấn luyện và dự đoán (trước đây là file2).
  
- **notebooks/**: Chứa notebook tổng hợp cuối cùng.
  - `final-and-image-gen.ipynb`: Notebook chạy toàn bộ quy trình từ trích xuất đặc trưng, huấn luyện model đến sinh các biểu đồ báo cáo.

- **submissions/**: Chứa các file kết quả nộp bài quan trọng nhất (`submission.csv`, `submission_Final.csv`).

- **dataset/**: Chứa dữ liệu log của cuộc thi (`train_log.csv`, `test_log.csv`).

- **images/**: Các biểu đồ và hình ảnh minh họa cho báo cáo.

- **old_models/**: Thư mục chứa các mô hình cũ, mã nguồn thực nghiệm, và các script phụ trợ khác không còn được sử dụng trong phiên bản cuối cùng.

- **old_submissions/**: Chứa các file kết quả nộp bài từ các phiên bản mô hình cũ.

## Hướng dẫn chạy
1. Đảm bảo dữ liệu `train_full_lightcurves.csv` và `test_full_lightcurves.csv` (nếu có) được đặt đúng đường dẫn trong code hoặc `dataset/`.
2. Chạy `notebooks/final-and-image-gen.ipynb` để tái hiện kết quả.

---
*Dự án thực hiện bởi nhóm sinh viên UET-INT3405E.*
