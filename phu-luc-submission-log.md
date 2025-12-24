# PHỤ LỤC: NHẬT KÝ CÁC THỬ NGHIỆM ĐÃ NỘP (SUBMISSION LOG)

Dưới đây là bảng tổng hợp các kết quả nộp bài (submission) trên hệ thống Leaderboard để đánh giá hiệu quả của các phương pháp thử nghiệm khác nhau. Phần này minh chứng cho quá trình nghiên cứu và thử sai (trial-and-error) của nhóm trong việc tìm ra đặc trưng và mô hình tối ưu.

| Tên File Submission | Score (F1) | Phương pháp / Mô tả thử nghiệm |
|---------------------|------------|--------------------------------|
| **POST-PROCESSING EXPERIMENTS** | | *Thử nghiệm chiến lược chọn Top-N thay vì Threshold cố định* |
| `submission_refined_top400.csv` | **0.5171** | Chọn Top 400 mẫu có xác suất cao nhất |
| `submission_refined_top420.csv` | 0.5154 | Chọn Top 420 mẫu có xác suất cao nhất |
| `submission_refined_top300.csv` | 0.5090 | Chọn Top 300 mẫu có xác suất cao nhất |
| **PHYSICS & ADVANCED FEATURES** | | *Thử nghiệm các đặc trưng vật lý và Gaussian Process* |
| `submission_physics_v4_GP_withZ.csv`| **0.4501**| Thêm đặc trưng Redshift (Z) + Gaussian Process |
| `submission_physics_v3_noZ.csv` | 0.4157 | Sử dụng đặc trưng Vật lý, loại bỏ Redshift |
| `submission_f1_optimal.csv` | 0.4146 | Tối ưu hóa trực tiếp hàm mục tiêu F1 |
| `submission_f1_higher.csv` | 0.4102 | Tối ưu hóa F1 (phiên bản thử nghiệm khác) |
| **ENSEMBLE METHODS** | | *Kết hợp nhiều mô hình* |
| `submission_ensemble_5fold.csv` | 0.4079 | Kết hợp 5-Fold Cross Validation tiêu chuẩn |
| `submission_v2.csv` | 0.4060 | Version 2 của model cơ bản |
| `submission_ensemble_optimal.csv` | 0.3975 | Ensemble với trọng số tối ưu hóa |
| `submission_ultimate_top300.csv` | 0.3915 | Ensemble "Ultimate" + Top 300 |
| `submission_ensemble_final.csv` | 0.3893 | Bản Ensemble cuối cùng của giai đoạn 2 |
| `submission_ensemble_weighted.csv`| 0.3226 | Weighted Average đơn giản |
| **MODEL TUNING & BASELINES** | | *Các thử nghiệm Baseline và Tuning khác* |
| `submission_full.csv` | 0.4068 | Train trên toàn bộ features (chưa select) |
| `submission_hybrid_v2.csv` | 0.4000 | Mô hình lai (Hybrid) |
| `submission_binary_optimal.csv` | 0.3919 | Binary classification tối ưu ngưỡng |
| `submission_robust_oversample.csv`| 0.3809 | Thử nghiệm kỹ thuật Oversampling (SMOTE/ADASYN) |
| `submission_optimized.csv` | 0.2837 | Phiên bản tối ưu hóa (nhưng bị overfit/lỗi) |
| `submission_xgboost_robust.csv` | 0.2792 | Thử nghiệm XGBoost thay vì LightGBM |
| `submission_raw_reversed.csv` | 0.2730 | Data Augmentation (đảo ngược chiều light curve) |
| `submission_metadata_only.csv` | 0.0995 | Baseline chỉ sử dụng Metadata (Z, SpecType) |
| `submission.csv` | 0.0743 | Random Forest Baseline sơ khai |

*Lưu ý: Bảng trên chỉ liệt kê các cột mốc quan trọng đại diện cho từng phương pháp. Nhiều thử nghiệm cho kết quả thấp hoặc lỗi (như `submission_physics_full.csv`) đã được loại bỏ để bảng súc tích hơn.*
