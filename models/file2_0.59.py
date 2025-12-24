import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# 1. CHUẨN BỊ DỮ LIỆU
all_features = [c for c in X_full.columns if c not in ['object_id', 'target', 'split', 'SpecType', 'English Translation'] 
            and pd.api.types.is_numeric_dtype(X_full[c])]

X = X_full[all_features]
y = X_full['target']
X_test_raw = X_test_final[all_features]

print(f"📦 Dữ liệu gốc: {X.shape[1]} đặc trưng")

# --- 2. GIAI ĐOẠN 1: CHỌN LỌC ĐẶC TRƯNG (Lọc rác) ---
print("🔍 Đang chạy Chọn Lọc Đặc Trưng (dùng LightGBM nhanh)...")

# Dùng LightGBM để tính importance nhanh
selector_model = lgb.LGBMClassifier(
    n_estimators=500, 
    learning_rate=0.05, 
    is_unbalance=True,
    verbose=-1,
    random_state=42
)

selector_model.fit(X, y)

# Chọn features có importance > ngưỡng trung bình (hoặc lấy Top K)
# Ở đây ta lấy Top 250 features tốt nhất để tránh overfit
importances = selector_model.feature_importances_
indices = np.argsort(importances)[::-1] # Sắp xếp giảm dần
top_k = 250 
top_features = [all_features[i] for i in indices[:top_k]]

print(f"✅ Đã chọn {len(top_features)} features quan trọng nhất.")
print(f"   Top 5: {top_features[:5]}")

# Cập nhật lại dữ liệu theo feature đã chọn
X = X[top_features]
X_test = X_test_raw[top_features]

# --- 3. GIAI ĐOẠN 2: XGBOOST TRAINING (Tập trung GAIN) ---

# Tính tỷ lệ Imbalance
scale_weight = np.sum(y == 0) / np.sum(y == 1)

xgb_params = {
    'n_estimators': 5000,           # Train sâu
    'learning_rate': 0.005,         # Học chậm
    'max_depth': 8,                 # Cây sâu hơn vì đã lọc feature rác
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'objective': 'binary:logistic',
    'scale_pos_weight': scale_weight,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
    'reg_alpha': 1.0,               # L1 Regularization
    'reg_lambda': 3.0,              # L2 Regularization (cao để tránh overfit)
    'eval_metric': 'aucpr',
    'importance_type': 'gain'
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print("\n🚀 Bắt đầu train XGBoost trên tập features đã chọn...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[tr_idx], X.iloc[va_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[va_idx]

    model = xgb.XGBClassifier(**xgb_params)
    
    # Early Stopping thủ công (do version mới sklearn đổi API)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Predict
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[va_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits
    
    # Check F1
    score = f1_score(y_val, (val_pred > 0.😎.astype(int)) # Check tạm threshold cao
    print(f"   Fold {fold+1}: F1 ~ {score:.4f} | Best Iter: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
    
    del model, X_train, X_val
    gc.collect()

# --- 4. TỐI ƯU NGƯỠNG & FILE NỘP BÀI ---
print("\n🎚️ Đang dò tìm ngưỡng tối ưu (Threshold Tuning)...")
best_f1 = 0
best_t = 0.5
for t in np.arange(0.1, 0.99, 0.005):
    score = f1_score(y, (oof_preds > t).astype(int))
    if score > best_f1:
        best_f1 = score
        best_t = t

print("="*40)
print(f"🏆 FINAL F1: {best_f1:.4f} @ Threshold {best_t:.3f}")
print("="*40)

# Xuất file
sub = pd.DataFrame({
    'object_id': X_test_final['object_id'],
    'target': (test_preds > best_t).astype(int)
})
sub.to_csv("submission_massive_select.csv", index=False)
print(f"✅ Đã lưu file: submission_massive_select.csv (TDEs: {sub['target'].sum()})")

# Vẽ Độ Quan Trọng Đặc Trưng
plt.figure(figsize=(10, 15))
# Lấy importance từ lần chạy cuối (hoặc có thể tích lũy)
# Lưu ý: Đây là importance sau khi đã lọc
sns.barplot(x=model.feature_importances_[:30], y=top_features[:30])
plt.title("Top 30 Selected Features (XGBoost Gain)")
plt.show()