import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# =========================================================
# ## Config tham số model
# =========================================================
TOP_K = 140          # giữ lại top k features (thử 180 / 200 / 220)
SCALE_POS_BASE = 1.5 # scale weight cho class imbalanced; thử 1.0 / 1.5 / 2.0
THRESH_MIN = 0.30
THRESH_MAX = 0.80
THRESH_STEP = 0.01

# =========================================================
# 1. Load Data & Xử lý
# =========================================================
all_features = [
    c for c in X_full.columns
    if c not in ["object_id", "target", "split", "SpecType", "English Translation"]
    and pd.api.types.is_numeric_dtype(X_full[c])
]

X = X_full[all_features]
y = X_full["target"].astype(int)
X_test_raw = X_test_final[all_features]

print(f"📦 Original data: {X.shape[1]} features")

# =========================================================
# 2. Feature Selection (dùng LightGBM) (BCE)
# =========================================================
print("🔍 Đang chạy feature selection... (BCE)...")
fs_model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.7,
    objective="binary",
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
)
fs_model.fit(X, y)

importances = fs_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [all_features[i] for i in indices[:TOP_K]]

print(f"✅ Chọn {len(top_features)} features.")
print("   Top 5:", top_features[:5])

X = X[top_features]
X_test = X_test_raw[top_features]

# =========================================================
# 3. Train LightGBM (Final) (BCE + scale_pos_weight)
# =========================================================
n_pos = (y == 1).sum()
n_neg = (y == 0).sum()
ratio = n_neg / n_pos
scale_weight = ratio * SCALE_POS_BASE   # ví dụ ~20 nếu SCALE_POS_BASE=1.0

print(f"Tổng train: {len(y)}, TDE: {n_pos}, non-TDE: {n_neg}, "
      f"scale_pos_weight≈{scale_weight:.1f}")

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.55,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.5,
    "lambda_l2": 4.0,
    "scale_pos_weight": scale_weight,
    "verbosity": -1,
    "seed": 42,
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

print("\n🚀 Start training LightGBM (BCE + scale_pos_weight)...")
for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_set = lgb.Dataset(X_va, label=y_va)

    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=4000,
        valid_sets=[valid_set],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_pred[va_idx] = val_pred
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

    tmp_f1 = f1_score(y_va, (val_pred > 0.5).astype(int))
    print(f" Fold {fold+1}: best_iter={model.best_iteration}, F1@0.5={tmp_f1:.4f}")

    if fold == 0:
        last_model = model

    del model, X_tr, X_va
    gc.collect()

# =========================================================
# 4. Tìm Threshold tối ưu
# =========================================================
print("\n🎚️ Tìm threshold ngon nhất trên OOF set......")
best_f1, best_t, best_frac = 0.0, 0.5, 0.0

for t in np.arange(THRESH_MIN, THRESH_MAX, THRESH_STEP):
    preds_bin = (oof_pred > t).astype(int)
    score = f1_score(y, preds_bin, pos_label=1)
    frac_pos = preds_bin.mean()
    if score > best_f1:
        best_f1, best_t, best_frac = score, t, frac_pos

print("=" * 40)
print(f"🏆 FINAL CV F1: {best_f1:.4f} @ Threshold {best_t:.3f}")
print(f"  Tỉ lệ TDE (train):: {best_frac:.4f}")
print("=" * 40)

# =========================================================
# 5. Tạo file Submission
# =========================================================
sub = pd.DataFrame({
    "object_id": X_test_final["object_id"],
    "target": (test_pred > best_t).astype(int),
})
sub.to_csv("submission_Final.csv", index=False)
print(f"✅ Save submission xong:_Final.csv (TDEs: {sub['target'].sum()})")

# =========================================================
# 6. Feature Importance
# =========================================================
plt.figure(figsize=(8, 12))
imp_idx = np.argsort(last_model.feature_importance())[::-1][:30]
imp_vals = last_model.feature_importance()[imp_idx]
imp_names = [top_features[i] for i in imp_idx]
sns.barplot(x=imp_vals, y=imp_names)
plt.title("Top 30 Important Features (LightGBM)")
plt.tight_layout()
plt.show()
