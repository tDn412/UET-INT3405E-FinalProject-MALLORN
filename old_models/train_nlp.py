
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

def main():
    print("Training NLP Model on English Translations...")
    
    # Load Data
    train_df = pd.read_csv('train_log.csv')
    test_df = pd.read_csv('test_log.csv')
    
    # Fill NaNs
    train_df['English Translation'] = train_df['English Translation'].fillna('')
    test_df['English Translation'] = test_df['English Translation'].fillna('')
    
    # TF-IDF Vectorization
    print("Vectorizing Text...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        stop_words='english', 
        max_features=1000,
        binary=True # Presence might be more important than frequency
    )
    
    full_text = pd.concat([train_df['English Translation'], test_df['English Translation']])
    vectorizer.fit(full_text)
    
    X = vectorizer.transform(train_df['English Translation'])
    X_test = vectorizer.transform(test_df['English Translation'])
    y = train_df['target']
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"Features: {len(feature_names)}")
    
    # Model Training
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X.toarray())) # LGBM handles sparse, but let's see
    test_preds_accum = np.zeros(len(X_test.toarray()))
    
    fold_f1s = []
    
    # Convert sparse to dataframe for LGBM convenience (or pass directly)
    # Passing directly is fine for LGBM
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        
        # Optimize threshold
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.02):
            val_pred = (val_proba >= thresh).astype(int)
            f1 = f1_score(y_val, val_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        fold_f1s.append(best_f1)
        print(f"Fold {fold+1} F1: {best_f1:.4f} (Thresh: {best_thresh:.2f})")
        
        test_preds_accum += model.predict_proba(X_test)[:, 1]
        
    avg_f1 = np.mean(fold_f1s)
    print(f"Average NLP CV F1: {avg_f1:.4f}")
    
    # Global Optimal Threshold
    global_best_thresh = 0.5
    global_best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        pred = (oof_preds >= thresh).astype(int)
        f1 = f1_score(y, pred)
        if f1 > global_best_f1:
            global_best_f1 = f1
            global_best_thresh = thresh
            
    print(f"Global Optimal Threshold: {global_best_thresh:.3f} with F1: {global_best_f1:.4f}")
    
    # Analyze Top Features
    # Train one full model to inspect importance
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y)
    
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Keywords:")
    for i in range(20):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]})")
        
    # Generate Submission
    avg_test_proba = test_preds_accum / 5
    final_preds = (avg_test_proba >= global_best_thresh).astype(int)
    
    sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_preds})
    sub.to_csv('submission_nlp_only.csv', index=False)
    
    prob_sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': avg_test_proba})
    prob_sub.to_csv('submission_nlp_probs.csv', index=False)
    
    print("Saved submission_nlp_only.csv")
    print(sub['prediction'].value_counts())

if __name__ == "__main__":
    main()
