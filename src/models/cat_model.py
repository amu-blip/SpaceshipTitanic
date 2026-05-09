# ============================================================
# 阶段 4 — CatBoost 单 fold 训练
# ============================================================
from catboost import CatBoostClassifier
from utils import accuracy


def train_cat_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_features):
    """在单个 fold 上训练 CatBoost，cat_features 指定类别列名/索引。"""
    params = params.copy()
    model = CatBoostClassifier(**params)
    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False,
    )

    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    val_score = accuracy(y_val, y_val_pred >= 0.5)

    return {
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'model': model,
        'val_score': val_score,
    }
