# ============================================================
# 阶段 4 — XGBoost 单 fold 训练
# ============================================================
import xgboost as xgb
from utils import accuracy


def train_xgb_fold(X_tr, y_tr, X_val, y_val, X_test, params):
    """在单个 fold 上训练 XGBoost，使用 enable_categorical 原生类别支持。"""
    params = params.copy()
    dtrain = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    y_val_pred = model.predict(dval)
    y_test_pred = model.predict(dtest)
    val_score = accuracy(y_val, y_val_pred >= 0.5)

    return {
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'model': model,
        'val_score': val_score,
    }
