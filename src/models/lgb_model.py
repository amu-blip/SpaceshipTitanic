# ============================================================
# 阶段 4 — LightGBM 单 fold 训练
# ============================================================
import lightgbm as lgb
from utils import accuracy


def train_lgb_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_features):
    """在单个 fold 上训练 LightGBM，返回 val_pred, test_pred, model, score。"""
    params = params.copy()
    callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]

    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_features)
    dval = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    val_score = accuracy(y_val, y_val_pred >= 0.5)

    return {
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'model': model,
        'val_score': val_score,
    }
