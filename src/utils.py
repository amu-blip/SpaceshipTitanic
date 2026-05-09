# ============================================================
# 基础工具（各阶段共用）：固定种子、评估、保存提交
# ============================================================
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from config import SEED, OUTPUT_DIR


def seed_everything(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def accuracy(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean()


def save_oof(df_oof, exp_name):
    """保存 OOF 预测"""
    path = os.path.join(OUTPUT_DIR, 'oof', f'{exp_name}.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_oof.to_csv(path, index=False)
    print(f'[save] OOF → {path} (shape={df_oof.shape})')
    return path


def save_preds(passenger_ids, proba, exp_name):
    """保存测试集预测概率"""
    path = os.path.join(OUTPUT_DIR, 'preds', f'{exp_name}.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({'PassengerId': passenger_ids, 'Transported_Prob': proba})
    df.to_csv(path, index=False)
    print(f'[save] preds → {path} (shape={df.shape})')
    return path


def save_submission(passenger_ids, proba, threshold, exp_name):
    """生成 Kaggle 提交文件"""
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    fname = f'{exp_name}_th{threshold:.3f}_{ts}.csv'
    path = os.path.join(OUTPUT_DIR, 'submissions', fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Transported': (proba >= threshold)
    })
    df.to_csv(path, index=False)
    print(f'[submit] {path}')
    print(f'  True:  {(df["Transported"] == True).sum()} / {len(df)} ({(df["Transported"] == True).mean():.1%})')
    print(f'  False: {(df["Transported"] == False).sum()} / {len(df)} ({(df["Transported"] == False).mean():.1%})')
    return path
