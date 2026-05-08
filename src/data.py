# ============================================================
# 数据加载
# ============================================================
import pandas as pd
from config import TRAIN_CSV, TEST_CSV


def load_raw():
    """加载原始数据，返回 train, test DataFrame"""
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f'[load] train: {train.shape}, test: {test.shape}')
    return train, test


def load_sample_submission():
    return pd.read_csv(TEST_CSV.replace('test.csv', 'sample_submission.csv'))
