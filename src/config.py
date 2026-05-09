# ============================================================
# 基础配置（各阶段共用）：路径、特征列表、超参
# ============================================================
import os

SEED = 42

# 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'spaceship-titanic')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# 原始数据
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_CSV = os.path.join(DATA_DIR, 'sample_submission.csv')

# 消费字段
SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# 类别特征（features.py 产出后存在）
CAT_COLS = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
            'Deck', 'Side', 'AgeGroup', 'DeckSide', 'CabinNumBin',
            'Route', 'Deck_HomePlanet']

# 特征列表（分组定义，训练时可按组拼装）
# preprocess 产出：base / id(部分) / cabin(部分) / age(部分)
# features  产出：其余所有组
FEAT_GROUPS = {
    'base': ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
             'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
    'id':   ['GroupSize', 'IsAlone', 'IsLargeGroup', 'PersonNum'],
    'cabin': ['Deck', 'CabinNum', 'CabinNumBin', 'Side', 'DeckSide'],
    'spend': ['TotalSpend', 'LogTotalSpend', 'HasSpend', 'NumSpendCategories',
              'MaxSpend', 'SpendStd', 'LuxurySpend', 'EssentialSpend', 'LuxuryRatio',
              'LogRoomService', 'LogFoodCourt', 'LogShoppingMall', 'LogSpa', 'LogVRDeck'],
    'age':  ['AgeGroup', 'IsChild', 'IsTeen', 'IsSenior'],
    'name': ['FamilySize', 'IsSurnameInGroup'],
    'group_agg': ['Group_TotalSpend_mean', 'Group_TotalSpend_sum',
                  'Group_Age_mean', 'Group_Age_min', 'Group_Age_max',
                  'Group_CryoRatio', 'Group_VIP_any', 'Group_HomePlanet_nunique'],
    'cabin_agg': ['Cabin_Size', 'Cabin_TotalSpend_sum', 'Cabin_CryoRatio'],
    'interact': ['Cryo_x_TotalSpend', 'Route', 'Deck_HomePlanet',
                 'Age_x_VIP', 'IsAlone_x_TotalSpend'],
}

# LightGBM 默认超参起点
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l2': 1.0,
    'verbose': -1,
    'random_state': SEED,
}

# XGBoost 默认超参起点
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'verbosity': 0,
    'random_state': SEED,
}

# CatBoost 默认超参起点
CAT_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'iterations': 5000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'bagging_temperature': 1,
    'verbose': 0,
    'random_seed': SEED,
}
