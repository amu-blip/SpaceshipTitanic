# ============================================================
# 数据预处理：解析 → 规则填充 → 组内填充 → 统计填充
# 处理顺序严格遵循 strategy 第四节
# ============================================================
import numpy as np
import pandas as pd
from config import SPEND_COLS

# ── 1. 字段解析 ───────────────────────────────────────────

def parse_ids(df):
    """PassengerId → GroupId, PersonNum, GroupSize"""
    df = df.copy()
    parts = df['PassengerId'].str.split('_', expand=True)
    df['GroupId'] = parts[0].astype(int)
    df['PersonNum'] = parts[1].astype(int)
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    return df


def parse_cabin(df):
    """Cabin (deck/num/side) → Deck, CabinNum, Side"""
    df = df.copy()
    parts = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = parts[0]
    df['CabinNum'] = pd.to_numeric(parts[1], errors='coerce')
    df['Side'] = parts[2]
    return df


def parse_name(df):
    """Name → FirstName, LastName, SurnameGroupSize"""
    df = df.copy()
    parts = df['Name'].str.strip().str.rsplit(' ', n=1, expand=True)
    df['FirstName'] = parts[0]
    df['LastName'] = parts[1].fillna('Unknown')
    df['SurnameGroupSize'] = df.groupby('LastName')['LastName'].transform('count')
    return df


# ── 2. 规则推断填充（最高优先级）────────────────────────────

def rule_fill(df):
    """
    基于业务逻辑的硬填充，不依赖统计：
    - 消费均为 0 → CryoSleep 很可能为 True
    - CryoSleep=True → 消费应为 0
    - 儿童 (Age<13) → VIP=False
    """
    df = df.copy()

    # 总消费（处理 NaN：先填 0 计算）
    total = df[SPEND_COLS].fillna(0).sum(axis=1)

    # 规则 1: 消费全 0 + Age>12 + CryoSleep 缺失 → CryoSleep=True
    mask_cryo = (
        (total == 0)
        & (df['Age'].notna()) & (df['Age'] > 12)
        & df['CryoSleep'].isna()
    )
    df.loc[mask_cryo, 'CryoSleep'] = True
    n1 = mask_cryo.sum()
    if n1:
        print(f'  [rule] 消费=0 + Age>12 + CryoSleep 缺失 → CryoSleep=True: {n1}')

    # 规则 2: 消费全 0 + Age 也缺失 + CryoSleep 缺失 → CryoSleep=True
    mask_cryo2 = (
        (total == 0)
        & df['Age'].isna()
        & df['CryoSleep'].isna()
    )
    df.loc[mask_cryo2, 'CryoSleep'] = True
    n1b = mask_cryo2.sum()
    if n1b:
        print(f'  [rule] 消费=0 + Age缺失 + CryoSleep 缺失 → CryoSleep=True: {n1b}')

    # 规则 3: CryoSleep=True → 消费缺失填充 0
    cryo_true = df['CryoSleep'].fillna(False)
    for col in SPEND_COLS:
        mask_spend = (cryo_true == True) & df[col].isna()
        n = mask_spend.sum()
        if n:
            df.loc[mask_spend, col] = 0.0
            print(f'  [rule] CryoSleep=True + {col} 缺失 → 0.0: {n}')

    # 规则 4: 儿童 Age < 13 → VIP=False（儿童不可能 VIP）
    mask_child_vip = (df['Age'] < 13) & (df['VIP'].isna())
    n3 = mask_child_vip.sum()
    if n3:
        df.loc[mask_child_vip, 'VIP'] = False
        print(f'  [rule] Age<13 + VIP 缺失 → False: {n3}')

    return df


# ── 3. 组内填充（GroupId 层级）───────────────────────────────

def group_fill(df):
    """
    同组成员特征高度一致，用组内众数拉通：
    HomePlanet, Destination, Deck, Side
    """
    df = df.copy()
    group_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']

    for col in group_cols:
        if col not in df.columns:
            continue
        na_before = df[col].isna().sum()
        if na_before == 0:
            continue
        # 用同 GroupId 的众数填充
        mode_map = (
            df.groupby('GroupId')[col]
            .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
        df[col] = df[col].fillna(df['GroupId'].map(mode_map))
        na_after = df[col].isna().sum()
        if na_after < na_before:
            print(f'  [group_fill] {col}: {na_before} → {na_after}')

    # CryoSleep：组内如果大多数人在冷冻，缺失的人也冷冻
    if 'CryoSleep' in df.columns:
        cryo_ratio = df.groupby('GroupId')['CryoSleep'].transform(
            lambda x: x.mean() if x.notna().any() else np.nan
        )
        mask = df['CryoSleep'].isna() & (cryo_ratio.notna())
        n = mask.sum()
        if n:
            # 组内 ≥ 50% 冷冻 → True
            df.loc[mask & (cryo_ratio >= 0.5), 'CryoSleep'] = True
            df.loc[mask & (cryo_ratio < 0.5), 'CryoSleep'] = False
            print(f'  [group_fill] CryoSleep by group majority: {n}')

    # Age：同组内用中位数填充（家人年龄有相关但不是完全相同）
    if 'AgeGroup' not in df.columns:
        df['AgeGroup'] = pd.cut(
            df['Age'],
            bins=[-1, 12, 17, 25, 40, 60, 200],
            labels=['<13', '13-17', '18-25', '26-40', '41-60', '60+']
        )

    return df


# ── 4. 统计填充（兜底）──────────────────────────────────────

def statistical_fill(df):
    """
    前面的规则和组内填充之后，剩余缺失用统计方法填：
    - 类别特征：按 HomePlanet 分层众数，再全局众数
    - 数值特征：按 HomePlanet 分层中位数，再全局中位数
    """
    df = df.copy()

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'LastName']
    num_cols = ['Age'] + SPEND_COLS

    for col in cat_cols:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        # 先尝试按 HomePlanet 分层
        if 'HomePlanet' in df.columns and df['HomePlanet'].notna().any():
            strat_fill = df.groupby('HomePlanet')[col].transform(
                lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
            )
            df[col] = df[col].fillna(strat_fill)
        # 剩余用全局众数
        if df[col].isna().any():
            global_mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(global_mode)
            print(f'  [stat_fill] {col}: 全局众数 → {global_mode}')

    for col in num_cols:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        # 先尝试按 HomePlanet 分层
        if 'HomePlanet' in df.columns and df['HomePlanet'].notna().any():
            strat_fill = df.groupby('HomePlanet')[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(strat_fill)
        # 剩余用全局中位数
        if df[col].isna().any():
            global_median = df[col].median()
            df[col] = df[col].fillna(global_median)
            print(f'  [stat_fill] {col}: 全局中位数 → {global_median:.2f}')

    return df


# ── 5. 基础衍生特征 ────────────────────────────────────────

def add_base_features(df):
    """计算不依赖目标变量的基础特征：消费聚合、年龄标记等"""
    df = df.copy()

    # 消费
    df['TotalSpend'] = df[SPEND_COLS].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['NumSpendCategories'] = (df[SPEND_COLS] > 0).sum(axis=1).astype(int)

    # 年龄标记
    df['IsChild'] = (df['Age'] < 13).astype(int)
    df['IsSenior'] = (df['Age'] >= 60).astype(int)

    # 独行标记
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    return df


# ── 6. 类型规范化 ──────────────────────────────────────────

def finalize_types(df):
    """统一字段类型，确保模型 ingest 正确"""
    df = df.copy()
    for col in ['CryoSleep', 'VIP']:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    for col in ['Age', 'TotalSpend'] + SPEND_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


# ── 7. 主入口 ─────────────────────────────────────────────

def preprocess(train_raw, test_raw):
    """
    train + test 拼接后统一预处理，避免两次处理不一致。
    返回 (train_processed, test_processed)
    """
    print('=== preprocess ===')
    train = train_raw.copy()
    test = test_raw.copy()

    is_train = pd.Series(True, index=train.index)
    is_test = pd.Series(False, index=test.index)

    df = pd.concat([train, test], ignore_index=True)
    is_train_all = pd.concat([is_train, is_test], ignore_index=True)

    n_before = len(df)

    # Step 1: 解析
    print('[1/5] 解析 PassengerId / Cabin / Name...')
    df = parse_ids(df)
    df = parse_cabin(df)
    df = parse_name(df)

    # Step 2: 规则填充（不含 Transported 列）
    print('[2/5] 规则推断填充...')
    df = rule_fill(df)

    # Step 3: 组内填充
    print('[3/5] 组内填充...')
    df = group_fill(df)

    # Step 4: 统计填充
    print('[4/5] 统计填充...')
    df = statistical_fill(df)

    # Step 5: 衍生特征 + 类型规范
    print('[5/5] 基础衍生特征 + 类型规范化...')
    df = add_base_features(df)
    df = finalize_types(df)

    # 确认无缺失：按列类型分别填充
    missing_after = df.isna().sum()
    missing_after = missing_after[missing_after > 0]
    if len(missing_after) > 0:
        print(f'  [WARN] 剩余缺失列（将填充）: {missing_after.to_dict()}')
        for col in missing_after.index:
            if df[col].dtype.name == 'category':
                # Categorical 加入新模式再 fill
                df[col] = df[col].cat.add_categories('missing')
                df[col] = df[col].fillna('missing')
            elif df[col].dtype in ['int64', 'int32', 'int8', 'float64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('Unknown')

    # 拆分回 train / test
    train_clean = df.loc[is_train_all.values].reset_index(drop=True)
    test_clean = df.loc[~is_train_all.values].reset_index(drop=True)

    print(f'  done: train={train_clean.shape}, test={test_clean.shape} (n={n_before})')
    return train_clean, test_clean
