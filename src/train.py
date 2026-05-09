# ============================================================
# 阶段 4 — 训练主流程：多种子 CV、OOF 保存、测试集预测
# 支持 StratifiedKFold / GroupKFold，模型通过 CLI 参数选择
# ============================================================
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from config import FEAT_GROUPS, CAT_COLS, LGB_PARAMS, XGB_PARAMS, CAT_PARAMS, OUTPUT_DIR, SEED
from data import load_raw
from preprocess import preprocess
from features import build_features
from utils import accuracy, save_oof, save_preds, seed_everything


VALID_MODELS = {'lgb', 'xgb', 'cat'}
VALID_CV_SCHEMES = {'stratified', 'group'}


def get_feature_names(groups=None):
    """从 FEAT_GROUPS 组装特征列名。传入 groups 列表只取部分组，默认全部。"""
    if groups is None:
        groups = list(FEAT_GROUPS.keys())

    unknown_groups = sorted(set(groups) - set(FEAT_GROUPS))
    if unknown_groups:
        raise ValueError(f'Unknown feature group(s): {unknown_groups}. Available: {sorted(FEAT_GROUPS)}')

    return [col for g in groups for col in FEAT_GROUPS[g]]


def prepare_data(feature_groups=None):
    """加载 → 预处理 → 特征工程 → 返回训练所需对象。"""
    print('=' * 60)
    print('Phase 4: Training Pipeline')
    print('=' * 60)

    seed_everything(SEED)
    train_raw, test_raw = load_raw()
    y = train_raw['Transported'].astype(int)

    train_c, test_c = preprocess(train_raw, test_raw)
    train_f, test_f = build_features(train_c, test_c)

    feature_cols = get_feature_names(feature_groups)
    missing_features = [c for c in feature_cols if c not in train_f.columns]
    feature_cols = [c for c in feature_cols if c in train_f.columns]
    if missing_features:
        print(f'  [warn] missing configured features ignored: {missing_features}')
    print(f'  feature count: {len(feature_cols)}')

    # PassengerId 用于保存 OOF / 提交
    train_ids = train_f['PassengerId']
    test_ids = test_f['PassengerId']

    X_train = train_f[feature_cols].copy()
    X_test = test_f[feature_cols].copy()

    # 类别列转换为 category dtype（LGB/XGB/CAT 原生支持）
    cat_cols_present = [c for c in CAT_COLS if c in X_train.columns]
    for col in cat_cols_present:
        # bool → int 先（XGBoost 无法处理 bool categories）
        if X_train[col].dtype == bool:
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    # GroupId 用于 GroupKFold
    group_ids = train_f['GroupId'].values

    print(f'  categorical columns: {len(cat_cols_present)}')
    return X_train, y, X_test, test_ids, train_ids, cat_cols_present, group_ids


def run_cv(models=None, seeds=None, n_splits=5, cv_scheme='stratified', feature_groups=None):
    """
    主训练入口。

    Parameters
    ----------
    models : list[str] | None
        要训练的模型列表，可选 'lgb', 'xgb', 'cat'。默认 ['lgb']。
    seeds : list[int] | None
        CV 随机种子列表。默认 [42, 2024]。
    n_splits : int
        折数，默认 5。
    cv_scheme : str
        'stratified' 或 'group'。
    feature_groups : list[str] | None
        使用的特征组列表，默认全部（FEAT_GROUPS 所有 key）。

    Returns
    -------
    list[dict]
        每个模型的 OOF、CV、输出文件路径等摘要。
    """
    if models is None:
        models = ['lgb']
    if seeds is None:
        seeds = [42, 2024]

    _validate_run_args(models, seeds, n_splits, cv_scheme)

    X_train, y, X_test, test_ids, train_ids, cat_cols, all_groups = prepare_data(feature_groups)
    groups = all_groups if cv_scheme == 'group' else None
    summaries = []

    for model_type in models:
        print(f'\n{"─" * 50}')
        print(f'[{model_type.upper()}] seeds={seeds} folds={n_splits} scheme={cv_scheme}')
        print(f'{"─" * 50}')

        oof_preds_sum = np.zeros(len(X_train))
        test_preds_list = []
        fold_scores = []

        for seed in seeds:
            split_iter = _get_split_iterator(cv_scheme, X_train, y, groups, n_splits, seed)

            for fold_idx, (tr_idx, val_idx) in enumerate(split_iter, 1):
                X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
                y_tr = y.iloc[tr_idx].reset_index(drop=True)
                X_val = X_train.iloc[val_idx].reset_index(drop=True)
                y_val = y.iloc[val_idx].reset_index(drop=True)

                # 分发到各模型训练函数；每个 seed/fold 使用不同随机态，提升多模型多折独立性
                model_seed = seed * 100 + fold_idx
                params, cat_cols_override = _get_model_params(model_type, model_seed)
                cat = cat_cols_override if cat_cols_override is not None else cat_cols

                result = _train_one_fold(model_type, X_tr, y_tr, X_val, y_val, X_test, params, cat)

                # 多 seed OOF 必须累加后平均，避免后一个 seed 覆盖前一个 seed
                oof_preds_sum[val_idx] += result['y_val_pred']
                test_preds_list.append(result['y_test_pred'])
                fold_scores.append({
                    'model': model_type,
                    'cv_scheme': cv_scheme,
                    'seed': seed,
                    'model_seed': model_seed,
                    'fold': fold_idx,
                    'val_size': int(len(val_idx)),
                    'acc': round(float(result['val_score']), 5),
                    'best_iteration': _get_best_iteration(result['model']),
                })
                print(f'  [seed={seed} fold={fold_idx}] acc={result["val_score"]:.5f}')

        oof_preds = oof_preds_sum / len(seeds)
        test_avg = np.mean(test_preds_list, axis=0)

        oof_acc = accuracy(y, oof_preds >= 0.5)
        cv_mean = float(np.mean([s['acc'] for s in fold_scores]))
        cv_std = float(np.std([s['acc'] for s in fold_scores]))
        print(f'  → OOF acc={oof_acc:.5f}  CV mean={cv_mean:.5f}±{cv_std:.5f}')

        exp_name = f'{model_type}_{cv_scheme}_{n_splits}fold_{len(seeds)}seed'
        oof_path = save_oof(pd.DataFrame({
            'PassengerId': train_ids.values,
            'Transported_Prob': oof_preds,
        }), exp_name)
        pred_path = save_preds(test_ids.values, test_avg, exp_name)
        metrics_path = save_metrics(exp_name, fold_scores, {
            'model': model_type,
            'cv_scheme': cv_scheme,
            'seeds': seeds,
            'n_splits': n_splits,
            'feature_groups': feature_groups or list(FEAT_GROUPS.keys()),
            'n_features': int(X_train.shape[1]),
            'oof_acc': round(float(oof_acc), 5),
            'cv_mean': round(cv_mean, 5),
            'cv_std': round(cv_std, 5),
            'oof_path': oof_path,
            'pred_path': pred_path,
        })

        summaries.append({
            'exp_name': exp_name,
            'model': model_type,
            'oof_acc': round(float(oof_acc), 5),
            'cv_mean': round(cv_mean, 5),
            'cv_std': round(cv_std, 5),
            'oof_path': oof_path,
            'pred_path': pred_path,
            'metrics_path': metrics_path,
        })

    print(f'\n{"=" * 60}')
    print('Phase 4 complete.')
    print(f'{"=" * 60}')
    print(pd.DataFrame(summaries)[['exp_name', 'oof_acc', 'cv_mean', 'cv_std']].to_string(index=False))
    return summaries


def _validate_run_args(models, seeds, n_splits, cv_scheme):
    if not models:
        raise ValueError('At least one model is required.')
    unknown_models = sorted(set(models) - VALID_MODELS)
    if unknown_models:
        raise ValueError(f'Unknown model(s): {unknown_models}. Available: {sorted(VALID_MODELS)}')
    if cv_scheme not in VALID_CV_SCHEMES:
        raise ValueError(f'Unknown cv_scheme: {cv_scheme}. Available: {sorted(VALID_CV_SCHEMES)}')
    if not seeds:
        raise ValueError('At least one seed is required.')
    if n_splits < 2:
        raise ValueError('n_splits must be >= 2.')


def _get_split_iterator(cv_scheme, X_train, y, groups, n_splits, seed):
    if cv_scheme == 'group':
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(X_train, y, groups=groups)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cv.split(X_train, y)


def _get_model_params(model_type, model_seed):
    """返回 (params_dict, cat_cols_override_or_None)。"""
    if model_type == 'lgb':
        p = LGB_PARAMS.copy()
        p['random_state'] = model_seed
        return p, None  # cat_cols from prepare_data
    if model_type == 'xgb':
        p = XGB_PARAMS.copy()
        p['random_state'] = model_seed
        return p, None
    if model_type == 'cat':
        p = CAT_PARAMS.copy()
        p['random_seed'] = model_seed
        return p, None
    raise ValueError(f'Unknown model_type: {model_type}')


def _train_one_fold(model_type, X_tr, y_tr, X_val, y_val, X_test, params, cat_cols):
    if model_type == 'lgb':
        from models.lgb_model import train_lgb_fold
        return train_lgb_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols)
    if model_type == 'xgb':
        from models.xgb_model import train_xgb_fold
        return train_xgb_fold(X_tr, y_tr, X_val, y_val, X_test, params)
    if model_type == 'cat':
        from models.cat_model import train_cat_fold
        return train_cat_fold(X_tr, y_tr, X_val, y_val, X_test, params, cat_cols)
    raise ValueError(f'Unknown model_type: {model_type}')


def _get_best_iteration(model):
    """兼容 LightGBM / XGBoost / CatBoost 的最佳迭代数提取。"""
    if hasattr(model, 'best_iteration'):
        value = model.best_iteration
    elif hasattr(model, 'best_iteration_'):
        value = model.best_iteration_
    elif hasattr(model, 'get_best_iteration'):
        value = model.get_best_iteration()
    else:
        value = None
    return None if value is None else int(value)


def save_metrics(exp_name, fold_scores, summary):
    """保存 fold 级分数和实验摘要，便于复查阶段四结果。"""
    metrics_dir = os.path.join(OUTPUT_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    fold_path = os.path.join(metrics_dir, f'{exp_name}_folds.csv')
    summary_path = os.path.join(metrics_dir, f'{exp_name}_summary.json')
    pd.DataFrame(fold_scores).to_csv(fold_path, index=False)

    payload = summary.copy()
    payload['created_at'] = datetime.now().isoformat(timespec='seconds')
    payload['fold_metrics_path'] = fold_path
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'[save] metrics → {fold_path}')
    print(f'[save] summary → {summary_path}')
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 4 training pipeline')
    parser.add_argument('--models', default='lgb', help='Comma-separated models: lgb,xgb,cat')
    parser.add_argument('--cv', default='stratified', choices=sorted(VALID_CV_SCHEMES))
    parser.add_argument('--seeds', default='42,2024', help='Comma-separated random seeds')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument(
        '--feature-groups',
        default=None,
        help=f'Comma-separated feature groups. Default uses all: {",".join(FEAT_GROUPS.keys())}',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    feature_groups = None
    if args.feature_groups:
        feature_groups = [g.strip() for g in args.feature_groups.split(',') if g.strip()]

    print(f'Models: {models} | CV: {args.cv} | Seeds: {seeds} | Folds: {args.folds}')
    run_cv(
        models=models,
        seeds=seeds,
        n_splits=args.folds,
        cv_scheme=args.cv,
        feature_groups=feature_groups,
    )


if __name__ == '__main__':
    main()
