# ============================================================
# 阶段 5 — 集成：OOF 加权 blend、LogisticRegression stacking、提交生成
# ============================================================
import argparse
import json
import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import OUTPUT_DIR, SAMPLE_CSV, SEED
from data import load_raw
from utils import accuracy, save_submission, seed_everything


DEFAULT_EXPERIMENTS = [
    'lgb_stratified_5fold_2seed',
    'xgb_stratified_5fold_2seed',
    'cat_stratified_5fold_2seed',
]


def load_prediction_pair(exp_name):
    """加载一个基模型实验的 OOF 和测试集概率。"""
    oof_path = os.path.join(OUTPUT_DIR, 'oof', f'{exp_name}.csv')
    pred_path = os.path.join(OUTPUT_DIR, 'preds', f'{exp_name}.csv')
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f'OOF file not found for {exp_name}: {oof_path}')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f'Prediction file not found for {exp_name}: {pred_path}')

    oof = pd.read_csv(oof_path)
    pred = pd.read_csv(pred_path)
    _validate_prediction_frame(oof, oof_path)
    _validate_prediction_frame(pred, pred_path)
    return oof, pred


def _validate_prediction_frame(df, path):
    required = {'PassengerId', 'Transported_Prob'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} missing required column(s): {sorted(missing)}')
    if df['PassengerId'].duplicated().any():
        dupes = df.loc[df['PassengerId'].duplicated(), 'PassengerId'].head().tolist()
        raise ValueError(f'{path} contains duplicated PassengerId values, e.g. {dupes}')
    if df['Transported_Prob'].isna().any():
        raise ValueError(f'{path} contains NaN probabilities')
    if not df['Transported_Prob'].between(0, 1).all():
        raise ValueError(f'{path} contains probabilities outside [0, 1]')


def build_prediction_matrices(experiments):
    """
    将多个基模型 OOF/test 预测按 PassengerId 对齐为矩阵。

    Returns
    -------
    tuple[pd.Index, pd.Index, pd.DataFrame, pd.DataFrame]
        train_ids, test_ids, oof_matrix, test_matrix。
    """
    if not experiments:
        raise ValueError('At least one experiment is required for ensembling.')

    train_raw, _ = load_raw()
    train_ids = train_raw['PassengerId']
    sample = pd.read_csv(SAMPLE_CSV)
    test_ids = sample['PassengerId']

    oof_cols = {}
    test_cols = {}
    for exp_name in experiments:
        oof, pred = load_prediction_pair(exp_name)
        oof_aligned = oof.set_index('PassengerId').reindex(train_ids)
        pred_aligned = pred.set_index('PassengerId').reindex(test_ids)
        if oof_aligned['Transported_Prob'].isna().any():
            raise ValueError(f'OOF predictions for {exp_name} do not cover all train PassengerId values')
        if pred_aligned['Transported_Prob'].isna().any():
            raise ValueError(f'Test predictions for {exp_name} do not cover all sample PassengerId values')
        oof_cols[exp_name] = oof_aligned['Transported_Prob'].to_numpy(dtype=float)
        test_cols[exp_name] = pred_aligned['Transported_Prob'].to_numpy(dtype=float)

    oof_matrix = pd.DataFrame(oof_cols, index=train_ids)
    test_matrix = pd.DataFrame(test_cols, index=test_ids)
    return train_ids, test_ids, oof_matrix, test_matrix


def optimize_threshold(y_true, proba, low=0.35, high=0.65, step=0.001):
    """在 OOF 概率上搜索 Accuracy 最优阈值。"""
    thresholds = np.round(np.arange(low, high + step / 2, step), 6)
    scores = np.array([accuracy(y_true, proba >= t) for t in thresholds])
    best_idx = int(scores.argmax())
    return float(thresholds[best_idx]), float(scores[best_idx])


def iter_simplex_weights(n_models, step=0.05):
    """生成非负且和为 1 的网格权重。"""
    if n_models < 1:
        raise ValueError('n_models must be >= 1')
    units = int(round(1 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError('step must divide 1.0 exactly, e.g. 0.1, 0.05, 0.02')

    if n_models == 1:
        yield np.array([1.0])
        return

    # 对前 n-1 个模型枚举整数份额，最后一个模型使用剩余份额。
    for combo in product(range(units + 1), repeat=n_models - 1):
        used = sum(combo)
        if used <= units:
            weights = np.array([*combo, units - used], dtype=float) / units
            yield weights


def search_blend_weights(y_true, oof_matrix, step=0.05, threshold=0.5, tune_threshold=False):
    """网格搜索 OOF accuracy 最优的 blend 权重。"""
    values = oof_matrix.to_numpy(dtype=float)
    best = {
        'weights': None,
        'threshold': float(threshold),
        'score': -np.inf,
        'proba': None,
    }

    for weights in iter_simplex_weights(values.shape[1], step):
        blended = values @ weights
        if tune_threshold:
            candidate_threshold, score = optimize_threshold(y_true, blended)
        else:
            candidate_threshold = float(threshold)
            score = float(accuracy(y_true, blended >= threshold))
        if score > best['score']:
            best = {
                'weights': weights,
                'threshold': float(candidate_threshold),
                'score': float(score),
                'proba': blended,
            }

    return best


def run_stacking(y_true, oof_matrix, test_matrix, n_splits=5, seed=SEED):
    """用 LogisticRegression 对基模型 OOF 概率做二层 stacking。"""
    X = oof_matrix.to_numpy(dtype=float)
    X_test = test_matrix.to_numpy(dtype=float)
    stack_oof = np.zeros(len(X), dtype=float)
    stack_test_folds = []
    fold_scores = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y_true), 1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=seed + fold_idx),
        )
        model.fit(X[tr_idx], y_true[tr_idx])
        val_proba = model.predict_proba(X[val_idx])[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        stack_oof[val_idx] = val_proba
        stack_test_folds.append(test_proba)
        fold_scores.append({
            'fold': fold_idx,
            'acc_0_5': round(float(accuracy(y_true[val_idx], val_proba >= 0.5)), 5),
            'val_size': int(len(val_idx)),
        })

    stack_test = np.mean(stack_test_folds, axis=0)
    threshold, tuned_score = optimize_threshold(y_true, stack_oof)
    return {
        'oof_proba': stack_oof,
        'test_proba': stack_test,
        'threshold': threshold,
        'score': tuned_score,
        'score_0_5': float(accuracy(y_true, stack_oof >= 0.5)),
        'fold_scores': fold_scores,
    }


def save_ensemble_outputs(exp_name, train_ids, test_ids, oof_proba, test_proba, y_true, threshold, metadata):
    """保存阶段五集成 OOF、测试概率、提交和 metadata。"""
    ensemble_dir = os.path.join(OUTPUT_DIR, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    oof_path = os.path.join(ensemble_dir, f'{exp_name}_oof.csv')
    pred_path = os.path.join(ensemble_dir, f'{exp_name}_preds.csv')
    summary_path = os.path.join(ensemble_dir, f'{exp_name}_summary.json')

    pd.DataFrame({
        'PassengerId': train_ids,
        'Transported_Prob': oof_proba,
        'Transported': y_true.astype(bool),
        'Predicted': oof_proba >= threshold,
    }).to_csv(oof_path, index=False)
    pd.DataFrame({
        'PassengerId': test_ids,
        'Transported_Prob': test_proba,
    }).to_csv(pred_path, index=False)
    submission_path = save_submission(test_ids, test_proba, threshold, exp_name)

    payload = {
        **metadata,
        'threshold': round(float(threshold), 6),
        'oof_acc': round(float(accuracy(y_true, oof_proba >= threshold)), 5),
        'oof_path': oof_path,
        'pred_path': pred_path,
        'submission_path': submission_path,
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'[save] ensemble OOF → {oof_path}')
    print(f'[save] ensemble preds → {pred_path}')
    print(f'[save] ensemble summary → {summary_path}')
    return {
        'oof_path': oof_path,
        'pred_path': pred_path,
        'submission_path': submission_path,
        'summary_path': summary_path,
    }


def run_ensemble(experiments=None, blend_step=0.05, threshold=0.5, tune_threshold=True, n_splits=5):
    """阶段五主入口：执行单模型评估、加权 blend 和 LR stacking。"""
    seed_everything(SEED)
    if experiments is None:
        experiments = DEFAULT_EXPERIMENTS

    train_ids, test_ids, oof_matrix, test_matrix = build_prediction_matrices(experiments)
    train_raw, _ = load_raw()
    y = train_raw.set_index('PassengerId').loc[train_ids, 'Transported'].astype(int).to_numpy()

    print('=' * 60)
    print('Phase 5: Ensemble Pipeline')
    print('=' * 60)
    print(f'experiments: {experiments}')
    print(f'OOF matrix: {oof_matrix.shape}, test matrix: {test_matrix.shape}')

    model_rows = []
    for col in oof_matrix.columns:
        proba = oof_matrix[col].to_numpy(dtype=float)
        t, tuned_score = optimize_threshold(y, proba)
        model_rows.append({
            'method': col,
            'acc_0_5': round(float(accuracy(y, proba >= 0.5)), 5),
            'best_threshold': round(t, 6),
            'best_acc': round(tuned_score, 5),
        })
    print('\n[base models]')
    print(pd.DataFrame(model_rows).to_string(index=False))

    blend = search_blend_weights(
        y,
        oof_matrix,
        step=blend_step,
        threshold=threshold,
        tune_threshold=tune_threshold,
    )
    blend_test = test_matrix.to_numpy(dtype=float) @ blend['weights']
    blend_exp = 'blend_' + '_'.join(oof_matrix.columns)
    blend_paths = save_ensemble_outputs(
        blend_exp,
        train_ids,
        test_ids,
        blend['proba'],
        blend_test,
        y,
        blend['threshold'],
        {
            'method': 'weighted_blend',
            'experiments': experiments,
            'weights': {name: round(float(w), 6) for name, w in zip(oof_matrix.columns, blend['weights'])},
            'blend_step': blend_step,
            'threshold_tuned': tune_threshold,
            'base_model_scores': model_rows,
        },
    )

    stack = run_stacking(y, oof_matrix, test_matrix, n_splits=n_splits, seed=SEED)
    stack_exp = 'stack_lr_' + '_'.join(oof_matrix.columns)
    stack_paths = save_ensemble_outputs(
        stack_exp,
        train_ids,
        test_ids,
        stack['oof_proba'],
        stack['test_proba'],
        y,
        stack['threshold'],
        {
            'method': 'logistic_regression_stacking',
            'experiments': experiments,
            'n_splits': n_splits,
            'score_0_5': round(float(stack['score_0_5']), 5),
            'fold_scores': stack['fold_scores'],
            'base_model_scores': model_rows,
        },
    )

    results = pd.DataFrame([
        {
            'method': 'weighted_blend',
            'oof_acc': round(float(blend['score']), 5),
            'threshold': round(float(blend['threshold']), 6),
            **{f'w_{name}': round(float(w), 3) for name, w in zip(oof_matrix.columns, blend['weights'])},
        },
        {
            'method': 'logistic_regression_stacking',
            'oof_acc': round(float(stack['score']), 5),
            'threshold': round(float(stack['threshold']), 6),
        },
    ])
    print('\n[ensemble results]')
    print(results.to_string(index=False))
    return {
        'base_model_scores': model_rows,
        'blend': {**blend, 'paths': blend_paths},
        'stacking': {**stack, 'paths': stack_paths},
        'results': results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 5 ensemble pipeline')
    parser.add_argument(
        '--experiments',
        default=','.join(DEFAULT_EXPERIMENTS),
        help='Comma-separated experiment names present under outputs/oof and outputs/preds',
    )
    parser.add_argument('--blend-step', type=float, default=0.05, help='Blend weight grid step, e.g. 0.1, 0.05, 0.02')
    parser.add_argument('--threshold', type=float, default=0.5, help='Fixed threshold if --no-tune-threshold is set')
    parser.add_argument('--no-tune-threshold', action='store_true', help='Disable OOF threshold tuning')
    parser.add_argument('--folds', type=int, default=5, help='Stacking CV folds')
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = [e.strip() for e in args.experiments.split(',') if e.strip()]
    run_ensemble(
        experiments=experiments,
        blend_step=args.blend_step,
        threshold=args.threshold,
        tune_threshold=not args.no_tune_threshold,
        n_splits=args.folds,
    )


if __name__ == '__main__':
    main()
