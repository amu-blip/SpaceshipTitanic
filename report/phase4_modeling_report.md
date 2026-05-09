# 阶段四：建模 报告

## 1. 训练架构

```
train_f (8693×63) + test_f (4277×63)
         │
         ▼  assemble 55 features from FEAT_GROUPS
         │  convert 11 categorical cols → category dtype
         ▼
   ╔══════════════════════════════════════╗
   ║  2 seeds × 5 folds = 10 training runs  ║
   ║  StratifiedKFold (shuffle per seed)    ║
   ╚══════════════════════════════════════╝
         │
         ├──→ LightGBM: num_leaves=63, lr=0.02, ES=100
         ├──→ XGBoost:  max_depth=6,  lr=0.02, ES=100
         └──→ CatBoost: depth=6,     lr=0.03, ES=100
         │
         ▼
   OOF predictions → outputs/oof/
   Test predictions → outputs/preds/
```

所有模型使用 `config.py` 中的默认超参（未调优），early_stopping=100 轮。

---

## 2. 单模型结果

### 2.1 10-Fold 汇总（2 seeds × 5 folds）

| Seed | Fold | LGB | XGB | CAT |
|------|------|-----|-----|-----|
| 42   | 1    | 0.83209 | 0.81771 | 0.82174 |
| 42   | 2    | 0.80909 | 0.80851 | 0.80909 |
| 42   | 3    | 0.82806 | 0.82116 | 0.82404 |
| 42   | 4    | 0.82969 | 0.82048 | 0.83487 |
| 42   | 5    | 0.80265 | 0.81358 | 0.80610 |
| 2024 | 1    | 0.82864 | 0.82634 | 0.82519 |
| 2024 | 2    | 0.80219 | 0.79413 | 0.80736 |
| 2024 | 3    | 0.80449 | 0.79758 | 0.80334 |
| 2024 | 4    | 0.81703 | 0.80725 | 0.81243 |
| 2024 | 5    | 0.82278 | 0.81991 | 0.81991 |

### 2.2 汇总指标

| 模型 | OOF Acc | CV Mean | CV Std | 与策略目标差距 |
|------|---------|---------|--------|--------------|
| **LightGBM** | **0.81525** | **0.81767** | ±0.01149 | +0.00567 ✓ |
| **XGBoost**  | **0.81525** | 0.81266 | ±0.01008 | +0.00066 ✓ |
| CatBoost  | 0.81456 | 0.81641 | ±0.00972 | +0.00441 ✓ |

> 策略目标：CV ≥ 0.812。三个模型均达标，LGB CV 最高（0.81767），LGB/XGB OOF 持平（0.81525）。  
> OOF Acc = 两 seed 的 OOF 概率平均后再阈值化的准确率，与 CV Mean（各 fold 独立计算后平均）含义不同。

---

## 3. 模型间相关性

| | LGB | XGB | CAT |
|------|------|------|------|
| LGB | 1.00 | 0.950 | 0.941 |
| XGB | — | 1.00 | **0.987** |
| CAT | — | — | 1.00 |

- **XGB vs CAT 几乎一致（0.987）**，两者在绝大多数样本上给出相同预测，集成价值有限
- **LGB vs CAT 相关性（0.941）**，略低于 LGB-XGB，是集成的最优搭档
- LGB vs XGB 相关性 0.950，差异有限

### 3.1 简单平均集成

| 集成方式 | OOF Acc | vs LGB 单模 |
|---------|---------|-----------|
| LGB only | 0.81525 | — |
| (LGB+XGB+CAT)/3 | 0.81514 | −0.00011 ↓ |
| (LGB+CAT)/2 | 0.81514 | −0.00011 ↓ |

三模型简单平均与 LGB 单模几乎持平（差距仅 0.0001），但未超越单模。**Phase 5 需要加权 blend 或 stacking** 来充分利用模型多样性。

---

## 4. 单模型分析

### 4.1 LightGBM（最佳单模）

- CV 均值最高（0.81767），但标准差也最大（0.01149），fold 间波动较大
- Fold 1（seed 42）达到 0.83209，Fold 5 仅 0.80265，需要关注分割质量
- 建议后续使用 GroupKFold 重新评估，消除同组泄漏导致的 CV 膨胀

### 4.2 CatBoost

- CV 0.81641，标准差最小（0.00972），最稳定
- 与 LGB 相关性 0.941，低于 XGB-CAT（0.987），是集成设计的最优搭档

### 4.3 XGBoost

- CV 垫底（0.81266），与 CAT 高度重叠（0.987），OOF 与 LGB 持平但 CV 更低
- **建议**：在 Phase 5 集成中降低 XGB 权重或考虑将其替换为其他模型（如 HistGradientBoosting 或 ExtraTrees）

---

## 5. 与竞争对手对比

| 维度 | 本方案 (LGB) | 对手方案 (最佳单模) |
|------|-------------|-------------------|
| CV 方法 | 2 seeds × 5 folds | 2 seeds × 5 folds |
| 单模 CV | 0.81767 (LGB) | 0.81353 (CatBoost) |
| 集成 CV | 待 Phase 5 | 0.81709 (LR stack) |
| 模型数 | 3 | 5 (ET+HGB+XGB+LGB+CAT) |
| CV/LB gap | 待提交 | ~0.016 |

我们的 LGB 单模 CV 已经超过对手的 CatBoost 单模（0.81767 vs 0.81353），且比对手集成 CV（0.81709）略高。**差距主要来自特征工程而非模型**。

---

## 6. Fold 波动分析

LGB 的 fold 得分范围 0.802–0.832（极差 0.030），部分 fold 得分显著偏低。可能原因：

1. **StratifiedKFold 不防止同组泄漏**：同一个 GroupId 的成员可能分散在 train/val 中，val 中的成员可从 train 中的同组特征获利，导致 CV 分数被人为抬高。但同时，某些 fold 可能恰巧分到了困难的组，导致 fold 分偏低。
2. **GroupKFold 应作为最终评估方案**：将同组成员完全放在同一 fold，CV 分数更能反映真实测试场景。

---

## 7. 产出文件

| 文件 | 内容 |
|------|------|
| `outputs/oof/lgb_stratified_5fold_2seed.csv` | LGB OOF 预测 (8693 条) |
| `outputs/oof/xgb_stratified_5fold_2seed.csv` | XGB OOF 预测 (8693 条) |
| `outputs/oof/cat_stratified_5fold_2seed.csv` | CAT OOF 预测 (8693 条) |
| `outputs/preds/lgb_stratified_5fold_2seed.csv` | LGB 测试集预测 (4277 条) |
| `outputs/preds/xgb_stratified_5fold_2seed.csv` | XGB 测试集预测 (4277 条) |
| `outputs/preds/cat_stratified_5fold_2seed.csv` | CAT 测试集预测 (4277 条) |

---

## 8. 对应代码

| 文件 | 职责 |
|------|------|
| `src/train.py` | 主训练流程：数据准备、CV 分裂、OOF/测试集保存、CLI 入口 |
| `src/models/lgb_model.py` | LightGBM 单 fold 训练（native categorical support） |
| `src/models/xgb_model.py` | XGBoost 单 fold 训练（`enable_categorical=True`） |
| `src/models/cat_model.py` | CatBoost 单 fold 训练（`cat_features` 参数） |
| `src/config.py` | 模型超参字典 (`LGB_PARAMS`, `XGB_PARAMS`, `CAT_PARAMS`) |

运行方式：
```bash
cd src

# 单模型
python train.py --models lgb --seeds 42,2024 --cv stratified

# 全部模型
python train.py --models lgb,xgb,cat --seeds 42,2024 --cv stratified

# GroupKFold（推荐最终评估用）
python train.py --models lgb --seeds 42 --cv group
```

---

## 9. Bug 修复记录

事后代码审查发现并修复两处 bug：

### Bug 1（Critical）：多 seed OOF 覆盖而非平均

**问题**：`run_cv()` 中 `oof_preds[val_idx] = result['y_val_pred']`，第二个 seed（2024）的预测完全覆盖第一个 seed（42）的 OOF，最终 OOF 只来自最后一个 seed。

**影响**：OOF 准确率反映单 seed 而非真实 2-seed 平均，对 XGB 影响最大（旧值 0.80904，修正后 0.81525）。

**修复**：改为累加后除以 seed 数量：
```python
oof_preds_sum[val_idx] += result['y_val_pred']
...
oof_preds = oof_preds_sum / len(seeds)
```

### Bug 2（Minor）：模型函数内部重复覆盖 seed

**问题**：`_get_model_params` 已向 `params` 注入 seed，但各模型训练函数内部又执行 `params['random_state'] = seed`，导致注入值被覆盖。

**修复**：删除三个模型文件中的 seed 覆盖行，seed 由 `_get_model_params` 统一管理。

---

## 10. Phase 5 建议

1. **加权 blend**：在 OOF 上搜索最优权重（`scipy.optimize.minimize`），以 LGB 为主（预期权重 0.6+），CAT 辅助（0.2-0.3），XGB 低权重或直接排除
2. **Stacking**：用 LogisticRegression 对 3 组 OOF 概率做元学习，可补充少量强特征（CryoSleep、TotalSpend）
3. **GroupKFold 重评**：用 GroupKFold 对所有模型重新评估，确认 CV 分数在防止组泄漏后仍达标
4. **阈值优化**：在 OOF 上搜索 0.35–0.65 最优阈值（对手方案关键技巧）

---

## 11. 阶段四补全与调试记录

本轮补全将阶段四训练入口从临时脚本式参数解析升级为可复现实验入口，覆盖策略文档中要求的 StratifiedKFold / GroupKFold 双验证方案、三模型训练、OOF/测试集预测保存和 fold 级指标留档。

### 11.1 训练入口补全

- CLI 改为 `argparse`，支持：
  - `--models lgb,xgb,cat`
  - `--cv stratified|group`
  - `--seeds 42,2024`
  - `--folds 5`
  - `--feature-groups base,id,cabin,...`
- 增加参数校验：非法模型名、非法 CV 方案、空 seed、`folds < 2` 会立即报错。
- 增加特征组校验：传入不存在的 feature group 时直接报错；配置中暂未产出的特征会以 warning 打印并跳过。
- `run_cv()` 现在返回每个实验的摘要，便于 notebook 或后续阶段直接复用。

### 11.2 CV 与随机种子修复

- 保留阶段四主方案：`StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`。
- 补全最终评估方案：`GroupKFold(n_splits=5)`，可通过 `--cv group` 运行，避免同一 `GroupId` 同时进入 train/validation。
- 每个 `seed/fold` 使用独立 `model_seed = seed * 100 + fold_idx` 注入模型参数，避免不同 fold 使用完全相同的模型随机态。
- 保留并强化多 seed OOF 平均逻辑：每个 seed 的 validation 预测先累加，最终除以 seed 数量，避免后一个 seed 覆盖前一个 seed。

### 11.3 指标与产物补全

除原有 OOF 与测试集预测外，阶段四现在额外保存：

| 输出目录 | 内容 |
|---------|------|
| `outputs/oof/` | 每个实验的 OOF 概率 |
| `outputs/preds/` | 每个实验的测试集概率 |
| `outputs/metrics/` | fold 级分数 CSV 与实验 summary JSON |

fold 级指标包含：模型名、CV 方案、seed、model_seed、fold、validation 样本数、accuracy、best_iteration。summary JSON 包含 OOF accuracy、CV mean/std、特征组、特征数量和输出文件路径。

### 11.4 模型配置补全

- XGBoost 默认参数补上 `tree_method='hist'`，与策略文档阶段四要求一致。
- LightGBM / XGBoost / CatBoost 单 fold 函数移除未使用的 `numpy` 导入和冗余 `seed` 参数，模型随机性统一由 `train.py` 注入后的参数控制。
- `save_oof()` 与 `save_preds()` 现在返回保存路径，便于训练摘要和 metrics JSON 记录完整产物链路。

### 11.5 运行方式（补全后）

```bash
cd src

# 单模型 StratifiedKFold
python train.py --models lgb --seeds 42,2024 --folds 5 --cv stratified

# 三模型 StratifiedKFold（阶段四主结果）
python train.py --models lgb,xgb,cat --seeds 42,2024 --folds 5 --cv stratified

# GroupKFold 最终评估
python train.py --models lgb,xgb,cat --seeds 42,2024 --folds 5 --cv group

# 只跑部分特征组做消融
python train.py --models lgb --feature-groups base,id,cabin,spend --seeds 42 --folds 5 --cv stratified
```

> 环境说明：当前容器默认 Python 3.14 环境未安装 `numpy/pandas/scikit-learn/lightgbm/xgboost/catboost`，且网络访问 PyPI 失败，因此本轮完成了静态语法检查与代码路径审查；完整训练需先按 `requirements.txt` 安装依赖后运行上述命令。
