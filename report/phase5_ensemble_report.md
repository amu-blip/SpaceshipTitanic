# 阶段五：集成 报告

## 1. 目标

阶段五的目标是基于阶段四保存的 OOF 与测试集预测完成模型集成，覆盖策略文档中的三项核心内容：

1. **Blending**：对多个基模型的 OOF 概率搜索最优非负权重，权重和为 1。
2. **Stacking**：使用 LogisticRegression 作为二层模型，在基模型 OOF 概率矩阵上做交叉验证训练。
3. **种子平均兼容**：阶段四产出的 `*_2seed` 或未来 `*_5seed` 实验都可作为阶段五输入；阶段五通过 `--experiments` 接收任意同结构预测文件。

---

## 2. 输入文件

默认读取阶段四三模型产物：

| 实验名 | OOF | 测试集预测 |
|--------|-----|------------|
| `lgb_stratified_5fold_2seed` | `outputs/oof/lgb_stratified_5fold_2seed.csv` | `outputs/preds/lgb_stratified_5fold_2seed.csv` |
| `xgb_stratified_5fold_2seed` | `outputs/oof/xgb_stratified_5fold_2seed.csv` | `outputs/preds/xgb_stratified_5fold_2seed.csv` |
| `cat_stratified_5fold_2seed` | `outputs/oof/cat_stratified_5fold_2seed.csv` | `outputs/preds/cat_stratified_5fold_2seed.csv` |

所有输入文件必须包含：

- `PassengerId`
- `Transported_Prob`

阶段五会校验：

- 必需列是否存在
- `PassengerId` 是否重复
- 概率是否为空
- 概率是否在 `[0, 1]`
- OOF 是否覆盖全部训练集 PassengerId
- 测试预测是否覆盖 sample submission 的全部 PassengerId

---

## 3. 加权 Blending

实现位置：`src/ensemble.py`。

搜索方式：

- 使用网格枚举非负权重；例如 `--blend-step 0.05` 表示权重粒度为 5%。
- 所有权重满足 `sum(weights) = 1`。
- 对每组权重，计算 blended OOF 概率。
- 默认开启阈值搜索，在 `[0.35, 0.65]` 内以 `0.001` 步长搜索最佳 Accuracy 阈值。

运行示例：

```bash
cd src
python ensemble.py \
  --experiments lgb_stratified_5fold_2seed,xgb_stratified_5fold_2seed,cat_stratified_5fold_2seed \
  --blend-step 0.05
```

如果需要固定阈值 0.5：

```bash
cd src
python ensemble.py \
  --experiments lgb_stratified_5fold_2seed,xgb_stratified_5fold_2seed,cat_stratified_5fold_2seed \
  --blend-step 0.05 \
  --no-tune-threshold
```

---

## 4. LogisticRegression Stacking

实现方式：

- 输入矩阵：`[lgb_oof, xgb_oof, cat_oof]`
- 二层模型：`StandardScaler + LogisticRegression(C=1.0, solver='lbfgs')`
- 验证方式：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 测试集预测：每个 fold 的二层模型预测测试集概率，最后取平均。
- 阈值：同样在 OOF 上搜索最优阈值。

Stacking 的 OOF 由二层 CV 产生，避免直接在全量 OOF 上训练后又在同一批 OOF 上评估造成过度乐观。

---

## 5. 输出文件

阶段五会输出到两个目录：

| 目录 | 文件类型 | 说明 |
|------|----------|------|
| `outputs/ensemble/` | `*_oof.csv` | 集成后的训练集 OOF 概率、真实标签、阈值化预测 |
| `outputs/ensemble/` | `*_preds.csv` | 集成后的测试集概率 |
| `outputs/ensemble/` | `*_summary.json` | 方法、输入实验、权重、阈值、OOF accuracy、输出路径 |
| `outputs/submissions/` | `*.csv` | Kaggle 提交文件 |

Blending 输出名前缀：

```text
blend_<experiment_1>_<experiment_2>_...
```

Stacking 输出名前缀：

```text
stack_lr_<experiment_1>_<experiment_2>_...
```

---

## 6. CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiments` | `lgb_stratified_5fold_2seed,xgb_stratified_5fold_2seed,cat_stratified_5fold_2seed` | 用逗号分隔的基模型实验名 |
| `--blend-step` | `0.05` | blend 权重搜索步长 |
| `--threshold` | `0.5` | 关闭阈值搜索时使用的固定阈值 |
| `--no-tune-threshold` | `False` | 是否禁用 OOF 阈值搜索 |
| `--folds` | `5` | LogisticRegression stacking 的 CV 折数 |

---

## 7. 当前调试状态

本轮已完成：

- 阶段五代码入口 `src/ensemble.py`
- OOF/test 预测对齐与输入校验
- 加权 blend 网格搜索
- OOF 阈值搜索
- LogisticRegression stacking
- ensemble OOF、test 概率、summary JSON、submission 保存
- 静态语法检查

当前容器默认 Python 3.14 环境缺少 `numpy/pandas/scikit-learn` 等运行依赖，且 PyPI 拉取失败，因此未能在容器内执行完整阶段五运行。安装 `requirements.txt` 后建议执行：

```bash
cd src
python ensemble.py --blend-step 0.05
```

若阶段四后续补跑 5-seed LGB，可以将对应实验加入输入，例如：

```bash
cd src
python ensemble.py \
  --experiments lgb_stratified_5fold_5seed,xgb_stratified_5fold_2seed,cat_stratified_5fold_2seed \
  --blend-step 0.05
```
