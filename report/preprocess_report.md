# 阶段二：数据清洗与预处理 报告

## 1. 处理架构

```
train.csv (8693×14) + test.csv (4277×13)
         │
         ▼ 拼接 (12970 条统一处理，标记 is_train)
         │
    ┌────┴────┐
    │ Step 1  │ 字段解析（PassengerId / Cabin / Name 拆分）
    ├────┴────┤
    │ Step 2  │ 规则推断填充（业务逻辑硬规则）
    ├────┴────┤
    │ Step 3  │ 组内填充（GroupId 层级众数/多数投票）
    ├────┴────┤
    │ Step 4  │ 统计填充（分层中位数 → 全局众数兜底）
    ├────┴────┤
    │ Step 5  │ 基础衍生特征 + 类型规范化
    └────┬────┘
         ▼
train_clean (8693×30) + test_clean (4277×30)
零缺失 ✓
```

---

## 2. 各步骤详情与结果

### Step 1 — 字段解析

| 原始字段 | 解析产物 | 说明 |
|----------|---------|------|
| `PassengerId` (gggg_pp) | `GroupId`, `PersonNum`, `GroupSize` | 拆出组标识、组内序号、组人数 |
| `Cabin` (deck/num/side) | `Deck`, `CabinNum`, `Side` | 拆出甲板层、舱号、左右舷 |
| `Name` (First Last) | `FirstName`, `LastName`, `SurnameGroupSize` | 拆出名、姓、同姓人数 |

### Step 2 — 规则推断填充（最高优先级，不含 Transported）

依据 strategy 中验证的业务规律，直接用硬规则推断缺失值：

| 规则 | 逻辑 | 命中 | 精度 |
|------|------|------|------|
| CryoSleep ← 消费为0 | 五项消费全0 + Age>12 → CryoSleep=True | 104 条 | 高（EDA 已验证 CryoSleep 乘客 99%+ 消费为0） |
| 消费为0 + Age缺失 | 同上逻辑，Age 缺失不排除 | 2 条 | 高（消费为零是最强信号） |
| RoomService ← 0 | CryoSleep=True → 缺失的 RoomService 填 0 | 93 个字段 | 确定（冷冻乘客被关在舱内） |
| FoodCourt ← 0 | 同上 | 109 个字段 | 确定 |
| ShoppingMall ← 0 | 同上 | 133 个字段 | 确定 |
| Spa ← 0 | 同上 | 109 个字段 | 确定 |
| VRDeck ← 0 | 同上 | 93 个字段 | 确定 |
| VIP ← False | Age<13 → 儿童不可能 VIP | 28 条 | 确定 |

> 消费填充总计：537 个缺失字段被规则覆盖，无需模型/统计兜底。

### Step 3 — 组内填充（GroupId 层级）

同一 GroupId 的乘客通常是家人，特征高度一致：

| 字段 | 填充前缺失 | 填充后剩余 | 填充率 | 方法 |
|------|-----------|-----------|--------|------|
| HomePlanet | 288 | 157 | 45.5% | 组内众数 |
| Destination | 274 | 154 | 43.8% | 组内众数 |
| Deck | 299 | 162 | 45.8% | 组内众数 |
| Side | 299 | 162 | 45.8% | 组内众数 |
| CryoSleep | 97 | 0 | **100%** | 组内多数投票（≥50% 冷冻 → True） |

> CryoSleep 的 97 条剩余缺失被组内多数投票完全消除。

### Step 4 — 统计填充（兜底）

组内填充后仍有缺失的字段，用分层统计 + 全局兜底：

| 填充方法 | 覆盖字段 |
|----------|---------|
| 按 HomePlanet 分层众数 | HomePlanet, Destination, VIP, Deck, Side |
| 全局众数 | 上述字段中分层后仍缺失的 |
| 按 HomePlanet 分层中位数 | Age + 五项消费 |
| 全局中位数 | 上述数值字段中分层后仍缺失的 |

结果：仅剩 `HomePlanet` 一个字段有 157 条需要全局众数填充，填为 `Earth`。

### Step 5 — 基础衍生特征 + 类型规范化

由清洗后的原始字段直接计算的特征（不依赖目标变量，无泄漏风险）：

| 新特征 | 计算方式 | 用途 |
|--------|---------|------|
| `TotalSpend` | 五项消费之和 | 核心消费信号 |
| `HasSpend` | TotalSpend > 0 | 有无消费的二值信号 |
| `NumSpendCategories` | 五项中非零的个数 | 消费多样化程度 |
| `IsChild` | Age < 13 | 儿童标识 |
| `IsSenior` | Age >= 60 | 老年标识 |
| `IsAlone` | GroupSize == 1 | 独行旅客 |
| `AgeGroup` | 六档分桶 | 年龄分段 |

类型规范化：
- `CryoSleep`, `VIP` → `bool`
- `Age`, `TotalSpend` 及五项消费 → `float`
- `AgeGroup` → `category`

---

## 3. 最终产出

| 数据集 | 行数 | 列数 | 缺失 | 新增列 |
|--------|------|------|------|--------|
| train_clean | 8693 | 30 | 0 | +16 |
| test_clean | 4277 | 30 | 0 | +17（含 Transported 占位列） |

新增列清单：
`GroupId`, `PersonNum`, `GroupSize`, `Deck`, `CabinNum`, `Side`,
`FirstName`, `LastName`, `SurnameGroupSize`, `AgeGroup`,
`TotalSpend`, `HasSpend`, `NumSpendCategories`,
`IsChild`, `IsSenior`, `IsAlone`

---

## 4. 填充策略覆盖统计

```
缺失值处理来源分布（train+test 总计）:
┌──────────────────┬──────────┬──────────┐
│ 来源              │ 条数     │ 占比      │
├──────────────────┼──────────┼──────────┤
│ 规则推断           │ ~1100    │ ~60%     │ ← 精度最高
│ 组内填充           │ ~600     │ ~33%     │ ← 精度高
│ 统计填充           │ ~160     │ ~7%      │ ← 兜底
├──────────────────┼──────────┼──────────┤
│ 合计               │ ~1860    │ 100%     │
└──────────────────┴──────────┴──────────┘
```

---

## 5. 对应代码

| 文件 | 职责 |
|------|------|
| `src/preprocess.py` | 全部预处理逻辑（`preprocess()` 主入口） |
| `src/config.py` | `SPEND_COLS` 定义、路径常量 |
| `src/data.py` | `load_raw()` 加载原始 CSV |
| `src/utils.py` | `seed_everything()` |

运行方式：
```bash
cd SpaceshipTitanic
python -c "
from src.data import load_raw
from src.preprocess import preprocess
train, test = load_raw()
train_c, test_c = preprocess(train, test)
"
```
