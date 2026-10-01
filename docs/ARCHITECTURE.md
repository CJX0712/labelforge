# LabelForge 架构说明

## 1. 分层与依赖方向

```
        cli.py
          │
      pipeline/pipeline.py  ── 编排 / 基准 / 门禁 / 消融
          │
   ┌──────┼──────────┬─────────────┐
   ▼      ▼          ▼             ▼
 data/  labelmodel/  eval/      models/
   │      │          │             │
   └──────┴──────────┴─────────────┘
                     │
                  core/   (类型/错误/配置/接口/种子)
```

**单向无环**：`core` 不依赖任何上层；`labelmodel` 只依赖 `core`，不反向依赖 `pipeline`。

## 2. 模块职责

| 模块 | 职责 |
|---|---|
| `core/types` | `Dataset` / `LabelModelResult` / `EvalReport` |
| `core/config` | `Config` + `LABELFORGE_*` 环境变量覆盖 + schema 校验 |
| `core/seed` | 全局确定性种子 |
| `core/errors` | 配置/模型错误类型 |
| `core/interfaces` | `LabelModel` / `LabelingFunctionBank` 抽象契约 |
| `data/synthetic` | 8 个生成器，**已知真实 LF 准确率 α 与覆盖率 β** |
| `labelmodel/dawid_skene` | Dawid-Skene EM（MLE/MAP 双模式）+ 对数似然 + 后验闭式 |
| `labelmodel/aggregators` | MajorityVote / RandomLF / GoldOracle / **AdaWS** |
| `eval/metrics` | 准确率 / F1 / 参数恢复误差 / EM 单调性违反计数 |
| `models` | 下游分类器（纯 numpy 逻辑回归 + sklearn 可选后端） |
| `pipeline` | 多 seed 基准、确定性校验、消融、失败案例派生 |

## 3. 生成模型与 EM

```
Y_i ~ Bernoulli(π)
第 j 个 LF 以概率 β_j 参与投票（弃权 = −1）
投票时 P(L_ij = Y_i) = α_j

E 步: q_i = P(Y_i=1 | L_i, θ)
M 步: α_j = Σ_i [q_i·1(L_ij=1) + (1−q_i)·1(L_ij=0)] / Σ_i 1(L_ij≠−1)
      π   = mean(q)
```

MAP 模式额外加 Beta 先验（默认强度 0 = 纯 MLE，原因见 README 第 4 节）。

**关键实现细节（踩过的坑）**：

1. **投票必须编码为 ±1**。若把「投负类」记作 0，所有投票和恒为正，聚合器退化为
   「恒定预测正类」——准确率恰好等于正类比例，极易被误当成"模型还行"。
2. **标签翻转对称性**。`(α, π, Y)` 与 `(1−α, 1−π, 1−Y)` 似然完全相同，EM 可能收敛到倒置解，
   预测整体翻转。本项目以多数投票方向为锚消解该对称性。
3. **护栏判据必须用留出似然**。样本内似然下 EM 永远优于退化模型（自由度更多），
   连纯随机数据都能凭空拟合出远离 0.5 的 α，判据形同虚设。

## 4. 旗舰 AdaWS

```
1. Dawid-Skene EM（多起点重启）估计 α 与 π
2. 权重 w_j = log(α_j/(1−α_j)) · β_j      （准确率 → 对数几率，按覆盖率缩放）
3. 争议门控：|聚合 logit| 低于分位阈值时向先验收缩（共识不足不强行二分）
护栏 A: 留出似然非劣守护（不优于随机猜则回退多数投票）
护栏 B: 权重裁剪 |w_j| ≤ 4，防止 α→0/1 时单个 LF 主导
```

消融显示：准确率加权贡献巨大（adversarial 上 +0.2708），争议门控贡献接近零
（+0.0042 / −0.0042，属噪声水平）——这一点如实写入消融报告。

## 5. 评测设计

- **主指标 = 聚合标签准确率**。下游准确率会被下游分类器的去噪能力压平（金标上界 0.96，
  各方法挤在 0.90 附近），不适合作为区分性主指标。
- **下游训练用软标签双样本加权**：把 (x, p) 拆成 (x, 1, w=p) 与 (x, 0, w=1−p)。
  既完整利用概率输出，又恒含两个类别，避免退化软标签导致训练失败。
- **门禁统计**：跨数据集配对差重尾 → 主检验单侧 Wilcoxon 符号秩；配对 t 同时披露。

## 6. 复杂度

| 阶段 | 复杂度 |
|---|---|
| EM 单次迭代 | `O(n·m)` |
| EM 全流程 | `O(restarts · max_iter · n · m)` |
| 冗余相关矩阵（默认关闭） | `O(m²·n)` |
| 下游训练 | `O(2n·d·iter)` |
