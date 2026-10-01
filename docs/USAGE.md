# LabelForge 使用说明

## 安装

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt    # Windows
# source .venv/bin/pip install -r requirements.lock.txt  # Linux/macOS
pip install -e .          # 可选：安装 labelforge 命令
```

最低依赖仅 `numpy` + `scipy`；`scikit-learn` 为可选下游后端（缺失时自动用纯 numpy 逻辑回归）。

## 命令行

```bash
python -m labelforge.cli benchmark                                   # 全基准 -> benchmark.json
python -m labelforge.cli explain --dataset adversarial_lfs --method adaws
```

## Python API

```python
from labelforge.data import synthetic
from labelforge.labelmodel import DawidSkeneEM, AdaWS, MajorityVote

ds = synthetic.make_heterogeneous(seed=0)     # L ∈ {-1 弃权, 0 负类, 1 正类}

em = DawidSkeneEM(restarts=3, seed=0).fit(ds.L)   # 全程不接触 ds.y
print(em.alpha_)                              # 估计的每个 LF 准确率
print(em.pi_)                                 # 估计的正类先验
probs = em.predict_proba(ds.L)                # P(Y=1|L)

# 与真实 LF 准确率对照（仅评测用，训练不可见）
from labelforge.eval import metrics as E
print(E.param_recovery_error(em.alpha_, ds.true_lf_accuracy))
```

跑完整基准：

```python
from labelforge.core.config import Config
from labelforge.pipeline.pipeline import LabelForgePipeline

rep = LabelForgePipeline(Config(n_seeds=3).validate()).benchmark()
rep["performance_gate"]       # 门禁（Wilcoxon + 配对 t）
rep["ablation"]               # 组件贡献 + 被否决组件
rep["determinism"]            # 逐位一致性
```

## 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `LABELFORGE_SEED` | 42 | 全局种子 |
| `LABELFORGE_N_SEEDS` | 3 | 基准重复种子数 |
| `LABELFORGE_N_SAMPLES` | 800 | 样本数 |
| `LABELFORGE_N_FEATURES` | 12 | 特征维度 |
| `LABELFORGE_N_LFS` | 12 | 标注函数个数 |
| `LABELFORGE_EM_MAX_ITER` | 100 | EM 最大迭代 |
| `LABELFORGE_EM_RESTARTS` | 3 | EM 随机重启次数 |
| `LABELFORGE_PERF_BUDGET_SEC` | 60 | 端到端耗时预算 |

## 数据集

| 名称 | n | d | m | 检验点 |
|---|---|---|---|---|
| `uniform` | 800 | 12 | 12 | LF 同质（加权应退化为 MV） |
| `heterogeneous` | 800 | 12 | 12 | 质量参差 0.55~0.95 |
| `sparse_coverage` | 800 | 12 | 12 | 覆盖率仅 0.15~0.45 |
| `correlated_lfs` | 800 | 12 | 12 | 每 3 个 LF 互为噪声副本 |
| `adversarial_lfs` | 800 | 12 | 12 | 一半 LF 反相关（α 0.15~0.40） |
| `class_imbalanced` | 800 | 12 | 12 | 正类仅 20% |
| `high_dim_few_lf` | 600 | 50 | 5 | 高维少 LF |
| `noisy_many_lf` | 800 | 12 | 30 | 30 个弱 LF（α 0.50~0.65） |

## 结果解读

- `label_accuracy`：聚合标签 vs 金标（主指标）
- `downstream_accuracy`：用聚合标签训练的下游模型在测试集上的准确率（次指标）
- `param_recovery_mae` / `param_recovery_corr`：无金标估计的 LF 准确率 vs 生成真值
- `ll_monotonicity_violations`：EM 目标单调性违反次数（必须为 0）
- `performance_gate.wilcoxon_p_one_sided`：门禁主检验 p 值
