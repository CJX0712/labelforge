"""弱监督合成数据生成器：特征 X、金标 y（仅评测用）、标注函数输出 L。

每个生成器**已知真实的 LF 准确率 α 与覆盖率 β**，用于「参数恢复」这一硬金标准：
标签模型在无金标条件下估计出的 α 必须与生成时用的真值吻合。
生成器固定 seed -> 逐位可复现。

L 的取值约定：{-1 弃权, 0 负类, 1 正类}。
"""

import numpy as np

from ..core.types import Dataset


def _rng(seed):
    return np.random.default_rng(seed)


def _base_xy(rng, n, d, prevalence=0.5, snr=1.0):
    """生成特征与金标 y。prevalence 控制正类比例，snr 控制信号强度。

    snr 决定「金标上界」的高低。若设得过高（如 2.0），下游分类器能凭特征结构
    自行修正标签噪声，各聚合方法的差距会被压缩到噪声水平（基准饱和、无区分度）。
    这里取 1.0，使金标上界落在 ~0.80，标签质量成为真正的瓶颈 —— 这正是弱监督
    方法要解决的问题设定。
    """
    w = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    logits = X @ w * snr
    thr = np.quantile(logits, 1.0 - prevalence)
    y = (logits >= thr).astype(int)
    return X, y


def _emit_lf(rng, y, alpha, coverage, group_src=None, copy_flip=0.0):
    """按 (准确率 alpha, 覆盖率 coverage) 生成单个 LF 输出列。

    group_src 非空时，该 LF 是源 LF 的噪声副本（模拟相关/冗余标注函数）。
    """
    n = len(y)
    base = group_src if group_src is not None else y
    vote = rng.random(n) < coverage
    agree = rng.random(n) < alpha
    out = np.where(agree, base, 1 - base)
    if group_src is not None and copy_flip > 0:
        flip = rng.random(n) < copy_flip
        out = np.where(flip, 1 - out, out)
    return np.where(vote, out, -1).astype(int)


def _assemble(rng, X, y, acc, cov, groups=None, copy_flip=0.0):
    m = len(acc)
    L = np.full((len(y), m), -1, dtype=int)
    for j in range(m):
        src = None
        if groups is not None:
            g = groups[j]
            if g is not None and j not in _group_leaders(groups):
                src = None  # 由 leader 填好后用其输出作为源
        L[:, j] = _emit_lf(rng, y, acc[j], cov[j], src, copy_flip)
    if groups is not None:
        # 组内非 leader 的 LF = leader 输出的噪声副本
        seen = {}
        for j in range(m):
            g = groups[j]
            if g is None:
                continue
            if g not in seen:
                seen[g] = j                       # leader
            else:
                lead = seen[g]
                L[:, j] = np.where(L[:, lead] == -1, -1,
                                   np.where(rng.random(len(y)) < copy_flip,
                                            1 - L[:, lead], L[:, lead])).astype(int)
    return L


def _group_leaders(groups):
    seen = set()
    leaders = set()
    for j, g in enumerate(groups):
        if g is None:
            continue
        if g not in seen:
            seen.add(g)
            leaders.add(j)
    return leaders


def _empirical_acc(L, y):
    """真实（经验）LF 准确率：在投票样本上与金标一致的比例。"""
    m = L.shape[1]
    acc = np.zeros(m)
    for j in range(m):
        voted = L[:, j] != -1
        acc[j] = float(np.mean(L[voted, j] == y[voted])) if voted.any() else 0.5
    return acc


def _build(X, y, L, name, meta=None):
    return Dataset(
        X=X, y=y, L=L, name=name,
        true_lf_accuracy=_empirical_acc(L, y),
        true_lf_propensity=(np.mean(L != -1, axis=0)),
        meta=meta or {},
    )


# ---------------- 数据集工厂 ----------------

def make_uniform(seed=0, n=800, d=12, m=12):
    """所有 LF 同质（准确率/覆盖率相同）—— 加权无法带来增益，方法应退化为多数投票。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = np.full(m, 0.70)
    cov = np.full(m, 0.80)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "uniform", {"note": "同质 LF，加权退化为 MV 才是正确行为"})


def make_heterogeneous(seed=0, n=800, d=12, m=12):
    """LF 质量差异大（0.55~0.95）—— 准确率加权应显著胜过多数投票。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = rng.uniform(0.55, 0.95, size=m)
    cov = rng.uniform(0.5, 1.0, size=m)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "heterogeneous", {"note": "LF 质量异质"})


def make_sparse_coverage(seed=0, n=800, d=12, m=12):
    """低覆盖率（0.15~0.45）—— 大量弃权，考验弃权处理与小样本 LF 估计稳定性。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = rng.uniform(0.65, 0.90, size=m)
    cov = rng.uniform(0.15, 0.45, size=m)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "sparse_coverage", {"note": "高弃权率"})


def make_correlated_lfs(seed=0, n=800, d=12, m=12, group_size=3, copy_flip=0.15):
    """组内 LF 互为噪声副本 —— 独立性假设被打破，朴素加权会重复计数。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = rng.uniform(0.65, 0.90, size=m)
    cov = rng.uniform(0.6, 1.0, size=m)
    groups = np.array([j // group_size for j in range(m)])
    L = _assemble(rng, X, y, acc, cov, groups=groups, copy_flip=copy_flip)
    return _build(X, y, L, "correlated_lfs",
                  {"note": f"每组 {group_size} 个相关 LF（副本翻转率 {copy_flip}）"})


def make_adversarial_lfs(seed=0, n=800, d=12, m=12):
    """混入反相关 LF（准确率 < 0.5）—— 必须识别并降权/反向利用，否则被带偏。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = np.empty(m)
    half = m // 2
    acc[:half] = rng.uniform(0.70, 0.90, size=half)
    acc[half:] = rng.uniform(0.15, 0.40, size=m - half)
    cov = rng.uniform(0.6, 1.0, size=m)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "adversarial_lfs", {"note": f"后 {m-half} 个 LF 反相关"})


def make_class_imbalanced(seed=0, n=800, d=12, m=12, prevalence=0.2):
    """类别不平衡（正类 20%）—— 先验估计与阈值选择成为关键。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d, prevalence=prevalence)
    acc = rng.uniform(0.60, 0.85, size=m)
    cov = rng.uniform(0.5, 0.9, size=m)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "class_imbalanced", {"note": f"正类比例 {prevalence}"})


def make_high_dim_few_lf(seed=0, n=600, d=50, m=5):
    """高维少 LF —— 特征多、标注源少，考验标签噪声对下游的影响。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = rng.uniform(0.60, 0.90, size=m)
    cov = rng.uniform(0.7, 1.0, size=m)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "high_dim_few_lf", {"note": f"d={d}, m={m}"})


def make_noisy_many_lf(seed=0, n=800, d=12, m=30):
    """大量弱 LF（准确率 0.50~0.65）—— 信号微弱，需要稳健聚合。"""
    rng = _rng(seed)
    X, y = _base_xy(rng, n, d)
    acc = rng.uniform(0.50, 0.65, size=m)
    cov = rng.uniform(0.5, 1.0, size=m)
    L = _assemble(rng, X, y, acc, cov)
    return _build(X, y, L, "noisy_many_lf", {"note": f"m={m} 个弱 LF"})
