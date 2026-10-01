"""Dawid-Skene 标签模型（纯 numpy EM，无金标）。

生成模型（二分类）：
  Y_i ~ Bernoulli(π)
  第 j 个 LF 以概率 β_j 参与投票（弃权记作 -1）
  投票时 P(L_ij = Y_i) = α_j（准确率）

EM:
  E 步: q_i = P(Y_i=1 | L_i, θ)
  M 步: α_j, β_j, π 的加权闭式更新

硬不变量：观测数据对数似然 ℓ(θ) = Σ_i log Σ_y P(Y=y)Π_j P(L_ij|Y=y) 逐轮单调不减（EM 定理）。
"""

import numpy as np

from ..core.errors import ModelError
from ..core.types import LabelModelResult


def _log_likelihood(L, alpha, beta, pi):
    """观测数据对数似然（对 Y 求边际）。数值上按行做 log-sum-exp。"""
    n, m = L.shape
    voted = L != -1
    # 对每个 LF: log P(L_ij | Y=y)
    # y=1: 若 L=1 -> alpha ; 若 L=0 -> 1-alpha
    a = np.clip(alpha, 1e-9, 1 - 1e-9)[None, :]
    lp1 = np.where(voted, np.where(L == 1, np.log(a), np.log(1 - a)), 0.0)
    lp0 = np.where(voted, np.where(L == 1, np.log(1 - a), np.log(a)), 0.0)
    s1 = lp1.sum(axis=1) + np.log(max(pi, 1e-9))
    s0 = lp0.sum(axis=1) + np.log(max(1 - pi, 1e-9))
    mx = np.maximum(s1, s0)
    return float(np.sum(mx + np.log(np.exp(s1 - mx) + np.exp(s0 - mx))))


def _posterior(L, alpha, beta, pi):
    """q_i = P(Y_i=1 | L_i, θ)（逐行闭式，beta 不进入行内后验）。"""
    voted = L != -1
    a = np.clip(alpha, 1e-9, 1 - 1e-9)[None, :]
    lp1 = np.where(voted, np.where(L == 1, np.log(a), np.log(1 - a)), 0.0)
    lp0 = np.where(voted, np.where(L == 1, np.log(1 - a), np.log(a)), 0.0)
    s1 = lp1.sum(axis=1) + np.log(max(pi, 1e-9))
    s0 = lp0.sum(axis=1) + np.log(max(1 - pi, 1e-9))
    d = s1 - s0
    return 1.0 / (1.0 + np.exp(-np.clip(d, -60, 60)))


class DawidSkeneEM:
    """Dawid-Skene EM 标签模型。"""

    name = "dawid_skene"

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, restarts: int = 3,
                 seed: int = 0, init_alpha: float = 0.75, alpha_smoothing: float = 0.0):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.restarts = int(restarts)
        self.seed = int(seed)
        self.init_alpha = float(init_alpha)
        # MAP-EM：以 Beta 先验把 α 向 0.5 收缩（伪计数），默认 0 = 纯 MLE。
        # 实测：开启收缩会削弱「反相关 LF」的关键信号（把 α<0.5 拉向 0.5，
        # 其负向投票权重趋近 0），在 adversarial_lfs 上显著变差，故默认关闭。
        # 该结论已写入消融报告（rejected components）。
        self.alpha_smoothing = float(alpha_smoothing)
        self.alpha_ = None
        self.beta_ = None
        self.pi_ = None
        self.log_likelihood_ = None
        self.history_ = []

    def fit(self, L, **kwargs):
        L = np.asarray(L, dtype=int)
        n, m = L.shape
        rng = np.random.default_rng(self.seed)
        voted = L != -1
        self.beta_ = voted.mean(axis=0)                     # 覆盖率可闭式估计，与 Y 无关
        mv = _mv_statistic(L)
        pi0 = float(np.clip(np.mean(mv > 0), 0.05, 0.95))   # 用多数投票初始化先验

        best = None
        for r in range(self.restarts):
            if r == 0:
                alpha = np.full(m, self.init_alpha)
            else:
                alpha = np.clip(rng.uniform(0.55, 0.95, size=m), 1e-3, 1 - 1e-3)
            pi = pi0
            prev_ll = -np.inf
            hist = []
            for _ in range(self.max_iter):
                q = _posterior(L, alpha, self.beta_, pi)
                # M 步：加权闭式
                num = np.zeros(m)
                den = np.zeros(m)
                for j in range(m):
                    v = voted[:, j]
                    agree = (L[v, j] == 1)
                    num[j] = np.sum(np.where(agree, q[v], 1 - q[v]))
                    den[j] = float(np.sum(v))
                # MAP 更新：向 0.5 收缩 alpha_smoothing 个伪计数
                kappa = self.alpha_smoothing
                alpha = (num + kappa * 0.5) / (np.maximum(den, 1e-12) + kappa)
                alpha = np.where(den > 0, alpha, 0.5)
                alpha = np.clip(alpha, 1e-3, 1 - 1e-3)
                pi = float(np.clip(np.mean(q), 1e-3, 1 - 1e-3))
                # MAP 目标 = 观测数据对数似然 + Beta 先验对数密度（不含常数项）
                ll = _log_likelihood(L, alpha, self.beta_, pi)
                c = 0.5 * self.alpha_smoothing
                obj = ll + c * float(np.sum(np.log(alpha) + np.log(1 - alpha)))
                hist.append(obj)
                if obj - prev_ll < self.tol:
                    prev_ll = obj
                    break
                prev_ll = obj
            if best is None or prev_ll > best[0]:
                best = (prev_ll, alpha.copy(), pi, hist)
        if best is None:
            raise ModelError("EM 拟合失败")
        ll, alpha, pi, hist = best

        # ---- 标签翻转识别（label switching）----
        # 生成模型对 (α, π, Y) 与 (1−α, 1−π, 1−Y) 给出完全相同的似然，
        # 因此 EM 可能收敛到「整体倒置」的解 —— 此时所有预测被翻转，准确率跌到 1−acc。
        # 必须以外部锚点消解该对称性：取与多数投票方向一致的解
        # （这是无金标条件下唯一可用的先验：假定 LF 总体上比随机猜更准）。
        self.flipped_ = False
        mv_hard = (_mv_statistic(L) > 0).astype(int)
        q_hard = (_posterior(L, alpha, self.beta_, pi) >= 0.5).astype(int)
        if len(mv_hard) and float(np.mean(q_hard == mv_hard)) < 0.5:
            alpha = 1.0 - alpha
            pi = 1.0 - pi
            self.flipped_ = True

        self.log_likelihood_, self.alpha_, self.pi_, self.history_ = ll, alpha, pi, hist
        return self

    def predict_proba(self, L):
        L = np.asarray(L, dtype=int)
        return _posterior(L, self.alpha_, self.beta_, self.pi_)

    def result(self, L) -> LabelModelResult:
        return LabelModelResult(
            method=self.name, probs=self.predict_proba(L),
            lf_accuracy=self.alpha_.copy(), log_likelihood=float(self.log_likelihood_),
            n_iter=len(self.history_),
            meta={"pi": float(self.pi_), "beta": self.beta_.tolist(),
                  "ll_history": [float(h) for h in self.history_],
                  "label_flip_corrected": bool(self.flipped_)},
        )


def _mv_statistic(L):
    """多数投票统计量：Σ_j 带符号投票（负类 −1，弃权 0，正类 +1）。"""
    L = np.asarray(L, dtype=int)
    Z = np.where(L == 1, 1.0, np.where(L == 0, -1.0, 0.0))
    return Z.sum(axis=1)
