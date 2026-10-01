"""聚合方法：多数投票（强基线）/ 随机 LF（弱基线）/ 金标上界 / 旗舰 AdaWS。"""

import numpy as np

from ..core.types import LabelModelResult


def _signed_votes(L):
    """把 L∈{-1弃权,0负类,1正类} 转成 {-1, 0, +1} 的带符号投票矩阵。

    必须把「投负类」映射为 −1 而不是 0：否则所有投票和恒为正，
    聚合器会退化成「恒定预测正类」（准确率≈正类比例，一个极易被忽略的陷阱）。
    """
    L = np.asarray(L, dtype=int)
    return np.where(L == 1, 1.0, np.where(L == 0, -1.0, 0.0))


def _mv_statistic(L):
    return _signed_votes(L).sum(axis=1)


class MajorityVote:
    """多数投票（强基线）：等权聚合，按票数符号决策，弃权记 0。"""

    name = "majority_vote"

    def fit(self, L, **kwargs):
        self.pi_ = float(np.clip(np.mean(_mv_statistic(L) > 0), 0.05, 0.95))
        return self

    def predict_proba(self, L):
        L = np.asarray(L, dtype=int)
        s = _mv_statistic(L)
        # 用票数做软化（避免平票处概率跳变），斜率为 1
        return 1.0 / (1.0 + np.exp(-np.clip(s, -60, 60)))

    def result(self, L) -> LabelModelResult:
        return LabelModelResult(
            method=self.name, probs=self.predict_proba(L),
            meta={"pi": getattr(self, "pi_", 0.5)},
        )


class RandomLF:
    """最弱基线：随机选一个 LF，直接采用它的投票（弃权处随机猜）。"""

    name = "random_lf"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def fit(self, L, **kwargs):
        rng = np.random.default_rng(self.seed)
        self.j_ = int(rng.integers(0, L.shape[1]))
        return self

    def predict_proba(self, L):
        L = np.asarray(L, dtype=int)
        col = L[:, self.j_]
        rng = np.random.default_rng(self.seed + 1)
        out = np.where(col == -1, rng.random(len(col)), col.astype(float))
        return np.clip(out, 1e-6, 1 - 1e-6)

    def result(self, L) -> LabelModelResult:
        return LabelModelResult(method=self.name, probs=self.predict_proba(L),
                                meta={"lf_index": self.j_})


class GoldOracle:
    """金标上界（仅作参照，不参与竞争）：直接用真实标签。"""

    name = "gold_oracle"

    def fit(self, L, y=None, **kwargs):
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict_proba(self, L):
        return np.clip(self.y_, 1e-6, 1 - 1e-6)

    def result(self, L) -> LabelModelResult:
        return LabelModelResult(method=self.name, probs=self.predict_proba(L),
                                meta={"note": "上界参照，不使用于排名"})


class AdaWS:
    """旗舰 AdaWS（Adaptive Weighted Supervision）。

    三步 + 两道护栏：
      1. 用 Dawid-Skene EM（多起点重启）在无金标条件下估计每个 LF 的准确率 α 与先验 π；
      2. 把准确率转成对数几率权重 w_j = log(α_j/(1−α_j))，并按覆盖率 β_j 做置信缩放；
      3. 争议门控：聚合后的 |logit| 低于阈值时（LF 意见分歧），把结果向先验收缩，
         避免在无共识样本上强行二分。

    护栏：
      · 非劣守护：若 EM 的观测数据似然低于「所有 LF 同质 α=0.5」的退化模型，
        判定准确率估计不可信，整体回退到多数投票；
      · 权重裁剪：|w_j| 上限截断，防止 α 估计接近 0/1 时单个 LF 主导。
    """

    name = "adaws"

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, restarts: int = 3,
                 seed: int = 0, max_weight: float = 4.0, gate_quantile: float = 0.25,
                 redundancy_aware: bool = False, redundancy_threshold: float = 0.3):
        self.max_iter = max_iter
        self.tol = tol
        self.restarts = restarts
        self.seed = seed
        self.max_weight = float(max_weight)
        self.gate_quantile = float(gate_quantile)
        # 实测：在 LF 多且弱（noisy_many_lf）时，大量伪相关会把权重压到接近 0，
        # 反而丢失信号，故默认关闭。该结论已写入消融报告（rejected components）。
        self.redundancy_aware = bool(redundancy_aware)
        self.redundancy_threshold = float(redundancy_threshold)
        self.fallback_reason_ = None

    def _redundancy_factor(self, L):
        """冗余感知收缩：相关/冗余的 LF 会被重复计数，按有效独立度降权。

        用带符号投票的相关矩阵度量 LF 间冗余：对每个 LF j，统计与它强相关的
        其他 LF 数量 r_j，收缩因子取 1/(1+r_j)（冗余团内的 LF 平分话语权）。
        这是 Dawid-Skene 条件独立假设失效时的实用修正。
        """
        Z = _signed_votes(L)
        both = (L != -1)
        m = Z.shape[1]
        corr = np.zeros((m, m))
        for j in range(m):
            for k in range(j + 1, m):
                mask = both[:, j] & both[:, k]
                if mask.sum() < 10:
                    continue
                a, b = Z[mask, j], Z[mask, k]
                sa, sb = a.std(), b.std()
                if sa < 1e-9 or sb < 1e-9:
                    continue
                c = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
                corr[j, k] = corr[k, j] = c
        redundant = np.abs(corr) > self.redundancy_threshold
        r = redundant.sum(axis=1).astype(float)
        return 1.0 / (1.0 + r)

    def fit(self, L, **kwargs):
        from .dawid_skene import DawidSkeneEM, _log_likelihood

        L = np.asarray(L, dtype=int)
        m = L.shape[1]
        em = DawidSkeneEM(max_iter=self.max_iter, tol=self.tol,
                          restarts=self.restarts, seed=self.seed)
        em.fit(L)
        alpha, pi = em.alpha_, em.pi_

        # --- 护栏 1：留出似然非劣守护 ---
        # 不能用样本内似然做判据：EM 在纯噪声数据上也能靠自由度凭空拟合出远离 0.5 的 α，
        # 样本内似然必然优于退化模型，判据永不触发。改用留出似然（泛化检查）：
        # 在一半样本上拟合、在另一半上评估；若 EM 的留出似然不优于「所有 LF 都是随机猜」
        # 的退化模型，说明 α 只是过拟合产物，准确率加权不可用 -> 回退多数投票。
        beta = em.beta_
        rng = np.random.default_rng(self.seed + 7919)
        idx = rng.permutation(L.shape[0])
        half = len(idx) // 2
        A, B = idx[:half], idx[half:]
        if half >= 10:
            em_a = DawidSkeneEM(max_iter=self.max_iter, tol=self.tol,
                                restarts=self.restarts, seed=self.seed).fit(L[A])
            ll_b_em = _log_likelihood(L[B], em_a.alpha_, em_a.beta_, em_a.pi_)
            ll_b_degen = _log_likelihood(L[B], np.full(m, 0.5), em_a.beta_, em_a.pi_)
            if not np.isfinite(ll_b_em) or ll_b_em <= ll_b_degen:
                self.fallback_reason_ = "heldout_loglik_not_better_than_degenerate"
                alpha = np.full(m, 0.5 + 1e-3)   # 退化为（近似）等权 = 多数投票
        self.alpha_ = alpha
        self.pi_ = pi
        self.beta_ = beta

        # --- 步骤 2：准确率 -> 对数几率权重，覆盖率缩放，冗余感知收缩，再裁剪 ---
        a = np.clip(alpha, 1e-3, 1 - 1e-3)
        w = np.log(a / (1 - a)) * np.clip(beta, 1e-3, 1.0)
        if self.redundancy_aware:
            w = w * self._redundancy_factor(L)
        self.weights_ = np.clip(w, -self.max_weight, self.max_weight)
        self.ll_ = float(em.log_likelihood_)
        self.ll_history_ = list(em.history_)

        # --- 步骤 3：争议门控阈值（按分位数在聚合统计量上标定）---
        logits = self._raw_logits(L)
        self.gate_ = float(np.quantile(np.abs(logits), self.gate_quantile))
        return self

    def _raw_logits(self, L):
        Z = _signed_votes(L)          # ±1 投票（弃权为 0）
        return Z @ self.weights_ + np.log(max(self.pi_, 1e-6) / max(1 - self.pi_, 1e-6))

    def predict_proba(self, L):
        L = np.asarray(L, dtype=int)
        logits = self._raw_logits(L)
        # 争议门控：|logit| 小于阈值时按线性比例向先验收缩（共识不足不强行二分）
        shrink = np.clip(np.abs(logits) / max(self.gate_, 1e-9), 0.0, 1.0)
        prior_logit = np.log(max(self.pi_, 1e-6) / max(1 - self.pi_, 1e-6))
        gated = prior_logit + shrink * (logits - prior_logit)
        return 1.0 / (1.0 + np.exp(-np.clip(gated, -60, 60)))

    def result(self, L) -> LabelModelResult:
        return LabelModelResult(
            method=self.name, probs=self.predict_proba(L),
            lf_accuracy=self.alpha_.copy(), log_likelihood=float(self.ll_),
            n_iter=len(self.ll_history_),
            meta={"weights": self.weights_.tolist(), "pi": float(self.pi_),
                  "gate": float(self.gate_), "fallback": self.fallback_reason_},
        )
