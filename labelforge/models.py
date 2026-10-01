"""下游分类器（被弱监督标签训练的模型）。

- NumpyLogisticModel：纯 numpy 逻辑回归，零依赖 = 离线兜底。
- sklearn LogisticRegression 为可选 Tier-0 后端（available_sklearn() 探测）。
- 支持软标签（概率）加权训练，这是标签模型相对硬标签多数投票的主要增益通道。
"""

import numpy as np

from .core.errors import ModelError


class NumpyLogisticModel:
    """纯 numpy 逻辑回归（标准化 + 梯度下降 + L2 + 支持样本权重）。"""

    name = "numpy_logistic"

    def __init__(self, lr: float = 0.5, n_iter: int = 800, l2: float = 1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0) + 1e-8
        Xs = (X - self.mean_) / self.scale_
        Xb = np.hstack([np.ones((n, 1)), Xs])
        w = np.zeros(Xb.shape[1])
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        sw = sw / max(sw.mean(), 1e-12)
        for _ in range(self.n_iter):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = Xb.T @ (sw * (p - y)) / n
            g[1:] += self.l2 * w[1:] / n
            w_new = w - self.lr * g
            if np.max(np.abs(w_new - w)) < 1e-8:
                w = w_new
                break
            w = w_new
        self.coef_scaled_ = w[1:]
        self.intercept_ = float(w[0])
        self.coef_ = self.coef_scaled_ / self.scale_
        self.intercept_raw_ = float(self.intercept_ - (self.coef_scaled_ / self.scale_) @ self.mean_)
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        z = Xs @ self.coef_scaled_ + self.intercept_
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


class SklearnLogistic:
    """sklearn 逻辑回归后端（可用时优先，便于与主流生态对齐）。"""

    name = "sklearn_logistic"

    def __init__(self, seed: int = 0, C: float = 1.0):
        self.seed = seed
        self.C = C

    def fit(self, X, y, sample_weight=None):
        if not available_sklearn():
            raise ModelError("scikit-learn 不可用")
        from sklearn.linear_model import LogisticRegression

        self._est = LogisticRegression(max_iter=2000, C=self.C, random_state=self.seed)
        self._est.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int),
                      sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self._est.predict(np.asarray(X, dtype=float)), dtype=int)

    def predict_proba(self, X):
        return np.asarray(self._est.predict_proba(np.asarray(X, dtype=float)), dtype=float)


def make_downstream(seed: int = 0, prefer_sklearn: bool = True):
    if prefer_sklearn and available_sklearn():
        return SklearnLogistic(seed=seed)
    return NumpyLogisticModel()
