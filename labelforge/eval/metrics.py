"""评测指标：标签准确率、下游准确率、F1、参数恢复误差。"""

import numpy as np


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def f1_binary(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def param_recovery_error(alpha_hat, alpha_true) -> float:
    """LF 准确率参数恢复误差（平均绝对偏差）。"""
    a = np.asarray(alpha_hat, dtype=float).ravel()
    t = np.asarray(alpha_true, dtype=float).ravel()
    return float(np.mean(np.abs(a - t))) if len(a) else float("nan")


def param_recovery_corr(alpha_hat, alpha_true) -> float:
    """估计准确率与真值准确率的 Spearman 相关（排序恢复能力）。"""
    from scipy.stats import spearmanr

    a = np.asarray(alpha_hat, dtype=float).ravel()
    t = np.asarray(alpha_true, dtype=float).ravel()
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    r = spearmanr(a, t).correlation
    return float(0.0 if np.isnan(r) else r)


def ll_monotonicity_violations(history, tol: float = 1e-8) -> int:
    """EM 对数似然历史中的单调性违反次数（EM 定理要求: 应为 0）。"""
    h = np.asarray(history, dtype=float)
    if h.size < 2:
        return 0
    return int(np.sum(h[1:] - h[:-1] < -tol))
