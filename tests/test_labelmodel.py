"""标签模型公理级不变量测试（金标准）。

不写「跑通就算过」的烟测，只写可独立交叉验证的硬不变量：
  · EM 单调性        MAP 目标逐轮不减（EM 定理）
  · 参数可恢复性     无金标估计出的 LF 准确率须逼近生成真值
  · 同质 LF 退化     LF 准确率全相同时，加权聚合应退化为多数投票
  · 对称性           两条完全相同的 LF 列须得到完全相同的估计准确率
  · 反相关识别       准确率 < 0.5 的 LF 须被识别（估计值也 < 0.5）
  · 弃权不变性       追加一列全弃权 LF 不应改变聚合结果
  · 数值独立性       EM 与 scipy 数值优化（不同算法）应收敛到同一组参数
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from labelforge.data import DATASET_FACTORY
from labelforge.eval import metrics as E
from labelforge.labelmodel import AdaWS, DawidSkeneEM, MajorityVote
from labelforge.labelmodel.dawid_skene import _log_likelihood

ALL = list(DATASET_FACTORY.keys())


def _ds(name, seed=0):
    return DATASET_FACTORY[name](seed)


# ---------------- 数据层 ----------------

@pytest.mark.parametrize("name", ALL)
def test_dataset_deterministic(name):
    a, b = _ds(name, 3), _ds(name, 3)
    assert np.array_equal(a.L, b.L) and np.array_equal(a.y, b.y)
    assert np.array_equal(a.true_lf_accuracy, b.true_lf_accuracy)


@pytest.mark.parametrize("name", ALL)
def test_dataset_shapes_and_lf_domain(name):
    ds = _ds(name)
    n, m = ds.L.shape
    assert ds.y.shape == (n,) and ds.X.shape[0] == n
    assert set(np.unique(ds.L)).issubset({-1, 0, 1})
    assert set(np.unique(ds.y)).issubset({0, 1})
    assert np.all((ds.true_lf_accuracy >= 0) & (ds.true_lf_accuracy <= 1))


def test_empirical_lf_accuracy_matches_design_on_uniform():
    """uniform 数据集设计准确率 0.70，经验统计应逼近该值。"""
    ds = _ds("uniform", 0)
    assert abs(float(np.mean(ds.true_lf_accuracy)) - 0.70) < 0.05


def test_adversarial_dataset_has_subchance_lfs():
    ds = _ds("adversarial_lfs", 0)
    assert np.any(ds.true_lf_accuracy < 0.5), ds.true_lf_accuracy


# ---------------- EM 公理 ----------------

@pytest.mark.parametrize("name", ALL)
def test_em_objective_is_monotone(name):
    """EM 定理：MAP 目标逐轮单调不减（违反次数必须为 0）。"""
    ds = _ds(name, 1)
    em = DawidSkeneEM(restarts=1, seed=0, max_iter=60).fit(ds.L)
    assert E.ll_monotonicity_violations(em.history_) == 0, em.history_[-5:]


@pytest.mark.parametrize("name", ["heterogeneous", "adversarial_lfs", "sparse_coverage"])
def test_parameter_recovery(name):
    """参数可恢复性：估计的 LF 准确率须逼近生成真值（MAE 小且排序高度一致）。"""
    ds = _ds(name, 0)
    em = DawidSkeneEM(restarts=3, seed=0).fit(ds.L)
    mae = E.param_recovery_error(em.alpha_, ds.true_lf_accuracy)
    corr = E.param_recovery_corr(em.alpha_, ds.true_lf_accuracy)
    assert mae < 0.05, (mae, em.alpha_, ds.true_lf_accuracy)
    assert corr > 0.85, corr


@pytest.mark.parametrize("name", ["heterogeneous", "adversarial_lfs"])
def test_em_matches_independent_numerical_optimizer(name):
    """交叉验证：EM 与 scipy L-BFGS（完全不同的算法）应收敛到同一组 α。"""
    ds = _ds(name, 0)
    L = ds.L
    m = L.shape[1]
    em = DawidSkeneEM(restarts=5, seed=0, max_iter=300).fit(L)
    beta, pi = em.beta_, em.pi_

    def neg_obj(theta):                    # alpha 用 logit 参数化保证在 (0,1)
        alpha = 1.0 / (1.0 + np.exp(-theta))
        return -_log_likelihood(L, alpha, beta, pi)

    best = None
    for s in range(3):
        rng = np.random.default_rng(s)
        x0 = rng.uniform(-1.0, 1.0, size=m)
        r = minimize(neg_obj, x0, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    alpha_opt = 1.0 / (1.0 + np.exp(-best.x))
    assert np.allclose(np.sort(em.alpha_), np.sort(alpha_opt), atol=0.05), (em.alpha_, alpha_opt)


def test_symmetry_identical_lf_columns():
    """对称性：两条完全相同的 LF 列须得到完全相同的估计准确率。"""
    ds = _ds("heterogeneous", 0)
    L = np.hstack([ds.L, ds.L[:, [0]]])     # 复制第 0 列
    em = DawidSkeneEM(restarts=2, seed=0).fit(L)
    assert abs(em.alpha_[0] - em.alpha_[-1]) < 1e-9, em.alpha_


def test_abstain_only_column_is_ignored():
    """弃权不变性：追加一列全弃权 LF 不改变聚合结果。"""
    ds = _ds("heterogeneous", 0)
    before = DawidSkeneEM(restarts=2, seed=0).fit(ds.L).predict_proba(ds.L)
    L2 = np.hstack([ds.L, np.full((len(ds.L), 1), -1, dtype=int)])
    after = DawidSkeneEM(restarts=2, seed=0).fit(L2).predict_proba(L2)
    assert np.allclose(before, after, atol=1e-6)


# ---------------- 聚合口径 ----------------

def test_signed_vote_encoding_prevents_constant_prediction():
    """回归测试：投票必须编码为 ±1，否则聚合统计量恒为正 -> 恒定预测正类。"""
    ds = _ds("uniform", 0)
    p = MajorityVote().fit(ds.L).predict_proba(ds.L)
    assert p.min() < 0.4 and p.max() > 0.6, (p.min(), p.max())
    acc = E.accuracy(ds.y, (p >= 0.5).astype(int))
    assert acc > 0.75, acc


def test_majority_vote_beats_random_lf():
    from labelforge.labelmodel import RandomLF

    ds = _ds("heterogeneous", 0)
    mv = E.accuracy(ds.y, (MajorityVote().fit(ds.L).predict_proba(ds.L) >= 0.5).astype(int))
    rl = E.accuracy(ds.y, (RandomLF(seed=0).fit(ds.L).predict_proba(ds.L) >= 0.5).astype(int))
    assert mv > rl


def test_homogeneous_lfs_make_weighting_degenerate_to_mv():
    """同质 LF：所有准确率相同 -> 加权聚合在数学上退化为多数投票（预测应一致）。"""
    ds = _ds("uniform", 0)
    mv_pred = (MajorityVote().fit(ds.L).predict_proba(ds.L) >= 0.5).astype(int)
    aw = AdaWS(seed=0, gate_quantile=0.0).fit(ds.L)     # 关闭门控以隔离加权效应
    aw.weights_ = np.full_like(aw.weights_, 1.0)        # 等权
    aw_pred = (aw.predict_proba(ds.L) >= 0.5).astype(int)
    agreement = float(np.mean(mv_pred == aw_pred))
    assert agreement > 0.99, agreement


def test_adaws_recovers_anti_correlated_lfs():
    """反相关识别：真值 α<0.5 的 LF，估计值也必须 < 0.5（权重为负）。"""
    ds = _ds("adversarial_lfs", 0)
    aw = AdaWS(seed=0).fit(ds.L)
    bad = ds.true_lf_accuracy < 0.5
    assert np.all(aw.alpha_[bad] < 0.5), aw.alpha_[bad]
    assert np.all(aw.weights_[bad] < 0), aw.weights_[bad]


def test_adaws_falls_back_when_em_uninformative():
    """非劣守护：EM 似然不优于退化模型时，AdaWS 应记录回退原因。"""
    rng = np.random.default_rng(0)
    n, m = 200, 6
    L = rng.choice([-1, 0, 1], size=(n, m))      # 纯随机噪声，无可用信号
    aw = AdaWS(seed=0).fit(L)
    assert aw.fallback_reason_ == "heldout_loglik_not_better_than_degenerate", aw.fallback_reason_


@pytest.mark.parametrize(
    "name", ["uniform", "heterogeneous", "sparse_coverage", "correlated_lfs",
             "adversarial_lfs", "class_imbalanced", "noisy_many_lf"]
)
def test_guard_does_not_fire_on_real_signal(name):
    """反向检查：有真实信号的数据集上，护栏不得误触发回退。"""
    ds = _ds(name, 0)
    aw = AdaWS(seed=0).fit(ds.L)
    assert aw.fallback_reason_ is None, (name, aw.fallback_reason_)


@pytest.mark.parametrize("name", ALL)
def test_label_switching_is_anchored(name):
    """标签翻转识别：EM 有 (α,π,Y)↔(1−α,1−π,1−Y) 的似然对称性，
    若收敛到倒置解会把预测整体翻转。须与多数投票方向保持一致。"""
    ds = _ds(name, 0)
    em = DawidSkeneEM(restarts=3, seed=0).fit(ds.L)
    mv_hard = (MajorityVote().fit(ds.L).predict_proba(ds.L) >= 0.5).astype(int)
    em_hard = (em.predict_proba(ds.L) >= 0.5).astype(int)
    assert float(np.mean(em_hard == mv_hard)) >= 0.5, (name, em.alpha_)


@pytest.mark.parametrize("name", ALL)
def test_em_label_accuracy_safely_above_chance(name):
    """端到端护栏：任何数据集上 EM 标签准确率都必须显著高于随机（防止翻转未被纠正）。"""
    ds = _ds(name, 0)
    em = DawidSkeneEM(restarts=3, seed=0).fit(ds.L)
    acc = E.accuracy(ds.y, (em.predict_proba(ds.L) >= 0.5).astype(int))
    assert acc > 0.6, (name, acc)


def test_determinism_same_seed():
    ds = _ds("heterogeneous", 2)
    a = DawidSkeneEM(restarts=3, seed=7).fit(ds.L).predict_proba(ds.L)
    b = DawidSkeneEM(restarts=3, seed=7).fit(ds.L).predict_proba(ds.L)
    assert np.array_equal(a, b)


def test_probabilities_are_valid():
    ds = _ds("sparse_coverage", 0)
    for model in (MajorityVote(), DawidSkeneEM(seed=0), AdaWS(seed=0)):
        p = model.fit(ds.L).predict_proba(ds.L)
        assert np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p))
