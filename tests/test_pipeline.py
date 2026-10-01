"""pipeline / config / models / metrics 层测试：确定性、配置校验、端到端门禁。"""

import numpy as np
import pytest

from labelforge.core.config import Config, ConfigError
from labelforge.eval import metrics as E
from labelforge.models import NumpyLogisticModel, available_sklearn
from labelforge.pipeline.pipeline import ALL_METHODS, LabelForgePipeline, _train_downstream


def _cfg(**kw):
    base = dict(n_seeds=1, n_samples=300, n_features=8, em_max_iter=30, em_restarts=2)
    base.update(kw)
    return Config(**base).validate()


# ---------------- config ----------------

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LABELFORGE_SEED", "5")
    monkeypatch.setenv("LABELFORGE_N_LFS", "9")
    c = Config.from_env().validate()
    assert c.seed == 5 and c.n_lfs == 9


def test_config_rejects_bad_env(monkeypatch):
    monkeypatch.setenv("LABELFORGE_SEED", "xyz")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_config_validate():
    with pytest.raises(ConfigError):
        Config(n_seeds=0).validate()
    with pytest.raises(ConfigError):
        Config(test_ratio=1.0).validate()


# ---------------- metrics ----------------

def test_accuracy_and_f1():
    y = np.array([1, 1, 0, 0])
    assert E.accuracy(y, y) == 1.0
    assert E.accuracy(y, 1 - y) == 0.0
    assert abs(E.f1_binary(y, y) - 1.0) < 1e-12
    assert E.f1_binary(np.array([1, 1]), np.array([0, 0])) == 0.0


def test_param_recovery_metrics():
    a = np.array([0.9, 0.8, 0.3])
    assert E.param_recovery_error(a, a) == 0.0
    assert E.param_recovery_corr(a, a) == 1.0
    assert E.param_recovery_corr(a, -a) == -1.0


def test_ll_monotonicity_violations():
    assert E.ll_monotonicity_violations([1.0, 2.0, 3.0]) == 0
    assert E.ll_monotonicity_violations([1.0, 0.5, 3.0]) == 1


# ---------------- downstream model ----------------

def test_soft_label_training_handles_degenerate_labels():
    """软标签双样本加权：即使软标签全为 1（单类退化）也能训练完成。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    Xt = rng.standard_normal((20, 4))
    pred = _train_downstream(lambda: NumpyLogisticModel(n_iter=50), X,
                             np.ones(60), Xt, 0)
    assert pred.shape == (20,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_numpy_logistic_learns_separable_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3))
    y = (X @ np.array([2.0, -1.0, 0.5]) > 0).astype(int)
    clf = NumpyLogisticModel(n_iter=1500).fit(X, y)
    assert E.accuracy(y, clf.predict(X)) > 0.9


# ---------------- pipeline ----------------

def test_run_is_deterministic():
    p = LabelForgePipeline(_cfg())
    a, b = p.run(0), p.run(0)
    for d in a:
        for m in a[d]:
            assert a[d][m]["label_accuracy"] == b[d][m]["label_accuracy"]
            assert a[d][m]["downstream_accuracy"] == b[d][m]["downstream_accuracy"]


def test_benchmark_gate_passes_and_is_honest():
    """端到端门禁：系统须在多数投票强基线上显著胜出（主检验 Wilcoxon）。"""
    rep = LabelForgePipeline(_cfg(n_seeds=2)).benchmark()
    g = rep["performance_gate"]
    assert rep["determinism"]["identical"] is True
    assert g["significant"] is True, g
    assert g["system_macro_label_accuracy"] > g["majority_vote_macro_label_accuracy"]
    assert set(rep["datasets"]) == set(["uniform", "heterogeneous", "sparse_coverage",
                                        "correlated_lfs", "adversarial_lfs", "class_imbalanced",
                                        "high_dim_few_lf", "noisy_many_lf"])
    # 金标准：所有数据集上 EM 目标单调性零违反
    for d, dm in rep["per_dataset"].items():
        for m in ("dawid_skene", "adaws"):
            if "ll_monotonicity_violations" in dm.get(m, {}):
                assert dm[m]["ll_monotonicity_violations"] == 0, (d, m)


def test_benchmark_reports_all_methods_and_bounds():
    rep = LabelForgePipeline(_cfg()).benchmark()
    agg = rep["per_method_aggregate"]
    for m in ALL_METHODS:
        assert m in agg, m
    # 金标上界必须最高，随机基线最低
    assert agg["gold_oracle"]["label_accuracy"] >= 0.99
    assert (agg["random_lf"]["label_accuracy"]
            < agg["majority_vote"]["label_accuracy"])


def test_benchmark_values_finite():
    rep = LabelForgePipeline(_cfg()).benchmark()
    for m, v in rep["per_method_aggregate"].items():
        assert np.isfinite(v["label_accuracy"]), m
        assert 0.0 <= v["label_accuracy"] <= 1.0, m


def test_ablation_records_rejected_components():
    """消融必须同时记录被否决的组件（负结果不得隐藏）。"""
    rep = LabelForgePipeline(_cfg()).benchmark()
    ab = rep["ablation"]
    assert "rejected_components" in ab and len(ab["rejected_components"]) >= 2
    for row in ab["rejected_components"]:
        assert "redundancy_delta" in row and "smoothing_delta" in row


def test_explain_single_returns_param_recovery():
    out = LabelForgePipeline(_cfg()).explain_single("heterogeneous", "adaws")
    assert out["param_recovery_mae"] is not None
    assert 0.0 <= out["param_recovery_mae"] < 0.1
    assert len(out["estimated_lf_accuracy"]) == len(out["true_lf_accuracy"])
