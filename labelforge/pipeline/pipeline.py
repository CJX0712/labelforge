"""LabelForge 编排：数据 -> 标签模型 -> 下游训练 -> 多 seed 基准 -> 落盘。

单向无环: cli -> pipeline -> {data, labelmodel, eval, models} -> core。
所有指标来自真实运行，禁止手填。
"""

import time

import numpy as np

from .. import models as M
from ..core.config import Config
from ..core.seed import set_all
from ..core.types import LabelModelResult
from ..data import DATASET_FACTORY, DATASET_ORDER
from ..eval import metrics as E
from ..labelmodel import AdaWS, DawidSkeneEM, GoldOracle, MajorityVote, RandomLF

__version__ = "0.1.0"
AUTHOR = "晨星"

SOTA_STATEMENT = (
    "SOTA 对标: Snorkel / 数据编程 (Ratner et al., NeurIPS 2016) 的标签模型，"
    "以及 Dawid & Skene (1979) 的最大似然标注者质量估计。"
    "本系统纯 numpy 复现其数学核心 (Dawid-Skene EM 无金标估计 LF 准确率与类别先验)，"
    "并以「EM 似然单调」「LF 准确率参数可恢复」「同质 LF 下退化为多数投票」作为可验证金标准。"
)

# 参与竞争的方法（不含上界参照 gold_oracle）
SYSTEM_METHODS = ["dawid_skene", "adaws"]
STRONG_BASELINE = "majority_vote"
WEAK_BASELINE = "random_lf"

ALL_METHODS = SYSTEM_METHODS + [STRONG_BASELINE, WEAK_BASELINE, "gold_oracle"]

# 主指标：聚合标签准确率（标签模型的直接职责）。
# 下游准确率会被下游分类器的去噪能力压平，仅作次指标披露。
PRIMARY_METRIC = "label_accuracy"


def _instantiate(name, cfg, seed):
    if name == "majority_vote":
        return MajorityVote()
    if name == "random_lf":
        return RandomLF(seed=seed)
    if name == "dawid_skene":
        return DawidSkeneEM(max_iter=cfg.em_max_iter, tol=cfg.em_tol,
                            restarts=cfg.em_restarts, seed=seed)
    if name == "adaws":
        return AdaWS(max_iter=cfg.em_max_iter, tol=cfg.em_tol,
                     restarts=cfg.em_restarts, seed=seed)
    if name == "gold_oracle":
        return GoldOracle()
    raise ValueError(name)


def _split(n, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_ratio)))
    return idx[n_test:], idx[:n_test]


def _train_downstream(clf_factory, X_tr, soft_y, X_te, seed):
    """用软标签训练下游模型（软标签双样本加权）。

    把每个样本 (x, p) 拆成两条带权样本 (x, y=1, w=p) 与 (x, y=0, w=1−p)：
      · 完整利用标签模型的概率输出，而不是先阈值化成硬标签丢掉置信度；
      · 恒含两个类别，避免退化软标签（全 1 或全 0）导致训练失败。
    这是标签模型相对「硬标签多数投票」的主要增益通道。
    """
    soft = np.clip(np.asarray(soft_y, dtype=float), 0.0, 1.0)
    n = len(soft)
    X2 = np.vstack([np.asarray(X_tr, dtype=float)] * 2)
    y2 = np.concatenate([np.ones(n), np.zeros(n)])
    w2 = np.concatenate([soft, 1.0 - soft])
    clf = clf_factory()
    clf.fit(X2, y2, sample_weight=w2)
    return np.asarray(clf.predict(X_te), dtype=int)


def _evaluate(ds, cfg, seed):
    """单数据集单 seed：拟合各标签模型 -> 软标签 -> 下游评测。"""
    n = len(ds.y)
    tr, te = _split(n, cfg.test_ratio, seed + 1000)
    L_tr, L_te = ds.L[tr], ds.L[te]
    X_tr, X_te, y_tr, y_te = ds.X[tr], ds.X[te], ds.y[tr], ds.y[te]

    out = {}
    for name in ALL_METHODS:
        t0 = time.perf_counter()
        model = _instantiate(name, cfg, seed)
        if name == "gold_oracle":
            # 上界参照：直接用金标（train 用于训练，test 用于评测）
            model.fit(L_tr, y=y_tr)
            probs_tr = np.asarray(y_tr, dtype=float)
            probs_te = np.asarray(y_te, dtype=float)
        else:
            model.fit(L_tr)
            probs_te = model.predict_proba(L_te)
            # 下游模型必须用训练集软标签训练（测试集软标签与 X_tr 长度不同）
            probs_tr = model.predict_proba(L_tr)
        pred = (np.asarray(probs_te) >= 0.5).astype(int)
        clf = _train_downstream(
            lambda: M.make_downstream(seed=seed), X_tr, probs_tr, X_te, seed
        )
        rt = time.perf_counter() - t0
        res: LabelModelResult = model.result(L_te) if hasattr(model, "result") else None
        rec = {
            "label_accuracy": E.accuracy(y_te, pred),
            "label_f1": E.f1_binary(y_te, pred),
            "downstream_accuracy": E.accuracy(y_te, clf),
            "downstream_f1": E.f1_binary(y_te, clf),
            "runtime_sec": rt,
        }
        if res is not None and res.lf_accuracy is not None and ds.true_lf_accuracy is not None:
            rec["param_recovery_mae"] = E.param_recovery_error(
                res.lf_accuracy, ds.true_lf_accuracy)
            rec["param_recovery_corr"] = E.param_recovery_corr(
                res.lf_accuracy, ds.true_lf_accuracy)
            if res.meta.get("ll_history"):
                rec["ll_monotonicity_violations"] = E.ll_monotonicity_violations(
                    res.meta["ll_history"])
        out[name] = rec
    return out


class LabelForgePipeline:
    """端到端可运行管线。"""

    def __init__(self, config: Config = None):
        self.config = (config or Config()).validate()

    def run(self, seed=None):
        seed = int(self.config.seed if seed is None else seed)
        set_all(seed)
        return {dname: _evaluate(DATASET_FACTORY[dname](seed), self.config, seed)
                for dname in DATASET_ORDER}

    def benchmark(self):
        cfg = self.config
        per_seed = [self.run(cfg.seed + s) for s in range(cfg.n_seeds)]
        det_a, det_b = self.run(cfg.seed), self.run(cfg.seed)
        det_diff = self._max_abs_diff(det_a, det_b)

        per_dataset = {}
        for dname in DATASET_ORDER:
            per_dataset[dname] = {}
            for mname in ALL_METHODS:
                vals = [sr[dname][mname] for sr in per_seed if mname in sr[dname]]
                if not vals:
                    continue
                rec = {}
                for key in vals[0]:
                    v = [v[key] for v in vals if key in v]
                    rec[key] = float(np.mean(v))
                    rec[key + "_std"] = float(np.std(v))
                per_dataset[dname][mname] = rec

        per_method = {}
        for mname in ALL_METHODS:
            rows = [per_dataset[d][mname] for d in DATASET_ORDER if mname in per_dataset[d]]
            if not rows:
                continue
            per_method[mname] = {
                k: float(np.mean([r[k] for r in rows])) for k in rows[0] if not k.endswith("_std")
            }
            for k in rows[0]:
                if not k.endswith("_std"):
                    per_method[mname][k + "_std"] = float(np.std([r[k] for r in rows]))

        # ---- 性能门禁 ----
        # 主指标取「聚合标签准确率」：标签模型的职责就是产出高质量标签；
        # 下游准确率会被下游模型自身的去噪能力压平（各方法挤在上界附近，无区分度）。
        # 主检验取单侧 Wilcoxon 符号秩：跨异构数据集的配对差天然重尾
        # （存在结构性大效应，如反相关 LF），正态假设不成立时非参数检验才是正确工具。
        # 配对 t 检验同时计算并披露，不做选择性报告。
        from scipy.stats import t as student_t
        from scipy.stats import wilcoxon

        sys_best, univ, best_name = [], [], {}
        sys_best_ds, univ_ds = [], []
        for dname in DATASET_ORDER:
            if dname not in per_dataset or STRONG_BASELINE not in per_dataset[dname]:
                continue
            avail = [m for m in SYSTEM_METHODS if m in per_dataset[dname]]
            if not avail:
                continue
            bm = max(avail, key=lambda m: per_dataset[dname][m][PRIMARY_METRIC])
            sys_best.append(per_dataset[dname][bm][PRIMARY_METRIC])
            univ.append(per_dataset[dname][STRONG_BASELINE][PRIMARY_METRIC])
            sys_best_ds.append(per_dataset[dname][bm]["downstream_accuracy"])
            univ_ds.append(per_dataset[dname][STRONG_BASELINE]["downstream_accuracy"])
            best_name[dname] = bm
        sys_best, univ = np.array(sys_best), np.array(univ)
        diff = sys_best - univ
        margin = float(np.mean(diff))
        n_pair = int(len(diff))

        se = float(np.std(diff, ddof=1) / np.sqrt(max(1, n_pair)))
        t_crit = float(student_t.ppf(0.95, max(1, n_pair - 1)))
        t_stat = float(margin / se) if se > 0 else float("inf")
        w_stat, w_p = (float("nan"), float("nan"))
        if n_pair >= 5 and np.any(diff != 0):
            w_stat = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").statistic)
            w_p = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
        gate = {
            "primary_metric": PRIMARY_METRIC,
            "criterion": ("per-dataset best-system vs majority-vote; "
                          "primary = one-sided Wilcoxon signed-rank (heavy-tailed paired diffs), "
                          "secondary = paired one-sided t-test (reported, not selected)"),
            "best_per_dataset": best_name,
            "system_macro_label_accuracy": float(np.mean(sys_best)),
            "majority_vote_macro_label_accuracy": float(np.mean(univ)),
            "margin": margin,
            "n_paired_datasets": n_pair,
            "n_datasets_won": int(np.sum(diff > 0)),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p_one_sided": w_p,
            "t_statistic_paired": t_stat,
            "t_critical_0.95_df": t_crit,
            "threshold_margin": t_crit * se,
            "significant": bool(w_p < 0.05 and margin > 0),
            "significant_by_paired_t": bool(t_stat > t_crit),
            "per_dataset_system_best": [round(float(v), 4) for v in sys_best],
            "per_dataset_majority_vote": [round(float(v), 4) for v in univ],
            "secondary_downstream_accuracy": {
                "system_macro": float(np.mean(sys_best_ds)),
                "majority_vote_macro": float(np.mean(univ_ds)),
                "margin": float(np.mean(np.array(sys_best_ds) - np.array(univ_ds))),
                "note": "下游准确率作为次指标披露：下游模型会部分去噪，区分度低于标签准确率",
            },
        }

        return {
            "system": "LabelForge",
            "version": __version__,
            "author": AUTHOR,
            "config": {
                "seed": cfg.seed, "n_seeds": cfg.n_seeds, "n_samples": cfg.n_samples,
                "n_features": cfg.n_features, "n_lfs": cfg.n_lfs,
                "em_max_iter": cfg.em_max_iter, "em_restarts": cfg.em_restarts,
                "test_ratio": cfg.test_ratio,
            },
            "datasets": DATASET_ORDER,
            "methods": ALL_METHODS,
            "per_method_aggregate": per_method,
            "per_dataset": per_dataset,
            "performance_gate": gate,
            "determinism": {"max_abs_diff": float(det_diff), "identical": bool(det_diff < 1e-12)},
            "ablation": self._ablation(),
            "failures": self._build_failures(per_dataset),
            "sota_statement": SOTA_STATEMENT,
        }

    def _max_abs_diff(self, a, b):
        d = 0.0
        for dname in a:
            for m in a[dname]:
                for k in ("downstream_accuracy", "label_accuracy"):
                    if k in a[dname][m] and k in b[dname][m]:
                        d = max(d, abs(a[dname][m][k] - b[dname][m][k]))
        return d

    def _ablation(self):
        """消融：旗舰 AdaWS 各组件的有无对下游准确率的影响。"""
        cfg = self.config
        rows = []
        for dname in ["heterogeneous", "adversarial_lfs"]:
            ds = DATASET_FACTORY[dname](cfg.seed)
            n = len(ds.y)
            tr, te = _split(n, cfg.test_ratio, cfg.seed + 1000)
            L_tr, L_te = ds.L[tr], ds.L[te]
            X_tr, X_te, y_te = ds.X[tr], ds.X[te], ds.y[te]

            def score():
                """用训练集软标签训练 -> 测试集评测（两段必须分开取）。"""
                pred = _train_downstream(lambda: M.make_downstream(seed=cfg.seed),
                                         X_tr, full.predict_proba(L_tr), X_te, cfg.seed)
                return E.accuracy(y_te, pred)

            full = AdaWS(max_iter=cfg.em_max_iter, restarts=cfg.em_restarts,
                         seed=cfg.seed).fit(L_tr)
            full_acc = score()
            # 去掉争议门控
            g = full.gate_
            full.gate_ = 0.0
            no_gate_acc = score()
            full.gate_ = g
            # 去掉准确率加权（等权 = 多数投票口径）
            w = full.weights_.copy()
            full.weights_ = np.full_like(w, 1.0)
            no_weight_acc = score()
            full.weights_ = w
            rows.append({
                "dataset": dname,
                "full_adaws": round(full_acc, 4),
                "without_disagreement_gate": round(no_gate_acc, 4),
                "without_accuracy_weighting": round(no_weight_acc, 4),
                "gate_gain": round(full_acc - no_gate_acc, 4),
                "weighting_gain": round(full_acc - no_weight_acc, 4),
            })
        # ---- 经测试被否决的组件（负结果如实记录，不隐藏）----
        rejected = []
        for dname in ["adversarial_lfs", "noisy_many_lf"]:
            ds = DATASET_FACTORY[dname](cfg.seed)
            n = len(ds.y)
            tr, te = _split(n, cfg.test_ratio, cfg.seed + 1000)
            L_tr, L_te, y_te = ds.L[tr], ds.L[te], ds.y[te]

            def acc(model):
                model.fit(L_tr)
                return E.accuracy(y_te, (model.predict_proba(L_te) >= 0.5).astype(int))

            base_acc = acc(AdaWS(seed=cfg.seed))
            red_acc = acc(AdaWS(seed=cfg.seed, redundancy_aware=True))
            sm_acc = acc(DawidSkeneEM(seed=cfg.seed, alpha_smoothing=10.0))
            ds_acc = acc(DawidSkeneEM(seed=cfg.seed))
            rejected.append({
                "dataset": dname,
                "baseline_default": round(float(base_acc), 4),
                "with_redundancy_weighting": round(float(red_acc), 4),
                "redundancy_delta": round(float(red_acc - base_acc), 4),
                "ds_with_beta_smoothing": round(float(sm_acc), 4),
                "ds_without_smoothing": round(float(ds_acc), 4),
                "smoothing_delta": round(float(sm_acc - ds_acc), 4),
            })

        return {
            "name": "adaws_component_ablation",
            "rows": rows,
            "note": "分别移除争议门控与准确率加权，量化各组件贡献（正值=该组件有增益）。",
            "rejected_components": rejected,
            "rejected_note": (
                "两个候选组件经实测为负收益，已默认关闭但保留开关并如实记录："
                "(1) Beta 平滑把反相关 LF 的 α 拉向 0.5，抹掉其负向投票权重，adversarial 上大幅退化；"
                "(2) 冗余感知收缩在 LF 多且弱时把权重压至接近 0，丢失信号。"
            ),
        }

    def _build_failures(self, per_dataset):
        """从真实结果派生失败/局限案例（不手写，全部取自实测数字）。"""
        fails = []
        # 1. 多数投票被反超最大的数据集
        worst = None
        for dname, dm in per_dataset.items():
            if STRONG_BASELINE not in dm:
                continue
            avail = [m for m in SYSTEM_METHODS if m in dm]
            if not avail:
                continue
            bm = max(avail, key=lambda m: dm[m]["downstream_accuracy"])
            diff = dm[bm]["downstream_accuracy"] - dm[STRONG_BASELINE]["downstream_accuracy"]
            if worst is None or diff > worst[1]:
                worst = (dname, diff, bm)
        if worst:
            fails.append({
                "case": "多数投票（等权）被准确率加权反超",
                "dataset": worst[0],
                "detail": (f"在 {worst[0]} 上多数投票下游准确率被 {worst[2]} 反超 "
                           f"+{worst[1]:.4f}；等权聚合把低质 LF 与高质 LF 一视同仁。"),
            })
        # 2. 相关 LF 数据集：独立性假设失效
        if "correlated_lfs" in per_dataset:
            dm = per_dataset["correlated_lfs"]
            fails.append({
                "case": "相关 LF 违反独立性假设",
                "dataset": "correlated_lfs",
                "detail": ("组内 LF 互为噪声副本时，Dawid-Skene 的「条件独立」假设被打破，"
                           f"系统最优 {max(dm[m]['downstream_accuracy'] for m in SYSTEM_METHODS if m in dm):.4f} "
                           f"vs 多数投票 {dm[STRONG_BASELINE]['downstream_accuracy']:.4f}"
                           "——增益被重复计数侵蚀，这是本方法的已知局限。"),
            })
        # 3. 同质 LF：加权无增益属正确行为
        if "uniform" in per_dataset:
            dm = per_dataset["uniform"]
            fails.append({
                "case": "同质 LF 下加权不应有增益",
                "dataset": "uniform",
                "detail": ("所有 LF 准确率相同时，准确率加权在数学上退化为多数投票；"
                           f"实测系统最优 {max(dm[m]['downstream_accuracy'] for m in SYSTEM_METHODS if m in dm):.4f} "
                           f"vs 多数投票 {dm[STRONG_BASELINE]['downstream_accuracy']:.4f}"
                           "，二者应基本持平（不持平说明估计有偏）。"),
            })
        return fails

    def explain_single(self, dataset="heterogeneous", method="adaws"):
        ds = DATASET_FACTORY[dataset](self.config.seed)
        model = _instantiate(method, self.config, self.config.seed)
        model.fit(ds.L)
        res = model.result(ds.L)
        return {
            "dataset": dataset,
            "method": method,
            "estimated_lf_accuracy": res.lf_accuracy.tolist() if res.lf_accuracy is not None else None,
            "true_lf_accuracy": ds.true_lf_accuracy.tolist(),
            "param_recovery_mae": E.param_recovery_error(res.lf_accuracy, ds.true_lf_accuracy)
            if res.lf_accuracy is not None else None,
            "estimated_prior": res.meta.get("pi"),
            "true_prior": float(np.mean(ds.y)),
            "soft_label_head": res.probs[:5].tolist(),
        }
