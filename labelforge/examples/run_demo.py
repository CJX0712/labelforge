"""端到端演示：合成弱监督数据 -> 标签模型聚合 -> 下游评测 -> 落盘 benchmark.json。

运行: python labelforge/examples/run_demo.py  (仓库根目录)
"""

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from labelforge.core.config import Config
from labelforge.pipeline.pipeline import LabelForgePipeline


def main():
    t0 = time.perf_counter()
    cfg = Config.from_env().validate()
    pipe = LabelForgePipeline(cfg)
    report = pipe.benchmark()
    with open("benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    dt = time.perf_counter() - t0
    g = report["performance_gate"]
    print(f"[LabelForge] demo 完成, 端到端 {dt:.1f}s (预算 <= {cfg.perf_budget_sec:.0f}s)")
    print(f"  方法: {len(report['methods'])} | 数据集: {len(report['datasets'])} | seeds: {cfg.n_seeds}")
    print(f"  确定性逐位一致: {report['determinism']['identical']} "
          f"(max|Δ|={report['determinism']['max_abs_diff']:.2e})")
    print(f"{'method':<18}{'label_acc':>12}{'downstream':>12}{'param_mae':>12}")
    for m, v in sorted(report["per_method_aggregate"].items(),
                       key=lambda kv: -kv[1]["label_accuracy"]):
        mae = v.get("param_recovery_mae", float("nan"))
        print(f"{m:<18}{v['label_accuracy']:>12.4f}{v['downstream_accuracy']:>12.4f}{mae:>12.4f}")
    print(f"  门禁: 标签准确率 {g['system_macro_label_accuracy']:.4f} vs 多数投票 "
          f"{g['majority_vote_macro_label_accuracy']:.4f} "
          f"({g['margin']:+.4f}), Wilcoxon p={g['wilcoxon_p_one_sided']:.5f}, "
          f"显著={g['significant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
