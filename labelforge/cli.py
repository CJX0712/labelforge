"""LabelForge 命令行入口。"""

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="labelforge",
        description="LabelForge · 弱监督 / 程序化标注标签模型系统 (作者: 晨星)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="运行全基准并落盘 benchmark.json")
    b.add_argument("--out", default="benchmark.json")

    e = sub.add_parser("explain", help="查看单个数据集的 LF 准确率估计与参数恢复")
    e.add_argument("--dataset", default="heterogeneous")
    e.add_argument("--method", default="adaws",
                   choices=["dawid_skene", "adaws"])

    args = ap.parse_args(argv)

    if args.cmd == "benchmark":
        from .core.config import Config
        from .pipeline.pipeline import LabelForgePipeline

        rep = LabelForgePipeline(Config.from_env().validate()).benchmark()
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        g = rep["performance_gate"]
        print(f"[LabelForge] 基准完成 -> {args.out}")
        print(f"  系统宏平均标签准确率 : {g['system_macro_label_accuracy']:.4f}")
        print(f"  多数投票基线         : {g['majority_vote_macro_label_accuracy']:.4f}")
        print(f"  差值                 : {g['margin']:+.4f} "
              f"(赢下 {g['n_datasets_won']}/{g['n_paired_datasets']} 个数据集)")
        print(f"  Wilcoxon 单侧 p      : {g['wilcoxon_p_one_sided']:.5f} -> "
              f"{'显著' if g['significant'] else '不显著'}")
        print(f"  配对 t（次检验）     : {g['t_statistic_paired']:.3f} "
              f"(临界 {g['t_critical_0.95_df']:.3f})")
        return 0

    if args.cmd == "explain":
        from .core.config import Config
        from .pipeline.pipeline import LabelForgePipeline

        out = LabelForgePipeline(Config.from_env().validate()).explain_single(
            args.dataset, args.method)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
