"""数据层：弱监督合成数据生成器（已知真值 LF 准确率，供参数恢复校验）。"""

from .synthetic import (
    make_adversarial_lfs,
    make_class_imbalanced,
    make_correlated_lfs,
    make_heterogeneous,
    make_high_dim_few_lf,
    make_noisy_many_lf,
    make_sparse_coverage,
    make_uniform,
)

DATASET_FACTORY = {
    "uniform": make_uniform,
    "heterogeneous": make_heterogeneous,
    "sparse_coverage": make_sparse_coverage,
    "correlated_lfs": make_correlated_lfs,
    "adversarial_lfs": make_adversarial_lfs,
    "class_imbalanced": make_class_imbalanced,
    "high_dim_few_lf": make_high_dim_few_lf,
    "noisy_many_lf": make_noisy_many_lf,
}

DATASET_ORDER = list(DATASET_FACTORY.keys())

__all__ = ["DATASET_FACTORY", "DATASET_ORDER"] + [f.__name__ for f in DATASET_FACTORY.values()]
