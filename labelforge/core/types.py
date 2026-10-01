"""核心数据结构。"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Dataset:
    """弱监督数据集：特征 X、金标 y（仅评测用，训练时不可见）、标注函数输出 L。"""

    X: np.ndarray
    y: np.ndarray
    L: np.ndarray
    name: str = "dataset"
    true_lf_accuracy: Optional[np.ndarray] = None
    true_lf_propensity: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        self.L = np.asarray(self.L, dtype=int)


@dataclass
class LabelModelResult:
    """标签模型输出：软标签概率 + 估计出的 LF 质量参数。"""

    method: str
    probs: np.ndarray                       # P(Y=1 | L)，形状 (n,)
    lf_accuracy: Optional[np.ndarray] = None  # 估计的每个 LF 准确率
    log_likelihood: Optional[float] = None
    n_iter: int = 0
    meta: dict = field(default_factory=dict)

    def predict(self, threshold: float = 0.5) -> np.ndarray:
        return (self.probs >= threshold).astype(int)


@dataclass
class WeakSupervisionData:
    """仅含标注函数矩阵与可选金标的轻量容器（供 CLI / 单测使用）。"""

    L: np.ndarray
    y: Optional[np.ndarray] = None
    name: str = "ws"


@dataclass
class EvalReport:
    """下游评测结果。"""

    method: str
    dataset: str
    label_accuracy: float
    downstream_accuracy: float
    f1: float
    runtime_sec: float
    meta: dict = field(default_factory=dict)
