"""抽象接口。所有标签模型遵循同一契约，便于替换与交叉验证。"""

from abc import ABC, abstractmethod

import numpy as np


class LabelModel(ABC):
    """标签模型契约：输入标注函数矩阵 L (n, m)，输出 P(Y=1|L)。"""

    name: str = "label_model"

    @abstractmethod
    def fit(self, L: np.ndarray, **kwargs) -> "LabelModel":
        """拟合标签模型（不使用任何金标 y）。"""

    @abstractmethod
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """返回 P(Y=1|L)，形状 (n,)。"""


class LabelingFunctionBank(ABC):
    """标注函数库契约：持有 m 个标注函数及其在样本上的输出。"""

    @abstractmethod
    def apply(self, X: np.ndarray) -> np.ndarray:
        """返回 L (n, m)，取值 {-1, 0, 1}（-1 表示弃权）。"""
