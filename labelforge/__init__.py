"""LabelForge · 弱监督 / 程序化标注（数据编程）标签模型系统。

纯 numpy 复现 Dawid-Skene 无金标标注者质量估计与 Snorkel 式标签聚合，
以「EM 似然单调」「LF 准确率参数可恢复」「同质 LF 退化为多数投票」为可验证金标准。

作者: 晨星
"""

__version__ = "0.1.0"
__author__ = "晨星"

from .core.config import Config
from .core.types import Dataset, LabelModelResult
from .labelmodel import AdaWS, DawidSkeneEM, MajorityVote, RandomLF

__all__ = ["Config", "Dataset", "LabelModelResult", "AdaWS", "DawidSkeneEM",
           "MajorityVote", "RandomLF", "__version__"]
