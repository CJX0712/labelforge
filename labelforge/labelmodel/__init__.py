"""标签模型包。全部纯 numpy，零第三方依赖 = 离线兜底。"""

from .aggregators import AdaWS, GoldOracle, MajorityVote, RandomLF
from .dawid_skene import DawidSkeneEM, _log_likelihood, _posterior

# 参与排名的聚合方法（gold_oracle 只作上界参照，不参与竞争）
RANKED_METHODS = [MajorityVote, DawidSkeneEM, AdaWS]

__all__ = [
    "AdaWS",
    "DawidSkeneEM",
    "GoldOracle",
    "MajorityVote",
    "RandomLF",
    "RANKED_METHODS",
    "_log_likelihood",
    "_posterior",
]
