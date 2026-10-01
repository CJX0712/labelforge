"""core: 类型 / 错误 / 配置 / 接口 / 确定性种子。不依赖任何上层模块。"""

from .config import Config
from .errors import ConfigError, LabelForgeError, ModelError
from .interfaces import LabelModel, LabelingFunctionBank
from .seed import set_all
from .types import Dataset, LabelModelResult, WeakSupervisionData

__all__ = [
    "Config",
    "ConfigError",
    "LabelForgeError",
    "ModelError",
    "LabelModel",
    "LabelingFunctionBank",
    "set_all",
    "Dataset",
    "LabelModelResult",
    "WeakSupervisionData",
]
