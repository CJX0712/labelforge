"""自定义错误类型。"""


class LabelForgeError(Exception):
    """LabelForge 基础错误。"""


class ConfigError(LabelForgeError):
    """配置非法。"""


class ModelError(LabelForgeError):
    """模型/后端不可用或使用不当。"""
