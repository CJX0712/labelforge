"""配置：支持 ENV 覆盖 + schema 校验。"""

import os
from dataclasses import dataclass, field

from .errors import ConfigError


@dataclass
class Config:
    seed: int = 42
    n_samples: int = 800
    n_features: int = 12
    n_lfs: int = 12
    n_seeds: int = 3
    em_max_iter: int = 100
    em_tol: float = 1e-6
    em_restarts: int = 3
    perf_budget_sec: float = 60.0
    test_ratio: float = 0.3
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        mapping = {
            "LABELFORGE_SEED": ("seed", int),
            "LABELFORGE_N_SAMPLES": ("n_samples", int),
            "LABELFORGE_N_FEATURES": ("n_features", int),
            "LABELFORGE_N_LFS": ("n_lfs", int),
            "LABELFORGE_N_SEEDS": ("n_seeds", int),
            "LABELFORGE_EM_MAX_ITER": ("em_max_iter", int),
            "LABELFORGE_EM_RESTARTS": ("em_restarts", int),
            "LABELFORGE_PERF_BUDGET_SEC": ("perf_budget_sec", float),
        }
        for env_key, (attr, caster) in mapping.items():
            raw = os.environ.get(env_key)
            if raw is not None:
                try:
                    setattr(cfg, attr, caster(raw))
                except (ValueError, TypeError) as exc:
                    raise ConfigError(f"无效环境变量 {env_key}={raw!r}: {exc}")
        return cfg

    def validate(self) -> "Config":
        if self.seed < 0:
            raise ConfigError("seed 必须 >= 0")
        if self.n_seeds < 1:
            raise ConfigError("n_seeds 必须 >= 1")
        if self.n_lfs < 1:
            raise ConfigError("n_lfs 必须 >= 1")
        if self.em_max_iter < 1:
            raise ConfigError("em_max_iter 必须 >= 1")
        if self.em_restarts < 1:
            raise ConfigError("em_restarts 必须 >= 1")
        if not (0.0 < self.test_ratio < 1.0):
            raise ConfigError("test_ratio 必须在 (0,1)")
        if self.perf_budget_sec <= 0:
            raise ConfigError("perf_budget_sec 必须 > 0")
        return self
