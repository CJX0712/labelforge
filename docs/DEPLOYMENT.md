# LabelForge 部署与复现说明

## 1. 环境要求

| 项 | 要求 |
|---|---|
| Python | ≥ 3.10（实测 3.13.14 / cp313） |
| 必需依赖 | numpy、scipy（锁版本见 `requirements.lock.txt`） |
| 可选依赖 | scikit-learn（下游分类器后端，缺失自动用纯 numpy 逻辑回归） |
| 开发依赖 | pytest（77 项测试） |
| 内存 | < 300 MB |
| 磁盘 | < 15 MB |

## 2. 一键复现

```bash
git clone https://github.com/cjx0712/labelforge.git
cd labelforge
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt
python -m pytest tests -q
python labelforge/examples/run_demo.py
```

预期输出要点：

- `77 passed`
- `确定性逐位一致: True (max|Δ|=0.00e+00)`
- `benchmark.json` 中 `performance_gate.significant == true`
- 端到端耗时 ≈ 8 s（默认 3 seeds × 8 数据集）

## 3. 离线部署

标签模型核心（Dawid-Skene EM、聚合器、指标）**全部纯 numpy**，不依赖任何第三方包：

- 无 scikit-learn 时，下游分类器自动降级为 `NumpyLogisticModel`（纯 numpy）；
- 所有 EM / 聚合逻辑与指标计算保持不变；
- 结论与指标不受影响（下游后端不同会带来极小幅度数值差异）。

## 4. CI

`.github/workflows/ci.yml` 在 Python 3.11 / 3.13 上执行：

1. 安装锁定依赖
2. `pytest tests -q`（公理不变量）
3. `run_demo.py`（端到端基准）
4. 上传 `benchmark.json` 产物

## 5. 性能预算

| 配置 | 端到端耗时 |
|---|---|
| 默认（3 seeds × 8 数据集） | ≈ 8 s |
| 单 seed | ≈ 2.5 s |

调优旋钮：`LABELFORGE_EM_RESTARTS`（最敏感）、`LABELFORGE_EM_MAX_ITER`、`LABELFORGE_N_SEEDS`。

## 6. 故障排查

| 现象 | 检查 |
|---|---|
| 标签准确率接近正类比例 | 投票编码错误（负类必须记 −1 而非 0）→ 恒定预测正类 |
| 标签准确率远低于 0.5 | EM 标签翻转未被识别；检查多数投票锚点是否生效 |
| `alpha` 估计全在 0.5 附近 | 数据无信号；确认 LF 与 Y 存在真实关联 |
| 结果不可复现 | 确认未混用全局随机源；所有采样走 `np.random.default_rng(seed)` |
| 下游训练报单类错误 | 已用软标签双样本加权规避；若仍出现检查软标签是否全部相等 |
