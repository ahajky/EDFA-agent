# EDFA Factory Diagnosis & Calibration Agent

面向 EDFA 出厂测试的链路放大仿真与智能诊断原型。当前仓库已有一个可运行的三级 EDFA 灰盒仿真 MVP，用于在厂商参数和真实案例到位前验证数据生成、故障传播和溯源流程。

> **参数状态：** 演示配置全部是公开示例和工程假设，`vendor_confirmed: false`，不能当作厂商规格或物理精度结论。

## 当前能力

- 40 通道、三级 EDFA、级间 GFF/VOA/隔离器、输入/级间/输出 tap。
- 逐通道信号、ASE、有效 NF、增益纹波、DGT、温度和简化输出饱和。
- 逐级 AGC 根据实际增益调整泵浦，支持泵浦限流。
- 泵浦健康度、额外插损、增益谱偏移、PD 偏置/掉线和输入功率变化。
- 光学真值、tap/PD/OSA 测量值、故障传播账本和诊断证据分离保存。
- 固定 seed 可复现，输出 JSON 案例、CSV 汇总和运行清单。

## 快速开始

推荐使用项目根目录的 `run.sh`，它通过 Conda 环境 `edfa-agent` 执行命令，
不要求当前 shell 已经执行过 `conda init`：

```bash
./run.sh setup          # 创建或更新 Conda 环境
./run.sh check          # 校验默认配置并运行全部测试
./run.sh run            # 运行全部场景
./run.sh case stage2_pump_health_60
./run.sh summary        # 查看 cases.csv
./run.sh doctor         # 查看 Python 和依赖版本
./run.sh help           # 查看全部命令
```

也可以直接进入环境：

```bash
conda activate edfa-agent
```

原有免安装方式仍然可用：

```bash
PYTHONPATH=src python3 -m edfa_greybox.cli validate \
  --config configs/three_stage_demo.json

PYTHONPATH=src python3 -m edfa_greybox.cli run \
  --config configs/three_stage_demo.json \
  --output outputs/three_stage_demo
```

也可以安装命令行入口：

```bash
python3 -m pip install -e .
edfa-greybox run --config configs/three_stage_demo.json
```

只运行指定案例：

```bash
PYTHONPATH=src python3 -m edfa_greybox.cli run \
  --case stage2_pump_health_60 \
  --case out_tap_bias_2db \
  --output outputs/selected_cases
```

运行测试：

```bash
python3 -m pytest -q
```

## 输出怎么看

`outputs/three_stage_demo/` 中的主要文件：

| 文件 | 用途 |
| --- | --- |
| `config_snapshot.json` | 本次运行完整参数快照 |
| `<case>.json` | 单案例真值、测量值、传播链和诊断证据 |
| `cases.csv` | 所有案例的对比索引 |
| `run_manifest.json` | seed、基线案例和参数来源 |

单个案例 JSON 的关键层次：

```text
truth.events                 根因参数变化
truth.propagation            各节点输入/输出、增益、泵浦和饱和
measurements.taps            真实 tap 功率与 PD 观测值
measurements.output_osa      逐通道光学真值与 OSA 观测值
comparison_to_baseline       相对正常案例的变化
diagnostic_evidence          基于方向性变化的可解释证据
```

## 仿真流程

```text
配置光路与工作点
  -> 注入故障，只修改器件/测量参数
  -> 逐通道通过 EDFA、无源器件和 tap
  -> AGC 迭代调整各级泵浦
  -> 计算输出饱和下的信号与 ASE
  -> 独立模拟 PD/OSA
  -> 保存真值、测量、传播账本和诊断证据
```

## 厂商资料到位后

1. 用厂商光路替换 `configs/three_stage_demo.json` 的拓扑。
2. 用 BOM、数据手册或测得的增益/NF/DGT 曲线替换工程假设参数。
3. 把真实测试字段映射到 `truth`、`measurements` 和 `comparison_to_baseline`。
4. 用正常案例校准器件参数，用故障案例校验传播方向。
5. 再决定是否引入 GNPy 链路预算对照、GPR 增益谱代理或更高保真的 Giles-Desurvire 锚点。

## 文档与协作规则

- 前期方案：[EDFA 链路放大仿真方案调研](.agent/reports/20261019-EDFA链路放大仿真方案调研.md)。
- 仿真入门：[EDFA 仿真基础介绍](.agent/reports/20261021-EDFA仿真基础介绍.md)。
- 当前实现说明：[EDFA 灰盒仿真 MVP 说明](.agent/reports/20261023-EDFA灰盒仿真MVP说明.md)。
- 目录说明与协作规则见 [AGENTS.md](AGENTS.md)。

## 两套并行仿真方案

- `src/edfa_greybox/`:灰盒链路仿真(故障注入、AGC、测量链),配置 `configs/three_stage_demo.json`,入口 `edfa-greybox`。
- `src/edfa_whitebox/`:Giles-Desurvire 速率-传播白盒模型,配置 `configs/whitebox_three_stage_demo.json`,入口 `edfa-whitebox` 或 `python -m edfa_whitebox <config>`。

两者参数均为工程假设,未经厂商确认。
