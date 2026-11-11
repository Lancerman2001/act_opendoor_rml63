# ACT OpenDoor · RML63

基于 **LeRobot 0.7.0 / LeRobotDataset v3.0** 的单 GPU ACT 训练工程。提供数据划分、配置管理、训练监控、checkpoint、损失记录、Matplotlib 绘图与离线评估。

默认使用新初始化的 ACT 和 ImageNet ResNet18 视觉骨干。训练参数集中在 [`configs/train.yaml`](configs/train.yaml)，可用命令行覆盖。

| 默认设置 | 值 |
|---|---|
| 总训练步数 | 50,000 |
| Batch size / 混合精度 | 32 / BF16 |
| 训练 / 测试划分 | 按完整 episode 划分 80% / 20% |
| Checkpoint | 每 10,000 步，并额外保存最后一步 |
| 损失记录 | 每 100 步的窗口均值，并保存结束时的剩余窗口 |
| 图像输出 | 训练结束自动输出 PNG、PDF，160 DPI |

数据、模型权重、虚拟环境和运行日志由用户在本地生成，均不包含在源码发布包中。

## 快速开始

准备一台能被 `nvidia-smi` 识别的 NVIDIA GPU、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和已转换的 LeRobotDataset v3 数据。以下命令使用 Linux / WSL2 Bash；将 `/path/to/lerobot_v3` 替换为你的数据集目录。

```bash
git clone https://github.com/Lancerman2001/act_opendoor_rml63.git
cd act_opendoor_rml63
sudo apt-get update
sudo apt-get install -y ffmpeg
./setup.sh
mkdir -p data
ln -s /path/to/lerobot_v3 data/source

# 先跑 20 步，检查数据读取、训练、checkpoint 和绘图
./run.sh all --config configs/smoke.yaml

# 再运行默认 50,000 步训练和结束后的离线评估
./run.sh all --config configs/train.yaml
```

训练模型位于 `runs/<run_name>/checkpoints/`，损失、曲线和结果报告位于 `reports/<run_name>_<UTC时间>/`。首次试跑后可先阅读下方配置说明，再启动正式训练。

快速导航：[环境准备](#1-环境准备) · [数据准备](#2-准备-lerobot-v3-数据) · [训练与续训](#3-配置与训练) · [损失曲线](#4-查看损失与效果图) · [参数手册](docs/CONFIGURATION.md)

## 1. 环境准备

验证环境：Ubuntu 22.04 / WSL2、Python 3.12、RTX 5080 Laptop 16 GB、NVIDIA 驱动 592.00。其他 NVIDIA GPU 的 batch size 和精度需要按设备能力设置；当前启动器要求单 CUDA GPU。

[`requirements.txt`](requirements.txt) 固定直接依赖：LeRobot 0.7.0、PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130、TorchCodec 0.17.0+cu130、Matplotlib 3.11.2 等。[`requirements.lock.txt`](requirements.lock.txt) 固定完整依赖版本。驱动与 FFmpeg 共享库属于系统依赖。

先确认 `nvidia-smi` 能识别 GPU，并安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在 Ubuntu 上准备 FFmpeg：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

克隆本仓库，或者下载源码 ZIP 后解压：

```bash
git clone https://github.com/Lancerman2001/act_opendoor_rml63.git
cd act_opendoor_rml63
./setup.sh
```

安装脚本创建项目内 `.venv`，下载依赖及 ResNet18 权重，验证真实 CUDA 运算、BF16、FFmpeg/TorchCodec 解码和无界面绘图。没有数据集也可完成环境检查。首次安装需要网络；CUDA 运行库需要数 GB 磁盘空间。

不支持 BF16 的 GPU 可运行 `./setup.sh --precision fp16`，并把配置中的 `training.mixed_precision` 改为 `fp16`；全精度使用 `--precision no` 和 YAML 字符串 `"no"`。`./run.sh all` 会按所选配置检查精度。

脚本会隔离继承的 `PYTHONPATH`、`PYTHONHOME`，避免 ROS、OpenPI 或用户 site-packages 混入训练环境。旧环境缓存复用属于可选加速：

```bash
./setup.sh --reuse-from /path/to/old/.venv
```

如需手动使用 pip，先安装系统依赖，再执行：

```bash
python3.12 -m venv .venv
source env.sh
.venv/bin/python -m pip install -r requirements.txt -c requirements.lock.txt
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/check_environment.py
```

## 2. 准备 LeRobot v3 数据

输入是已经转换好的本地 LeRobotDataset v3 数据集，至少包含两个 episode：

```text
data/source/
├── meta/info.json
├── meta/stats.json
├── meta/episodes/...parquet
├── data/...parquet
└── videos/...mp4
```

需要图像观测、`observation.state` 和 `action` 向量。episode 元数据需包含统计量。帧率、轨迹数、向量维度和相机形状从数据元信息读取，不依赖特定的 110 条数据或本机绝对路径。

把数据放入 `data/source/`，或修改配置中的 `dataset.source` 为自己的路径，然后执行：

```bash
./run.sh prepare
```

脚本按完整轨迹固定随机划分数据，复制一份底层存储，生成 `data/split.json`、`train.json`、`test.json`。两组共享这份视频/Parquet 存储，以 episode 索引隔离；视频不会重新编码。归一化统计量只使用训练集，完整原统计量另作备份。

相同数据与配置可校验后复用。改变划分比例、seed 或来源时，请设置新的 `dataset.root` 和 `dataset.split_file`，保留已有实验的划分。

## 3. 配置与训练

编辑 [`configs/train.yaml`](configs/train.yaml)，重点参数如下：

```yaml
training:
  run_name: act_50k
  steps: 50000
  batch_size: 32
  mixed_precision: bf16
logging:
  every_steps: 100
checkpoint:
  every_steps: 10000
plot:
  enabled: true
  formats: [png, pdf]
  dpi: 160
  smoothing_window: 20
```

上面是配置片段；完整配置保存在仓库中。所有相对路径以仓库根目录为基准。配置拼写错误或非法值会报错，不会静默忽略。详细说明见 [配置与输出说明](docs/CONFIGURATION.md)。

```bash
# 查看实际训练命令，校验数据；不启动训练
./train.sh --dry-run

# 默认训练，并自动保存日志、checkpoint 和图形
./train.sh

# 独立配置 + 临时命令行覆盖
./train.sh --config configs/train.yaml --run-name act_custom \
  --steps 50000 --batch-size 32 --save-every 10000 --log-every 100
```

如果需要先做 20 步连通性检查：

```bash
./run.sh all --config configs/smoke.yaml
```

运行完整默认流程：

```bash
./run.sh all
```

完整流程依次检查环境、准备数据、执行可选 batch 基准、训练、执行可选留出评估和生成报告。只有与当前配置、数据和最终步数匹配的已完成训练才会复用；历史 2,000 步结果不会被当成新的 50,000 步训练。每个新实验使用新的 `training.run_name`。

恢复训练时指定已保存的配置，`steps` 表示恢复后的总步数：

```bash
./train.sh --config configs/train.yaml \
  --resume-config runs/act_50k/checkpoints/last/pretrained_model/train_config.json \
  --steps 100000 --run-name act_50k_resume
```

恢复使用 checkpoint 的模型、优化器、batch 和数据设置；日志及保存间隔采用本次 YAML/CLI 设置。每次启动会保存 `resolved_config.yaml`，记录实际运行参数。

## 4. 查看损失与效果图

每次训练的独立目录为 `reports/<run_name>_<UTC时间>/`：

```text
resolved_config.yaml          实际运行配置
training_result.json           本次训练结果与固定 checkpoint 路径
loss.csv / loss.jsonl          完整精度的损失和学习率
console.log / metrics.jsonl    控制台与概览指标
telemetry.csv / summary.json   GPU、温度、内存与退出状态
plots/training_curves.png      自动生成的训练曲线
plots/training_curves.pdf      矢量格式图形
plots/plot_summary.json        绘图输入、平滑参数和文件信息
```

损失记录包括 `step`、`loss`、`l1_loss`、`kld_loss`、`weighted_kl_loss`、`lr`、`grad_norm`、`epoch`、`interval_steps` 等。当前默认 ACT 目标为 `L1 + 10 × KL`，KL 权重由 `policy.kl_weight` 调整。默认每 100 步保存该窗口的均值；`--log-every 1` 可逐步记录。训练结束时不足一个窗口的记录也会保存。

```bash
tail -f reports/act_50k_*/loss.csv
```

Matplotlib 使用无界面后端，不需要桌面或显示器。图中展示总 loss、L1、KL 和学习率，保留原始曲线并叠加平滑线。平滑窗口单位是**日志记录条数**，不会修改原始 CSV。

重新绘图，或调整导出格式：

```bash
./run.sh plot --log-dir reports/ACT_RUN_TIMESTAMP \
  --formats png pdf --dpi 200 --smoothing-window 10
```

旧日志也可通过 `--input /path/to/metrics.jsonl` 或 `--input /path/to/console.log` 绘图；缺失分量会明确标注，控制台舍入值不会被当成完整精度数据。

## 5. Checkpoint 与评估

默认 50,000 步训练保存 10,000 / 20,000 / 30,000 / 40,000 / 50,000 步，`checkpoints/last` 指向最后一次保存。每个 checkpoint 包含 `pretrained_model/` 与 `training_state/`；前者用于模型加载，后者用于恢复优化器和训练进度。总步数不是保存间隔的整数倍时，最后一步仍会保存。

单独评估指定模型：

```bash
./evaluate.sh --config configs/train.yaml \
  --checkpoint runs/act_50k/checkpoints/last/pretrained_model \
  --report reports/act_50k_test.json
```

默认从每条测试轨迹均匀取 64 个位置，排除动作块尾部 padding，报告归一化 L1、原始单位 MAE 与保持当前位置基线。该基线要求 state/action 向量维度及物理语义一致；其他数据表示请关闭 `evaluation.enabled` 并使用合适的评估器。测试集不用于 batch 选择、早停或训练中调参。损失曲线和离线误差不等于真实机器人任务成功率。

## 6. Batch 基准与参考

```bash
./benchmark.sh --batches 16 32 48 --warmup 20 --steps 100 \
  --report-dir reports/batch_comparison
```

基准使用训练集样本和完整 ACT 的前向、反向、梯度裁剪及优化更新；图像解码与预处理单独计时。正式计时记录功率、温度和显存。不同 GPU 任务应依次运行。

本项目在单路 640×480、100 帧动作块、RTX 5080 Laptop 16 GB、BF16、150 W 功率上限下测得：batch 32 约 183.5 样本/秒、整卡峰值 9.84 GiB；batch 48 约 181.3 样本/秒、12.57 GiB。这是缓存样本计算基准，不是任意数据集的端到端训练速度，因此默认选择 batch 32。

例如训练集为 77,677 个观测起点时，batch 32 的 50,000 步约遍历 20.6 轮。`chunk_size=100` 是重叠未来动作窗口，不把 epoch 数再乘以 100。

## 7. 测试与发布

```bash
source env.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/export_release.py --check-only
.venv/bin/python scripts/export_release.py
```

发布脚本生成 `dist/act-training-src.zip`、`.tar.gz`、文件清单和 SHA256。使用源码白名单，排除数据、checkpoint、权重、虚拟环境、缓存、日志和本机私有路径；重复生成需加 `--force`。[发布说明](docs/PUBLISHING.md) 包含 GitHub 初始化命令。

本仓库脚本采用 [MIT License](LICENSE)。第三方依赖和下载权重保留各自许可证，见 [第三方说明](THIRD_PARTY.md)。

参考：[LeRobot ACT](https://huggingface.co/docs/lerobot/act)、[PyTorch 安装](https://pytorch.org/get-started/locally/)、[Matplotlib 后端](https://matplotlib.org/stable/users/explain/figure/backends.html)。本项目运行接口以锁定版本及随附配置为准。
