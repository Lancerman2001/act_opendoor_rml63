# 配置与运行产物

默认配置是 `configs/train.yaml`。`--config another.yaml` 将该文件作为局部覆盖与默认配置合并；命令行训练参数最后覆盖。未知键、负数间隔和不支持的精度会直接报错。

| 配置项 | 默认值 | 含义 |
|---|---|---|
| dataset.source | data/source | 输入 LeRobotDataset v3 目录 |
| dataset.root / split_file | data/full / data/split.json | 准备后的数据与轨迹划分清单 |
| dataset.train_ratio / seed | 0.8 / 42 | 按完整 episode 划分；小数据集至少各保留一条 |
| training.steps / batch_size | 50000 / 32 | 优化步数和每批观测起点数 |
| training.run_name | act_50k | 新实验输出名；已有目录需要恢复或换名 |
| training.mixed_precision | bf16 | 可选 bf16、fp16、字符串 "no" |
| training.num_workers / prefetch_factor | 2 / 2 | 数据读取并行度和预取量 |
| training.resume_config | null | 恢复用 pretrained_model/train_config.json |
| logging.every_steps | 100 | 聚合多少个更新后记录一次，1 为逐步记录 |
| checkpoint.every_steps | 10000 | 保存间隔；结束时另保存最后一步 |
| policy.chunk_size / n_action_steps | 100 / 100 | 预测和执行的动作块长度 |
| policy.optimizer_lr / optimizer_lr_backbone | 1e-5 / 1e-5 | ACT 与视觉骨干学习率 |
| policy.kl_weight | 10 | VAE 的 KL 项权重 |
| plot.enabled / formats | true / [png,pdf] | 自动出图开关与文件格式（亦支持 svg） |
| plot.dpi / smoothing_window | 160 / 20 | 位图分辨率与平滑记录条数 |
| monitor.interval_seconds | 2 | 硬件采样间隔 |
| monitor.temperature_limit_c | 90 | 连续高温保护阈值 |
| monitor.min_host_free_mib | 750 | 连续主机低内存保护阈值 |
| evaluation.enabled | true | all 流程结束后是否做离线测试 |
| evaluation.samples_per_episode / all_frames | 64 / false | 均匀测试采样或全部测试帧 |
| evaluation.batch_size / num_workers | 8 / 2 | 评估批量和读取进程数 |
| benchmark.enabled | false | all 流程是否先执行 batch 基准 |
| benchmark.headroom_gib | 2 | 自动 batch 推荐要求的整卡显存余量 |

`training.batch_size: auto` 需要与当前数据、模型设置相匹配的成功基准报告。模型、分辨率或相机数发生变化时需要重新测速，并使用新的 `benchmark.report_dir`。

## 日志语义

`loss.csv` 和 `loss.jsonl` 从官方 `MetricsTracker` 的原始浮点数据读取，不经过控制台三位小数格式。每行的 `interval_steps` 是实际聚合更新数，`elapsed_seconds` 是本次进程内的相对时间。恢复训练会创建新的报告目录，`step` 保留全局优化步数。

`weighted_kl_loss = loss - l1_loss`，等同于当前 KL 权重乘原始 `kld_loss`，可能有小量浮点误差。`epoch` 沿用 LeRobot 固定 batch 的近似计数，末尾小批次会使其与精确遍历数略有偏差。

未记录的分量不会由历史日志凭空补全。绘图中的平滑线仅用于阅读趋势；训练目标的窗口均值不等于独立测试的推理 L1。

## 可复现记录

每次运行保存 `resolved_config.yaml`，并保留官方 checkpoint 中的 `train_config.json`、优化器状态及训练步数。归一化统计量与划分清单有 SHA256 校验。数据集内容、相机/关节映射和机器人控制语义仍由数据提供者负责。

## 常见问题

- **CUDA unavailable**：检查 NVIDIA 驱动、WSL GPU 映射及 PyTorch `+cu130` 构建；启动器不自动切换 CPU。
- **TorchCodec/FFmpeg 加载失败**：安装发行版 FFmpeg 及共享库；只有一个可执行文件不够。运行 `scripts/check_environment.py` 查看具体加载错误。
- **输出目录已存在**：设置新的 run_name，或者指定 resume_config；不覆盖旧模型。
- **绘图失败但 checkpoint 存在**：先检查本次报告目录的 plot.log，修复依赖后单独运行 `./run.sh plot --log-dir ...`。监控报告会区分训练状态和出图状态。
- **修改了划分比例或随机种子**：选择新的数据输出目录与 split_file；训练与测试轨迹不应跨实验混淆。
- **笔记本吞吐波动**：同时检查驱动报告的功率上限、温度与显存，而非只看 GPU 利用率。
