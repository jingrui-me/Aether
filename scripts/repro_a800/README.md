# Aether + WorldCache 单卡复现

已在 Ubuntu 22.04.4 / A800 80GB PCIe / Python 3.10.21 上跑通同输入同种子的原始/缓存配对。

## 来源与改动

- 原始 Aether 复现基线：`3892c3c3292dc449095b2b8f287615897636e4b8`；原版保留在 `codex/aether-a800-repro` 分支。
- WorldCache：`FofGofx/WorldCache@b921368f7dfbd7ca5d7cfcd0276fec1d1cfd7d91`。
- 带入官方 `aether/worldcache_aether` 包及其修改过的 pipeline，在 demo 的 build_pipeline 中调用两个 apply 函数。
- Diffusers 0.32.2 的 CogVideoXBlock 不接受 attention_kwargs；本分支仅移除 block 调用处这一空参数。当前支持的是官方 demo 的 attention_kwargs=None 设置，缓存预测和跳步算法未改。
- 安装补充 protobuf 4.25.6 和 loguru 0.7.3。constraints.txt 固定实际依赖版本。
- WorldCache 来源与 Apache-2.0 许可见 `aether/worldcache_aether/UPSTREAM.md` 和 LICENSE。

## 安装与权重

已有原始 Aether 环境和权重时无需重复下载。新环境需先按上游说明建立 Python 3.10 环境，并设置数据盘路径：

```bash
export WC_ROOT=/root/autodl-tmp/worldcache-repro
mkdir -p "$WC_ROOT"/{envs,models,cache/conda,tmp,outputs,logs,wheelhouse}
export CONDA_PKGS_DIRS="$WC_ROOT/cache/conda"
conda create -p "$WC_ROOT/envs/aether" python=3.10 pip -y
bash scripts/repro_a800/install_dependencies.sh
"$WC_ROOT/envs/aether/bin/python" scripts/repro_a800/download_weights.py
```

下载器固定 AetherV1 transformer 与 CogVideoX-5b-I2V 的 tokenizer、text_encoder、vae、scheduler 版本，合计约 20.1 GiB。运行时从本地路径读取。

## 运行

在本分支仓库目录执行：

```bash
set -o pipefail
WORLDCACHE_MODE=worldcache \
WORLDCACHE_PERCENTILE_STABLE=0.30 \
WORLDCACHE_PERCENTILE_CHAOTIC=0.60 \
WORLDCACHE_N_MAX=2 \
WORLDCACHE_ERROR_THRESHOLD=0.2 \
  bash scripts/repro_a800/run_prediction.sh 2>&1 | tee wc.log
```

图像 car.png、轨迹 raymap_forward_right.npy、seed=42、480×720、41 帧、50 步与原版一致。保留原 demo 的额外 4 步重建和点云导出。输出另存至 `$WC_ROOT/outputs/aether_worldcache_car_seed42`，重复执行会使用同一输出目录。

## 实测（2026-09-22）

| 项目 | 原版 | WorldCache |
|---|---:|---:|
| 预测阶段完整前向 | 50* | 30 |
| 预测阶段缓存预测 | 0 | 20 |
| 重建阶段完整前向 | 4* | 4 |
| 整进程秒数 | 242 | 179 |
| 每秒采样观测显存峰值 MiB | 36391 | 36613 |

原版使用未修改的完整前向实现，其 50/4 步由源码路径与完成日志确认，没有 Calculation Type 探针。缓存版逐步日志确认 50 步预测和 4 步重建没有漏计或重复计数。

整份日志执行 `grep -c 'Calculation Type: full' wc.log` 得到 34，`grep -c 'Calculation Type: worldcache' wc.log` 得到 20；前者包含 4 次额外重建。预测 FULL=30，包括 10 次预热、19 次误差超阈值刷新和最后一步。

两组退出码均为 0，各生成 41 帧、720×480、12 fps 的 RGB 和视差视频，以及 5 个 GLB 文件。抽帧可见车辆位置和场景细节存在差异，不能据此宣称质量无损。

242/179≈1.35 是单次整进程耗时比，包含模型加载、预测、重建和导出，且缓存版包含详细日志；未做重复预热统计或 WorldScore 质量评估，不作为正式加速比结论。
