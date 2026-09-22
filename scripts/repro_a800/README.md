# Aether 单卡 A800 80GB 复现

目标：运行官方动作条件预测示例。保留上游模型和 demo 代码，新增环境约束、必要权重下载及本地权重运行入口。

上游源码：`InternRobotics/Aether@871c6f7ebcd66f3571ed27db5d65264a51d624a3`。

## 环境

Ubuntu 22.04，Python 3.10，单张 A800 80GB。`constraints.txt` 记录此次实际安装的 145 个依赖版本，满足上游 requirements 的要求；它不是上游发布的锁文件。安装器额外补充 `protobuf==4.25.6`，这是 T5 分词器转换实际需要、但上游 requirements 未显式列出的依赖。

在 Linux 服务器执行，先进入 Aether 仓库目录：

```bash
export WC_ROOT=/root/autodl-tmp/worldcache-repro
mkdir -p "$WC_ROOT"/{envs,cache/conda,tmp,models,outputs,logs,wheelhouse}
export CONDA_PKGS_DIRS="$WC_ROOT/cache/conda"
conda create -p "$WC_ROOT/envs/aether" python=3.10 pip -y
bash scripts/repro_a800/install_dependencies.sh
```

依赖安装使用阿里云 PyPI 镜像。若预置 `$WC_ROOT/wheelhouse`，安装器会将其作为本地依赖候选；也可提前安装其中已校验的 Linux/Python 3.10 wheel。模型和环境目录都放在数据盘；持久性以租赁平台说明为准。

## 权重

仅下载官方 demo 所用组件，约 20.1 GiB：

- AetherWorldModel/AetherV1：transformer；版本 `6c53ba75e398c8b91623ed86757b2289eb45f1ce`。
- THUDM/CogVideoX-5b-I2V：tokenizer、text_encoder、vae、scheduler；版本 `a6f0f4858a8395e7429d82493864ce92bf73af11`。

```bash
export HF_HOME="$WC_ROOT/cache/huggingface"
# 仅在官方站不可访问且选择使用镜像时设置：
# export HF_ENDPOINT=https://hf-mirror.com
"$WC_ROOT/envs/aether/bin/python" scripts/repro_a800/download_weights.py
```

## 运行

```bash
bash scripts/repro_a800/run_prediction.sh
```

使用 car.png 和 raymap_forward_right.npy，seed=42，480×720，41 帧，50 步。权重从本地路径加载，运行脚本开启离线模式。输出目录为 `$WC_ROOT/outputs/aether_car_seed42`。再次运行会复用同一输出目录。

官方 demo 默认还执行 4 步 reconstruction 并导出点云；本脚本保留这个行为。因此该示例用于跑通环境，整个进程的耗时不能直接用作 WorldCache 的公平加速对照。正式对照还需统一生成入口、后处理、精度和计时边界。

## 实机验证（2026-09-22）

- Ubuntu 22.04.4；A800 80GB PCIe；Python 3.10.21。
- PyTorch 2.5.1+cu124、CUDA runtime 12.4、Diffusers 0.32.2、Transformers 4.48.3。
- `pip check` 通过；官方 prediction 示例退出码为 0。
- RGB 与视差视频均完整解码：41 帧、720×480、12 fps；生成 5 个 GLB 点云文件。
- 抽查 RGB 第 0、20、40 帧，画面正常显示。
- 完整进程约 252 秒，包含模型加载、50 步预测、4 步重建和文件导出；不是稳定推理时延或加速比。
- `nvidia-smi` 每秒采样观测到的显存峰值为 36,011 MiB（约 35.2 GiB）；不等同于 PyTorch allocator 精确峰值。

本次仅验证原始 Aether 流程；WorldCache 配对测速与 WorldScore 评估尚未执行。服务器采用稀疏检出，仅保留当前预测示例需要的两个输入，fork 中仍保留其余上游资产。
