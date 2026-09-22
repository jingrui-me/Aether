# Aether 单卡 A800 80GB 复现

目标：运行官方动作条件预测示例。保留上游模型和 demo 代码，新增环境约束、必要权重下载及本地权重运行入口。

上游源码：`InternRobotics/Aether@871c6f7ebcd66f3571ed27db5d65264a51d624a3`。

## 环境

Ubuntu 22.04，Python 3.10，单张 A800 80GB。`constraints.txt` 为此次复现选择的核心版本，满足上游 requirements 的要求；它不是上游发布的锁文件。

在 Linux 服务器执行，先进入 Aether 仓库目录：

```bash
export WC_ROOT=/root/autodl-tmp/worldcache-repro
mkdir -p "$WC_ROOT"/{envs,cache/conda,tmp,models,outputs,logs,wheelhouse}
export CONDA_PKGS_DIRS="$WC_ROOT/cache/conda"
conda create -p "$WC_ROOT/envs/aether" python=3.10 pip -y
bash scripts/repro_a800/install_dependencies.sh
```

依赖安装使用阿里云 PyPI 镜像。若预置 `$WC_ROOT/wheelhouse`，安装器会复用其中匹配版本的 Linux/Python 3.10 安装包。模型和环境目录都放在数据盘；持久性以租赁平台说明为准。

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

验证状态：安装和 GPU 示例验证进行中，以后续验证记录为准。
