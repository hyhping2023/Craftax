# 安装指南

Craftax 具身交互层（FastAPI 服务 / Pygame GUI / 数据录制 / 数据集导出）的安装与验证。

## 环境要求

- Python 3.10–3.12（推荐 3.12；conda 环境示例见下）
- 可选：NVIDIA GPU（H800 / RTX 3090 等，需驱动 ≥ CUDA 12.x，`nvidia-smi` 确认）
- 本机开发（CPU）与 GPU 服务器共用同一套代码，无需平台分支

## 1. 创建环境（conda 示例）

```bash
conda create -n craftax python=3.12 -y
conda activate craftax
```

不使用 conda 时，用 `python3.12 -m venv .venv` 等价即可。

## 2. 安装 JAX

### CPU（本地开发 / Mac）

```bash
pip install -U "jax[cpu]"
```

### GPU（H800 / RTX 3090 等 NVIDIA 卡）

```bash
pip install -U "jax[cuda12]"
```

`jax[cuda12]` 自带 CUDA/cuDNN 运行库，不需要系统 CUDA toolkit，只需 NVIDIA 驱动 ≥ 535（CUDA 12.x）。
验证：

```bash
python -c "import jax; print(jax.devices())"
```

期望输出包含 `CudaDevice(...)`；CPU 环境输出 `CpuDevice(...)` 即可。

## 3. 安装项目

```bash
git clone https://github.com/hyhping2023/Craftax.git
cd Craftax
pip install -e ".[server,dataset]"
```

或仅安装运行时最小依赖：

```bash
pip install -e .                     # 仅游戏环境
pip install -e ".[server]"           # + FastAPI 服务端
pip install -e ".[dataset]"          # + Zarr/Parquet/视频录制
pip install -e ".[dev]"              # + ruff / pytest / pre-commit
```

可选：视频编码依赖（H.264 MP4 录制需要 ffmpeg 可执行文件，`pip install imageio-ffmpeg` 已含二进制；也可用系统 ffmpeg）。

## 4. 验证安装

```bash
# 游戏环境 + 全部新模块可导入
python -c "
import craftax
from craftax.contracts import list_task_ids
print('tasks:', len(list_task_ids()))   # 期望 77
from craftax.service.app import create_app
from craftax.gui.pygame_client import PygameGUI
from craftax.recording.recorder import AsyncRecorder
from craftax.dataset.reader import ShardReader
print('embodied modules OK')
"

# 运行测试套件（可选，约 30s）
python -m pytest craftax/gui/tests craftax/service/tests craftax/tasks/tests \
  craftax/recording/tests craftax/dataset/tests tests -q
```

## 5. 启动与使用

### 启动 API 服务

```bash
python -m uvicorn "craftax.service.app:create_app" --factory \
  --host 0.0.0.0 --port 8321
```

首次 reset/step 有 10–40s JIT 编译（生成/加载纹理缓存后变快）。

### 一键体验（demo）

```bash
# 模型式 API 交互（需服务运行中）
python scripts/demos/demo_api.py --steps 10

# 数据录制 -> 封存 -> 校验（需服务运行中）
python scripts/demos/demo_record.py --steps 60

# 读取最新数据集并导出 VLA/World Model 样本 / WebDataset
python scripts/demos/demo_dataset.py --latest --export data/webdataset/

# Pygame GUI：本机 remote 连接，或 --embedded 免服务
python scripts/demos/demo_gui.py --base http://127.0.0.1:8321
python scripts/demos/demo_gui.py --embedded
```

详见 `scripts/demos/README.md`。

## 6. GPU 服务器部署要点（H800 / RTX 3090）

- 用第 2 节的 `jax[cuda12]`；`nvidia-smi` 确认驱动 ≥ 535
- 服务器无头环境只跑 API + 录制（`demo_api` / `demo_record` / 批量采集），GUI 在本地开发机远程连接：`python scripts/demos/demo_gui.py --base http://<服务器>:8321`
- 批量录制建议分辨率：demo 用 `--block-pixel-size 64`（720p 级），正式采集用 `--block-pixel-size 18`（≈240p，体积小 ~13 倍）
- 录制输出写入 `data/spool/`（已 gitignore），不会进入仓库

## 常见问题

| 现象 | 处理 |
|---|---|
| `ImportError: No module named 'craftax'` | 确认在 `Craftax` 目录内运行，或 `pip install -e .` |
| GPU 环境 `jax.devices()` 只有 CPU | 驱动版本过低，升级 NVIDIA 驱动到 ≥ 535；或确认 `jax[cuda12]` 而非 `jax[cpu]` |
| 首次渲染/step 慢 | 正常 JIT 编译；纹理缓存 `texture_cache.pbz2` 生成后显著加速；强制重建：`CRAFTAX_RELOAD_TEXTURES=true` |
| 录制视频播放器无法打开 | 帧尺寸非 16 倍数（240p 的 198×234），管线内 imageio/ffmpeg 读写正常，外部播放器个别不支持 |
| `os.fork()` RuntimeWarning | 来自 imageio 子进程编码，无死锁，可忽略 |
