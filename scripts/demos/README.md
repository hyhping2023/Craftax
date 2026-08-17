# Demos

一组可直接运行的示例，覆盖「GUI 交互 → 模型 API → 数据录制 → 数据集读取」全流程。
所有脚本从仓库根目录运行；产生的数据统一写入 `data/`（已 gitignore）。

## 前置：启动服务

```bash
conda run -n craftax python -m uvicorn "craftax.service.app:create_app" --factory \
  --host 127.0.0.1 --port 8321
```

## 1. Pygame GUI

```bash
# remote 模式（连接上面的服务，需要显示环境）
python scripts/demos/demo_gui.py --base http://127.0.0.1:8321

# embedded 模式（同进程，无需服务）
python scripts/demos/demo_gui.py --embedded
```

键位：WASD 移动 · 空格 互动 · R 重置 · C 切换 human/model · 1-8 合成工具/武器 · 其余见 controls.py。

## 2. 模型式 API 交互

```bash
python scripts/demos/demo_api.py --steps 10 --seed 42
python scripts/demos/demo_api.py --save-frames --steps 5   # 保存每步 PNG 到 data/demo_frames/
```

展示 reset → step 循环 → 状态摘要 / 事件 token / 场景帧引用的完整协议。

## 3. 数据录制

```bash
# demo 高清（默认 64px/格 ≈ 720p，704x832）
python scripts/demos/demo_record.py --steps 40 --seed 2026 --task native.collect_wood

# 真实批量录制：240p（18px/格，198x234），存储节省约 13 倍
python scripts/demos/demo_record.py --steps 40 --block-pixel-size 18
```

随机策略录制一个 episode（40 步），DELETE 会话触发异步封存，
自动运行 validators 校验时间轴不变量。输出 `data/spool/<run-id>/` 下的 sealed shard。

**分辨率策略**：demo/演示用高分辨率（`--block-pixel-size 64`），
正式数据采集按 240p（`--block-pixel-size 18`）录制以控制存储成本；
两者都通过 `render.block_pixel_size` 配置，训练时按固定分辨率消费。

## 4. 数据集读取与样本导出

```bash
python scripts/demos/demo_dataset.py --latest
python scripts/demos/demo_dataset.py --latest --window 8 --export data/webdataset/
```

读取最新 shard：episode 摘要、视频帧对齐校验、VLA/World Model 窗口样本，
可选导出 WebDataset tar（`<episode_id>.mp4 + .json + .actions.json`）。

## 建议顺序

```bash
# 1) 起服务 -> 2) GUI 人工示范 -> 3) 录制一批 -> 4) 读取并导出训练样本
python scripts/demos/demo_record.py --run-id run-001 --steps 100 --gold-frames
python scripts/demos/demo_dataset.py --latest --export data/webdataset/
```

## 训练数据布局（与 embodied_environment_plan.md 第 7 节一致）

```
data/spool/<run-id>/<shard>/
├── shard_manifest.json      # 不可变 manifest（hash + 计数 + 配置）
├── tensors.zarr/            # 每 step 完整状态/动作/奖励（Zarr v3）
├── episodes.parquet         # episode 边界、seed、任务
├── transitions.parquet      # action metadata / 事件 / 指令
├── frame_index.parquet      # 视频帧 ↔ timestep 显式映射
├── video-<id>.mp4           # CFR H.264（默认 10 FPS，R=2）
└── gold-frames/             # 可选：采样帧的无损 PNG 副本
```
