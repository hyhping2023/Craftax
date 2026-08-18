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
# demo 高清（默认 64px/格 ≈ 720p，704x832）；启发式策略：移动+砍树，
# collect_wood 任务约十几步真实完成，视频可见任务过程
python scripts/demos/demo_record.py --steps 60 --seed 2026

# 真实批量录制：240p（18px/格，198x234），存储节省约 13 倍
python scripts/demos/demo_record.py --steps 60 --block-pixel-size 18
```

随机策略录制一个 episode，DELETE 会话触发异步封存，
自动运行 validators 校验时间轴不变量。输出 `data/spool/<run-id>/` 下的 sealed shard。

> **注意**：不要使用 `--god-mode` 做任务演示——god_mode 会让玩家开局满背包
> （99 木材+全套装备），收集类成就第 0 步即"达成"，任务演示失去意义。

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

## 5. 任务依赖图与分级任务

```bash
# 实时从 registry 构建 92 个任务的静态图：统计/DAG 校验/类别/拓扑与语义层级/依赖链
python scripts/demos/demo_task_graph.py

# 展开某个根目标的前置依赖树
python scripts/demos/demo_task_graph.py --tree native.conquer_dungeon_bosses

# 打印静态产物 craftax/tasks/task_graph.json
python scripts/demos/demo_task_graph.py --json
```

背景：77 个基础任务 + 15 个由并发子 agent 提议的分级复合任务
（`craftax/tasks/builtin/hierarchy_tasks.py`），按 `TaskSpec.dependencies`
构建 DAG，`TaskGraph` 提供拓扑层级、语义层级（atomic/composite/root_goal）与
前置闭包，供后续 runtime planner（survey/滚动规划）消费。

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
