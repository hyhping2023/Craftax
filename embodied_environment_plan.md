# Craftax 具身交互环境改造方案

## 1. 目标与范围

将当前 Craftax 从一个本地 JAX/Gymnax 强化学习环境，扩展为可供人类、脚本和具身模型统一交互的平台。该项目的首要训练用途是 **VLA（视觉-语言-动作）** 与 **World Model（世界模型）**；设计须同时保证视觉、动作、任务语义和完整状态的时间对齐。目标包括：

1. 提供 **Pygame GUI**：展示游戏场景、物品栏、角色状态、动作与任务状态，用于人类示范、模型调试和回放；Web GUI 不属于当前范围。
2. 提供服务端 API：模型可以创建环境会话、reset、提交动作、取得操作后的状态摘要、结构化观察和场景 frame。
3. 仅使用原生 Craftax 世界生成、游戏规则、奖励和终止规则；任务系统在首期只做标签、目标解释和评估，不改变初始世界或游戏规则。
4. 支持在一台配备 NVIDIA RTX 3090 的服务器上录制数据：每一步持久化完整状态和动作 metadata；画面按确定的模拟时间轴采样、编码为 MP4，以降低储存成本并保持与状态/动作的精确对应。
5. 不重写现有游戏逻辑。服务层、GUI、任务和录制层均作为现有 Craftax 环境外的适配层实现。

非目标（当前阶段）：

- 不更改 `game_logic.py` 的游戏规则；
- 不将 JAX 环境状态放入 Redis 或数据库；
- 不在 JIT 函数内部进行网络、Pygame、图像编码或磁盘 I/O；
- 不实现 Web GUI、浏览器 WebSocket 前端、多 GPU/多机调度或集群采集；
- 不在首期支持自定义初始世界、任意 `EnvState` 修改、任务专属奖励或任务专属终止规则。

### 1.1 已确定的设计基线

| 维度 | 已确定选择 | 直接影响 |
|---|---|---|
| GUI | 仅 Pygame | GUI 可与环境同进程或作为 HTTP 客户端；不建设浏览器前端。 |
| 首要训练任务 | VLA + World Model | 每个样本必须携带视觉时间轴、动作、原生任务/事件语义、完整状态及可切分的连续序列。 |
| 部署硬件 | 单台 RTX 3090 服务器 | 第一版采用单 GPU runner、CPU 图像/视频编码 worker、本地 NVMe spool；不提前引入多 GPU 路由。 |
| RGB 存储 | 定频采样后以 MP4 保存 | MP4 是状态的视觉派生数据；Zarr 状态与 Parquet 索引仍是时间轴权威来源。 |
| 游戏语义 | 原生 Craftax | 不改变 world generation、`craftax_step`、原生 reward 或终局条件；首期任务只读取并标注原生事件。 |
| 自动重置 | 显式 reset | API 和采集使用 `NoAutoReset`，完整保留 terminal state。 |

---

## 2. 当前代码基础与接入点

Craftax 已具备完整的纯 JAX 交互核心。调用路径如下：

```text
reset(key, params)
  -> generate_world(...)
  -> EnvState

step(key, state, action, params)
  -> craftax_step(...)
  -> next_state, reward, done, info
  -> renderer(next_state)
  -> RGB frame / symbolic observation
```

### 2.1 可直接复用的模块

| 能力 | 现有位置 | 使用方式 |
|---|---|---|
| 环境工厂 | `craftax/craftax_env.py:1` | 从环境名称选择新版/Classic、像素/符号环境与自动重置策略。 |
| 环境状态 | `craftax/craftax/craftax_state.py:39` | `EnvState` 是完整世界状态的 flax PyTree，包含地图、角色、背包、怪物、投射物、成就、随机数和时间步。 |
| 游戏单步逻辑 | `craftax/craftax/game_logic.py:3006` | `craftax_step` 为纯函数；应保持其与服务/存储无耦合。 |
| 无自动重置基类 | `craftax/environment_base/environment_bases.py:105` | `step` 在终局时返回 terminal state，适合 API 与录制。 |
| 自动重置基类 | `craftax/environment_base/environment_bases.py:12` | 适合常规 RL 训练，但终局状态会被 reset state 替换，不适合作为录制/API 默认。 |
| 像素环境 | `craftax/craftax/envs/craftax_pixels_env.py:19` | `CraftaxPixelsEnvNoAutoReset` 返回显示用像素观察。 |
| 渲染器 | `craftax/craftax/renderer.py:201` | `make_craftax_pixel_renderer` 可用不同方块像素大小生成 RGB renderer。 |
| Pygame 试玩客户端 | `craftax/craftax/play_craftax.py:77` | 已有键盘映射、Pygame 窗口和更高分辨率渲染，可拆分复用。 |
| 动作定义 | `craftax/craftax/constants.py:76` | 当前有 43 个离散动作，枚举名称可作为 action metadata 的权威来源。 |

### 2.2 关键现状

像素环境的 `get_obs()` 以 agent 分辨率渲染 RGB 图像：`craftax/craftax/envs/craftax_pixels_env.py:62`。试玩客户端则在 host 上将更高分辨率的 JAX render 结果送进 Pygame：`craftax/craftax/play_craftax.py:106`。

当前试玩轨迹仅用压缩 pickle 保存 `state/action/reward/done`：`craftax/craftax/play_craftax.py:165` 与 `craftax/craftax/play_craftax.py:194`。这可以作为调试工具保留，但不能作为正式训练数据格式：它没有 schema、任务版本、frame、分片、校验、断点恢复或跨语言读取能力。

---

## 3. 推荐总体架构

```text
Pygame GUI Client ─┐
Model / Script ────┴─ REST ── FastAPI Gateway
                                      │
                                      ▼
                           Single-host Session Manager
                                      │
                                      ▼
                         Authoritative Session Actor
                         - EnvState / EnvParams
                         - JAX PRNG key
                         - revision / action log
                         - native event/task labels
                         - recorder hooks
                           │          │
                           ▼          ▼
                     JAX step/render  Async Recorder
                                      - per-step state/action/event
                                      - sampled RGB -> CFR MP4
                                      - frame index / shard validation
```

### 3.1 Session Actor：环境状态唯一所有者

每一个环境会话由一个串行 actor 管理，actor 是该会话的唯一可变状态所有者：

```python
SessionState(
    session_id,
    env_name,
    state,          # JAX EnvState
    params,         # JAX EnvParams
    key,            # JAX PRNG key
    task_instance,
    revision,       # 成功 transition 后单调递增
    terminated,
)
```

每次命令必须按以下顺序原子执行：

```text
校验请求与控制权
-> split PRNG key
-> JAX env.step
-> 得到 terminal state / reward / info
-> render（如请求或录制要求）
-> host 化所需的 frame 与摘要
-> recorder 追加 transition
-> revision + 1
-> 返回 Snapshot
```

这样可避免多人/模型并发控制时动作乱序、随机数复用、重复提交和“画面不是操作后状态”的问题。

### 3.2 为什么必须默认使用 NoAutoReset

API 和录制层必须创建 `CraftaxPixelsEnvNoAutoReset` 或 `CraftaxSymbolicEnvNoAutoReset`。自动重置基类会在 `done` 时用 reset 后的 state 替换状态，具体逻辑见 `craftax/environment_base/environment_bases.py:33`。

对于具身训练数据，必须保留以下明确语义：

```text
state[t] + action[t] -> reward[t], done[t], state[t + 1]
```

当 `done[t]` 为真时，`state[t + 1]` 必须仍是 terminal state；下一 episode 应当由独立 `reset` 命令开始。

### 3.3 JAX 边界规则

- `reset`、`step`、renderer 应维持固定 shape，并在启动或首次创建会话时预热编译。
- `EnvState` 是 JAX/Flax PyTree，不可直接 JSON 序列化。
- `jax.device_get()` 或 `np.asarray()` 必须在 JIT 外发生；它会同步所需 device 计算并把数据带回 host。
- frame 编码、Pygame 绘制、文件写入、JSON 构造和 HTTP/WebSocket 推送必须在 host 层进行。
- FastAPI event loop 不应执行 JAX 推理、图像编码或大规模写盘；这些操作应在 session runner、受控线程池或独立 worker 中运行。
- session 内 transition 必须串行；多会话扩展时通过多个 runner/进程分配会话，而非并发修改同一 state。

---

## 4. GUI 方案

### 4.1 第一阶段：Pygame 本地调试 GUI（推荐）

从现有 `CraftaxRenderer` 拆分独立 Pygame GUI client。当前只支持两种 **Pygame** 运行方式：

1. **embedded 模式**：与环境进程直接传递 `uint8 HWC RGB` 数组，用于低延迟人类示范、模型调试和回放；
2. **HTTP polling 模式**：通过 REST API 获取 Snapshot 与 PNG frame，用于验证服务协议和在服务器端运行环境、在带显示的开发机运行 GUI。

当前不实现 WebSocket GUI 流；若将来需要远程低延迟 Pygame 控制，再在 API 层增加该传输方式。

GUI 应包含：

- 场景画布：使用现有 renderer，当前 frame 已覆盖地图和物品栏；
- 状态面板：生命、饥饿、饮水、能量、法力、楼层、经验与属性；
- 背包/装备摘要：资源、工具、护甲、药水、附魔；
- 任务面板：任务 ID、目标、进度、成功/失败状态；
- 控制面板：键盘动作、暂停、reset、controller 切换（human/model/replay）；
- 调试面板：session ID、seed、revision、timestep、reward、最近成就和录制状态。

Pygame 仅为 frame 消费者，不能成为 state 的权威持有者。

### 4.2 非当前范围：Web GUI

当前阶段明确不实现 Web GUI，也不实现浏览器专用的 WebSocket frame stream。服务端保留 REST 的 frame reference 与结构化 Snapshot，保证未来添加浏览器客户端时不需要修改环境核心或会话协议。

### 4.3 Frame 格式

| 场景 | 默认格式 | 说明 |
|---|---|---|
| 同进程 Pygame | raw RGB `uint8` | 低延迟，无编码/解码。 |
| REST API / Pygame remote | PNG | Craftax 为像素风格并有文字/物品栏，PNG 比 JPEG 更适合保真。 |
| 录制管线输入 | raw RGB `uint8` + `frame_index` | 在 host 编码前保留无歧义的离散模拟帧编号。 |
| 训练 RGB | MP4（H.264，固定 FPS） | 以视频降低储存成本；每个视频帧由 manifest 显式映射到环境 timestep。 |
| 对齐/回归金标集 | PNG frame bundle | 小规模保存每一步无损帧，用于验证 MP4 编码、renderer 和 reader。 |
| 深度/分割/掩码 | Zarr 或 PNG | 未来如增加这些模态，必须无损，不能随 RGB 混入有损视频。 |

---

## 5. 服务端 API 设计

### 5.1 组件边界

- **FastAPI Gateway**：鉴权、请求 schema 校验、限流和 REST 协议；
- **Session Manager**：创建/销毁 session、查找 actor、维护控制权 lease 与 command 去重；
- **Session Actor/Runner**：独占环境状态并执行 reset/step/render；
- **Frame Encoder**：把 host RGB frame 编为 PNG/JPEG；
- **Recorder**：接收不可变 transition record，异步写入数据集。

单机开发版可以使用进程内 session registry。多 runner 时使用 Redis 仅保存 `session_id -> runner`、TTL、lease、幂等结果索引；**绝不将每步 EnvState 写入 Redis**。

### 5.2 REST API 草案

#### 创建 session

```http
POST /v1/sessions
```

```json
{
  "env_name": "Craftax-Pixels-v1",
  "seed": 42,
  "task": {"task_id": "survive", "version": "1.0.0", "params": {}},
  "render": {"format": "png", "mode": "human"},
  "recording": {"enabled": true, "dataset_run_id": "demo-001"}
}
```

返回 revision `0` 的 Snapshot 与 frame reference。

#### 显式 reset

```http
POST /v1/sessions/{session_id}/reset
```

请求带 `seed`（可选）、`expected_revision` 和 `command_id`。Reset 产生一个新 episode，并记录新 episode 的 task 参数和 seed。

#### 执行动作

```http
POST /v1/sessions/{session_id}/step
```

```json
{
  "action": {"id": 1, "name": "LEFT"},
  "expected_revision": 12,
  "command_id": "uuid",
  "return": {"frame": "reference", "observation": "summary"}
}
```

响应包含：

```json
{
  "session_id": "sess_...",
  "revision": 13,
  "transition_id": "tr_...",
  "action": {
    "requested": {"id": 1, "name": "LEFT"},
    "applied": {"id": 1, "name": "LEFT"}
  },
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "info": {"discount": 1.0},
  "frame": {
    "revision": 13,
    "url": "/v1/sessions/sess_.../frames/13",
    "content_type": "image/png",
    "width": 0,
    "height": 0
  },
  "state_summary": {"timestep": 13}
}
```

frame 采用独立端点，避免 Base64 塞入 JSON：

```http
GET /v1/sessions/{session_id}/frames/{revision}
Accept: image/png
```

如果集成方必须让同一个 HTTP 响应包含图片，可额外支持 `Accept: multipart/mixed`，由 JSON metadata part 和 `image/png` part 组成；这应是兼容模式而非默认协议。

#### 获取结构化 observation 与状态

```http
GET /v1/sessions/{session_id}/observations/{revision}
GET /v1/sessions/{session_id}/state?revision=13&detail=summary
```

小型摘要使用 JSON；高维 symbolic observation 可以提供 `application/msgpack`，且协议必须明确 dtype、shape、字节序和 schema version。

### 5.3 一致性、重试与控制权

每个变更命令必须包含：

- `expected_revision`：乐观并发控制。旧 revision 返回 `409 Conflict` 与最新摘要；
- `command_id`：同一客户端重试时返回原结果，不重复推进环境；
- `controller lease`：控制者为 `human`、`model:<run_id>` 或 `replay`，避免多个输入源争用。

### 5.4 非当前范围：WebSocket

当前 Pygame GUI 和模型客户端通过 REST 轮询或同步 `step` 响应工作，不实现 WebSocket。保留 `revision`、`command_id` 和 frame reference 语义；未来如需低延迟远程 GUI 或观测流，可在不更改 Session Actor 的前提下添加 WebSocket 适配器。

---

## 6. 原生 Craftax 任务与语义标注

当前阶段固定使用原生 Craftax 规则：`generate_world`、`craftax_step`、原生 reward、原生终局和 43 个动作均不改变。任务系统的职责不是改变环境，而是把原生状态、成就和事件解释成可供 VLA/World Model 训练的可复现标签。

任务定义仍须版本化、可序列化、可复现，但当前 `TaskSpec` 不得包含初始状态覆盖、reward 覆盖、终局覆盖或动作约束。建议定义 `TaskSpec` 与只读 `TaskAdapter`：

```python
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    version: str
    instruction: str
    objective: str
    success_predicate: dict       # 仅评估 EnvState / achievements
    annotation_predicates: list[dict]
    renderer_config: dict
    action_vocabulary_version: str
```

只读 `TaskAdapter` 必须在不改动 `craftax_step`、world generation 或 `EnvParams` 的前提下：

- 从 state、原生 achievement、原生 reward 和终局状态计算任务进度及成功/失败标记；
- 生成稳定的任务指令、子目标、事件和成就标签；
- 输出 VLA 所需的 `(instruction, visual frame, action)` 语义关联；
- 输出 World Model 所需的环境事件 token（资源获得、合成、伤害、楼层变化、终局等）；
- **不得**修改 reset seed、初始状态、reward、终局、可执行动作或原生 action mask。

每个 task bundle 必须保存：

```json
{
  "task_id": "collect_wood",
  "task_version": "1.0.0",
  "task_hash": "sha256:...",
  "environment_code_revision": "git:...",
  "state_schema_hash": "sha256:...",
  "action_schema_hash": "sha256:...",
  "reward_definition_version": "...",
  "termination_definition_version": "...",
  "renderer_config": {"...": "..."}
}
```

推荐的第一批任务：

| 任务类别 | 示例 | 训练价值 |
|---|---|---|
| 生存 | 在 N 步内存活 | 长时序控制、资源维持。 |
| 原子技能 | 收集木材、制作木镐、放置火把 | 视觉-动作对齐与行为克隆。 |
| 组合任务 | 收集资源后制作指定装备 | 分层规划。 |
| 目标导航 | 到达楼层、找到熔炉、打开宝箱 | 目标条件策略。 |
| 战斗 | 击败指定敌人或 Boss 阶段 | 反应控制与风险管理。 |
| 演示任务 | 人类完成指定流程 | 高质量 imitation / VLA 数据。 |

---

## 7. VLA 与 World Model 数据集录制设计

### 7.1 数据集分层

采用“双层数据集”，并把 **Zarr/Parquet 的 transition 时间轴设为唯一权威**：

1. **Canonical（权威录制层）**：逐步完整 state、action、reward、终局、原生事件、任务文本/标签、seed、版本和精确的 frame-to-timestep 映射。它可审计、可回放，是视频对齐的真相来源。
2. **Derived VLA 层**：按固定上下文窗口导出 `(instruction, RGB frame/clip, action, action name, task progress)`；采样只能基于 canonical 索引，不能根据视频容器的近似时间戳推断 action 对齐。
3. **Derived World Model 层**：导出连续 `(state_t, frame_t, action_t, state_t+1, frame_t+1, event_t)` 序列，并按 episode 级别切分 train/validation/test，防止同一轨迹泄漏。

不要尝试以一种格式同时解决状态随机访问、视频压缩、列式查询和训练顺序吞吐。

### 7.2 时间对齐规范

对有 `T` 个动作的 episode，强制使用：

```text
state[0..T]           # T + 1 个完整状态
frame[0..T]           # T + 1 个 frame
action[0..T-1]        # T 个动作
reward[0..T-1]        # T 个 reward
terminated[0..T-1]
truncated[0..T-1]
```

严格语义：

```text
action[t] 作用于 state[t]
得到 reward[t]、terminated[t]、truncated[t]、state[t + 1]
frame[t] 描述 state[t]
```

### 7.3 MP4 的精确时间轴对齐

MP4 的压缩时间戳不是训练数据的权威时间轴。每个录制 episode 必须维护离散 `timestep` 和 `frame_index`，并在索引中显式记录二者的映射。

设环境每秒执行 `S` 个 step，录制视频帧率为 `F`，且每隔 `R` 个环境 step 保存一个 frame。第一阶段必须要求：

```text
S % F == 0
R == S / F
```

例如 `S=20`、`F=10`、`R=2`：视频第 `i` 帧总是对应 `state[timestep=2*i]`。这避免按 wall-clock 时间、浮点 PTS 或近似 seek 推断对齐关系。

每个 episode 另写一个 `frame_index` 表（Parquet 或 Zarr），至少包含：

```text
episode_id, video_id, frame_index, timestep, state_index,
sim_time_ns, is_initial_frame, is_terminal_frame, encoder_pts
```

规则：

- `frame_index=0` 必须对应 reset 后的 `state[0]`；
- 常规帧只在 `timestep % R == 0` 时保存，且对应当前的 `state[timestep]`；
- terminal state 即使不落在采样周期上，也必须单独保存为 `is_terminal_frame=true`，以便把最后一个 action 与结果对齐；
- 所有 action 和 state 仍每 step 保存，绝不因视频降采样而减少；
- MP4 使用固定帧率（CFR），禁止可变帧率（VFR）；encoder 必须确认解码帧数等于 `frame_index` 表的行数；
- 禁止用按秒 seek 的方式构建训练样本；reader 必须按 `frame_index` 解码或在顺序读取时校验 frame ordinal。

建议 RGB 视频编码为 H.264，使用 yuv420p、固定 `F`、明确 GOP（例如 16 或 32）。H.264 是有损视觉观测；完整 state、文本任务标签和 action 仍从 canonical 存储读取。对每个 dataset revision 保留小型 PNG 金标集，用于校验“原始 renderer RGB -> MP4 解码 RGB”失真是否在预设阈值内。

### 7.4 格式选择

| 内容 | 推荐格式 | 原因 |
|---|---|---|
| 固定 shape 的完整状态 | Zarr | 每 step 保存；多维数组、时间 chunk、压缩、连续窗口读取。 |
| 固定 shape action/reward/done/event token | Zarr | 每 step 保存；与 state 时间维严格对齐。 |
| transition/episode/action/frame index metadata | Parquet | 列筛选、任务标签、VLA 指令、视频帧到 timestep 的显式索引。 |
| RGB 画面 | CFR MP4（H.264） | 每 `R` step 采样一次，显著降低储存；必须由 `frame_index` 映射到 state/action 时间轴。 |
| 对齐验证子集 | PNG frame bundle | 每 step 无损保存少量 episode，用于测试 renderer/encoder/reader。 |
| 深度/分割/法线 | Zarr/PNG | 未来增加这些模态时保持无损。 |
| schema/task/manifest | JSON/YAML + SHA-256 | 可读、可审计、稳定。 |

不要将整个 `EnvState` pickle 作为正式格式。应把 PyTree 展开为稳定路径，例如：

```text
state/map
state/item_map
state/player_position
state/player_health
state/inventory/wood
state/inventory/armour
state/achievements
state/state_rng
state/timestep
```

每个字段应在 schema 中声明 dtype、shape、单位/语义和版本。任何 state/action 语义变动都创建新的 schema hash，不覆盖历史数据。

### 7.5 Action metadata

除了离散 action id，还必须记录执行上下文：

```json
{
  "action_id": 1,
  "action_name": "LEFT",
  "source": "human",
  "requested": {"id": 1},
  "applied": {"id": 1},
  "command_id": "uuid",
  "policy_id": "optional-policy-name",
  "policy_checkpoint_hash": "optional-sha256",
  "instruction_id": "native.collect_wood.v1",
  "instruction_text": "Collect wood.",
  "event_tokens": ["COLLECT_WOOD"],
  "log_prob": null,
  "value_estimate": null,
  "latency_ms": 0,
  "valid": true
}
```

`source` 至少支持：`human`、`scripted`、`policy`、`random`、`replay`、`curriculum_generator`。

### 7.6 推荐目录布局

```text
datasets/
└── craftax-embodied-v1/
    ├── dataset.yaml
    ├── schemas/
    │   ├── state-v1.json
    │   ├── action-v1.json
    │   └── transition-v1.json
    ├── tasks/
    │   └── collect_wood/1.0.0.json
    ├── revisions/
    │   └── r000001/
    │       ├── manifest.json
    │       ├── catalog.parquet
    │       └── splits.parquet
    └── shards/
        └── task=collect_wood/
            └── shard=<uuid>/
                ├── tensors.zarr/
                ├── episodes.parquet
                ├── transitions.parquet
                ├── frame_index.parquet
                ├── video-000.mp4
                ├── gold-frames/            # 可选：无损对齐验证子集
                └── shard_manifest.json
```

### 7.7 RTX 3090 单机采集流水线

```text
RTX 3090 上的 JAX batch step / render
  -> 每 K 步 device-to-host 微批传输
  -> 有界内存队列
  -> state/action writer + frame sampler + episode assembler
  -> CPU/硬件视频 encoder（固定 FPS CFR MP4）
  -> 本地 NVMe spool
  -> shard seal + validator
  -> 数据集目录或对象存储
  -> 单一 committer 更新 dataset revision manifest
```

单机部署边界：

- 运行一个 JAX rollout runner，独占 RTX 3090；交互 API/GUI session 与大规模 rollout 不应同时抢占该 GPU；
- Pygame GUI 用于人类演示和小规模调试。大规模 VLA/World Model 数据通过批量 `vmap` rollout 收集，而不是创建海量 API session；
- 使用 JAX GPU 后端执行 step/render；host 收到的数据必须分成“每 step 的 state/action 批”和“按 `R` 采样的 RGB 批”；
- 视频编码作为独立、有界的 worker。优先在服务器可用时使用 NVIDIA NVENC；若运行环境/库不稳定，则先使用 FFmpeg CPU 编码以保证正确性，再测量升级 NVENC；
- recorder 队列触顶时不可静默丢失 state/action 或跳过应保存的采样 frame；必须暂停/减速 rollout 并计数背压事件；
- producer 写入私有本地 NVMe spool，多个 producer 不得 append 同一 Zarr/Parquet/MP4；
- shard 目标压缩体积先设为 0.5--4 GiB，最终根据 3090 上实际 step/s、D2H 带宽、CPU/NVENC 编码吞吐和磁盘吞吐调整。

### 7.8 3090 基准与默认参数确定流程

在实现视频写入前，先运行 5--10 分钟的单机基准，记录：环境 step/s、render step/s、GPU 显存、device-to-host 带宽、CPU/NVENC 编码 fps、NVMe 写入 MB/s、队列最大水位及每 transition 的压缩字节数。

初始建议不是最终承诺：

```text
environment step rate S = 20 Hz
video rate F = 10 FPS
frame stride R = 2 steps
MP4 = H.264, yuv420p, CFR, GOP 20
host transfer micro-batch K = 32 steps
```

该组合满足 `S % F == 0`，并使每个视频帧对应偶数 timestep。若基准显示 render/编码受限，应优先降低 `F` 且保持整除关系（如 `20 Hz -> 5 FPS, R=4`），而不是丢弃已声明需要保存的帧。VLA/World Model 训练 reader 必须读取 `frame_index`，不能假定所有 action 都有直接对应 frame。

### 7.9 封存、恢复与数据完整性

1. Producer 写私有目录：`spool/{run_id}/{producer_id}/{attempt_id}/`。
2. 每个 episode 维护 WAL：episode ID、已写 transition 范围、seed、任务参数与可选 state checkpoint。
3. shard 满后 flush 所有状态/元数据/frame 文件，验证时间轴不变量并生成 hash。
4. 写不可变 `shard_manifest.json`，上传所有对象并验证远端 hash/size。
5. 只有被 dataset revision manifest 引用的 sealed shard 才对训练可见。
6. 已上传但未提交的数据可按 `shard_id + content_hash` 幂等提交；损坏或不完整 episode 必须标注 `truncated/invalid`，不伪造成完整数据。

必做校验：

- 文件内容 hash 与大小；
- Zarr shape、dtype、chunk 与时间长度；
- Parquet schema、row count、`episode_id/timestep` 唯一性；
- `len(state) == len(action) + 1`；
- `frame_index` 的每行对应存在的 `state_index/timestep`，且每个 `(episode_id, video_id, frame_index)` 唯一；
- 视频解码的帧数严格等于 `frame_index` 对应行数；固定采样帧满足 `timestep % R == 0`，终局帧除外；
- `frame_index=0` 对应 reset 后 `state[0]`，终局 action 的结果 state 总可定位到一个 frame；
- terminal/truncated 不变量；
- 抽样 episode 将 MP4 解码帧与 PNG 金标 frame 比较，验证尺寸、颜色格式、frame order 和失真阈值；
- task/schema/renderer/asset 版本与 hash 完整存在。

训练任务必须引用确定的 dataset revision（如 `r000001`），不能只引用可变的 `latest`。

---

## 8. 推荐项目结构

```text
craftax/
├── service/
│   ├── app.py                 # FastAPI application
│   ├── api_models.py          # Pydantic request/response schema
│   ├── session_actor.py       # 原子 session 状态机
│   ├── session_manager.py     # 单机生命周期、lease、幂等
│   └── frame_encoder.py       # RGB -> PNG（API/Pygame remote）
├── gui/
│   ├── pygame_client.py       # 本地 GUI
│   ├── controls.py            # 键盘/模型/replay 控制
│   └── view_models.py         # 展示数据模型
├── tasks/
│   ├── base.py                # 只读 TaskSpec / TaskAdapter
│   ├── registry.py
│   └── builtin/               # 原生成就/事件的任务指令和标签
├── recording/
│   ├── recorder.py            # transition recorder API
│   ├── state_codec.py         # EnvState PyTree -> schema arrays
│   ├── frame_sampler.py       # timestep -> frame index 映射
│   ├── video_writer.py        # RGB -> CFR MP4
│   ├── shard_writer.py
│   ├── manifest.py
│   └── validators.py
└── dataset/
    ├── reader.py
    ├── vla_windows.py         # instruction/RGB/action 序列
    ├── world_model_windows.py # state/frame/action 连续序列
    └── export_webdataset.py
```

在第一阶段不修改 `craftax/craftax/game_logic.py`、现有 EnvState 定义或默认 Gymnax API。新的 service/GUI/task/recording 模块依赖现有 `reset`、`step` 与 renderer。

---

## 9. 分阶段实施与验收标准

### Phase 1：Pygame 与单机 API 交互 MVP

范围：Pygame、FastAPI、单 session actor、`NoAutoReset`、PNG API frame 和轻量调试录制。运行目标是一台 RTX 3090 服务器，但此阶段只验证单环境交互正确性。

实现内容：

- `SessionActor`，明确拥有 `(key, state, params, revision)`；
- `POST /sessions`、`POST /reset`、`POST /step`、`GET /frames/{revision}`；
- Pygame embedded/HTTP polling 客户端；
- Snapshot、离散 action schema、revision CAS、command id 幂等；
- terminal state 显式返回；
- JSONL + PNG 的开发期录制器与 deterministic replay test；
- 原生 Craftax event/achievement 到文本指令与标签的最小 registry。

验收：

- 同一 seed 与 action 序列可重放到同一状态；
- API 的 step 后 PNG frame 与本地调用 renderer 的画面一致；
- terminal state 完整可见、可保存，且不会自动 reset；
- 重复相同 command id 不会额外推进状态；
- 旧 expected revision 返回 409；
- Pygame 人类动作、API 模型动作和 replay 动作使用同一 action metadata schema。

### Phase 2：VLA/World Model 训练级录制

范围：原生任务语义标注、每 step 的 Zarr/Parquet canonical 时间轴、按固定采样率写 CFR MP4、shard manifests、reader 与回放/对齐校验。

实现内容：

- 原生成就/事件到 task instruction、progress 和 event token 的 registry；
- EnvState PyTree 稳定展开与 schema；
- 每 step state/action/reward/done/event 的 Zarr writer；
- transition/episode/frame index 的 Parquet writer；
- `frame_sampler` 与 `video_writer`：固定 `S/F/R`，CFR H.264 MP4，强制 terminal frame；
- PNG 金标 episode、immutable shard/revision manifests；
- VLA window reader 与 World Model sequence reader；
- episode replay、frame-index、视频解码和对齐 validator；
- 3090 上的 step/render/encode/storage 基准。

验收：

- 可读取任意 episode 的连续状态和视频帧窗口；
- 每个 episode 满足 `states = actions + 1`；
- state、action、事件均逐 step 对齐；每个 MP4 frame 都可由 `frame_index` 定位到精确 state timestep；
- reset state 和 terminal state 均有可定位的视频 frame；
- task/schema/renderer/encoder/hash 可确定数据语义；
- 可导出 VLA `(instruction, RGB, action)` 样本与 World Model 连续预测样本；
- 基准结果决定正式的 `S/F/R/K/GOP/shard size` 默认值。

### Phase 3：RTX 3090 批量采集与稳健写入

范围：批量 JAX rollout、异步录制、NVMe spool、恢复与吞吐优化；不包含 Web GUI 或多 GPU/多机路由。

实现内容：

- 批量 `vmap` rollout runner 和固定大小 host 微批搬运；
- 有界队列、CPU/NVENC 视频编码、NVMe spool、WAL、断点恢复；
- MP4 + frame index 的训练导出与可选 WebDataset 封装；
- 运行时指标：step/s、render/s、D2H 带宽、encoder fps、NVMe MB/s、队列水位、每 transition 字节数；
- 不可变 dataset revision 与对象存储上传（如后续需要）。

验收：

- recorder 背压不会静默丢失 state/action 或声明应写的视频帧；
- producer 崩溃后不会发布不完整 shard；
- 每个封存 MP4 的 frame count 与 index 完全一致；
- 3090 的实际吞吐和存储占用达到基准中定义的可接受目标；
- 同一 dataset revision 能可复现地生成 VLA 与 World Model 训练样本。

---

## 10. 已确认决策与剩余参数

### 10.1 已确认

| 决策 | 已确认方案 | 设计结果 |
|---|---|---|
| GUI | 仅 Pygame | 不实现 Web GUI；Pygame 支持 embedded 与 REST polling。 |
| 模型服务 | FastAPI REST | API 返回 Snapshot + PNG frame reference；不把 frame Base64 放入 JSON。 |
| 训练用途 | VLA + World Model | 录制状态、动作、原生事件、任务文本和可索引视频帧；实现两个独立训练 reader。 |
| 部署 | 单台 RTX 3090 | 单 GPU rollout runner、异步 host writer、CPU/NVENC 视频 worker、本地 NVMe spool。 |
| RGB 录制 | 定频采样、CFR MP4 | 以 `frame_index` 显式对齐 timestep；保留 PNG 金标子集。 |
| 状态和动作 | 每 step 保存 | Zarr 是完整 state/action 时间轴；Parquet 保存 metadata/index。 |
| 游戏规则 | 原生 Craftax | 不修改世界生成、原生 reward、终局或动作；任务只读取和标注。 |
| 终局 | 显式 reset | 一律使用 `NoAutoReset`，完整保存 terminal state。 |

### 10.2 实现前仍需通过基准确定的参数

以下不是架构方向选择，而是必须在 RTX 3090 实机测量后固化到 dataset manifest 的运行参数：

| 参数 | 暂定值 | 如何确定 |
|---|---|---|
| 环境 step rate `S` | 20 Hz | 以真实交互需求与 3090 上 JAX step/render 吞吐确定。 |
| 视频帧率 `F` | 10 FPS | 必须满足 `S % F == 0`；若瓶颈明显，优先尝试 5 FPS。 |
| frame stride `R` | `S / F`（初始为 2） | 固定映射，不接受运行时随意变化。 |
| 视频 codec / GOP | H.264 / GOP 20 | 比较 CPU FFmpeg 与 NVENC 的编码速度、质量、可用性和可复现性。 |
| render 分辨率 | agent / human 两档待测 | VLA 需要足够读懂物品栏和局部环境；以真实模型输入尺寸和压缩后数据成本决定。 |
| JAX batch 与微批 `K` | `K=32` 起测 | 以显存、D2H 带宽、NVMe 速度和 queue 背压确定。 |
| 并行环境数 | 待测 | 从小批量扩至 3090 可稳定持续录制且无队列积压的上限。 |
| shard size | 0.5--4 GiB | 以恢复速度、文件数量、训练读取吞吐和上传特性确定。 |

所有最终值均写入 `dataset.yaml`、task bundle 和 shard manifest，避免训练时不知道视频、状态和动作的采样语义。

---

## 11. 关键技术依据

- JAX 异步 dispatch 与 host 同步：<https://docs.jax.dev/en/latest/async_dispatch.html>
- JAX `device_get`：<https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html>
- JAX 并发限制：<https://docs.jax.dev/en/latest/concurrency.html>
- FastAPI 异步与 CPU 密集工作：<https://fastapi.tiangolo.com/async/>
- Pygame image API：<https://www.pygame.org/docs/ref/image.html>
- FFmpeg H.264 encoder 文档：<https://ffmpeg.org/ffmpeg-codecs.html>
- NVIDIA Video Codec SDK（NVENC）：<https://developer.nvidia.com/video-codec-sdk>
- Apache Parquet 数据模型：<https://parquet.apache.org/docs/concepts/>
- Zarr v3 sharding codec：<https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html>
- WebDataset：<https://github.com/webdataset/webdataset>
- RLDS episode/step 语义：<https://github.com/google-research/rlds>
- MCAP 格式：<https://mcap.dev/spec>
