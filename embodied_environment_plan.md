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
    dependencies: list[str]       # 严格前置 task_id；用于静态 DAG，不改变游戏规则
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

### 6.1 面向地图实例的任务规划与录制闭环

自动录制不能将单个 builtin task 直接等同于一个 episode。推荐以“**可复现世界实例中的根目标尝试**”为录制和规划单位：先基于地图 seed 观察周边世界，再从任务依赖图中实例化一个当前可执行的子图，执行有限长度的计划片段，并在关键状态变化后验证和重规划。

这既保留了 seed 级复现能力，也能保留同一地图中不同目标、不同路线和失败恢复的有效样本。

#### 6.1.1 标识层级：seed 不是唯一 episode ID

| 层级 | 标识/组成 | 语义与用途 |
|---|---|---|
| `world_instance_id` | `environment_code_revision + env_name + map_seed + generation_config_hash` | 可重复生成的世界实例；`map_seed` 是核心键，但不单独充当唯一 ID。 |
| `episode_id` | UUID/ULID；关联 `world_instance_id + root_task_id + task_graph_version + planner_version + controller/policy version` | 一次从 reset 到终局、主动停止或预算耗尽的完整目标尝试。 |
| `plan_run_id` | UUID；关联一次 survey、候选计划集与选中计划 | 一个 episode 内的一次规划决策；重规划必须创建新的记录，而不是覆盖旧计划。 |
| `task_execution_id` | UUID；关联 `plan_run_id + task_id + occurrence_index` | 一个任务节点的一次实际执行，记录起止状态、动作范围和结果。 |

同一个 `world_instance_id` 可以有多个 episode，例如同一 seed 下尝试不同根目标、同一根目标使用不同规划器、或在不同初始决策后形成不同路线。重复运行相同 `world_instance_id`、根目标、控制器版本和行动决策时，仍应使用独立 `episode_id`，并用 `attempt_index` 表示重复尝试。

#### 6.1.2 两层任务图

任务图必须分成静态语义层与运行时实例层，避免把“游戏规则上的依赖”与“本次地图的资源、距离和风险”混为一谈。

1. **静态任务依赖图（Task Dependency Graph）**
   - 节点是版本化 `TaskSpec.task_id`；边来自现有 `TaskSpec.dependencies`，表达不可绕过的严格前置任务。
   - 图必须为 DAG；注册 builtin task 时校验依赖目标存在、无环且版本固定。
   - `TaskSpec` 继续承担原生成功谓词和语义标注；不向游戏规则注入任务奖励、初始状态或动作约束。
   - 规划器额外读取只读的 `TaskPlanningDescriptor`（可由 builtin registry 派生），描述每个任务的逻辑前置条件、预期效果/物资变化、候选资源类型、工具要求、预估风险和替代任务组。它是规划元数据，不是环境真相来源。

2. **运行时计划图（Runtime Plan Graph）**
   - 每次 survey 后，以根目标为反向起点裁剪静态 DAG，只保留当前尚未满足、可能可执行的任务与其依赖。
   - 节点带运行时属性：`status`、前置条件是否满足、所需/已知可用物资、候选地点、路径代价、风险、置信度及估计收益。
   - 边除严格任务依赖外，还可包含 `produces -> requires` 的资源/能力边和可替代分支。例如“获得食物”可以有采集、狩猎或制作等 OR 分支；选中一种路径不应删除其他候选路径。
   - 若扫描证明某资源不可达或某节点失败，标记该实例节点/边为不可行并重新求解；禁止修改静态 `TaskSpec.dependencies` 来适应单次地图。

建议的运行时节点结构：

```json
{
  "task_id": "native.craft_wood_pickaxe",
  "occurrence_index": 0,
  "strict_dependencies": ["native.collect_wood", "native.place_table"],
  "preconditions": [
    {"kind": "inventory", "item": "wood", "min_count": 2, "status": "satisfied"},
    {"kind": "reachable_station", "station": "crafting_table", "status": "unknown"}
  ],
  "candidate_locations": [{"position": [x, y], "distance": 17, "confidence": 0.9}],
  "estimated_cost": {"steps": 35, "risk": 0.1},
  "status": "candidate"
}
```

#### 6.1.3 Survey：有边界、可审计、非全知的充分性扫描

“充分性扫描”不应表示读取完整 `EnvState.map` 后做全图最优规划；这会让录制数据含有 agent 在正常交互中不可获得的特权信息。默认 planner 只能消费与 agent 一致的观测、背包、原生事件和已经访问的位置。完整 `EnvState` 仍按 canonical 录制，但仅供回放、离线评估和调试。

每轮 `survey` 必须有明确预算，而非无限探索：

```text
输入：当前位置、当前 observation/history、inventory、根目标、scan_budget
输出：SurveySnapshot + 已消耗动作区间 + 未知区域/不确定项
预算：最多步骤数、最大探索半径或风险上限、最大目标候选数
```

`SurveySnapshot` 至少记录：

- 起止 `state_index`、位置、背包、生命/饥饿等生存状态，以及使用的观测 schema；
- 已观察的资源、工作站、敌对实体、地形、出口和可达路径；
- 每项发现的数量/距离估计、观测来源、置信度和发现时 timestep；
- 资源状态的三值或四值语义：`confirmed_available`、`not_observed`、`confirmed_unreachable`、`confirmed_absent`；
- 未访问区域、扫描预算消耗和提前停止原因。

只有当游戏规则或已完成的覆盖证明支持时才可写 `confirmed_absent`；“当前扫描范围未发现”必须写为 `not_observed`。如需使用完整状态做 oracle/课程生成，必须设置 `planner_information_mode="privileged"`，与正常部分可观测录制分开标记、分割和评估，不能混入同一示范分布。

#### 6.1.4 规划与执行状态机

采用滚动规划，而非“先生成一条完整长计划后不再验证地执行”。每个 episode 按下列状态机运行：

```text
RESET
  -> INITIAL_SURVEY
  -> BUILD_RUNTIME_GRAPH
  -> PLAN
  -> EXECUTE_CHUNK
  -> VALIDATE
       -> SUCCESS / TERMINAL / BUDGET_EXHAUSTED
       -> RESURVEY -> REPLAN -> EXECUTE_CHUNK
```

1. **INITIAL_SURVEY**：在 `scan_budget.initial` 内建立初始 `SurveySnapshot`。
2. **BUILD_RUNTIME_GRAPH**：从根任务反向展开静态依赖，结合已满足前置条件、可达资源和替代分支生成运行时图。
3. **PLAN**：输出候选计划及被选中计划。计划必须包含有序任务节点、每个节点的前置条件、预期后置条件、目标地点、最大动作预算和回退分支。
4. **EXECUTE_CHUNK**：只执行到下一个任务节点完成、关键检查点或 `execution_chunk_budget`，不得盲目执行整条长链。
5. **VALIDATE**：基于最新 observation 和任务成功谓词核验实际后置条件，比较预期与实际状态差异。
6. **RESURVEY / REPLAN**：若需要，创建新的 `survey_id`、`plan_run_id` 和计划版本；保留历史计划和触发原因，绝不原地覆盖。

以下条件必须触发校验，通常也应触发重扫或重规划：

- 当前任务成功、失败或达到其动作预算；
- 关键资源、工具、工作站或路径与计划假设不一致；
- 生命、饥饿、能量或风险超过阈值；
- 发现新的低成本替代资源/路径；
- 进入新区域、楼层变化或可观测地图显著扩展；
- 环境终局、控制器中止或全局 episode 预算耗尽。

`PLAN` 的优化目标应由配置固定并版本化，例如按词典序最小化：`失败风险 -> 预计动作数 -> 不必要探索 -> 资源消耗`。首期不必追求全局最优；一个可解释、可回放、能正确重规划的启发式搜索优先于复杂但不可审计的最优求解器。

#### 6.1.5 录制数据与可回放性

现有 canonical transition 时间轴之外，必须增加以下规划元数据表；这些表均以 ID 关联 episode，不应把可变 JSON 塞进每一行 transition：

| 表 | 每行粒度 | 核心字段 |
|---|---|---|
| `world_instances` | 可复现世界实例 | `world_instance_id`, `map_seed`, `generation_config_hash`, 环境/资源版本。 |
| `episodes` | 根目标尝试 | `episode_id`, `world_instance_id`, `root_task_id`, `attempt_index`, planner/controller/version, information mode, 最终结果。 |
| `surveys` | 一次有限扫描 | `survey_id`, 起止 timestep, budget, observations summary, 未知项, 覆盖/置信度, stop reason。 |
| `plan_runs` | 一次规划求解 | `plan_run_id`, `survey_id`, runtime graph hash, candidate plan summaries, selected plan, objective/config version。 |
| `task_executions` | 一次节点执行 | `task_execution_id`, `plan_run_id`, `task_id`, 起止 timestep, expected/actual postconditions, outcome。 |
| `replan_events` | 计划转换 | 前后 `plan_run_id`, timestep, reason code, 计划偏差摘要。 |

每条 action metadata 增加可选的 `plan_run_id`、`task_execution_id`、`survey_id` 和 `decision_kind`（`survey`、`navigation`、`collection`、`craft`、`combat`、`validation`、`fallback`）。这样训练数据可以按需导出：

- 端到端 VLA 的 `(instruction, observation, action)`；
- 高层任务选择/子目标预测；
- 世界模型的状态转移与事件预测；
- 计划偏差、失败恢复和重规划决策；
- 同一 `world_instance_id` 下不同路线或不同 planner 的对比评估。

终局后必须运行一次 `FINAL_VALIDATE`：保存最终 observation、根目标成功谓词、未完成节点、资源/生存状态、最后一轮 survey 的新发现和停止原因。它不是为了继续执行，而是为了使失败、部分成功和替代路线都能被可靠分析。

#### 6.1.6 录制完整性不变量与分阶段落地

新增不变量：

- `world_instance_id` 的生成材料完整记录；同一 ID 必须重建出相同初始世界，环境或生成配置变化必须产生新 ID；
- 每个 `plan_run` 只引用一个明确的 `survey_id` 和静态任务图版本；
- 运行时图中的严格依赖必须是静态依赖图的子集，不能因扫描结果被删除；
- 每个计划节点均能追溯到对应的 action 区间、验证结果或明确的未执行原因；
- 每次重规划都有结构化 reason code，且前一计划仍可读取；
- `planner_information_mode`、扫描预算、规划目标和随机决策 seed 均进入 manifest；
- 同一 episode 中 `state/action/frame` 的既有时间对齐不因 survey、plan 或 replan 而改变。

实施顺序：

1. **先实现静态 DAG 校验与只读 planning descriptor**，并从现有 77 个 builtin `dependencies` 生成图；输出图版本/hash 和根任务反向依赖闭包。
2. **实现 observation-only survey 与 runtime graph builder**，先支持资源、工具、工作站、可达性和生存风险的最小集合；用固定扫描预算做 deterministic replay 测试。
3. **实现启发式 rolling planner**，每次只给出有限 action chunk，完成后调用 task adapter 验证；先覆盖采集 -> 制作 -> 采矿的链路。
4. **接入 recorder 的 survey/plan/task-execution/replan 表与 validator**，确保每次决策都能与 transition 时间轴关联。
5. **最后引入可选 privileged oracle/curriculum 模式**，严格隔离其数据集 split，比较其与 observation-only planner 的上限差异。

验收标准：固定 seed、根目标、规划器版本、信息模式和决策随机数后，系统能确定地产生相同的 survey、选中计划、动作序列和终局结果；任意失败或重规划样本可从 manifest 定位到其原始观测、计划假设、动作范围和触发原因。

### 6.2 环境感知的条件规划（Static Graph → WorldFacts → CombatModel → 楼层就绪门 → 滚动执行）

§6.1 定义了"静态图 + 运行时图 + rolling planner"的通用框架。本节落地一个可运行的具体实例（`craftax/planner/`），把**静态任务依赖图**与**环境事实**结合成"条件规划"：图决定任务顺序，环境决定每条边是否可行、需要补什么装备。

#### 6.2.1 分层架构

```text
静态任务依赖图 TaskGraph（task_id + dependencies，已存在）
  └─ closure(任务) → 拓扑排序的任务链
       └─ WorldFacts（环境事实：本 seed 各层可达矿石/梯子/水源/怪表）
            └─ CombatModel（战斗/生存模型：DPS、击杀回合、清层期望伤害、生存判定）
                 └─ Floor Readiness Gate（每层最低装备/属性/元素门槛）
                      └─ 条件规划器（有序 PlanSteps + 门控 + fallback + 不可行中止）
                           └─ 滚动执行器（step → validate → replan；战术原语）
```

各模块职责与关键机制：

| 模块 | 文件 | 职责 |
|---|---|---|
| `TaskGraph` | `craftax/tasks/graph.py` | 静态依赖图：closure、拓扑层级、DAG 校验（已存在，本节不改）。 |
| `WorldFacts` | `craftax/planner/world.py` | 每个 seed 的每层事实：可达矿石计数、梯子可达性、水源/食物、怪表；`SeedReadiness.evaluate(seed, target_floor)` 输出 `reach/armor_feasible/survival_ok/verdict`；`best_seeds(task, n)` 生成候选种子排序。 |
| `CombatModel` | `craftax/planner/combat_model.py` | 纯函数数值模型：玩家 DPS、`turns_to_kill`、`hits_per_kill`、清层期望伤害、`survival_verdict`（CLEARABLE/MARGINAL/INFEASIBLE）、`recommend_tactic`（stand/kite）。系数集中在模块顶部常量，供对真实运行标定。 |
| `FloorReadinessGate` | `craftax/planner/planner.py` | `FLOOR_GEAR_REQ` 每层最低装备（sword/armour/strength/elemental）；`check_floor_readiness` 返回缺失门槛；`resolve_gate` 由执行器补齐。 |
| `SkillChainExecutor` | `craftax/planner/executor.py` | 滚动执行：依赖图推导任务链，逐层派发原语技能；集成就绪门、批量清怪 + 锚点恢复、健康感知睡眠、升级策略。 |

#### 6.2.2 游戏机制数值（设计依据，已从 game_logic.py 核实）

- **下楼门**：`change_floor` 要求本层 `monsters_killed >= 8`（原生规则）；L0 初始 `monsters_killed=10`（已清，刷新率 1x）。
- **怪攻击**：相邻 + `attack_cooldown<=0` → 命中，冷却重置 5（每回合递减，无论是否相邻）；新刷怪槽冷却恒 <=0 → **每次接战约命中 1 次**，之后 5 回合内可安全击杀。→ 单层清 8 怪受击 ≈ `8 × mob_dmg × (1 − 护甲减免)`。
- **SLEEP**：受击 ×3.5、回血 2x（13 步/HP）、回能量（~11 步/点）、醒于能量满或被击。**REST(17)**：受击 ×1、回血 1x（26 步/HP）、不回能量、动作锁 NOOP 至血满。
- **护甲**：铁甲每件 3 铁 + 3 煤、10% 物免；4 件 40%。
- **属性**：上限 5；每下新层 +1 XP；力量每点 +25% 物伤 +1 血；敏捷每点 +2 能量上限、-12.5% 疲劳衰减。
- **怪表**：L1 orc(3伤/7HP) → L2 gnome(4/9) → L3 lizard(5/11) → L4 knight(6/12, 50%物免) → L5 troll(8/20, 20%物免) → L6 pigman(90%物免+火免) → L7 ice troll(90%物免+冰免) → L8 boss。
- **恢复锚点**：`ASCEND` 无 8 杀门槛 → 已清的上层（尤其 L0）是天然安全恢复区。

#### 6.2.3 双轨制种子策略

种子扫描数据（`data/seed_scan.json` + `data/seed_candidates.json`）显示：
- golden（梯子全可达）约 3%（30 扫 1、50 扫 1）；
- L0 装甲可行（铁≥3 且煤≥3）约 35%；二者交集很稀有。

因此分两条路线，`best_seeds(task)` 按 `(可达, 装甲可行, seed)` 排序：
1. **装甲路线**：种子在可达浅层能就地做铁甲 → 下楼前做甲（40% 物抗，清层受击减半）。
2. **风筝/锚点路线**：无甲种子 → 批量清怪 + 回 L0 锚点恢复 + 力量叠加 + 远程/风筝补生存。

#### 6.2.4 批量清怪 + 锚点恢复（L1+ 生存主机制）

单次清 8 怪的累积受击超出被动回血，因此不硬清：清到中止血量（<6）或能量将尽（<3）→ 回上一层锚点（已清、怪弱、有水/食物）→ SLEEP（回能量 + 回血）→ 再下继续，直至 `monsters_killed>=8`。该机制对战斗模型系数误差鲁棒（确定性，不依赖精确受击预估）。

#### 6.2.5 表层制备（下地牢前装备链）

按链上最深层需求（`_max_floor`）分级：
- 只需到达 L1（`enter_dungeon`）：木剑即可快速下行（不强制清怪）。
- 需清 L1+（`_max_floor>=2`）：木剑 → 木镐 → 石剑 → 石镐 → 铁剑（本层有铁时），绝不"为采铁先下楼"（递归保护）；铁甲仅在 `_max_floor>=2` 时按本层资源尽力做。

#### 6.2.6 已实现与已知瓶颈（2026-08 状态）

已实现：
- `combat_model.py` / `world.py` / `planner.py` 与各自单测（快速套件全绿）；
- 执行器集成就绪门、批量+锚点恢复、健康感知睡眠（血<8 不睡、SLEEP 回血副作用）、升级策略、`_collect_resource` 就地优先、工作台复用、`seed` 参数与 `abort_reason`；
- **风筝（2026-08 追加）**：跟踪怪攻击冷却（健康下降 + 相邻 → 计时 5，逐拍递减）；
  仅当 `recommend_tactic==kite`（慢速击杀层）且 `timer==1`（怪冷却将归零）时拉开
  2 步——规避"后续命中"；`timer==0`（新鲜怪）照常攻击承担必中首击。效果：击杀
  需 >5 回合的慢速怪受击周期 5→7 回合。
- **弓先制 bow-rush（2026-08 追加，§6.2.2"出路 a 远程击杀"落地）**：
  - `combat_model.py`：`Gear` 增加 `bow`/`bow_enchant`；`bow_arrow_damage`（ARROW2=5 物伤
    ×敏捷缩放 ×(1-物免) + 附魔元素半伤）、`turns_to_kill_bow`、`damage_per_kill_bow`、
    `damage_per_clear_bow`、`estimated_steps_bow`；`survival_verdict`/`recommend_tactic`
    的弓分支（L1-L3 弓 CLEARABLE、L4 需附魔/近战、L6/L7 需元素）。单测覆盖。
  - `executor.py`：`_bow_combat`（贴脸点射→同行列 <=14 格提前射杀→chase 追近身怪）、
    `_should_use_bow`、`_bow_rush`（木剑→下 L1→开首箱拿弓，确定性掉落）、`_ladder_descend`
    抽取、弓附魔 `ENCHANT_BOW=42`（`_acquire_elemental_capability` 首选弓附魔）、
    `_collect_resource` 补镐等级检查（无镐挖石会卡死的修复）、`_survival_action` 弓主动
    防御/安全点睡眠/箭补给/防御性回血、`_descend_to_inner` 有弓跳过深制备。
  - `planner.py`：`check_floor_readiness` 有弓时放宽 L1-L3 的剑/甲门槛。
  - `test_executor._summary` 补 `xp/strength/dexterity/intelligence`（缺失导致升级逻辑
    在测试中失效的修复——服务端 summary 一直有这些字段）。
- **安全睡眠点**：`_safe_sleep_spot_walk`（找 >=14 格安全点）已接入睡眠与回血主路径；
  但 L0 上"走位找点"仍不足以完全避免打醒（见下瓶颈）。
- **敏捷量化评估**：`awake_budget_steps`（dex1≈217 → dex5≈930 清醒步）、
  `energy_is_bottleneck`（按**单批工作段**能量 vs 预算）。结论：批量+锚点恢复下
  单批 ~46-64 步 << dex1 预算 217 → **能量不是瓶颈，力量优先**；敏捷只在力量满后
  （深层长程）或不可恢复的单批超长段有价值。
- **扩大种子扫描**：`scripts/scan_seeds_chunked.py`（每 chunk 独立子进程，规避 JAX
  CPU 编译内存映射泄漏）；1000 级扫描产出 golden∩L0 装甲可行交集（如 2011、2111）。

#### 6.2.6a 规划审查与修复（2026-08-18）

对图与执行逻辑做了一次审查，核心结论是：**图当时是"成就解锁顺序图"而不是因果
规划图，执行器是一层反应式规则级联，而三条"扩大观察边界"的通道全都接了线但没
通电**（WorldFacts 是死参数、`GET /map?floor=N` 从未被请求、模拟器可 fork 但从不
前瞻）。已修复项：

- **图的因果错误**（`tasks/builtin/*`）：骷髅在 L0 却依赖 `enter_dungeon`；兽人在 L1
  却依赖 `enter_sewers`（`FLOOR_MOB_MAPPING[1]=type 2`）；`learn_*` / `enchant_*` 的
  方向反了（书在 L3/L4 首箱、附魔台在 L3/L4，且元素能力是下 L6/L7 的**前置**）→
  已改正，并给 `enter_fire_realm`/`enter_ice_realm` 补上元素能力前置边；
  `place_furnace`/`place_stone`/`craft_tools` 的缺边（原 advisory）落成真实边。
- **成本感知排序**（`executor.build_task_chain`）：旧顺序按 `(拓扑层级, task_id)`
  字母序 → `enter_gnomish_mines` < `place_table` 使计划变成"先下 L2 再回头放工作台"，
  钻石剑被排到抵达 Boss 层之后（而它需要只在 L0 才有的木头）。新排序按"就地任务
  优先 → 距当前虚拟楼层最近 → 拓扑层级"贪心调度，采集/合成/下行自然按层成组。
- **完成判定归一**（`_task_is_complete`）：改为求值 registry 的 `success_predicate`
  （与数据集标注同源）。此前 28 个复合/根目标里有 25 个**永远无法判定完成**，
  即使成就全开也报失败。中间任务额外保留"执行层可用性"（储备量/工具等级/消耗品）。
- **崩溃**：`REACH_FLOOR.get(tid, ENTER_FLOOR[tid])` 的默认值被立即求值 →
  `explore_dungeon` / `dungeon_campaign` 一进 L1 即 KeyError，`build_plan` 对 5/92
  个任务崩溃。
- **`_max_floor` 取 max(刷新层)**：骷髅 `[0, 8]` 被当成 L8 任务，地表任务触发整套
  深层制备 → 改取就近层（与 `_combat` 的导航一致）。
- **生存优先级倒置（实测死因）**：`drink=0` 时执行器先"原地回血"再"睡觉"——而缺
  必需品会让 recover 变负（醒着 16 步/HP、睡着 31 步/HP 掉血），被动回血 26 步/HP
  追不上。补给（水/食物）现在排在回血/睡眠之前。
- **不可撤销决策的有限步前瞻**（`combat_model.project_sleep` /
  `projected_awake_health`）：SLEEP 会锁到能量回满（实测 60+ 步，期间不能喝水、
  不能反击），因此按真实衰减速率把整段时长走一遍，睡醒会低血/断补给就不睡；
  "原地等回血"同样要求投影为正。
- **恢复行为有界**：防御性回血改为滞回（<5 进入、>=8 退出）+ 150 步预算 + 200 步
  冷却。旧的裸阈值 `health<8` 在 L0（`monsters_killed` 初始 10）永久成立，子目标
  可以几百步不动。
- **弹药预算**（`combat_model.arrows_for_clear`）：清一层 8 怪需要的箭数（L1 兽人
  ~17 支、L2 ~20 支；L6/L7 数百支 → 说明不能靠普通箭清元素层）。下楼前若还要
  "穿过"下一层，就在**有木石的本层**备齐（1 木+1 石 → 2 箭）。
- **血线门**：到达即需清怪的层，先在当前层恢复到接近满血再下。
- **停滞看门狗**：进展签名（楼层/成就/背包/击杀）不变即计数，150 步换动作打破、
  600 步中止该 seed；另有上下楼振荡专项保护（楼层每步都变时签名也每步在变，
  普通停滞检测测不到）。
- **跨层观察真正接通**：`SkillChainExecutor(floor_map_provider=...)` 用
  `GET /map?floor=N` 在下楼前看目标层全图；`check_floor_readiness` 开始消费
  `WorldFacts`（梯子链断 → 硬中止；铁/煤不足 → 不再要求做不出来的甲）；
  采集换层跳过"已知没有该矿"的层；步数预算改由 `estimated_steps_bow` 逐层估算。
- **测试卫生**：`test_world.py` 过去依赖 gitignore 掉的 `data/seed_scan.json`，
  干净检出必失败 5 项 → 改用仓库内 fixture（`planner/tests/data/seed_scan.json`，
  由 `scan_seeds.py` 实测产出）；`WorldFacts._load_scan` 的聚合缓存按数据目录分键。

固定 seed 实测（金种子 3017/3050，同一批 rollout 对比）：

| 任务 | 修复前 | 修复后 |
|---|---|---|
| `enter_dungeon` | 49 步通过 | 49 步通过 |
| `defeat_skeleton` | 282 步暴死（当成 L8 任务） | **139 步通过** |
| `collect_diamond` | 282 步暴死 | **935 步通过** |
| `explore_dungeon` | 进 L1 即 KeyError | 不再崩溃，跑满预算 |
| `reach_floor_3` / `defeat_troll` / `enter_gnomish_mines` | 282-534 步死于 L0（渴着睡） | 死于 L1 清怪，1000-2000 步、成就 18-20（原 14） |

已修复项（续）：弹药经济学——**"备箭还是先造石/铁剑"由数值择优决定**
（`combat_model.recommend_clear_prep`，2026-08-19）：

- 旧假设"有弓就跳过深制备"在**弹药不可补的层**上不成立：`MAKE_ARROW` 要
  1 木 + 1 石，而地牢层（L1-L5）没有树 → 箭下楼后是不可再生资源，弹尽后剩下
  的怪只能用剑打。新规则把两个选项折算到同一货币——**每花 1 木 + 1 石 能省下
  的清层受击**（石剑与 2 支箭同价）：
  - 备箭：2 支箭把 `2/arrows_per_kill` 只怪从近战转为远程 →
    省 `Δcoverage × (damage_per_clear - damage_per_clear_bow)`；
  - 升剑：把**弹尽后残余击杀**的近战单价降一档 →
    省 `残余比例 × (damage_per_clear(旧剑) - damage_per_clear(新剑))`，
    再除以材料/采矿成本单位（石剑 1、铁剑 4、钻石剑 7）。
  L1 实例（弓 + 木剑、力量 1、8 支箭）：备箭 1.2 伤/单位 vs 石剑 2.4 伤/单位
  → **先造石剑**（一次 1 木 1 石，把清层受击 41.4 → 36.6），之后备箭反超
  （0.6 vs 铁剑 0.6/单位）→ 顺序自然是"石剑 → 备箭 → 下楼"。弹药可补（L0）或
  已备齐时收益恒为 0 → 自动退回旧的"跳过深制备"，即旧行为成为新规则的特例。
- 弹药预留分两档：`arrows_min`（1.25x 均值，值得专程采料）与 `arrows_target`
  （不可补层 1.75x 预留，只用手头余料补）。**单档会死锁**：地表持续刷怪、弓
  持续消耗，产量≈消耗使"补满 23 支"永远不满足（实测 1800 步里 635 步在合成箭、
  一次没下楼）。同理生存层补给只补到基础储备 8，清层预算交给下楼路径。
- 两个只有 rollout 才暴露的缺陷同时修掉：① 生存层的补箭分支无上限 → 地表箭
  工厂；② 择优若不设**绝对门槛**（`MIN_SWORD_GAIN_ABS`）与**可建上限**
  （`max_sword=3`），"相对更优"会一路推到钻石剑（0.3 伤/单位）并卡在制备链上。
- 过路层的木料：`_arrow_feedstock_target` 让"下一层之后还要清怪"时先带够木
  （+2 供深层放工作台），因为 L1+ 只能靠背包木料现做箭。
- 就绪门同步：`check_floor_readiness(..., arrows_restockable=)` 下，"弓豁免剑/甲
  门槛"以**弹药够清满 8 怪**为前提（不可补层箭不够 → 剑/甲门槛恢复）。

固定 seed A/B（同一批 rollout，12 次；`old` = monkeypatch 复原"有弓就跳过深制备"）：

| 任务 / seed | 旧 | 新 |
|---|---|---|
| `enter_gnomish_mines` 3017 | 1025 步暴死，L1 击杀 2，木剑，余箭 4，成就 17 | 1349 步暴死，L1 击杀 1，**石剑**，余箭 15，成就 18 |
| `enter_gnomish_mines` 3050 | 1704 步暴死，L1 击杀 0，木剑，余箭 2，成就 18 | 1719 步暴死，L1 击杀 2，**石剑**，余箭 13，成就 20 |
| `reach_floor_3` 3017 | 2546 步跑满预算（活着但停在 L0，从未清怪），木剑，余箭 8，成就 19 | 2291 步暴死于 L0 恢复环，**铁剑**，余箭 7，成就 21 |
| `reach_floor_3` 3050 | 1455 步暴死，木剑，余箭 8，成就 19 | 2038 步暴死，石剑，余箭 12，成就 19 |
| `defeat_gnome_warrior` 3017 | 980 步暴死，余箭 6，成就 16 | 1078 步暴死，余箭 14，成就 18 |
| `defeat_gnome_warrior` 3050 | 1113 步暴死，余箭 5，成就 20 | 1161 步暴死，余箭 18，成就 19 |
| 2011 / 2111（L0 即死的 seed） | 489 / 221 步 | 相同（择优未介入） |

结论：制备内容确实变了（木剑→石剑/铁剑、死亡时余箭 2-8→7-18，不再弹尽待宰），
生存时长 5/6 例上升、成就 +1 左右，**但 L1 清怪墙未闭合**（L1 击杀仍 0-2/8）。

已知瓶颈（游戏机制固有，尚未闭合）：
- **L1 清怪墙**：下 L2 需在 L1 清 8 只兽人（3 伤/7HP）。出路 a)（石/铁剑与弓按
  `arrows_for_clear` / `damage_per_clear` 择优）已实现，见上；余下的量化障碍是
  **受击次数而不是杀怪速度**：模型里每次接战固定 1 次命中（3 伤），满血 8-10 →
  清一层需 6-9 次锚点往返，每次往返都要穿过 L0 出生环。因此剩下的出路是
  b) 角落/走廊防御位（怪只能沿直线接近 → 14 格提前射杀，把"固定 1 次命中"降下来）；
  c) 路线规划器（§6.1.4）减少 L0/L1 走动暴露。
- **战斗模型对弓偏保守（标定待办）**：`hits_per_kill` 对弓与近战用同一个
  `HITS_PER_KILL_BASE=1.0`，即认为 14 格点射也必吃 1 击。这让"铁剑 = 弓"
  （L1 两者都 2 回合击杀 → 清层受击都是 31.8），于是有铁剑后模型判定"再备箭
  买不到减伤"。实测弓的主动防御可长时间零受击 → 该系数应给弓单列一档，但它会
  同时改变 `survival_verdict` / `estimated_steps_bow` / 所有步数预算，属于独立
  的标定项，本轮没有顺手改。
- **L0 表层制备生存墙（原有量化结论仍成立）**：每次接战固定受击 1 次（2 伤，与
  剑无关——怪刷新即冷却<=0）；1080 个 seed 无 L0 满铁甲（矿密度上限 ~8 铁，
  满甲需 12 铁）。弓 + 补给优先 + 有界恢复把它从"必死"降为"可长时间维持"。
- **仍未做的前瞻**：真正的 rollout 搜索（fork JAX 状态试探候选计划）没有落地；
  当前前瞻是解析式投影（`project_sleep` 等），确定性、可单测、不与环境耦合。

#### 6.2.7 验收标准

- 快速套件（`-m "not slow"`）保持全绿（当前 planner+tasks 342 项、其余 62 项通过）；
- 固定 seed + 任务 + 决策随机数下，执行器行为确定（同轨迹可重放）；
- 慢速深层套件在候选种子上逐项通过（当前 `enter_dungeon` / `defeat_skeleton` /
  `collect_diamond` 通过；L1 及更深的清怪任务为已知待办，见 6.2.6a）。

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
  "survey_id": "optional-survey-uuid",
  "plan_run_id": "optional-plan-uuid",
  "task_execution_id": "optional-task-execution-uuid",
  "decision_kind": "collection",
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
- 基于 `TaskSpec.dependencies` 的静态 DAG 校验、任务 planning descriptor、observation-only survey、运行时计划图与滚动重规划；
- `world_instance/episode/survey/plan_run/task_execution/replan` 元数据表及其与 transition 时间轴的关联；
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
- 对固定 seed/根目标/规划器版本，survey、计划、动作和终局均可确定性回放；失败和重规划可追溯到原始观测、计划假设、动作区间与 reason code；
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
