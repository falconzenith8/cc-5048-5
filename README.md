# Volumetric Performance Player

一个能绕着演员旋转观看的体积点云表演播放器。点云帧以**采集时刻 (pts)、采集
原点、逐点颜色**存储，每帧点的数量和排列都可以变化；播放位置由**音频输出
时钟**唯一驱动，读取/解码在独立线程进行，解码结果放进有界缓存。演员在两个
鼓点之间穿过一道纸屏风，其中两帧大帧被人为安排得比紧随其后的小帧更晚到达。

## 交付内容与需求对应

| 需求 | 实现位置 |
| --- | --- |
| 带 pts / 原点 / 逐点颜色的三维帧，变长且排列可变 | `volumeview/frame.py`、`volumeview/scene.py` |
| 绕到背面能看到正面被挡住的点（逐像素深度测试） | `volumeview/software_render.py`、`volumeview/gl_render.py` |
| 以音频播放位置安排读取、解码、有界缓存 | `volumeview/audio.py`、`volumeview/streaming.py`、`volumeview/cache.py` |
| 拖动时间轴 / 来回循环后只提交当前区间的帧 | `volumeview/scheduler.py`（region epoch）、`volumeview/player.py` |
| 两个鼓点间穿过屏风 | `volumeview/scene.py`（beat 8→12，z=0 屏风） |
| 大帧比后一小帧晚到 | `volumeview/streaming.OutOfOrderNetwork`（帧 42/43 加 0.55s） |
| 两种欠载处理 + 最大音画偏离 + 内存上限 | `volumeview/scheduler.py`、`tools/compare_policies.py` |
| 图形窗口拖慢读取、旋转视点、看屏风前后深度 | `volumeview/gl_render.py`、`volumeview/player.py` |
| 同一录制操作上复验 | `reproduce.sh`、`tests/`（固定随机种子） |
| 场景、音轨、乱序回放、启动入口随仓库交付 | `assets/`、`recordings/replay_plan.json`、`volumeview/cli.py` |

## 架构

```
 AudioClock (音频输出位置=唯一时钟)
      │ position
      ▼
 Scheduler ── 目标帧索引 ──▶ BoundedFrameCache (字节+帧数双上限, LRU 驱逐)
      ▲                         ▲ 解码后插入, 按 region epoch 丢弃过期帧
      │ 欠载策略 HOLD / SKIP    │
 ReadDecodeService (worker 线程)
      └─ FrameFile 读取压缩块 → OutOfOrderNetwork 延迟队列 → zlib 解码
 Renderer: OpenGLRenderer (交互窗口) / SoftwareRenderer (无头 z-buffer)
```

- **容器** `assets/scene.vpc`（+ `.idx` 边车索引）：每帧独立 zlib 压缩，
  头部含 magic、pts、origin、点数；点为相对原点坐标，颜色为逐点 RGB。
- **乱序模型**：每帧注入 `base_delay + jitter`，重帧 42/43 额外 +0.55s，
  因此帧 43（约 4.1k 点）比帧 44（约 1.1k 点）晚约 0.45s 就绪，尽管先被请求。
- **区域 epoch**：seek 或循环绕回时 `ReadDecodeService.reset_for_region()`
  递增 epoch，清空缓存与在途队列；旧 epoch 的帧到达缓存即被丢弃
  （`stale_frames_dropped` 计数），保证渲染器只拿到当前播放区间的帧。

## 两种欠载策略

目标帧尚未到达时：

- **HOLD（保持上一姿势）**：继续显示上一帧直到目标帧到达。音画偏离只随
  等待时间增长，画面不跳。
- **SKIP（跳到下一可用姿势）**：向前跳到缓存中最近的已到达帧。保持新鲜度，
  但会出现可见的姿势跳跃，偏离可能为多帧间隔。

同一次录制、同样的慢读取压力下的实测对比（见
`recordings/policy_comparison.json`）：

| 指标 | HOLD | SKIP |
| --- | --- | --- |
| 最大音画偏离 | ≈ 0.25 s | ≈ 1.3 s |
| 平均音画偏离 | ≈ 0.035 s | ≈ 0.045 s |
| 缓存峰值 | < 1 MiB（上限 64 MiB） | 同左 |
| 行为 | 等待，无跳帧 | 向前跳帧 |

缓存硬上限由 `BoundedFrameCache(max_bytes=64MiB, max_frames=48)` 强制，
超限时驱逐离播放头最远的帧并计入 `cache_evictions`。

## 安装与运行

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) 生成全部资产（场景、鼓点 WAV、节拍表、乱序计划）
python -m volumeview.cli assets

# 2) 交互式 OpenGL 窗口（需要显示与系统 libGL）
python -m volumeview.cli play --policy hold

# 3) 无头复验（无需显示/GPU/声卡，输出 PNG 与指标）
python -m volumeview.cli headless --policy hold --out recordings/hold_run
python -m volumeview.cli compare --read-rate 3.5

# 一键完整复验
./reproduce.sh
```

无音频设备时 `AudioClock` 自动回退到由墙钟驱动的虚拟时钟，接口不变；
无 libGL 时 `play` 会给出明确提示，请改用 `headless`。

## 窗口操作

| 操作 | 按键 |
| --- | --- |
| 旋转视点（绕演员轨道，可转到背面） | 鼠标左键拖动 |
| 缩放 | 滚轮 |
| 播放 / 暂停 | Space |
| 时间轴后退 / 前进 1s | ← / → |
| 切换 HOLD / SKIP 欠载策略 | H / J |
| 拖慢 / 加快读取（延迟倍率） | `-` / `+` |
| 开关 4s 循环区间 | L |
| 深度着色模式（屏风前后深度） | D |
| 开关屏风 | S |

标题栏实时显示播放时刻、当前帧、策略、读取倍率、缓存占用、最大音画偏离与
欠载次数。

## 测试

```bash
pytest -q
```

19 个测试覆盖容器往返、变长点云、乱序保证、缓存硬上限、epoch 失效、两种
欠载语义、z-buffer 遮挡（背面揭示）、seek/循环区域约束和端到端回放。
