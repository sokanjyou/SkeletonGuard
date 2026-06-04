# 老年人跌倒检测算法框架

大家好，我是谢静灵，来自成都理工大学。本项目是一个面向居家养老场景的单目视觉跌倒检测框架。系统从视频流中检测人体 2D 姿态，进行多人追踪、轻量 2D-to-3D 几何升维、置信度感知 Kalman 时序融合、环境语义修正、规则评分与可选 LSTM 特征序列融合，最终输出每个追踪目标的跌倒风险和告警。


## 核心流程

```text
video frame
  -> YOLO26-Pose 2D keypoints
  -> pose-aware multi-person tracking
  -> subspace sparse 2D-to-3D lifting
  -> confidence-aware Kalman filtering with runtime dt
  -> skeleton graph and bone-length calibration
  -> motion/posture/context feature extraction
  -> rule-based fall scoring
  -> optional LSTM feature-window fusion
  -> environment-aware alarm and optional MQTT publish
```

## 目录结构

```text
config/
  config.yaml              主配置
  zones_config.json        环境语义区域配置
data/
  models/
    yolo26m-pose.pt        YOLO26-Pose 权重
    lstm_fall.pt           可选 LSTM 特征分类权重
src/
  main.py                  主入口
  core/
    stream_loader.py       视频源读取、重连、FPS 获取
  models/
    yolo26_pose.py         Ultralytics YOLO 姿态封装
    lifter_3d.py           2D-to-3D 几何升维
    kalman_fusion.py       骨架 Kalman 滤波与图约束
    lstm_classifier.py     可选 LSTM 特征序列分类器
  pipeline/
    multi_person_tracker.py 多人 ID 追踪
    environment_context.py  环境语义推断
    feature_extractor.py    姿态/运动/上下文特征
    fall_detector.py        跌倒规则评分和判定
  utils/
    visualizer.py          可视化绘制
    logger_alerts.py       日志和 MQTT 告警
    device.py              CPU/CUDA 设备选择
```

## 安装

使用Python3.11，并在项目虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖：

- `numpy`
- `opencv-python`
- `paho-mqtt`
- `PyYAML`
- `torch`
- `torchvision`
- `ultralytics`

项目中的 `.gitignore` 已忽略 `__pycache__/`、`*.pyc`、`Ultralytics/` 和 `runs/`。

## 模型文件

将 YOLO26-Pose 权重放到：

```text
data/models/yolo26m-pose.pt
```

如需启用 LSTM 分支，将训练好的权重放到：

```text
data/models/lstm_fall.pt
```

注意：当前主流程传给 LSTM 的输入是 `features.vector`，默认维度为 `features.feature_size: 32`。LSTM 权重必须用相同维度的逐帧特征序列训练。代码会在加载权重和推理时检查输入维度，不匹配会直接报出明确错误。

## 运行

摄像头：

```bash
python -m src.main --config config/config.yaml --source 0 --show
```

视频文件：

```bash
python -m src.main --config config/config.yaml --source data/demo.mp4 --show
```

保存结果视频：

```bash
python -m src.main --config config/config.yaml --source data/demo.mp4 --output data/out.mp4
```

如果编辑器提示 `cv2.VideoWriter_fourcc` 不是已知属性，这通常是 OpenCV 类型存根不完整导致的静态提示；运行时 `opencv-python` 正常安装即可。

## 配置说明

主要配置位于 `config/config.yaml`。

- `source`: 默认视频源。可以是摄像头编号，也可以是视频文件路径。
- `zones`: 环境语义区域配置文件路径。
- `stream`: 视频源打开和读取重试参数。
- `pose`: YOLO26-Pose 权重、设备、置信度、IoU 阈值和是否强制要求 pose 权重。
- `tracker`: 多人追踪参数，使用 bbox IoU、中心距离和关键点相似度综合匹配 ID。
- `lifter`: 2D-to-3D 升维参数。当前是轻量几何/子空间方法，`fallback_on_nonconvergence` 会在优化未收敛时回退到上一帧姿态。
- `kalman`: Kalman 噪声、骨架图约束和速度融合参数。`dt` 是初始值；运行时会根据视频源 FPS 或实时摄像头帧间隔动态更新。
- `features`: 逐帧特征维度和在线标准化参数。默认 `feature_size: 32`。
- `fall`: 规则跌倒判定阈值、窗口长度、姿态/冲击/高度权重、环境修正权重。
- `lstm`: 可选 LSTM 特征序列融合分支。`fusion.window` 是输入特征序列长度，`fusion.score_window` 是 LSTM 融合分数平滑窗口。
- `alerts`: 本地日志和 MQTT 告警配置。
- `runtime`: 每个追踪 ID 的状态 TTL 和最大状态数量。

## 跌倒判定原理

规则分支提取三个核心证据：

1. 姿态水平化：肩部中心到髋部中心的躯干向量越接近水平，姿态分数越高。
2. 冲击速度：使用骨盆速度，综合向下速度和三轴速度幅值。
3. 低高度/压缩：使用 root-centered 3D 骨架的竖直跨度，身体越扁平越接近跌倒形态。

原始分数为三者加权和：

```text
raw_score =
  weight_posture * posture_score
  + weight_impact * impact_score
  + weight_height * height_score
```

随后根据环境上下文做修正：

- 在床、榻榻米等允许休息区域，且接触稳定、冲击不明显时降低分数。
- 在浴室、厨房、楼梯等高风险区域，或身体明确处于地面/地毯区域时提高风险。
- 若姿态和冲击证据很强，不让环境抑制过度压低风险。

最终使用滑动窗口均值、连续高风险帧数和快速跌倒峰值证据共同决定是否报警。

## 环境语义

`config/zones_config.json` 使用图像坐标多边形标注区域，例如地面、床、榻榻米、浴室入口等。每个区域包含：

- `name`: 区域名称
- `kind`: 区域类型，如 `floor`、`bed`、`tatami`、`bathroom`
- `polygon`: 图像坐标多边形
- `height_m`: 区域物理高度，仅作为语义特征保留
- `rest_allowed`: 是否允许正常躺卧休息
- `fall_risk`: 区域风险系数

当前 3D 骨架是以骨盆为中心的相对坐标，因此不会把 `height_m` 当作真实世界高度去和骨架坐标相减。地面接触主要由 2D 接触点、躯干水平化、深度方向展开和语义区域共同判断。

## LSTM 融合

LSTM 分支默认 `enabled: auto`：权重存在时尝试加载，不存在时自动跳过。

当前 LSTM 的输入是逐帧特征向量序列，形状为：

```text
(T, F)
```

默认 `T = lstm.fusion.window = 18`，`F = features.feature_size = 32`。

相关配置：

- `lstm.fusion.window`: 输入 LSTM 的特征窗口长度。
- `lstm.fusion.score_window`: LSTM 融合分数平滑窗口，独立于 `fall.window`。
- `lstm.fusion.weight`: LSTM 分数在规则分数融合中的权重。
- `lstm.model.input_size`: LSTM 期望输入维度，默认应和 `features.feature_size` 一致。

## MQTT 告警

启用 `alerts.mqtt.enabled` 后，系统会在跌倒告警时发布 JSON 消息：

```yaml
alerts:
  min_interval_s: 2.0
  mqtt:
    enabled: true
    host: 127.0.0.1
    port: 1883
    topic: fall3d/alerts
    client_id: fall3d-detector
    qos: 1
    retain: false
    min_interval_s: 2.0
```

消息包含：

- `event`
- `timestamp`
- `track_id`
- `score`
- `raw_score`
- `posture_score`
- `impact_score`
- `height_score`
- `context_factor`
- `reason`

如果 MQTT broker 不可用，程序会记录警告并继续本地检测。

## 注意事项！用之前先确认！

- 阈值需要根据真实摄像头视角、场景和数据调参。
- 环境语义多边形需要按实际摄像头画面标注。
- LSTM 权重需要用当前特征定义重新训练，不能直接混用骨架序列权重。
