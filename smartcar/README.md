# 智慧道路车辆跟踪监测系统

本项目是一个基于YOLOv8的智慧道路车辆跟踪监测系统，使用Tkinter构建图形用户界面（GUI）。系统能够处理视频文件或实时摄像头画面，进行车辆检测、速度估计、异常行为（超速、碰撞、异常轨迹）检测，并提供统计数据显示功能。

## 项目结构

```
.
├── show.py                     # GUI主程序入口
├── show_relative/              # GUI相关的模块
│   ├── __init__.py
│   ├── constants.py            # 存放全局常量和配置
│   ├── video_operations.py     # 视频源处理模块
│   ├── processing_operations.py # 视频帧处理与分析模块
│   ├── statistics_operations.py # 统计数据显示模块
│   └── parameter_settings.py   # 参数设置模块
├── predict_car.py              # (本地视频处理) 综合车辆分析脚本
├── predict_car_collision.py    # (本地视频处理) 车辆碰撞检测脚本
├── predict_car_flow.py         # (本地视频处理) 车流量统计脚本
├── predict_car_speed.py        # (本地视频处理) 车辆速度检测脚本
├── predict_car_speedcheck.py   # (本地视频处理) 车辆超速检测脚本
├── predict_car_trajactory.py   # (本地视频处理) 车辆轨迹异常检测脚本
├── model_data/                 # 存放YOLO模型文件 (例如 best.pt)
├── ultralytics/                # YOLOv8相关的库文件 (如果本地集成)
├── requirement.txt             # 项目依赖的Python包
├── bytetrack.yaml              # ByteTrack追踪器配置文件
└── README.md                   # 项目说明文件
```

## 主要功能

### 1. GUI 应用 (`show.py` 及 `show_relative/` 模块)
   - **视频源管理**:
     - 支持上传本地视频文件。
     - 支持连接实时摄像头。
     - 控制视频的播放、暂停和停止。
   - **车辆检测与跟踪**:
     - 使用YOLOv8模型进行实时车辆检测和跟踪。
     - 在画面上绘制检测框和车辆ID。
   - **速度检测**:
     - 估算被检测车辆的实时速度。
     - 可设置超速阈值，高亮显示超速车辆。
   - **异常行为检测**:
     - **超速检测**: 检测超过预设速度阈值的车辆。
     - **碰撞预警**: 基于检测框的IoU（交并比）判断潜在的碰撞风险。
     - **异常轨迹检测**: 分析车辆行驶轨迹，检测例如急转弯等异常变道行为。
   - **统计数据显示**:
     - **车流量统计**: 实时统计单位时间内的车流量，并以图表形式展示。
     - **速度分布统计**: 显示检测到的车辆速度分布情况。
     - **车型统计**: 统计不同类型车辆的数量。
   - **参数设置**:
     - 允许用户自定义车道数、速度阈值、轨迹角度阈值、检测置信度、IoU阈值等参数。
   - **控制台输出**:
     - 在GUI界面内嵌控制台，显示程序运行日志和检测信息。

### 2. 本地视频处理脚本 (`predict_*.py`系列)
   这些脚本主要用于对本地视频文件进行离线分析，不直接参与GUI应用的实时处理流程。
   - `predict_car.py`: 对视频进行综合分析，可能包含检测、跟踪、计数等。
   - `predict_car_collision.py`: 专门用于检测视频中的车辆碰撞事件。
   - `predict_car_flow.py`: 专门用于统计视频中的车流量。
   - `predict_car_speed.py`: 专门用于检测和记录视频中车辆的速度。
   - `predict_car_speedcheck.py`: 专门用于检测视频中的车辆超速情况。
   - `predict_car_trajactory.py`: 专门用于分析和检测视频中车辆的异常轨迹。

## 环境要求

- Python 3.x
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Tkinter (通常Python自带)
- Matplotlib
- Pandas
- NumPy

建议使用 `requirement.txt` 文件安装依赖：
```bash
pip install -r requirement.txt
```

## 使用方法

### 运行GUI应用

1.  确保已安装所有必要的依赖库。
2.  确保 `model_data/` 目录下有所需的YOLO模型文件 (例如 `best.pt`)。
3.  确保 `bytetrack.yaml` 配置文件存在且配置正确。
4.  运行主程序：
    ```bash
    python show.py
    ```
5.  通过GUI界面操作：
    - **视频源**: 上传视频或连接摄像头。
    - **视频处理**: 启动/停止车辆检测、速度检测、异常检测。
    - **统计数据**: 查看各类统计图表。
    - **参数设置**: 根据需要调整检测参数。

### 运行本地视频处理脚本

这些脚本通常直接通过命令行运行，例如：
```bash
python predict_car_flow.py --source path/to/your/video.mp4
```
具体参数请参考各个脚本内部的说明或使用 `--help` 参数。

## 注意事项

- 模型文件 (`best.pt`) 需要放置在 `model_data/` 目录下。用户可以替换为自己的YOLOv8模型。
- 配置文件 `bytetrack.yaml` 用于目标跟踪。
- 确保摄像头驱动正常，如果使用摄像头功能。
- 视频处理的性能取决于计算机硬件配置，特别是GPU。 