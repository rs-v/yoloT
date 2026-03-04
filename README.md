# YOLO11 RTSP Tracking

使用 **YOLO11** 对 RTSP 视频流进行目标跟踪，同时：
- 在本地窗口**实时展示**带注释的画面
- 通过 FFmpeg 将注释后的画面**推送为 RTSP 输出流**

参考文档：<https://docs.ultralytics.com/modes/track/>

---

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.8 | |
| ultralytics ≥ 8.3 | YOLO11 支持 |
| opencv-python ≥ 4.8 | 视频捕获与显示 |
| FFmpeg | 推送 RTSP 输出流（可选，系统级安装） |

```bash
pip install -r requirements.txt
# Ubuntu/Debian 安装 FFmpeg
sudo apt-get install -y ffmpeg
```

---

## 快速开始

> **安全提示**：避免将 RTSP 凭据硬编码在命令行历史中。
> 推荐通过环境变量传递输入地址：
> ```bash
> export RTSP_SOURCE="rtsp://<username>:<password>@<host>:<port>/stream"
> python rtsp_track.py
> ```

```bash
# 使用默认 RTSP 地址 + 默认自定义跟踪器（custom_tracker.yaml）
python rtsp_track.py

# 通过 --source 直接指定输入源
python rtsp_track.py --source "rtsp://<username>:<password>@<camera-ip>:554/cam/realmonitor?channel=1&subtype=1"

# 指定模型
python rtsp_track.py --model yolo11n.pt

# 使用内置 BoT-SORT 跟踪器
python rtsp_track.py --tracker botsort.yaml

# 使用内置 ByteTrack 跟踪器
python rtsp_track.py --tracker bytetrack.yaml

# 禁用本地显示窗口（纯推流模式）
python rtsp_track.py --no-show

# 禁用 RTSP 输出（仅本地显示）
python rtsp_track.py --no-output

# 指定输出 RTSP 地址
python rtsp_track.py --output-rtsp rtsp://localhost:8554/live/output
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `$RTSP_SOURCE` 或内置默认值 | 输入视频源（RTSP URL 或任何 OpenCV 兼容源）；建议通过 `RTSP_SOURCE` 环境变量传递凭据 |
| `--model` | `yolo11n.pt` | YOLO11 权重文件 |
| `--tracker` | `custom_tracker.yaml` | 跟踪器配置 YAML（可用 `botsort.yaml`、`bytetrack.yaml` 或自定义） |
| `--output-rtsp` | `rtsp://localhost:8554/live/tracking` | 注释流输出的 RTSP 地址 |
| `--conf` | `0.3` | 检测置信度阈值 |
| `--iou` | `0.5` | NMS IoU 阈值 |
| `--device` | 自动 | 推理设备（`cpu`、`0` 表示 GPU 0 等） |
| `--no-show` | — | 禁用本地显示窗口 |
| `--no-output` | — | 禁用 RTSP 输出流 |

---

## 自定义跟踪器配置

编辑 [`custom_tracker.yaml`](custom_tracker.yaml) 即可调整跟踪算法及参数：

```yaml
# 切换算法：botsort 或 bytetrack
tracker_type: botsort

track_high_thresh: 0.25   # 第一阶段匹配置信度阈值
track_low_thresh: 0.1     # 第二阶段低分匹配阈值
new_track_thresh: 0.25    # 新建轨迹的最低分数
match_thresh: 0.8         # 轨迹与检测关联的相似度阈值
track_buffer: 30          # 丢失轨迹保留的帧数
fuse_score: true          # 是否融合检测分数与 IoU

# BoT-SORT 专有：全局运动补偿（移动摄像头场景推荐）
gmc_method: sparseOptFlow

# BoT-SORT 专有：ReID 外观重识别
with_reid: false
```

完整字段说明参见：<https://docs.ultralytics.com/modes/track/#how-do-i-configure-a-custom-tracker-for-ultralytics-yolo>

---

## RTSP 输出服务器

脚本通过 FFmpeg 将注释后的画面推送到 RTSP 服务器。
如果本地没有 RTSP 服务器，可以使用 [MediaMTX](https://github.com/bluenviron/mediamtx) 快速搭建：

```bash
# 下载并启动 MediaMTX（默认监听 rtsp://localhost:8554）
./mediamtx
```

然后运行脚本，输出流即可通过 VLC 等播放器订阅：

```
rtsp://localhost:8554/live/tracking
```

---

## 运行原理

```
RTSP 输入流
    │
    ▼
YOLO11 model.track(
    tracker="custom_tracker.yaml",   ← 指定跟踪器算法和参数
    persist=True,                    ← 跨帧保持轨迹状态
    stream=True,                     ← 逐帧生成（内存高效）
)
    │
    ├──► result.plot()  ──► cv2.imshow()          本地显示
    │
    └──► result.plot()  ──► FFmpeg stdin ──► RTSP 输出流
```
