# YOLO11 RTSP Tracking

使用 **YOLO11** 对 RTSP 视频流进行目标跟踪，同时：
- 在本地窗口**实时展示**带注释的画面
- 通过 FFmpeg 将注释后的画面**推送为 RTSP 输出流**
- 检测到目标时**自动保存标注帧**到本地目录
- 提供**内置 Web 监控面板**，实时展示最新检测帧与识别结果文字说明

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

## CUDA 加速

脚本启动时会**自动检测 CUDA**：若检测到可用的 NVIDIA GPU，则自动使用 GPU 0 进行推理；否则回退到 CPU。

### 安装支持 CUDA 的 PyTorch

`ultralytics` 会自动安装 PyTorch，但默认安装的是 CPU 版本。  
要启用 CUDA 加速，请**在安装 `ultralytics` 之前**手动安装对应 CUDA 版本的 PyTorch：

```bash
# 查看适合您 CUDA 版本的命令：https://pytorch.org/get-started/locally/

# 示例：CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 示例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 然后安装其余依赖
pip install -r requirements.txt
```

### 验证 CUDA 是否可用

```bash
python -c "
import torch
available = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if available and torch.cuda.device_count() > 0 else 'N/A'
print('CUDA available:', available, '|', name)
"
```

### 手动指定推理设备

脚本启动时自动选择最优设备，也可通过 `--device` 手动覆盖：

```bash
# 使用第一块 NVIDIA GPU（默认自动选择）
python rtsp_track.py --device 0

# 多 GPU（GPU 0 和 GPU 1）
python rtsp_track.py --device 0,1

# 强制使用 CPU（调试时有用）
python rtsp_track.py --device cpu

# Apple Silicon（MPS，自动检测，也可手动指定）
python rtsp_track.py --device mps
```

启动时日志会显示当前使用的设备，例如：

```
[INFO] CUDA available   : NVIDIA GeForce RTX 3080 – using GPU 0
```

---

## NVIDIA Jetson 上使用 CUDA（Jetson Orin / Xavier / Nano）

Jetson 设备运行 **ARM64（aarch64）** 架构，pytorch.org 提供的标准 x86 轮子**无法在 Jetson 上使用**。
CUDA 已作为 JetPack 的一部分预装，只需安装与 JetPack 版本匹配的 PyTorch ARM64 轮子即可。

### 第一步：确认 JetPack 版本

```bash
cat /etc/nv_tegra_release
# 或
dpkg -l | grep jetpack
```

### 第二步：安装 Jetson 专用 PyTorch（在安装 ultralytics 之前）

**JetPack 6.x（Jetson Orin，CUDA 12.x）**

```bash
# 从 NVIDIA NGC 安装 Jetson 专用 PyTorch ARM64 轮子
# 请先确认您的 JetPack 6.x 小版本，并查阅 NVIDIA 文档确认兼容的 PyTorch 版本：
# https://developer.nvidia.com/embedded/jetpack
pip install torch torchvision --index-url https://pypi.ngc.nvidia.com
pip install -r requirements.txt
```

**JetPack 5.x（CUDA 11.x）**

从 NVIDIA 开发者官网下载对应版本的 ARM64 轮子：

```bash
# 示例（请根据 JetPack 5.x 的具体小版本和 Python 版本替换文件名）
# 文件名格式示例：torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
# 下载地址：https://developer.download.nvidia.com/compute/redist/jp/
pip install torch-2.x.x-cpXX-cpXX-linux_aarch64.whl
pip install -r requirements.txt
```

### 第三步：验证 CUDA 在 Jetson 上可用

```bash
python -c "
import torch
available = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if available and torch.cuda.device_count() > 0 else 'N/A'
print('CUDA available:', available, '|', name)
"
```

正常输出示例（Jetson Orin）：

```
CUDA available: True | Orin
```

### 第四步：运行脚本

CUDA 可用后，脚本启动时会自动检测并使用 GPU：

```bash
python rtsp_track.py
# [INFO] CUDA available   : Orin – using GPU 0
```

也可手动指定：

```bash
python rtsp_track.py --device 0
```

> **注意**：若 `torch.cuda.is_available()` 返回 `False`，请检查：
> - 是否安装的是 Jetson 专用轮子（而非 x86 版本）
> - JetPack 与 PyTorch 轮子版本是否匹配
> - 运行 `nvcc --version` 确认 CUDA 工具链已安装

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
| `--device` | 自动 | 推理设备（`cpu`、`0` 表示 GPU 0、`0,1` 多 GPU、`mps` 表示 Apple Silicon）；省略时自动选择 CUDA GPU，否则 CPU |
| `--no-show` | — | 禁用本地显示窗口 |
| `--no-output` | — | 禁用 RTSP 输出流 |
| `--crf` | `28` | 输出流的 libx264 恒定码率因子（0–51）。值越小画质越高、码率越大；值越大压缩率越高、码率越小。28 是流媒体场景的良好折中点，23 为 libx264 默认值 |
| `--preset` | `ultrafast` | 输出流的 libx264 编码速度预设（`ultrafast` / `superfast` / `veryfast` / `faster` / `fast` / `medium` / `slow` / `slower` / `veryslow`）。更快的预设延迟低但压缩率较低；更慢的预设压缩率更高但 CPU 占用更大 |
| `--names` | `zh_names.yaml` | YAML 映射文件（拼音/英文类别名 → 中文显示名）；文件存在时自动加载 |
| `--font` | 自动检测 | TrueType 字体文件路径；显示中文名时需支持 CJK 字符（未指定时自动在系统常见位置查找） |
| `--no-web` | — | 禁用 Web 监控面板 |
| `--web-port` | `8080` | Web 监控面板监听端口 |
| `--save-dir` | `saved_frames` | 保存检测帧的目录；设为空字符串 `''` 可禁用帧保存 |

---

## 检测帧保存与 Web 监控面板

### 自动保存检测帧

每当 YOLO11 在当前帧中检测到至少一个目标时，脚本会自动将带有标注框和标签的帧保存为 JPEG 图片，文件名包含检测时间戳：

```
saved_frames/
├── detection_20260316_024400_123456.jpg
├── detection_20260316_024401_789012.jpg
└── ...
```

通过 `--save-dir` 可自定义保存目录，设为空字符串可禁用保存：

```bash
# 自定义保存目录
python rtsp_track.py --save-dir /data/detections

# 禁用帧保存
python rtsp_track.py --save-dir ""
```

### Web 监控面板

脚本启动时默认在后台开启一个轻量 HTTP 服务，可在浏览器中实时查看最新检测结果：

```
http://localhost:8080/
```

面板内容（每 2 秒自动刷新）：
- 📷 **最新检测帧**：含目标框和中文标签的标注图像
- 📋 **识别结果列表**：每个目标的类别名称（中文）和置信度百分比
- 🕐 **检测时间戳**和累计检测事件数
- 🖼️ **历史检测照片**：按时间倒序展示最近 50 次检测事件的照片；相同缺陷类型只要跟踪 ID（rid）不同，均作为独立事件记录并展示

此外还提供 JSON API，方便外部系统集成：

```
GET /api/detections   → {"detections":[...], "timestamp":"...", "detection_count":42, "accumulated_detections":[...], "history":[{"index":0,"timestamp":"...","detections":[...],"image_url":"/history/0.jpg"},...]}
GET /latest.jpg       → 最新检测帧（JPEG）
GET /history/{n}.jpg  → 第 n 帧历史检测图像（JPEG，n 从 0 开始，最多保留 50 帧）
```

通过 `--web-port` 可更改端口，`--no-web` 可完全禁用：

```bash
# 更改 Web 面板端口
python rtsp_track.py --web-port 9090

# 禁用 Web 面板
python rtsp_track.py --no-web
```

---

## 目标框中文标注

脚本默认加载 [`zh_names.yaml`](zh_names.yaml)，将模型中的拼音或英文类别名替换为中文显示名，使视频画面上的目标框标签显示为中文。

### 配置中文名称

编辑 `zh_names.yaml`，将左侧的键替换为**模型实际使用的类别名**，右侧填写希望在画面上显示的中文名称：

```yaml
quepian: 缺片
wushan: 污闪
posun: 破损
junyahuan_defect: 均压环缺陷
kaikouxiao_queshi: 开口销缺失
xiushi: 锈蚀
kaikouxiao_defect: 开口销缺陷
```

### 安装 CJK 字体

显示中文标签需要支持 CJK 字符的字体。脚本会自动在系统常见路径中查找，也可通过 `--font` 手动指定：

```bash
# Ubuntu/Debian – 安装文泉驿微米黑字体
sudo apt-get install fonts-wqy-microhei

# 或手动指定字体路径
python rtsp_track.py --font /path/to/your/cjk-font.ttf
```

### 禁用中文标注

若不需要中文名称映射，可通过 `--names ""` 跳过名称文件加载：

```bash
python rtsp_track.py --names ""
```

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
    ├──► result.plot()  ──► FFmpeg stdin ──► RTSP 输出流
    │
    └──► 检测到目标时
         ├── cv2.imwrite()  ──► saved_frames/      保存标注帧
         └── JPEG 编码     ──► HTTP 服务           Web 监控面板
```
