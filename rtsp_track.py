"""
YOLO11 RTSP Tracking Script
Reference: https://docs.ultralytics.com/modes/track/

Features:
  - Reads from an RTSP (or any OpenCV-compatible) input stream
  - Performs object tracking with YOLO11 using a configurable tracker
    (botsort / bytetrack, or a custom YAML – see custom_tracker.yaml)
  - Displays annotated frames in a local window
  - Re-streams annotated frames to an RTSP output via FFmpeg
  - Saves annotated frames to disk whenever objects are detected
  - Serves a web dashboard (HTTP) showing the latest detection image and text descriptions
"""

import argparse
import datetime
import http.server
import json
import os
import socketserver
import subprocess
import threading
import time

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Override the input source via the RTSP_SOURCE environment variable to avoid
# embedding credentials directly in the codebase, e.g.:
#   export RTSP_SOURCE="rtsp://user:password@host:554/stream"
DEFAULT_SOURCE = os.environ.get(
    "RTSP_SOURCE",
    "rtsp://user:0000@192.168.144.108:554/cam/realmonitor?channel=1&subtype=0",
)
DEFAULT_MODEL = "best_640_s.pt"
DEFAULT_TRACKER = "custom_tracker.yaml"   # or "botsort.yaml" / "bytetrack.yaml"
DEFAULT_OUTPUT_RTSP = "rtsp://localhost:8554/live/tracking"
DEFAULT_NAMES = "zh_names.yaml"

# Candidate CJK-capable font paths (searched in order on Linux/macOS systems).
# Install one of the corresponding packages to enable Chinese label rendering,
# e.g. on Ubuntu/Debian:  sudo apt-get install fonts-wqy-microhei
#
# The ultralytics library downloads Arial.Unicode.ttf to its config directory
# on first use.  We list that cached copy first so that subsequent runs can
# reuse it without triggering another network download.
_CJK_FONT_CANDIDATES = [
    # Ultralytics cached font (downloaded automatically on first PIL render)
    os.path.expanduser("~/.config/Ultralytics/Arial.Unicode.ttf"),
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

# FPS sanity-check bounds: treat any value outside [1, MAX_REASONABLE_FPS] as
# unreliable and fall back to FALLBACK_FPS (common IP-camera default).
MAX_REASONABLE_FPS = 120
FALLBACK_FPS = 25.0

WINDOW_NAME = "YOLO11 RTSP Tracking"

# ---------------------------------------------------------------------------
# Global state shared between the tracking loop and the web server thread
# ---------------------------------------------------------------------------
_web_state: dict = {
    "lock": threading.Lock(),
    "latest_frame": None,       # bytes – JPEG-encoded annotated frame
    "latest_detections": [],    # list of {"name": str, "confidence": float}
    "latest_timestamp": None,   # ISO-8601 string
    "detection_count": 0,       # total detection events since start
}

_WEB_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YOLO11 目标检测监控</title>
  <style>
    body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#1a1a2e;color:#eee}}
    header{{background:#16213e;padding:16px 24px;display:flex;align-items:center;gap:12px}}
    header h1{{margin:0;font-size:1.4rem;color:#e94560}}
    .badge{{background:#0f3460;border-radius:12px;padding:4px 12px;font-size:.85rem}}
    .container{{display:flex;flex-wrap:wrap;gap:16px;padding:20px}}
    .card{{background:#16213e;border-radius:10px;padding:16px;flex:1;min-width:280px}}
    .card h2{{margin:0 0 12px;font-size:1rem;color:#e94560;border-bottom:1px solid #0f3460;padding-bottom:8px}}
    .frame-img{{width:100%;border-radius:6px;display:block}}
    .det-list{{list-style:none;margin:0;padding:0}}
    .det-list li{{display:flex;justify-content:space-between;align-items:center;
                  padding:8px 10px;margin-bottom:6px;background:#0f3460;border-radius:6px}}
    .det-name{{font-weight:bold;font-size:1rem}}
    .det-conf{{background:#e94560;border-radius:10px;padding:2px 10px;font-size:.8rem}}
    .no-det{{color:#888;font-style:italic}}
    .meta{{font-size:.8rem;color:#888;margin-top:10px}}
    .refresh-note{{text-align:center;padding:10px;color:#555;font-size:.8rem}}
  </style>
</head>
<body>
  <header>
    <h1>&#128247; YOLO11 目标检测监控</h1>
    <span class="badge">检测事件：{detection_count}</span>
    <span class="badge">{timestamp_label}</span>
  </header>
  <div class="container">
    <div class="card" style="flex:2;min-width:360px">
      <h2>最新检测帧</h2>
      {img_tag}
      <p class="meta">检测时间：{timestamp}</p>
    </div>
    <div class="card">
      <h2>识别结果</h2>
      {det_html}
    </div>
  </div>
  <p class="refresh-note">页面每 2 秒自动刷新</p>
  <script>setTimeout(()=>location.reload(),2000);</script>
</body>
</html>
"""


def _build_html() -> bytes:
    """Render the dashboard HTML from current *_web_state*."""
    with _web_state["lock"]:
        frame_available = _web_state["latest_frame"] is not None
        detections = list(_web_state["latest_detections"])
        ts = _web_state["latest_timestamp"] or "—"
        count = _web_state["detection_count"]

    img_tag = (
        '<img class="frame-img" src="/latest.jpg" alt="最新检测帧">'
        if frame_available
        else '<p class="no-det">暂无检测帧</p>'
    )
    if detections:
        items = "".join(
            f'<li><span class="det-name">{d["name"]}</span>'
            f'<span class="det-conf">{d["confidence"]:.1%}</span></li>'
            for d in detections
        )
        det_html = f'<ul class="det-list">{items}</ul>'
    else:
        det_html = '<p class="no-det">等待检测结果…</p>'

    ts_label = ts[:19].replace("T", " ") if ts != "—" else "—"
    html = _WEB_PAGE_TEMPLATE.format(
        detection_count=count,
        timestamp_label=ts_label,
        timestamp=ts,
        img_tag=img_tag,
        det_html=det_html,
    )
    return html.encode("utf-8")


class _DetectionHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler serving the detection dashboard."""

    def log_message(self, fmt, *args):  # suppress per-request console noise
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _build_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/latest.jpg"):
            with _web_state["lock"]:
                frame_bytes = _web_state["latest_frame"]
            if frame_bytes is None:
                self.send_error(404, "No detection frame available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame_bytes)))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            self.wfile.write(frame_bytes)

        elif self.path.startswith("/api/detections"):
            with _web_state["lock"]:
                data = {
                    "detections": _web_state["latest_detections"],
                    "timestamp": _web_state["latest_timestamp"],
                    "detection_count": _web_state["detection_count"],
                }
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)


def start_web_server(port: int) -> socketserver.TCPServer:
    """Start the detection web dashboard in a background daemon thread.

    Returns the *TCPServer* instance (call ``server.shutdown()`` to stop it).
    """
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("", port), _DetectionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def build_ffmpeg_push_cmd(
    width: int,
    height: int,
    fps: float,
    output_rtsp: str,
    crf: int = 28,
    preset: str = "ultrafast",
) -> list:
    """Return the FFmpeg command list that pushes raw BGR frames to an RTSP server.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate.
        output_rtsp: Destination RTSP URL.
        crf: libx264 Constant Rate Factor (0–51).  Lower = higher quality /
             larger bitrate; higher = more compression / smaller bitrate.
             28 is a good starting point for noticeably smaller streams while
             keeping acceptable quality.  Use 23 to match the libx264 default.
        preset: libx264 encoding speed preset.  Faster presets (e.g.
                ``ultrafast``) have lower latency but produce larger files;
                slower presets (e.g. ``veryfast``, ``fast``) compress better
                at the cost of more CPU.
    """
    return [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-tune", "zerolatency",
        "-crf", str(crf),
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        output_rtsp,
    ]


def find_cjk_font():
    """Return the path of the first available CJK-capable font, or None."""
    for path in _CJK_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def load_names_map(path: str) -> dict:
    """Load a YAML file mapping original class names to Chinese display names."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise OSError(f"Cannot read names file '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Names file '{path}' must be a YAML mapping, got {type(data).__name__}."
        )
    return {str(k): str(v) for k, v in data.items() if k is not None and v is not None}


def apply_names_map(model_names: dict, name_map: dict) -> dict:
    """Return a copy of *model_names* with values replaced according to *name_map*."""
    return {idx: name_map.get(name, name) for idx, name in model_names.items()}


def open_capture(source: str, retries: int = 5, delay: float = 2.0) -> cv2.VideoCapture:
    """Open a video source with automatic retries."""
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        print(f"[WARN] Cannot open source (attempt {attempt}/{retries}). Retrying in {delay}s…")
        time.sleep(delay)
    raise RuntimeError(f"Cannot open video source after {retries} attempts: {source}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 object tracking on an RTSP stream with live display and RTSP output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Input RTSP URL or any OpenCV-compatible video source.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO11 model weights (e.g. yolo11n.pt, yolo11s.pt).",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER,
        help=(
            "Tracker configuration YAML. "
            "Built-ins: 'botsort.yaml', 'bytetrack.yaml'. "
            "Custom: path to your own YAML (e.g. custom_tracker.yaml)."
        ),
    )
    parser.add_argument(
        "--output-rtsp",
        default=DEFAULT_OUTPUT_RTSP,
        help="Output RTSP URL for the annotated stream.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Inference device: 'cpu', '0' (first CUDA GPU), '0,1' (multi-GPU), "
            "'mps' (Apple Silicon). "
            "Omit to auto-select: CUDA GPU if available, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the local display window.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Display the result window in fullscreen mode.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable the RTSP output stream.",
    )
    parser.add_argument(
        "--names",
        default=DEFAULT_NAMES,
        help=(
            "YAML file mapping model class names (pinyin/English) to Chinese display names. "
            "Loaded automatically if the file exists; ignored silently if it does not."
        ),
    )
    parser.add_argument(
        "--font",
        default=None,
        help=(
            "Path to a TrueType font file used for label rendering. "
            "Must support CJK characters when Chinese names are in use. "
            "Auto-detected from common system font locations when not specified."
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        metavar="N",
        help=(
            "libx264 Constant Rate Factor for the output stream (0–51). "
            "Lower values produce higher quality at larger bitrates; "
            "higher values compress more aggressively at lower quality. "
            "28 is a good balance for streaming; use 23 to match the libx264 default."
        ),
    )
    parser.add_argument(
        "--preset",
        default="ultrafast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help=(
            "libx264 encoding speed preset for the output stream. "
            "Faster presets reduce CPU load and latency but compress less; "
            "slower presets improve compression at the cost of more CPU."
        ),
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the web dashboard (default: enabled on --web-port).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8080,
        metavar="PORT",
        help="Port for the HTTP detection dashboard.",
    )
    parser.add_argument(
        "--save-dir",
        default="saved_frames",
        metavar="DIR",
        help=(
            "Directory for saving annotated frames when objects are detected. "
            "Set to empty string ('') to disable frame saving."
        ),
    )
    args = parser.parse_args()

    show = not args.no_show
    push_output = not args.no_output
    fullscreen = args.fullscreen
    enable_web = not args.no_web
    save_dir = args.save_dir.strip() if args.save_dir else ""

    # Create the save directory early so the web server can mention its path.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve inference device
    # ------------------------------------------------------------------
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "0"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "unknown"
            print(f"[INFO] CUDA available   : {gpu_name} – using GPU 0")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("[INFO] MPS available    : using Apple Silicon GPU")
        else:
            device = "cpu"
            print("[INFO] No GPU detected  : using CPU (set --device 0 to force CUDA)")
    else:
        print(f"[INFO] Device (manual)  : {device}")

    # ------------------------------------------------------------------
    # Load YOLO11 model
    # Docs: https://docs.ultralytics.com/models/yolo11/
    # ------------------------------------------------------------------
    print(f"[INFO] Loading model  : {args.model}")
    print(f"[INFO] Tracker config : {args.tracker}")
    model = YOLO(args.model)

    # ------------------------------------------------------------------
    # Apply Chinese class-name mapping (pinyin → Chinese characters)
    # ------------------------------------------------------------------
    plot_kwargs: dict = {}
    override_names = None  # set when model-level name assignment fails
    if args.names and os.path.isfile(args.names):
        name_map = load_names_map(args.names)
        new_names = apply_names_map(model.names, name_map)
        # model.names is a read-only property in some ultralytics/PyTorch
        # versions (no setter).  Try progressively deeper attributes and, as a
        # last resort for TensorRT engine files, patch each result in the loop.
        names_set = False
        try:
            model.names = new_names
            names_set = True
        except AttributeError:
            pass
        if not names_set:
            try:
                model.model.names = new_names
                names_set = True
            except AttributeError:
                pass
        if not names_set:
            # TensorRT engines may expose names only on the Results objects
            # produced during inference.  Store the mapping and apply it
            # per-frame inside the tracking loop below.
            override_names = new_names
        print(f"[INFO] Names file     : {args.names} ({len(name_map)} entries)")

        # PIL rendering is required for Unicode (Chinese) label text.
        # Detect a CJK-capable font automatically when none is specified.
        font_path = args.font or find_cjk_font()
        if font_path:
            print(f"[INFO] CJK font       : {font_path}")
            plot_kwargs["font"] = font_path
        else:
            print(
                "[WARN] No CJK font found – Chinese labels may not render correctly. "
                "Install fonts-wqy-microhei (Ubuntu/Debian) or specify --font."
            )
        plot_kwargs["pil"] = True
    elif args.font:
        plot_kwargs["font"] = args.font
        plot_kwargs["pil"] = True

    # ------------------------------------------------------------------
    # Probe the input stream for frame dimensions and FPS
    # ------------------------------------------------------------------
    print(f"[INFO] Opening source : {args.source}")
    cap = open_capture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > MAX_REASONABLE_FPS:
        fps = FALLBACK_FPS
    cap.release()
    print(f"[INFO] Stream info    : {width}x{height} @ {fps:.1f} fps")

    # ------------------------------------------------------------------
    # Start HTTP detection dashboard (optional)
    # ------------------------------------------------------------------
    web_server = None
    if enable_web:
        try:
            web_server = start_web_server(args.web_port)
            print(f"[INFO] Web dashboard  : http://localhost:{args.web_port}/")
        except OSError as exc:
            print(f"[WARN] Cannot start web server on port {args.web_port}: {exc}")
            web_server = None

    # ------------------------------------------------------------------
    # Start FFmpeg RTSP push process (optional)
    # ------------------------------------------------------------------
    ffmpeg_proc = None
    if push_output:
        cmd = build_ffmpeg_push_cmd(width, height, fps, args.output_rtsp, crf=args.crf, preset=args.preset)
        print(f"[INFO] Output RTSP    : {args.output_rtsp}")
        print(f"[INFO] Video CRF      : {args.crf}  (preset: {args.preset})")
        try:
            ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("[WARN] FFmpeg not found – RTSP output disabled. Install FFmpeg to enable it.")
            ffmpeg_proc = None

    # ------------------------------------------------------------------
    # Run YOLO11 tracking in streaming mode
    #
    # Key parameters (see https://docs.ultralytics.com/modes/track/):
    #   tracker  – YAML file selecting the tracking algorithm and its settings
    #   persist  – keep tracker state across successive model.track() calls
    #   stream   – yield Results one-by-one (memory-efficient for long streams)
    #   conf     – minimum detection confidence to track
    #   iou      – NMS IoU overlap threshold
    # ------------------------------------------------------------------
    print("[INFO] Starting tracking. Press 'q' in the display window to quit.")
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
    results = None
    try:
        results = model.track(
            source=args.source,
            tracker=args.tracker,       # custom or built-in tracker YAML
            persist=True,               # maintain track IDs across frames
            stream=True,                # memory-efficient generator
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )

        for result in results:
            # For TensorRT engines the name mapping couldn't be injected into
            # the model directly; apply it to each result object instead.
            if override_names is not None:
                result.names = override_names
            # Render bounding boxes, class labels, and track IDs on the frame
            annotated_frame = result.plot(**plot_kwargs)

            # result.plot(pil=True) returns a PIL Image (RGB); convert to a
            # BGR numpy array so cv2.imshow() and the FFmpeg pipe work correctly.
            if not isinstance(annotated_frame, np.ndarray):
                annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

            # ---- Save frame and update web state when objects are detected ----
            if result.boxes is not None and len(result.boxes) > 0:
                names_map = result.names or {}
                detections = [
                    {
                        "name": names_map.get(int(box.cls[0]), str(int(box.cls[0]))),
                        "confidence": float(box.conf[0]),
                    }
                    for box in result.boxes
                ]

                ts = datetime.datetime.now(datetime.timezone.utc)
                ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")

                # Save annotated frame to disk
                if save_dir:
                    img_path = os.path.join(save_dir, f"detection_{ts_str}.jpg")
                    cv2.imwrite(img_path, annotated_frame)

                # Encode frame as JPEG for the web server
                if web_server is not None:
                    ok, buf = cv2.imencode(".jpg", annotated_frame)
                    if ok:
                        frame_bytes = buf.tobytes()
                        with _web_state["lock"]:
                            _web_state["latest_frame"] = frame_bytes
                            _web_state["latest_detections"] = detections
                            _web_state["latest_timestamp"] = ts.isoformat()
                            _web_state["detection_count"] += 1

            # ---- Local display window ----
            if show:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit key pressed – stopping.")
                    break

            # ---- Push annotated frame to output RTSP stream ----
            if ffmpeg_proc is not None and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.write(annotated_frame.tobytes())
                except BrokenPipeError:
                    print("[WARN] FFmpeg pipe closed – output stream stopped.")
                    try:
                        ffmpeg_proc.stdin.close()
                    except Exception:
                        pass
                    ffmpeg_proc.wait()
                    ffmpeg_proc = None
                    if not show:
                        break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")
        raise
    finally:
        # Explicitly close the generator so the YOLO/OpenCV background thread
        # is torn down in a controlled manner before the interpreter exits.
        # Without this, Python's GC closes it at an arbitrary point and the
        # C++ runtime raises "terminate called without an active exception".
        if results is not None:
            try:
                results.close()
            except Exception:
                pass
        if show:
            cv2.destroyAllWindows()
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.wait()
        if web_server is not None:
            web_server.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
