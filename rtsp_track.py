"""
YOLO11 RTSP Tracking Script
Reference: https://docs.ultralytics.com/modes/track/

Features:
  - Reads from an RTSP (or any OpenCV-compatible) input stream
  - Performs object tracking with YOLO11 using a configurable tracker
    (botsort / bytetrack, or a custom YAML – see custom_tracker.yaml)
  - Displays annotated frames in a local window
  - Re-streams annotated frames to an RTSP output via FFmpeg
"""

import argparse
import os
import subprocess
import time

import cv2
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


def build_ffmpeg_push_cmd(width: int, height: int, fps: float, output_rtsp: str) -> list:
    """Return the FFmpeg command list that pushes raw BGR frames to an RTSP server."""
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
        "-preset", "ultrafast",
        "-tune", "zerolatency",
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
    args = parser.parse_args()

    show = not args.no_show
    push_output = not args.no_output
    fullscreen = args.fullscreen

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
    if args.names and os.path.isfile(args.names):
        name_map = load_names_map(args.names)
        new_names = apply_names_map(model.names, name_map)
        # model.names is a read-only property in some ultralytics/PyTorch
        # versions (no setter).  Fall back to the inner nn.Module when needed.
        try:
            model.names = new_names
        except AttributeError:
            model.model.names = new_names
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
    # Start FFmpeg RTSP push process (optional)
    # ------------------------------------------------------------------
    ffmpeg_proc = None
    if push_output:
        cmd = build_ffmpeg_push_cmd(width, height, fps, args.output_rtsp)
        print(f"[INFO] Output RTSP    : {args.output_rtsp}")
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
            # Render bounding boxes, class labels, and track IDs on the frame
            annotated_frame = result.plot(**plot_kwargs)

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
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
