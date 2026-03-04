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
from ultralytics import YOLO

# Override the input source via the RTSP_SOURCE environment variable to avoid
# embedding credentials directly in the codebase, e.g.:
#   export RTSP_SOURCE="rtsp://user:password@host:554/stream"
DEFAULT_SOURCE = os.environ.get(
    "RTSP_SOURCE",
    "rtsp://user:0000@192.168.144.108:554/cam/realmonitor?channel=1&subtype=1",
)
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_TRACKER = "custom_tracker.yaml"   # or "botsort.yaml" / "bytetrack.yaml"
DEFAULT_OUTPUT_RTSP = "rtsp://localhost:8554/live/tracking"

# FPS sanity-check bounds: treat any value outside [1, MAX_REASONABLE_FPS] as
# unreliable and fall back to FALLBACK_FPS (common IP-camera default).
MAX_REASONABLE_FPS = 120
FALLBACK_FPS = 25.0


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
        "--no-output",
        action="store_true",
        help="Disable the RTSP output stream.",
    )
    args = parser.parse_args()

    show = not args.no_show
    push_output = not args.no_output

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
            annotated_frame = result.plot()

            # ---- Local display window ----
            if show:
                cv2.imshow("YOLO11 RTSP Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit key pressed – stopping.")
                    break

            # ---- Push annotated frame to output RTSP stream ----
            if ffmpeg_proc is not None and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.write(annotated_frame.tobytes())
                except BrokenPipeError:
                    print("[WARN] FFmpeg pipe closed – output stream stopped.")
                    ffmpeg_proc = None

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
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
