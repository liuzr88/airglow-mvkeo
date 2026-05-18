"""ffmpeg-piped H.264 video writer with deadlock-safe stderr drain."""
from __future__ import annotations
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Iterable
import numpy as np

class FFmpegError(RuntimeError):
    pass

def _find_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FFmpegError(
            "ffmpeg not found on PATH. Install: `brew install ffmpeg`, "
            "or `pip install imageio-ffmpeg`."
        ) from exc
    p = imageio_ffmpeg.get_ffmpeg_exe()
    return p

def write_h264_video(frames: Iterable[np.ndarray], out_path: str | Path, *,
                     fps: int, crf: int, pix_fmt: str = "yuv420p",
                     codec: str = "libx264", web: bool = True) -> None:
    """Stream RGB uint8 frames to ffmpeg, encoding H.264 mp4. Drains stderr on a
    background thread to avoid pipe deadlock."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()

    it = iter(frames)
    first = next(it, None)
    if first is None:
        raise FFmpegError("no frames provided")
    h, w, c = first.shape
    if c != 3 or first.dtype != np.uint8:
        raise FFmpegError(f"frames must be (H,W,3) uint8, got shape={first.shape} dtype={first.dtype}")

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", codec, "-crf", str(crf), "-pix_fmt", pix_fmt,
    ]
    if web and codec == "libx264":
        cmd.extend(["-preset", "medium", "-profile:v", "high", "-level", "4.0"])
    cmd.extend([
        "-movflags", "+faststart",
        str(out),
    ])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, bufsize=0)

    err_buf: deque[bytes] = deque(maxlen=256)
    def _drain():
        for line in iter(proc.stderr.readline, b""):
            err_buf.append(line)
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    try:
        proc.stdin.write(np.ascontiguousarray(first).tobytes())
        for frame in it:
            if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
                raise FFmpegError("inconsistent frame shape/dtype")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait()
        t.join(timeout=2.0)
        if rc != 0:
            err = b"".join(err_buf).decode("utf-8", errors="replace")
            raise FFmpegError(f"ffmpeg exited {rc}\n{err}")
