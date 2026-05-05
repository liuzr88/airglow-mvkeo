import shutil
import subprocess
import numpy as np
import pytest
from airglow_mvkeo.movie import write_h264_video, FFmpegError

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (ffmpeg and ffprobe), reason="ffmpeg/ffprobe not on PATH")

def _frames_iter(n, h, w):
    for i in range(n):
        yield np.full((h, w, 3), (i * 255 // n) % 256, dtype=np.uint8)

def test_write_video_produces_playable_mp4(tmp_path):
    out = tmp_path / "out.mp4"
    write_h264_video(_frames_iter(20, 64, 64), out, fps=10, crf=23, pix_fmt="yuv420p")
    assert out.exists() and out.stat().st_size > 0
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
                       capture_output=True, text=True, check=True)
    assert 1.5 < float(r.stdout.strip()) < 2.5

def test_write_video_raises_on_bad_args(tmp_path):
    out = tmp_path / "fail.mp4"
    with pytest.raises(FFmpegError):
        write_h264_video(_frames_iter(5, 64, 64), out, fps=10, crf=23,
                         pix_fmt="not_a_real_pix_fmt")
