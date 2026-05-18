"""Small web artifacts: contact sheets and per-night HTML reports."""
from __future__ import annotations
from html import escape
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_contact_sheet(frames: Iterable[tuple[np.ndarray, str]], out_path: str | Path,
                        *, columns: int = 4, thumb_size: int = 220) -> None:
    """Write a labeled JPEG contact sheet from RGB uint8 frames."""
    items = [(Image.fromarray(f, mode="RGB"), label) for f, label in frames]
    if not items:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = max(1, columns)
    rows = (len(items) + columns - 1) // columns
    label_h = 24
    W = columns * thumb_size
    H = rows * (thumb_size + label_h)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    for idx, (img, label) in enumerate(items):
        r, c = divmod(idx, columns)
        x = c * thumb_size
        y = r * (thumb_size + label_h)
        thumb = img.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 6, y + 4), label, fill="black", font=font)
    sheet.save(out, quality=88, optimize=True)


def write_html_report(path: str | Path, payload: dict, files: dict[str, str]) -> None:
    """Write a compact static report with links to generated artifacts."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"{payload.get('band', '')} {payload.get('date', '')} @ {payload.get('site', '')}"
    rows = [
        ("Frames", f"{payload.get('n_frames_used')} / {payload.get('n_frames_total')}"),
        ("UT span", f"{payload.get('start_ut')} to {payload.get('end_ut')}"),
        ("FPS", str(payload.get("fps"))),
        ("Style", str(payload.get("style", ""))),
    ]
    links = "\n".join(
        f'<li><a href="{escape(name)}">{escape(key)}</a></li>'
        for key, name in files.items()
        if name
    )
    meta_rows = "\n".join(
        f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #111; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    table {{ border-collapse: collapse; margin-bottom: 20px; }}
    th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; }}
    th {{ color: #555; font-weight: 600; }}
    ul {{ line-height: 1.8; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table>{meta_rows}</table>
  <ul>{links}</ul>
  {"<img src='" + escape(files["contact_sheet"]) + "' alt='contact sheet'>" if "contact_sheet" in files else ""}
</body>
</html>
"""
    out.write_text(html)
