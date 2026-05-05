"""PIL-based overlay primitives for movie frames. Pure: array → array."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

def apply_colormap(image: np.ndarray, vmin: float, vmax: float,
                   name: str, crop: tuple[int, int]) -> np.ndarray:
    """Scale `image` to [0,1], apply named matplotlib colormap with crop, → uint8 RGB."""
    cmap = matplotlib.colormaps[name].resampled(128)
    table = (cmap(np.arange(128))[:, :3] * 255).astype(np.uint8)
    table = table[crop[0]:crop[1]]
    n = table.shape[0]
    norm = np.clip((image - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    idx = (norm * (n - 1)).astype(np.uint16)
    return table[idx]

def _to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb, mode="RGB")

def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img)

def draw_zenith_ring(rgb: np.ndarray, cx: int, cy: int, radius: int,
                     color: tuple[int, int, int] = (0, 0, 0),
                     width: int = 1) -> np.ndarray:
    img = _to_pil(rgb)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
              outline=color, width=width)
    return _to_array(img)

def draw_crosshair(rgb: np.ndarray, cx: int, cy: int, half_len: int,
                   color_outer=(255, 255, 255), color_inner=(0, 0, 0),
                   width_outer=2, width_inner=1) -> np.ndarray:
    img = _to_pil(rgb)
    d = ImageDraw.Draw(img)
    # Draw shadow/outline (inner/dark) first at wider width, then bright color on top.
    d.line([cx - half_len, cy, cx + half_len, cy], fill=color_inner, width=width_outer + 2)
    d.line([cx, cy - half_len, cx, cy + half_len], fill=color_inner, width=width_outer + 2)
    d.line([cx - half_len, cy, cx + half_len, cy], fill=color_outer, width=width_inner)
    d.line([cx, cy - half_len, cx, cy + half_len], fill=color_outer, width=width_inner)
    return _to_array(img)

def draw_text_corner(rgb: np.ndarray, lines: list[str], anchor: str,
                     color=(255, 255, 255), font_size: int = 14) -> np.ndarray:
    """Draw text at one of the four corners. anchor in {'tl','tr','bl','br'}."""
    img = _to_pil(rgb)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    h, w = rgb.shape[:2]
    pad = 6
    for i, text in enumerate(lines):
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if anchor == "tl": x, y = pad, pad + i * (th + 2)
        elif anchor == "tr": x, y = w - tw - pad, pad + i * (th + 2)
        elif anchor == "bl": x, y = pad, h - (len(lines) - i) * (th + 2) - pad
        elif anchor == "br": x, y = w - tw - pad, h - (len(lines) - i) * (th + 2) - pad
        else: raise ValueError(f"bad anchor {anchor}")
        d.text((x, y), text, fill=color, font=font)
    return _to_array(img)

def draw_marker(rgb: np.ndarray, cx: int, cy: int, R: int,
                az_deg: float, r_frac: float, label: str,
                color=(255, 255, 255), font_size: int = 11) -> np.ndarray:
    """Place a '+' and a label at angular position (az_deg, r_frac·R) measured
    from East CCW (matches MATLAB cosd/sind convention)."""
    rad = np.deg2rad(az_deg)
    px = int(cx + r_frac * R * np.cos(rad))
    py = int(cy + r_frac * R * np.sin(rad))
    img = _to_pil(rgb)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    d.text((px - 4, py - 8), "+", fill=color, font=font)
    d.text((px - 12, py + 6), label, fill=color, font=font)
    return _to_array(img)

def compose_movie_frame(*, image: np.ndarray, vmin: float, vmax: float,
                        x0: int, y0: int, R: int,
                        date_str: str, time_str: str, site_label: str,
                        markers: list[dict],
                        colormap_name: str, colormap_crop: tuple[int, int],
                        is_first_or_last: bool) -> np.ndarray:
    """Compose one movie frame from a single 2D `image` (H,W) → uint8 RGB."""
    rgb = apply_colormap(image, vmin, vmax, colormap_name, colormap_crop)
    rgb = draw_zenith_ring(rgb, x0, y0, R, color=(0, 0, 0), width=1)
    if is_first_or_last:
        rgb = draw_zenith_ring(rgb, x0, y0, R // 3, color=(0, 0, 0), width=1)
        rgb = draw_zenith_ring(rgb, x0, y0, 2 * R // 3, color=(0, 0, 0), width=1)
    rgb = draw_crosshair(rgb, x0, y0, half_len=max(R // 25, 4))
    # Compass labels around the lower-left direction crosshair (matches the MATLAB
    # convention at ReadNcOrig.m:330-333: N at top, S at bottom, W left, E right).
    rgb = draw_text_corner(rgb, ["N"], "tl")
    rgb = draw_text_corner(rgb, [site_label, date_str + " " + time_str], "bl")
    for m in markers:
        rgb = draw_marker(rgb, x0, y0, R, m["az_deg"], m["r_frac"], m["name"])
    return rgb
