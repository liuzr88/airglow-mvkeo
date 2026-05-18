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


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            from matplotlib import font_manager
            path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()


def _draw_text_with_outline(d: ImageDraw.ImageDraw, x: int, y: int, text: str,
                             font, fill=(255, 255, 255),
                             outline=(0, 0, 0)) -> None:
    """Draw text with a 1-pixel black outline for readability."""
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        d.text((x + dx, y + dy), text, fill=outline, font=font)
    d.text((x, y), text, fill=fill, font=font)


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
    font = _load_font(font_size)
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
        _draw_text_with_outline(d, x, y, text, font=font, fill=color)
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
    font = _load_font(font_size)
    _draw_text_with_outline(d, px - 4, py - 8, "+", font=font, fill=color)
    _draw_text_with_outline(d, px - 12, py + 6, label, font=font, fill=color)
    return _to_array(img)


def draw_compass_rose(rgb: np.ndarray, cx: int = 40, cy: int = -1,
                      size: int = 15, color=(255, 255, 255)) -> np.ndarray:
    """Draw a compass rose (crosshair + N/S/W/E labels) at (cx, cy).

    Replicates ReadNcOrig.m lines 307-338.  MATLAB uses Ydir Normal so its
    y=65 is near the top; in PIL y=0 is the top, so N is placed *above* the
    crosshair centre (smaller y) and S *below* it.

    cx, cy are in PIL pixel coords.  Pass cy=-1 to auto-place at H-40.
    """
    H = rgb.shape[0]
    if cy < 0:
        cy = H - 40

    img = _to_pil(rgb)
    d = ImageDraw.Draw(img)
    half = size  # half-length of the crosshair arms

    # Outer white wide line, inner black thin line (matches MATLAB 307-314)
    white, black = (255, 255, 255), (0, 0, 0)
    d.line([cx, cy - half, cx, cy + half], fill=white, width=3)
    d.line([cx - half, cy, cx + half, cy], fill=white, width=3)
    d.line([cx, cy - half, cx, cy + half], fill=black, width=1)
    d.line([cx - half, cy, cx + half, cy], fill=black, width=1)

    # North pointer: small wedge above the crosshair (MATLAB lines 311-314)
    # Two diagonal lines meeting at the top of the N arm
    tip_y = cy - half
    base_offset = 4
    d.line([cx - base_offset, tip_y + 10, cx, tip_y], fill=white, width=3)
    d.line([cx + base_offset, tip_y + 10, cx, tip_y], fill=white, width=3)
    d.line([cx - base_offset, tip_y + 10, cx, tip_y], fill=black, width=1)
    d.line([cx + base_offset, tip_y + 10, cx, tip_y], fill=black, width=1)

    font = _load_font(13)
    label_offset = half + 12
    # N above centre (smaller y in PIL = up), S below, W left, E right
    _draw_text_with_outline(d, cx - 4, cy - label_offset - 8, "N", font=font,
                            fill=color, outline=black)
    _draw_text_with_outline(d, cx - 4, cy + label_offset - 2, "S", font=font,
                            fill=color, outline=black)
    _draw_text_with_outline(d, cx - label_offset - 8, cy - 6, "W", font=font,
                            fill=color, outline=black)
    _draw_text_with_outline(d, cx + label_offset, cy - 6, "E", font=font,
                            fill=color, outline=black)

    return _to_array(img)


def draw_zenith_angle_labels(rgb: np.ndarray, cx: int, cy: int,
                              R: int, color=(255, 255, 255)) -> np.ndarray:
    """Draw 30°/60°/90° zenith-angle labels on first/last frames.

    Replicates ReadNcOrig.m lines 281-301.  Labels are placed at angular
    position ~0° (East direction) at radii R/3, 2R/3, R, slightly inside each
    ring (factor 0.85 for radial offset; the text appears just inside the ring).
    In PIL coords the 0° direction (East) is the right side of the image, so
    we offset by +cos(angle)*radius from the centre.  A small upward nudge (-8)
    puts the text just above the ring intersection.
    """
    img = _to_pil(rgb)
    d = ImageDraw.Draw(img)
    font = _load_font(13)

    # MATLAB: text at (R + r*cosd(0°), R - r*sind(0°)) with Ydir Normal.
    # In PIL cy+... because y increases downward but the MATLAB formula
    # subtracts (upward in Ydir Normal).  With angle=0°, sin=0 so vertical
    # offset is 0; we just nudge upward a tiny bit for readability.
    offsets = [
        (R // 3, "30°"),
        (2 * R // 3, "60°"),
        (R, "90°"),
    ]
    for r, label in offsets:
        # Place slightly inside the ring (0.85 factor) toward the East side
        lx = int(cx + r * 0.85)
        ly = int(cy - r * 0.7) - 4  # nudge up to sit above the ring
        _draw_text_with_outline(d, lx, ly, label, font=font, fill=color)

    return _to_array(img)


def compose_movie_frame(*, image: np.ndarray, vmin: float, vmax: float,
                        x0: int, y0: int, R: int,
                        date_str: str, time_str: str,
                        site_label: str,
                        site_name: str = "",
                        band: str = "",
                        markers: list[dict],
                        colormap_name: str, colormap_crop: tuple[int, int],
                        is_first_or_last: bool,
                        draw_geometry: bool = False,
                        output_size: int | None = None,
                        clean_overlay: bool = False,
                        label_font_size: int = 13) -> np.ndarray:
    """Compose one movie frame from a single 2D `image` (H,W) → uint8 RGB.

    Replicates the active MATLAB CreateMovNC.m overlay layout:
    - Corner compass rose with N/S/W/E labels (lower-left)
    - Top-left: site/band and coordinate label
    - Top-right: date and UT time
    - Bottom-right: pixel-size badge

    The original source has zenith rings, crosshairs, and site markers present
    but commented out.  ``draw_geometry=True`` keeps that richer overlay
    available without changing the MATLAB-compatible default.
    """
    H, W = image.shape[:2]
    # MATLAB renders image row 1 at the bottom via ``Ydir='Normal'``.
    rgb = apply_colormap(np.flipud(image), vmin, vmax, colormap_name, colormap_crop)

    # --- Zenith ring(s) and central crosshair ---
    if draw_geometry:
        cy = H - 1 - y0
        rgb = draw_zenith_ring(rgb, x0, cy, R, color=(0, 0, 0), width=1)
        if is_first_or_last:
            rgb = draw_zenith_ring(rgb, x0, cy, R // 3, color=(0, 0, 0), width=1)
            rgb = draw_zenith_ring(rgb, x0, cy, 2 * R // 3, color=(0, 0, 0), width=1)
        rgb = draw_crosshair(rgb, x0, cy, half_len=max(R // 25, 4))

        # --- Zenith-angle labels (first/last frame only) ---
        if is_first_or_last:
            rgb = draw_zenith_angle_labels(rgb, x0, cy, R, color=(255, 255, 255))

    # --- Corner compass rose (lower-left, all frames) ---
    rgb = draw_compass_rose(rgb, cx=40, cy=H - 40)

    # --- Top-left: site/band/location (two lines) ---
    # Build the two label lines
    if site_name and band:
        airglow_line = f"{site_name} {band}"
        coord_line = site_label
    else:
        # Fallback for callers that only pass site_label (legacy/test)
        airglow_line = site_label
        coord_line = ""
    bl_lines = [airglow_line] if not coord_line else [airglow_line, coord_line]
    rgb = draw_text_corner(rgb, bl_lines, "tl", font_size=label_font_size)

    # --- Top-right: date and time (two lines) ---
    rgb = draw_text_corner(rgb, [date_str, time_str], "tr", font_size=label_font_size)

    if not clean_overlay:
        # --- Bottom-right: pixel-size badge ---
        rgb = draw_text_corner(rgb, [f"{H}x{W}"], "br", font_size=label_font_size)

    if draw_geometry:
        # --- Marker labels ---
        for m in markers:
            rgb = draw_marker(rgb, x0, cy, R, m["az_deg"], m["r_frac"], m["name"])

    if output_size is not None and output_size != H:
        resample = Image.Resampling.BILINEAR
        rgb = _to_array(_to_pil(rgb).resize((output_size, output_size), resample=resample))

    return rgb
