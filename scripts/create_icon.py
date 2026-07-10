"""生成应用图标 icon.ico —— 麦克风 + 音符 主题。"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter


BG_TOP = (94, 129, 244)
BG_BOTTOM = (66, 84, 214)
MIC_BODY = (255, 255, 255)
MIC_BODY_DARK = (222, 232, 255)
MIC_GRILL = (176, 195, 244)
NOTE_COLOR = (255, 209, 102)


def _radial_bg(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    cx = cy = size / 2
    max_dist = size * 0.72
    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            t = min(1.0, math.hypot(dx, dy) / max_dist)
            r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t)
            g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t)
            b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t)
            px[x, y] = (r, g, b, 255)
    return img


def _rounded_square_mask(size: int, radius_ratio: float = 0.22) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    r = int(size * radius_ratio)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=r, fill=255)
    return mask


def _draw_mic(canvas: Image.Image) -> None:
    size = canvas.size[0]
    d = ImageDraw.Draw(canvas, "RGBA")

    cx = size / 2 - size * 0.06
    top = size * 0.18
    body_w = size * 0.32
    body_h = size * 0.44
    x0 = cx - body_w / 2
    y0 = top
    x1 = cx + body_w / 2
    y1 = top + body_h

    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_layer)
    sd.rounded_rectangle(
        (x0 + size * 0.02, y0 + size * 0.03, x1 + size * 0.02, y1 + size * 0.03),
        radius=body_w / 2,
        fill=(0, 0, 0, 90),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(size * 0.02))
    canvas.alpha_composite(shadow_layer)

    d.rounded_rectangle((x0, y0, x1, y1), radius=body_w / 2, fill=MIC_BODY)

    grill_pad_x = body_w * 0.16
    grill_pad_y = body_h * 0.09
    gx0, gy0 = x0 + grill_pad_x, y0 + grill_pad_y
    gx1, gy1 = x1 - grill_pad_x, y0 + body_w * 0.7
    d.rounded_rectangle((gx0, gy0, gx1, gy1), radius=(gx1 - gx0) / 2, fill=MIC_GRILL)

    line_spacing = (gy1 - gy0) / 6
    for i in range(1, 6):
        yy = gy0 + line_spacing * i
        d.line((gx0 + 4, yy, gx1 - 4, yy), fill=(255, 255, 255, 130), width=max(1, size // 96))

    arc_pad = size * 0.02
    ax0 = x0 - body_w * 0.28 - arc_pad
    ax1 = x1 + body_w * 0.28 + arc_pad
    ay0 = y0 + body_h * 0.32
    ay1 = y1 + body_h * 0.22
    d.arc((ax0, ay0, ax1, ay1), start=20, end=160, fill=MIC_BODY, width=max(3, size // 24))

    stand_w = size * 0.05
    stand_top = ay1 - (ay1 - ay0) * 0.15
    stand_bot = size * 0.86
    d.rounded_rectangle(
        (cx - stand_w / 2, stand_top, cx + stand_w / 2, stand_bot),
        radius=stand_w / 2,
        fill=MIC_BODY,
    )
    base_w = size * 0.30
    base_h = size * 0.06
    d.rounded_rectangle(
        (cx - base_w / 2, stand_bot - base_h * 0.3, cx + base_w / 2, stand_bot + base_h * 0.7),
        radius=base_h / 2,
        fill=MIC_BODY_DARK,
    )


def _draw_note(canvas: Image.Image) -> None:
    size = canvas.size[0]
    d = ImageDraw.Draw(canvas, "RGBA")

    head_r = size * 0.09
    head_cx = size * 0.72
    head_cy = size * 0.66
    d.ellipse(
        (head_cx - head_r, head_cy - head_r * 0.75, head_cx + head_r, head_cy + head_r * 0.75),
        fill=NOTE_COLOR,
    )
    stem_w = max(2, int(size * 0.022))
    stem_top = head_cy - size * 0.26
    d.rounded_rectangle(
        (head_cx + head_r - stem_w, stem_top, head_cx + head_r, head_cy - head_r * 0.2),
        radius=stem_w / 2,
        fill=NOTE_COLOR,
    )
    flag_pts = [
        (head_cx + head_r, stem_top),
        (head_cx + head_r + size * 0.10, stem_top + size * 0.03),
        (head_cx + head_r + size * 0.08, stem_top + size * 0.11),
        (head_cx + head_r, stem_top + size * 0.05),
    ]
    d.polygon(flag_pts, fill=NOTE_COLOR)


def _render(size: int) -> Image.Image:
    bg = _radial_bg(size)
    mask = _rounded_square_mask(size)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(bg, (0, 0), mask=mask)

    _draw_mic(icon)
    _draw_note(icon)

    return icon


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ico_path = os.path.join(root, "icon.ico")
    png_path = os.path.join(root, "icon.png")

    base_size = 512
    master = _render(base_size)
    master.save(png_path, format="PNG")

    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    frames = [master.resize(s, Image.LANCZOS) for s in sizes]
    frames[0].save(ico_path, format="ICO", sizes=sizes, append_images=frames[1:])
    print(f"Created {ico_path} and {png_path}")


if __name__ == "__main__":
    main()
