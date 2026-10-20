"""Render one recorded episode to an animated GIF (for slides and the paper)."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

VERGE, ASPHALT, CELL = (227, 230, 222), (75, 80, 83), (86, 91, 94)
CRASH, BRAKE, INK, ACCENT = (207, 58, 44), (232, 70, 47), (27, 32, 31), (29, 118, 86)
CAT = [(42, 120, 214), (235, 104, 52), (18, 160, 108), (213, 81, 129),
       (217, 149, 0), (111, 85, 201), (140, 98, 57), (43, 155, 179)]


def _font(size):
    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(meta, frame, cs=30, color_by="entry"):
    H, W, top = meta["H"], meta["W"], 34
    img = Image.new("RGB", (W * cs, H * cs + top), VERGE)
    d = ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            if meta["road"][r][c]:
                y = r * cs + top
                d.rectangle([c * cs, y, c * cs + cs - 1, y + cs - 1], fill=ASPHALT)
                d.rectangle([c * cs + 1, y + 1, c * cs + cs - 2, y + cs - 2], fill=CELL)
    for r, c in frame["hit"]:
        cx, cy, rad = c * cs + cs / 2, r * cs + top + cs / 2, cs * 0.62
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=CRASH, width=3)
    small, label = _font(max(9, cs // 3)), _font(15)
    for car in frame["cars"]:
        key = {"route": car["p"], "net": car["k"] or 0}.get(color_by, meta["path_arrival"][car["p"]])
        col = CAT[key % 8]
        x, y, w = car["c"] * cs + cs * 0.12, car["r"] * cs + top + cs * 0.12, cs * 0.76
        d.rounded_rectangle([x, y, x + w, y + w], radius=cs * 0.16, fill=col,
                            outline=CRASH if car["x"] else None, width=2 if car["x"] else 0)
        P = meta["paths"][car["p"]]
        q = car["q"]
        dr, dc = ((P[q + 1][0] - P[q][0], P[q + 1][1] - P[q][1]) if q + 1 < len(P)
                  else (P[q][0] - P[q - 1][0], P[q][1] - P[q - 1][1]))
        if car["m"] == 1:
            if dr:
                by = y + 1 if dr > 0 else y + w - w * 0.16 - 1
                d.rectangle([x + w * 0.15, by, x + w * 0.85, by + w * 0.16], fill=BRAKE)
            else:
                bx = x + 1 if dc > 0 else x + w - w * 0.16 - 1
                d.rectangle([bx, y + w * 0.15, bx + w * 0.16, y + w * 0.85], fill=BRAKE)
        d.text((x + w / 2, y + w / 2), str(car["s"]), fill=(255, 255, 255), font=small, anchor="mm")
        if car["g"] is not None and meta["bits"] > 0:
            rad = max(6, cs * 0.22)
            d.ellipse([x + w - rad, y - rad, x + w + rad, y + rad],
                      fill=ACCENT if car["g"] else (255, 255, 255), outline=ACCENT, width=2)
            d.text((x + w, y), str(car["g"]), fill=(255, 255, 255) if car["g"] else INK, font=small, anchor="mm")
    hit_total = sum(1 for _ in frame["hit"])
    d.text((8, 9), f"t = {frame['t']:>2}   cars {len(frame['cars']):>2}   exited {frame['total']['arrived']:>2}"
                   + ("   COLLISION" if hit_total else ""), fill=CRASH if hit_total else INK, font=label)
    return img


def write_gif(meta, episode, out, fps=3, cs=30, color_by="entry"):
    frames = [render_frame(meta, f, cs, color_by) for f in episode["frames"]]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return out
