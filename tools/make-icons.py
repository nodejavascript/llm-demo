#!/usr/bin/env python3
"""Generate the icon set and the social card for llm-demo.nodejavascript.com.

The mark is a small network — five nodes, six edges — because the site is about
a model small enough to hold in your head.

The house rule the apple icon is checked against: 180x180, and NO alpha channel,
because iOS paints transparency black (a transparent icon arrives on a home
screen with black wedges). It is composited onto the badge's own flat colour.

    ~/Documents/git/gitlab.com/datavisionstudios/docker-compose-master/.venv/bin/python3 tools/make-icons.py

Run it from the repository root; it writes into site/.
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(os.path.dirname(HERE), "site")

BG = (8, 5, 13)          # the badge colour, same as nodejavascript.com
VIOLET = (167, 139, 250)
AMBER = (251, 191, 36)

# Geometry in a 32x32 space, shared by the SVG and the raster versions.
NODES = [(6.0, 10.0), (6.0, 22.0), (16.0, 16.0), (26.0, 10.0), (26.0, 22.0)]
EDGES = [(0, 2), (1, 2), (2, 3), (2, 4)]
R = 3.0
STROKE = 1.7

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" role="img" aria-label="llm-demo">
  <rect width="32" height="32" rx="7" fill="#08050d"/>
  <g stroke="#a78bfa" stroke-width="1.7" fill="none">
    <path d="M8.6 11.5 13.4 15M8.6 20.5 13.4 17M18.6 15 23.5 11.6M18.6 17 23.5 20.4"/>
  </g>
  <g fill="#a78bfa">
    <circle cx="6" cy="10" r="3"/><circle cx="6" cy="22" r="3"/>
    <circle cx="26" cy="10" r="3"/><circle cx="26" cy="22" r="3"/>
  </g>
  <circle cx="16" cy="16" r="3.4" fill="#fbbf24"/>
</svg>
"""


def draw_mark(img, size, inset=0.0):
    """Draw the mark into a square image of the given pixel size."""
    draw = ImageDraw.Draw(img)
    s = size / 32.0
    pad = inset * size
    usable = size - 2 * pad
    scale = usable / 32.0

    def pt(node):
        x, y = NODES[node]
        return (pad + x * scale, pad + y * scale)

    for a, b in EDGES:
        draw.line([pt(a), pt(b)], fill=VIOLET, width=max(1, round(STROKE * scale)))
    for i, (x, y) in enumerate(NODES):
        cx, cy = pt(i)
        r = R * scale
        colour = AMBER if i == 2 else VIOLET
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
    void = s  # silence linters; scale already derived above
    return void


def square(size, background=BG, inset=0.0):
    img = Image.new("RGB", (size, size), background)
    draw_mark(img, size, inset)
    return img


def find_font(bold=True, size=48):
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf",
    ]
    roots = [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts",
    ]
    for root in roots:
        for name in names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def social_card():
    width, height = 1200, 630
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # soft violet bloom, matching the page background
    bloom = Image.new("RGB", (width, height), BG)
    bd = ImageDraw.Draw(bloom)
    for i in range(60, 0, -1):
        t = i / 60.0
        r = int(120 + 520 * t)
        colour = (int(8 + 60 * (1 - t)), int(5 + 26 * (1 - t)), int(13 + 90 * (1 - t)))
        bd.ellipse([140 - r, 90 - r, 140 + r, 90 + r], fill=colour)
    img = Image.blend(img, bloom, 0.55)
    draw = ImageDraw.Draw(img)

    mark_size = 190
    mark = Image.new("RGB", (mark_size, mark_size), BG)
    draw_mark(mark, mark_size)
    img.paste(mark, (72, 72))

    title_font = find_font(True, 62)
    sub_font = find_font(False, 30)
    tag_font = find_font(True, 26)

    draw.text((72, 320), "Train a language model", font=title_font, fill=(236, 230, 247))
    draw.text((72, 396), "in your browser", font=title_font, fill=(236, 230, 247))
    draw.text(
        (72, 486),
        "A small GPT written from scratch in JavaScript. Nothing uploaded.",
        font=sub_font,
        fill=(168, 156, 196),
    )
    draw.text((72, 546), "llm-demo.nodejavascript.com", font=tag_font, fill=AMBER)
    return img


def main():
    os.makedirs(SITE, exist_ok=True)

    with open(os.path.join(SITE, "favicon.svg"), "w") as fh:
        fh.write(SVG)

    # Raster icons are drawn large and downsampled, so the edges are clean.
    master = square(1024)
    for size, name in [(512, "android-chrome-512x512.png"), (192, "android-chrome-192x192.png"), (32, "favicon-32.png")]:
        master.resize((size, size), Image.LANCZOS).save(os.path.join(SITE, name))

    # Apple: 180x180, RGB (no alpha), opaque background — checked by ~/.seo_audit.py
    square(180, inset=0.16).save(os.path.join(SITE, "apple-touch-icon.png"))

    # .ico with 16/32/48 entries
    ico_master = square(256)
    ico_master.save(
        os.path.join(SITE, "favicon.ico"),
        sizes=[(16, 16), (32, 32), (48, 48)],
    )

    social_card().save(os.path.join(SITE, "og.png"))

    for name in [
        "favicon.svg",
        "favicon-32.png",
        "favicon.ico",
        "apple-touch-icon.png",
        "android-chrome-192x192.png",
        "android-chrome-512x512.png",
        "og.png",
    ]:
        path = os.path.join(SITE, name)
        print(f"{name:32} {os.path.getsize(path):>8,} bytes")


if __name__ == "__main__":
    main()
