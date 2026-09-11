"""Generate the siftr app icon and its .iconset.

The mark is a sieve: unsorted shapes fall in at the top, pass through a mesh, and
one comes out below picked out in the accent colour — which is what the app does.

Everything is drawn as filled polygons rather than thick strokes. Stroked
diagonals leave mitre artifacts where they meet, and the shape has to stay clean
down to 16px. Rendered at 4x and downsampled for antialiasing, so no vector
renderer is needed.
"""

from __future__ import annotations

import pathlib
import subprocess

from PIL import Image, ImageDraw

S, SS = 1024, 4
W = S * SS

BG_TOP, BG_BOT = (34, 32, 45), (17, 16, 23)
AMBER, MESH = (255, 176, 32), (168, 112, 26)
INK, MUTED = (236, 234, 244), (104, 99, 128)


def rounded_ground() -> Image.Image:
    """macOS-style squircle-ish ground with a vertical gradient."""
    ground = Image.new("RGBA", (W, W))
    draw = ImageDraw.Draw(ground)
    for y in range(W):
        t = y / W
        draw.line(
            [(0, y), (W, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOT)) + (255,),
        )
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, W - 1, W - 1], radius=int(W * 0.2237), fill=255
    )
    out = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    out.paste(ground, (0, 0), mask)
    return out


def funnel_polygons(t: float):
    """Outer and inner funnel outlines, `t` thick, as point lists."""
    ml, mr = W * 0.165, W * 0.835
    nl, nr = W * 0.437, W * 0.563
    top, neck = W * 0.365, W * 0.660
    outer = [(ml, top), (mr, top), (nr, neck), (nl, neck)]
    # Inset by t on every side; the neck inset is widened slightly so the walls
    # keep an even visual weight where they converge.
    inner = [
        (ml + t * 1.25, top + t),
        (mr - t * 1.25, top + t),
        (nr - t * 0.55, neck - t * 0.15),
        (nl + t * 0.55, neck - t * 0.15),
    ]
    return outer, inner, (ml, mr, top)


def build() -> Image.Image:
    img = rounded_ground()
    draw = ImageDraw.Draw(img)
    cx = W / 2
    thickness = W * 0.038

    outer, inner, (ml, mr, top) = funnel_polygons(thickness)

    # The mesh: a cross-hatch drawn across the mouth, then masked to the funnel's
    # interior so it never spills past the walls.
    mesh_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    mesh_draw = ImageDraw.Draw(mesh_layer)
    bar = int(W * 0.011)
    band_top, band_bottom = top + thickness, top + thickness + W * 0.085
    for i in range(1, 9):
        x = ml + (mr - ml) * i / 9
        mesh_draw.line([(x, band_top), (x, band_bottom)], fill=MESH + (255,), width=bar)
    for j in range(1, 3):
        y = band_top + (band_bottom - band_top) * j / 3
        mesh_draw.line([(ml, y), (mr, y)], fill=MESH + (255,), width=bar)

    mesh_mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mesh_mask).polygon(inner, fill=255)
    img.alpha_composite(Image.composite(mesh_layer, Image.new("RGBA", (W, W), (0, 0, 0, 0)), mesh_mask))
    draw = ImageDraw.Draw(img)

    # The funnel wall: outer polygon in amber, inner punched back out to the
    # ground. Filling twice gives mitre-free corners that stroking cannot.
    wall = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wall)
    wd.polygon(outer, fill=AMBER + (255,))
    wd.polygon(inner, fill=(0, 0, 0, 0))
    img.alpha_composite(wall)
    draw = ImageDraw.Draw(img)

    # Unsorted things falling in.
    def diamond(x0, y0, r, fill):
        draw.polygon([(x0, y0 - r), (x0 + r, y0), (x0, y0 + r), (x0 - r, y0)], fill=fill)

    diamond(W * 0.300, W * 0.212, W * 0.046, MUTED)
    diamond(W * 0.700, W * 0.243, W * 0.038, MUTED)
    r = W * 0.033
    draw.ellipse([cx - r, W * 0.160 - r, cx + r, W * 0.160 + r], fill=INK)

    # The one that made it through.
    out_y = W * 0.812
    for spread, alpha in ((W * 0.112, 34), (W * 0.086, 62)):
        glow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse(
            [cx - spread, out_y - spread, cx + spread, out_y + spread], fill=AMBER + (alpha,)
        )
        img.alpha_composite(glow)
    draw = ImageDraw.Draw(img)
    r_out = W * 0.056
    draw.ellipse([cx - r_out, out_y - r_out, cx + r_out, out_y + r_out], fill=AMBER)

    return img.resize((S, S), Image.LANCZOS)


def main() -> None:
    here = pathlib.Path(__file__).parent
    icon = build()
    icon.save(here / "icon.png")

    # macOS wants an .iconset of specific sizes, which iconutil turns into .icns.
    iconset = here / "icon.iconset"
    iconset.mkdir(exist_ok=True)
    for size in (16, 32, 64, 128, 256, 512):
        icon.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(
            iconset / f"icon_{size}x{size}@2x.png"
        )
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(here / "icon.icns")], check=True
    )
    print(f"wrote {here / 'icon.png'} and icon.icns")


if __name__ == "__main__":
    main()
