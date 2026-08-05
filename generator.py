"""
GitHub profile banner generator — Phase 1 checkpoint 1 (portrait + chrome + intro).
Dot-matrix portrait via 1-bit Floyd-Steinberg dithering (serpentine), rendered as an
animated SVG terminal window. Logo-morph loop layer is a separate follow-up once the
portrait crop/contrast is approved.
"""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy import ndimage
import random

random.seed(7)
np.random.seed(7)

SRC = "profile_src/WhatsApp Image 2026-08-06 at 2.17.24 AM.jpeg"
GRID_W, GRID_H = 300, 340
PORTRAIT_PX_W, PORTRAIT_PX_H = 448, 508   # rendered box inside the banner (38% of 1180x610-ish)
DOT_R = 1.05

PALETTE = {
    "dark":  {"portrait": "#A78BFA", "chrome": "#22D3EE", "chrome_dim": "#0891B2",
              "accent": "#10B981", "bg": "#0A101F", "panel": "#0F1830", "text": "#94A3B8",
              "text_bright": "#F8FAFC"},
    "light": {"portrait": "#7C3AED", "chrome": "#0891B2", "chrome_dim": "#22D3EE",
              "accent": "#10B981", "bg": "#F8FAFC", "panel": "#FFFFFF", "text": "#475569",
              "text_bright": "#0A101F"},
}

INFO_ROWS = [
    ("Subject", "Anirudh Arora"),
    ("Role", "AI & Computer Vision Engineer"),
    ("Origin", "Gurugram, India"),
    ("Education", "B.Tech CSE, IILM University"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "Git · VS Code · Docker · Postman"),
    ("Core.Lang", "Python · C · JavaScript"),
    ("Core.Frontend", "Next.js · React · Tailwind"),
    ("Core.Backend", "Node.js · Express"),
    ("Core.Infra", "Docker · Vercel"),
    ("Grid.Mail", "anirudharoraarora2@gmail.com"),
    ("Grid.LinkedIn", "anirudh-arora-30646b37a"),
    ("Grid.GitHub", "anirudh657"),
]


def load_and_crop(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = np.asarray(im.convert("L"))
    rowmean = a.mean(axis=1)
    nondark = np.where(rowmean >= 15)[0]
    top, bottom = int(nondark.min()), int(nondark.max())
    im = im.crop((0, top, im.width, bottom + 1))

    # head+shoulders framing: keep full width, trim excess lower torso so the
    # crop reads as head+shoulders rather than half-body
    w, h = im.size
    target_ratio = GRID_W / GRID_H  # 0.882
    keep_h = min(h, int(w / target_ratio))
    # bias crop toward the top (hair/face) rather than centering
    top_bias = int(h * 0.03)
    y0 = max(0, top_bias)
    y1 = min(h, y0 + keep_h)
    if y1 - y0 < keep_h:
        y0 = max(0, y1 - keep_h)
    im = im.crop((0, y0, w, y1))

    # center-crop width to match ratio exactly if needed
    w, h = im.size
    want_w = int(h * target_ratio)
    if want_w < w:
        x0 = (w - want_w) // 2
        im = im.crop((x0, 0, x0 + want_w, h))
    return im


def segment_subject_mask(color_small):
    """Colour-based segmentation. The wall and its cast shadow are NEUTRAL grey
    (low saturation); the subject's skin, hair and maroon sweater all carry
    saturation or are very dark. So subject = saturated OR dark. This separates
    the grey wall-shadow from skin, which a luminance threshold cannot do."""
    hsv = np.asarray(color_small.convert("HSV")).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0

    mask = (sat > 0.18) | (val < 0.30)  # coloured subject OR dark hair/features

    # opening severs any faint speckle bridges to the background
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=2)

    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        biggest = 1 + int(np.argmax(sizes))
        mask = labeled == biggest

    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=1)
    return mask


def floyd_steinberg_serpentine(gray01):
    """gray01: float array in [0,1], 0=black..1=white. Returns bool array, True=ink dot."""
    h, w = gray01.shape
    buf = gray01.copy().astype(np.float32)
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        left_to_right = (y % 2 == 0)
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        for x in xs:
            old = buf[y, x]
            new = 0.0 if old < 0.5 else 1.0
            out[y, x] = (new == 0.0)  # ink where black
            err = old - new
            nxt = x + 1 if left_to_right else x - 1
            prv = x - 1 if left_to_right else x + 1
            if 0 <= nxt < w:
                buf[y, nxt] += err * 7 / 16
            if y + 1 < h:
                if 0 <= prv < w:
                    buf[y + 1, prv] += err * 3 / 16
                buf[y + 1, x] += err * 5 / 16
                if 0 <= nxt < w:
                    buf[y + 1, nxt] += err * 1 / 16
    return out


def build_dots(mode, cropped_im):
    """Returns list of (gx, gy) grid dot coordinates for the given mode."""
    small = cropped_im.resize((GRID_W, GRID_H), Image.LANCZOS)
    gray = small.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    arr = np.asarray(gray).astype(np.float32) / 255.0

    if mode == "light":
        ink = floyd_steinberg_serpentine(arr)  # dots on dark parts, bg stays empty
    else:
        subj_mask = segment_subject_mask(small)  # colour-based, on the RGB crop
        inv = 1.0 - arr  # invert so dithering targets LIT areas as "ink"
        ink_all = floyd_steinberg_serpentine(inv)
        ink = ink_all & subj_mask  # only draw within segmented subject silhouette

    ys, xs = np.where(ink)
    return list(zip(xs.tolist(), ys.tolist()))


DOT_SIZE = 1.7  # square side in output px; crispEdges keeps them sharp


def dots_to_path(dots, scale_x, scale_y, r=DOT_R):
    """Batch dots into one compact <path> of tiny filled squares.
    Square subpath 'Mx yh?v?h-?z' is ~4x smaller than an arc circle, which
    keeps the file near ~1MB even at high dot counts."""
    s = DOT_SIZE
    cmds = []
    for gx, gy in dots:
        x = gx * scale_x
        y = gy * scale_y
        cmds.append(f"M{x:.1f} {y:.1f}h{s}v{s}h-{s}z")
    return "".join(cmds)


def evenness_metric(dots, grid_w, grid_h, groups):
    """Rough check that intro groups are spatially interleaved (not regional)."""
    cell = 6
    gxn, gyn = grid_w // cell + 1, grid_h // cell + 1
    variances = []
    for g in groups:
        occ = np.zeros((gyn, gxn))
        for gx, gy in g:
            occ[gy // cell, gx // cell] = 1
        variances.append(occ.mean())
    return float(np.std(variances)) if variances else 0.0


def make_intro_groups(dots, n_groups=60):
    order = list(range(len(dots)))
    random.shuffle(order)  # interleaved random assignment, not spatial
    groups = [[] for _ in range(n_groups)]
    for i, idx in enumerate(order):
        groups[i % n_groups].append(dots[idx])
    return groups


def xml_escape(s):
    """Escape text for safe embedding in SVG/XML. An unescaped & or < makes the
    whole SVG invalid XML, which browsers render as a broken image."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def render_svg(mode, dots, out_path):
    pal = PALETTE[mode]
    W, H = 1180, 610
    portrait_x, portrait_y = 56, 66
    scale_x = PORTRAIT_PX_W / GRID_W
    scale_y = PORTRAIT_PX_H / GRID_H

    groups = make_intro_groups(dots, n_groups=60)
    ev = evenness_metric(dots, GRID_W, GRID_H, groups)

    intro_paths = []
    dur_each = 2.0
    total_intro = 3.2
    for i, g in enumerate(groups):
        begin = (i / len(groups)) * (total_intro - dur_each) if len(groups) > 1 else 0
        begin = max(0.0, begin)
        d = dots_to_path(g, scale_x, scale_y)
        intro_paths.append(
            f'<path d="{d}" fill="{pal["portrait"]}" opacity="0">'
            f'<animate attributeName="opacity" values="0;1" dur="0.9s" '
            f'begin="{begin:.3f}s" fill="freeze"/></path>'
        )

    rows_svg = []
    ry = 40
    row_h = 23
    for label, value in INFO_ROWS:
        leader_len = max(4, 46 - len(label) - len(value))
        leader = "." * leader_len
        elabel, evalue = xml_escape(label), xml_escape(value)
        rows_svg.append(
            f'<text x="0" y="{ry}" font-family="JetBrains Mono, monospace" font-size="13" '
            f'fill="{pal["chrome_dim"]}">{elabel}</text>'
            f'<text x="150" y="{ry}" font-family="JetBrains Mono, monospace" font-size="13" '
            f'fill="{pal["text"]}" opacity="0.5">{leader}</text>'
            f'<text x="360" y="{ry}" font-family="JetBrains Mono, monospace" font-size="14" '
            f'fill="{pal["text_bright"]}" text-anchor="end" textLength="150" '
            f'lengthAdjust="spacingAndGlyphs">{evalue}</text>'
        )
        ry += row_h

    svg = f'''<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
<rect width="{W}" height="{H}" rx="14" fill="{pal["bg"]}"/>
<rect x="0" y="0" width="{W}" height="38" rx="14" fill="{pal["panel"]}"/>
<circle cx="24" cy="19" r="6" fill="#F87171"/>
<circle cx="46" cy="19" r="6" fill="#FBBF24"/>
<circle cx="68" cy="19" r="6" fill="#34D399"/>
<text x="{W/2}" y="24" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="13" fill="{pal["text"]}">profile.sh --live</text>

<text x="{portrait_x}" y="60" font-family="JetBrains Mono, monospace" font-size="12" letter-spacing="2" fill="{pal["chrome"]}">VISUAL.MAP</text>
<rect x="{portrait_x - 10}" y="{portrait_y - 4}" width="{PORTRAIT_PX_W + 20}" height="{PORTRAIT_PX_H + 20}" rx="6" fill="none" stroke="{pal["chrome_dim"]}" stroke-opacity="0.35"/>
<g transform="translate({portrait_x},{portrait_y})" shape-rendering="crispEdges">
{''.join(intro_paths)}
</g>

<g transform="translate(560,72)">
{''.join(rows_svg)}
</g>

<circle cx="1100" cy="60" r="4" fill="#F87171">
<animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/>
</circle>
<text x="1112" y="64" font-family="JetBrains Mono, monospace" font-size="12" fill="#F87171">LIVE</text>

<rect x="1000" y="80" rx="10" ry="10" width="150" height="26" fill="{pal["accent"]}" opacity="0.15" stroke="{pal["accent"]}"/>
<text x="1075" y="97" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="14" fill="{pal["accent"]}">@anirudh657</text>
</svg>'''

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return ev, len(dots)


if __name__ == "__main__":
    cropped = load_and_crop(SRC)
    cropped.save("profile_src/cropped_preview.png")

    for mode in ("light", "dark"):
        dots = build_dots(mode, cropped)
        ev, n = render_svg(mode, dots, f"{mode}.svg")
        print(f"{mode}: {n} dots, evenness={ev:.4f} (target ~0.05, <0.15 acceptable)")
