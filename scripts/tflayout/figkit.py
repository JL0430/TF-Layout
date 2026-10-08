"""
figkit.py — shared style & drawing primitives for the Work-2 framework figures.

"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import (Circle, FancyBboxPatch, Polygon, Rectangle,
                                Arc)

# --------------------------------------------------------------------------
# Palette (sampled from CITRA Fig. 1)
# --------------------------------------------------------------------------
C = dict(
    cis="#3366AA", cis_f="#DAE3EF", cis_l="#ECF0F7", cis_m="#A4BAD9",
    tf="#C67625", tf_f="#F3E2D1", tf_l="#FBF5EF", tf_m="#DFB081",
    ctx="#4C7864", ctx_f="#D9E8E4", ctx_l="#EEF5F2", ctx_m="#AECEC6",
    fus="#7030A0", fus_f="#EEEAF2", fus_m="#B9A2CF",
    gray="#7F7F7F", gray_f="#F2F2F2", gray_m="#BFBFBF",
    ink="#1A1A1A", teal="#205867",
)
# TF identities used in schematics (warm hues so they read as "trans")
TFC = ["#C67625", "#B5452B", "#7A5230"]

FS = dict(letter=10.5, title=8.5, box=7.0, lab=6.4, small=5.8, tiny=5.2)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def setup(font="sans"):
    """font="sans": Arial if available (matches Fig. 1), else metric-compatible fallbacks.
    font="times": Times New Roman (ICBCB/IEEE template wants Times New Roman for figure labels);
                  falls back to Liberation Serif / Tinos / STIX, which are metric-compatible."""
    have = {f.name for f in fm.fontManager.ttflist}
    if font == "times":
        fam = next((f for f in ["Times New Roman", "Liberation Serif", "Tinos",
                                "Nimbus Roman", "STIXGeneral"] if f in have), "DejaVu Serif")
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": [fam, "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": fam, "mathtext.sf": fam,
            "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
            "mathtext.fallback": "stix",
            "pdf.fonttype": 42, "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.solid_capstyle": "butt",
        })
        return fam
    fam = next(f for f in ["Arial", "Helvetica", "Liberation Sans",
                           "DejaVu Sans"] if f in have or f == "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": fam, "mathtext.sf": fam,
        "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in AI/Visio
        "svg.fonttype": "none",
        "lines.solid_capstyle": "butt",
    })
    return fam


def canvas(W, H):
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, stem, png_dpi=400):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=png_dpi if ext == "png" else None,
                    facecolor="white")
    print("saved:", ", ".join(f"{stem}.{e}" for e in ("pdf", "svg", "png")))


# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------
def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return to_hex((1 - t) * a + t * b)


def light(c, t=0.5):
    return mix(c, "white", t)


def dark(c, t=0.2):
    return mix(c, "black", t)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
def T(ax, x, y, s, size=FS["lab"], c=C["ink"], w="normal", st="normal",
      ha="center", va="center", z=8, **kw):
    return ax.text(x, y, s, fontsize=size, color=c, weight=w, style=st,
                   ha=ha, va=va, zorder=z, **kw)


def rich(ax, x, y, parts, size=FS["box"], va="center", z=8):
    """Left-aligned run of differently styled strings.
    parts = [(text, dict(c=..., w=..., st=..., size=...)), ...]"""
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    for s, kw in parts:
        t = ax.text(x, y, s, fontsize=kw.get("size", size),
                    color=kw.get("c", C["ink"]), weight=kw.get("w", "normal"),
                    style=kw.get("st", "normal"), ha="left", va=va, zorder=z)
        bb = t.get_window_extent(renderer=r)
        x = inv.transform((bb.x1, bb.y0))[0] + kw.get("gap", 0)
    return x


def panel(ax, x, y, letter, title):
    rich(ax, x, y, [(letter, dict(w="bold", size=FS["letter"], gap=5)),
                    (title, dict(w="bold", size=FS["title"]))])


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
DASH = (0, (3.2, 2.0))
DOT = (0, (1.0, 1.6))


def rbox(ax, x, y, w, h, fc="white", ec=C["ink"], lw=0.8, ls="-", r=3.5,
         z=1, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha,
                       joinstyle="round")
    ax.add_patch(p)
    return p


def header(ax, x, y, w, h, text):
    rbox(ax, x, y, w, h, fc=C["gray_f"], ec=C["gray"], lw=0.7, r=3)
    T(ax, x + w / 2, y + h / 2, text, size=FS["box"], w="bold")


def arrow(ax, pts, c, lw=0.9, ls="-", head=True, hl=4.6, hw=3.4, z=4):
    """Poly-line arrow with a solid triangular head (works for dashed bodies)."""
    P = np.asarray(pts, float)
    body = P.copy()
    if head:
        d = P[-1] - P[-2]
        u = d / np.hypot(*d)
        base = P[-1] - u * hl
        body[-1] = base
        n = np.array([-u[1], u[0]])
        ax.add_patch(Polygon([P[-1], base + n * hw / 2, base - n * hw / 2],
                             closed=True, fc=c, ec=c, lw=0.2, zorder=z))
    ax.plot(body[:, 0], body[:, 1], color=c, lw=lw, ls=ls, zorder=z,
            solid_joinstyle="miter", dash_capstyle="butt")


def line(ax, pts, c, lw=0.8, ls="-", z=3):
    P = np.asarray(pts, float)
    ax.plot(P[:, 0], P[:, 1], color=c, lw=lw, ls=ls, zorder=z,
            solid_joinstyle="miter")


def dot(ax, x, y, c, r=1.5, z=5):
    ax.add_patch(Circle((x, y), r, fc=c, ec=c, lw=0, zorder=z))


def block3d(ax, x, y, w, h, fc, ec, d=6, lw=0.8, z=3):
    """Visio-style extruded block (front / top / side faces)."""
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d),
                          (x + w + d, y + h + d), (x + w, y + h)],
                         fc=light(fc, 0.35), ec=ec, lw=lw, zorder=z,
                         joinstyle="round"))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d),
                          (x + w + d, y + h + d), (x + w, y + h)],
                         fc=dark(fc, 0.10), ec=ec, lw=lw, zorder=z,
                         joinstyle="round"))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))


_rng = np.random.default_rng(7)


def vstack(ax, x, y, w, h, n, c, ec=None, seed=0, lw=0.45, z=4, lo=0.25,
           hi=0.95):
    """Embedding vector drawn as a column of shaded cells."""
    g = np.random.default_rng(seed).uniform(lo, hi, n)
    ch = h / n
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * ch), w, ch, fc=mix("white", c, g[i]),
                               ec=ec or dark(c, 0.1), lw=lw, zorder=z))


def op(ax, x, y, kind, c, r=6.5, lw=0.9, z=6, fc="white"):
    """Operator node: '+', 'x', '-', 'odot', 'cap', 'sigma'."""
    ax.add_patch(Circle((x, y), r, fc=fc, ec=c, lw=lw, zorder=z))
    k = r * 0.55
    kw = dict(color=c, lw=lw, zorder=z + 1, solid_capstyle="round")
    if kind == "+":
        ax.plot([x - k, x + k], [y, y], **kw)
        ax.plot([x, x], [y - k, y + k], **kw)
    elif kind == "x":
        q = k * 0.75
        ax.plot([x - q, x + q], [y - q, y + q], **kw)
        ax.plot([x - q, x + q], [y + q, y - q], **kw)
    elif kind == "-":
        ax.plot([x - k, x + k], [y, y], **kw)
    elif kind == "odot":
        ax.add_patch(Circle((x, y), r * 0.2, fc=c, ec=c, lw=0, zorder=z + 1))
    elif kind == "cap":                          # set intersection  ∩
        a = k * 0.62
        ax.add_patch(Arc((x, y + a * 0.15), 2 * a, 2 * a, theta1=0,
                         theta2=180, color=c, lw=lw * 1.1, zorder=z + 1))
        ax.plot([x - a, x - a], [y + a * 0.15, y - a * 1.05], **kw)
        ax.plot([x + a, x + a], [y + a * 0.15, y - a * 1.05], **kw)
    elif kind == "sigma":
        T(ax, x, y + 0.3, r"$\sigma$", size=FS["box"] + 0.6, c=c, z=z + 1)


def promoter(ax, x, y, w, h, tss_frac, c=C["cis"], fill=C["cis_m"], lw=0.8,
             z=3):
    """Fig.1-style promoter bar: open upstream, filled downstream of TSS."""
    xt = x + w * tss_frac
    ax.add_patch(Rectangle((x, y), xt - x, h, fc="white", ec=c, lw=lw,
                           zorder=z))
    ax.add_patch(Rectangle((xt, y), x + w - xt, h, fc=fill, ec=c, lw=lw,
                           zorder=z))
    return xt


def tss(ax, x, y, up, right, c=C["cis"], lw=0.9, z=5):
    arrow(ax, [(x, y), (x, y + up), (x + right, y + up)], c, lw=lw,
          hl=3.6, hw=2.8, z=z)


def site(ax, x, y, w, h, strand, c, hollow=False, lw=0.7, z=6):
    """Oriented binding-site glyph (pentagon arrow; strand 0 = unknown)."""
    t = h * 0.42
    if strand > 0:
        P = [(x - w / 2, y - h / 2), (x + w / 2 - t, y - h / 2), (x + w / 2, y),
             (x + w / 2 - t, y + h / 2), (x - w / 2, y + h / 2)]
    elif strand < 0:
        P = [(x + w / 2, y - h / 2), (x - w / 2 + t, y - h / 2), (x - w / 2, y),
             (x - w / 2 + t, y + h / 2), (x + w / 2, y + h / 2)]
    else:
        P = [(x - w / 2, y - h / 2), (x + w / 2, y - h / 2),
             (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)]
    ax.add_patch(Polygon(P, closed=True, fc="white" if hollow else c, ec=c,
                         lw=lw, ls=DASH if hollow else "-", zorder=z,
                         joinstyle="round"))


def chip(ax, x, y, w, h, c, z=5, crossed=False):
    """Layout token: identity band on top + field cells below."""
    col = C["gray_m"] if crossed else c
    rbox(ax, x, y, w, h, fc=light(col, 0.72), ec=dark(col, 0.05), lw=0.6,
         r=1.6, z=z)
    ax.add_patch(Rectangle((x + 0.8, y + h * 0.62), w - 1.6, h * 0.30,
                           fc=col, ec="none", zorder=z + 0.1))
    for k in range(3):
        yy = y + h * (0.10 + 0.16 * k)
        ax.plot([x + 2, x + w - 2], [yy + 1.3, yy + 1.3], color=light(col, 0.3),
                lw=0.8, zorder=z + 0.1, solid_capstyle="butt")
    if crossed:
        ax.plot([x - 1, x + w + 1], [y - 1, y + h + 1], color=C["ink"], lw=0.8,
                zorder=z + 1)
        ax.plot([x - 1, x + w + 1], [y + h + 1, y - 1], color=C["ink"], lw=0.8,
                zorder=z + 1)


def heat(ax, x, y, w, h, M, c_lo, c_hi, ec="white", lw=0.35, z=4,
         frame=None):
    n, m = M.shape
    cw, chh = w / m, h / n
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x + j * cw, y + (n - 1 - i) * chh), cw, chh,
                                   fc=mix(c_lo, c_hi, float(M[i, j])), ec=ec,
                                   lw=lw, zorder=z))
    if frame:
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=frame, lw=0.6,
                               zorder=z + 0.2))
    return cw, chh


def head_trap(ax, x, y, w, h, c=C["ctx"], fc=C["ctx_f"], z=4, label=None):
    """Prediction head: trapezoid narrowing to the right."""
    ax.add_patch(Polygon([(x, y - h / 2), (x + w, y - h * 0.22),
                          (x + w, y + h * 0.22), (x, y + h / 2)], closed=True,
                         fc=fc, ec=c, lw=0.8, zorder=z, joinstyle="round"))
    if label:
        T(ax, x + w * 0.42, y, label, size=FS["lab"], c=dark(c, 0.2), w="bold",
          z=z + 1)


def bars(ax, x, y, n, bw, gap, heights, c, ec=None, z=4, zero=None):
    for i in range(n):
        hh = 0 if (zero is not None and i == zero) else heights[i]
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, hh,
                               fc=light(c, 0.45), ec=ec or c, lw=0.55,
                               zorder=z))
    ax.plot([x - 2, x + n * (bw + gap)], [y, y], color=c, lw=0.6, zorder=z)
    if zero is not None:
        xc = x + zero * (bw + gap) + bw / 2
        q = 2.6
        ax.plot([xc - q, xc + q], [y + 2 - q, y + 2 + q], color=C["ink"],
                lw=0.9, zorder=z + 1)
        ax.plot([xc - q, xc + q], [y + 2 + q, y + 2 - q], color=C["ink"],
                lw=0.9, zorder=z + 1)
        return xc


def seq_ticks(ax, x0, x1, y, h, step=1.9, seed=3, z=4):
    """Nucleotide-coloured ticks that read as a DNA sequence."""
    cols = ["#5B9A68", "#3366AA", "#D8A13A", "#B5452B"]   # A C G T (muted)
    g = np.random.default_rng(seed)
    xs = np.arange(x0 + 1, x1 - 1, step)
    for xx in xs:
        ax.add_patch(Rectangle((xx, y), step * 0.62, h,
                               fc=cols[g.integers(4)], ec="none", zorder=z))
