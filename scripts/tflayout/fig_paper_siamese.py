"""
Run:  python fig_paper_siamese.py      (needs figkit.py in the same folder)
Out:  fig_paper_siamese.{pdf,svg,png}
"""
import itertools

import numpy as np
from matplotlib.patches import Rectangle
from figkit import (C, TFC, FS, DASH, setup, canvas, save, T, rbox, arrow, line,
                    dot, block3d, vstack, op, chip, head_trap, bars, dark)

setup("times")          # Times New Roman (template: figure labels in 8 pt Times New Roman)
# sizes in pt; 1 data unit = 0.01 in, so the canvas is 1:1 with the 3.40-in IEEE column
MIN_PT = 8.0
FS.update(box=8.5, lab=8.0, small=8.0, tiny=8.0)
TXT2 = "#404040"                 # secondary text: dark gray instead of #7F7F7F
TF_TXT = dark(C["tf"], 0.15)     # slightly darker orange for the L_g labels

W, H = 340, 214
fig, ax = canvas(W, H)
# crop the empty band above/below the drawing: 340 x 200 units = 3.40 x 2.00 in, still 1:1 with the column
fig.set_size_inches(3.40, 2.00)
ax.set_ylim(7, 207)

YW, YD = 170, 46          # wild-type row / depleted row
YB, YC = 125, 88          # head-B row / head-C row

# ---------------------------------------------------------------- inputs
D_TF = 1
chip_cols = [0, 1, 2, 1]
for yc, dep in ((YW, False), (YD, True)):
    rbox(ax, 3, yc - 22, 78, 44, fc="white", ec=C["gray"], lw=0.8)
    for k, ci in enumerate(chip_cols):
        chip(ax, 9 + k * 10.5, yc - 3, 8, 15, TFC[ci], crossed=dep and ci == D_TF)
    # condition marker: one slot per TF, all on in WT, D's slot off when depleted
    bars(ax, 55, yc - 2, 5, 3.2, 1.4, np.full(5, 12.0), C["ctx"],
         zero=1 if dep else None)
    if dep:
        T(ax, 28, yc - 13, r"$L_g\setminus D$", size=FS["lab"], c=TF_TXT, w="bold")
        T(ax, 66, yc - 13, r"$\mathrm{ctx}_D$", size=FS["lab"], c=C["ctx"], w="bold")
    else:
        T(ax, 28, yc - 13, r"$L_g$", size=FS["lab"], c=TF_TXT, w="bold")
        T(ax, 66, yc - 13, r"$\mathrm{ctx}_{\mathrm{WT}}$", size=FS["lab"], c=C["ctx"],
          w="bold")
T(ax, 42, YW + 29, "wild type", size=FS["small"], st="italic", c=TXT2)
T(ax, 42, YD - 30, "TF $D$ depleted", size=FS["small"], st="italic", c=TXT2)

# ---------------------------------------------------------------- shared encoder
EX, EW = 90, 38
for yc in (YW, YD):
    arrow(ax, [(81, yc), (EX, yc)], C["ink"], lw=0.9)
    block3d(ax, EX, yc - 13, EW, 26, C["fus_f"], C["fus"], d=5)
    T(ax, EX + EW / 2, yc, r"$f_\theta$", size=FS["box"] + 1.5, c=C["fus"], w="bold")
XS = EX + 14                                   # dashed "shared weights" arrow
arrow(ax, [(XS, YD + 18), (XS, YW - 17)], C["fus"], lw=0.8, ls=DASH)
arrow(ax, [(XS, YW - 17), (XS, YD + 18)], C["fus"], lw=0.8, ls=DASH)
# right-aligned block in the empty band between the two input boxes
YM, PITCH = (YW + YD) / 2, 11.5
T(ax, XS - 4, YM + 1.5 * PITCH, "shared", size=FS["small"], c=C["fus"], st="italic",
  ha="right")
for k_, s_ in enumerate(("cis +", "layout +", "condition")):
    T(ax, XS - 4, YM + (0.5 - k_) * PITCH, s_, size=FS["small"], c=TXT2, ha="right")

# ---------------------------------------------------------------- gene vectors z
HX = 147
for yc, lab, above in ((YW, r"$\mathbf{z}_{\mathrm{WT}}$", True),
                       (YD, r"$\mathbf{z}_{D}$", False)):
    arrow(ax, [(EX + EW + 5, yc), (HX, yc)], C["fus"], lw=0.9)
    vstack(ax, HX, yc - 13, 7, 26, 5, C["fus"], seed=int(yc))
    T(ax, HX + 3.5, yc + (19 if above else -19), lab, size=FS["lab"], c=C["fus"], w="bold")

# ---------------------------------------------------------------- head A on both passes
GX = 162
for yc, lab, above in ((YW, r"$\hat y_{\mathrm{WT}}$", True), (YD, r"$\hat y_{D}$", False)):
    arrow(ax, [(HX + 7, yc), (GX, yc)], C["fus"], lw=0.9)
    head_trap(ax, GX, yc, 17, 19, label=r"$\psi_A$")
    arrow(ax, [(GX + 17, yc), (GX + 25, yc)], C["ctx"], lw=0.9, hl=3.4, hw=2.8)
    ax.add_patch(Rectangle((GX + 25, yc - 4), 8, 8, fc=C["ctx"], ec=dark(C["ctx"]),
                           lw=0.6, zorder=6))
    T(ax, GX + 29, yc + (11 if above else -11), lab, size=FS["lab"], c=C["ctx"], w="bold")

# ---------------------------------------------------------------- output boxes
OX, OW = 266, 58


def out_box(yc, letter, text, loss):
    rbox(ax, OX, yc - 8, OW, 16, fc="white", ec=C["ctx"], lw=0.9, r=3)
    T(ax, OX + 7, yc, letter, size=FS["box"], c=C["ctx"], w="bold")
    T(ax, OX + 35, yc, text, size=FS["lab"], c=dark(C["ctx"], 0.25), w="bold")
    T(ax, OX + OW / 2, yc - 14, loss, size=FS["tiny"], c=TXT2)


arrow(ax, [(GX + 33, YW), (OX, YW)], C["ctx"], lw=0.9)
out_box(YW, "A", "log TPM", r"MSE + $(1-r)$")

# head B: y_D - y_WT + psi_corr(D, L_g)
SX = 205                                       # moved left to make room for psi_corr
dot(ax, SX, YW, C["ctx"], r=1.4)
op(ax, SX, YB, "-", C["ctx"], r=5.5)
arrow(ax, [(SX, YW), (SX, YB + 5.5)], C["ctx"], lw=0.9)
line(ax, [(GX + 33, YD), (SX, YD), (SX, YC - 3.5)], C["ctx"], lw=0.9)   # bridge at YC
arrow(ax, [(SX, YC + 3.5), (SX, YB - 5.5)], C["ctx"], lw=0.9)
T(ax, SX - 7.5, YB + 8.5, "\u2212", size=FS["lab"], c=C["ctx"], w="bold")
T(ax, SX - 7.5, YB - 8.5, "+", size=FS["lab"], c=C["ctx"], w="bold")
PX = 237
arrow(ax, [(SX + 5.5, YB), (PX - 5.5, YB)], C["ctx"], lw=0.9)
op(ax, PX, YB, "+", C["ctx"], r=5.5)
CW = 54                                        # psi_corr box, widened for 8 pt
# kept low so the 8-pt label clears "MSE + (1 - r)" under box A
rbox(ax, PX - CW / 2, YB + 9.5, CW, 14, fc=C["ctx_f"], ec=C["ctx"], lw=0.8, r=3)
T(ax, PX, YB + 16.5, r"$\psi_{\mathrm{corr}}(D, L_g)$", size=FS["tiny"],
  c=dark(C["ctx"], 0.25), w="bold")
arrow(ax, [(PX, YB + 9.5), (PX, YB + 5.5)], C["ctx"], lw=0.8, hl=3.0, hw=2.6)
arrow(ax, [(PX + 5.5, YB), (OX, YB)], C["ctx"], lw=0.9)
out_box(YB, "B", r"log$_2$FC", "Huber + aux.")

# head C: psi_C(z_D - z_WT)
ZX = HX + 3.5
op(ax, ZX, YC + 6, "-", C["fus"], r=5)
arrow(ax, [(ZX, YW - 13), (ZX, YC + 11)], C["fus"], lw=0.8)
arrow(ax, [(ZX, YD + 13), (ZX, YC + 1)], C["fus"], lw=0.8)
arrow(ax, [(ZX + 5, YC + 6), (ZX + 11, YC + 6), (ZX + 11, YC), (241, YC)], C["fus"],
      lw=0.9)
T(ax, ZX + 32, YC + 7, r"$\Delta\mathbf{z}$", size=FS["lab"], c=C["fus"], w="bold")
head_trap(ax, 241, YC, 17, 19, label=r"$\psi_C$")
arrow(ax, [(258, YC), (OX, YC)], C["ctx"], lw=0.9)
out_box(YC, "C", "\u2193 / ns / \u2191", "cross-entropy")

# sign-consistency coupling between B and C (right edge)
BXR = OX + OW + 3.5
line(ax, [(OX + OW, YB), (BXR, YB), (BXR, YC), (OX + OW, YC)], C["fus"], lw=0.7, ls=DASH)
T(ax, BXR + 5, (YB + YC) / 2, "sign", size=FS["tiny"], c=C["fus"], st="italic",
  rotation=90)


# ---------------------------------------------------------------- label check
def check_labels(fig, ax, min_pt=MIN_PT):
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes, problems = [], []
    for t in ax.texts:
        s, bb = t.get_text(), t.get_window_extent(renderer=r)
        if t.get_fontsize() < min_pt - 1e-6:
            problems.append(f"{s!r}: {t.get_fontsize():.1f} pt < {min_pt} pt")
        if bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1:
            problems.append(f"{s!r}: outside the canvas")
        boxes.append((s, bb))
    for (a, ba), (b, bb) in itertools.combinations(boxes, 2):
        if ba.overlaps(bb):
            problems.append(f"{a!r} overlaps {b!r}")
    print("label check:", "OK" if not problems else "")
    for p in problems:
        print("  -", p)


check_labels(fig, ax)
save(fig, "fig_paper_siamese", png_dpi=600)
