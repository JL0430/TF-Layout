# scripts/tflayout/fig_paper_within_tf.py

import os
import re
import sys

CONFIG = dict(
    numbers_md="out/results/_paper/paper_numbers.md",
    out_png="out/results/_paper/fig/fig_within_tf_delta.png",
    out_pdf="out/results/_paper/fig/fig_within_tf_delta.pdf",
    width_in=3.3,          # 跟论文 Fig. 1 同宽(IEEE 单栏约 3.5 in)
    height_in=2.38,
    xlim=(-0.075, 0.235),
    # (显示名, 来源类型, 来源里的基线名/行标签)；顺序 = 从上到下
    ens_rows=[
        ("B4", "ens", "B4_pair+TF先验"),
        ("B6 (+WT)", "ens", "B6_gbdt(同B5特征)"),
        ("B7a", "ens", "B7a_gbdt+flat(无WT)"),
        ("B7a-bag", "ens", "B7a_bag(无WT)"),
        ("B7b (+WT)", "ens", "B7b_gbdt+flat(+WT)"),
        ("B7b-bag (+WT)", "ens", "B7b_bag(+WT)"),
    ],
    single_rows=[
        ("B7a", "single", "单seed均值−B7a(单对单)"),
        ("B7b (+WT)", "single", "单seed均值−B7b"),
    ],
)


def _pick_font():
    """Times New Roman per the ICBCB/IEEE template (figure labels: 8 pt Times New Roman);
    Liberation Serif / Tinos are metric-compatible fallbacks on Linux servers."""
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in ("Times New Roman", "Liberation Serif", "Tinos", "Nimbus Roman",
                             "STIXGeneral") if f in have), "DejaVu Serif")


def run_fig_within_tf():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = CONFIG
    # optional:  python fig_paper_within_tf.py [paper_numbers.md] [out_dir]
    if len(sys.argv) > 1:
        c["numbers_md"] = sys.argv[1]
    if len(sys.argv) > 2:
        c["out_png"] = os.path.join(sys.argv[2], "fig_within_tf_delta.png")
        c["out_pdf"] = os.path.join(sys.argv[2], "fig_within_tf_delta.pdf")
    if not os.path.exists(c["numbers_md"]):
        raise SystemExit(f"找不到 {c['numbers_md']}(先 ./run_all.sh --tables，或把 _paper 文件夹拷过来)")
    with open(c["numbers_md"], encoding="utf-8-sig") as fh:
        lines = fh.read().split("\n")
    num = r"([+\-−]\d+\.\d+)"
    rx_ens = re.compile(r"TF 内 AUROC\(合并\) Δ 模型 \S+ [−-] (.+?) (dn|up): " + num + r" \[" + num + r", " + num + r"\]")
    rx_single = re.compile(r"^- (?:\([a-z]\) )?(.+?) / wTF_(dn|up): Δ " + num + r" \[" + num + r", " + num + r"\]")
    f = lambda s: float(s.replace("−", "-"))

    found = {}
    for ln in lines:
        m = rx_ens.search(ln)
        if m:
            found[("ens", m.group(1), m.group(2))] = (f(m.group(3)), f(m.group(4)), f(m.group(5)), ln.strip()[:90])
            continue
        m = rx_single.match(ln.strip())
        if m:
            found[("single", m.group(1), m.group(2))] = (f(m.group(3)), f(m.group(4)), f(m.group(5)), ln.strip()[:90])

    rows = []     # (y, label, dn, up)
    y = 0.0
    heads = []
    for gi, (title, group) in enumerate((("Five-seed ensemble minus:", c["ens_rows"]),
                                         ("Single-seed mean minus single GBDT:", c["single_rows"]))):
        heads.append((y - 0.95, title))
        for label, kind, key in group:
            vals = {}
            for d in ("dn", "up"):
                k = (kind, key, d)
                if k not in found:
                    raise SystemExit(f"paper_numbers.md 里没解析到 {k}：来源句式变了？先看 CONFIG 里的名字和上面的正则")
                vals[d] = found[k]
            rows.append((y, label, vals["dn"], vals["up"]))
            y += 1.0
        y += 0.85
    n_parsed = sum(1 for r in rows for _ in (r[2], r[3]))
    assert n_parsed == 2 * (len(c["ens_rows"]) + len(c["single_rows"])), "解析个数不对"

    FONT = _pick_font()
    FS_ = 8.0     # template: 8 pt for all figure text
    plt.rcParams.update({"font.size": FS_, "axes.unicode_minus": True, "font.family": "serif",
                         "font.serif": [FONT, "DejaVu Serif"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0})
    fig, ax = plt.subplots(figsize=(c["width_in"], c["height_in"]))
    col_dn, col_up = "#1F4E79", "#C55A11"
    off = 0.17
    for yy, label, dn, up in rows:
        for yo, (v, lo, hi, _), col, mk, fill in ((-off, dn, col_dn, "o", True), (off, up, col_up, "s", False)):
            ax.plot([lo, hi], [yy + yo, yy + yo], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([lo, lo], [yy + yo - 0.08, yy + yo + 0.08], color=col, lw=0.8, zorder=2)
            ax.plot([hi, hi], [yy + yo - 0.08, yy + yo + 0.08], color=col, lw=0.8, zorder=2)
            ax.plot([v], [yy + yo], marker=mk, ms=3.4, mfc=(col if fill else "white"), mec=col, mew=0.9, ls="none", zorder=3)
    ax.axvline(0, color="black", lw=0.7, ls="--", zorder=1)
    ax.set_xlim(*c["xlim"])
    ymin, ymax = heads[0][0] - 0.55, rows[-1][0] + 1.05
    ax.set_ylim(ymax, ymin)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    for hy, title in heads:
        ax.text(c["xlim"][0] + 0.004, hy, title, fontsize=FS_, fontweight="bold", va="center", ha="left")
    # shortened so the label stays inside the 3.3-in figure (the old, longer label lost its closing parenthesis)
    ax.set_xlabel("Pooled within-TF AUROC difference (model \u2212 baseline)", fontsize=FS_, labelpad=2)
    xt = [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    ax.set_xticks(xt)
    ax.set_xticklabels([(f"{t:+.2f}".replace("-", "\u2212")) if t else "0" for t in xt])
    ax.tick_params(axis="both", labelsize=FS_)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", lw=0.4, zorder=0)
    ax.text(c["xlim"][0] + 0.004, ymax - 0.1, r"$\leftarrow$ favors baseline", fontsize=FS_, color="#555555", va="bottom", ha="left", style="italic",
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    ax.text(c["xlim"][1] - 0.002, ymax - 0.1, r"favors model $\rightarrow$", fontsize=FS_, color="#555555", va="bottom", ha="right", style="italic",
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    h1, = ax.plot([], [], marker="o", ms=3.4, mfc=col_dn, mec=col_dn, ls="-", lw=1.0, color=col_dn)
    h2, = ax.plot([], [], marker="s", ms=3.4, mfc="white", mec=col_up, mew=0.9, ls="-", lw=1.0, color=col_up)
    ax.legend([h1, h2], ["down-regulation", "up-regulation"], loc="center right", bbox_to_anchor=(1.0, 0.43),
              frameon=True, framealpha=0.95, edgecolor="#BBBBBB", fontsize=FS_, handlelength=1.8, borderpad=0.35, labelspacing=0.3)
    fig.tight_layout(pad=0.25)
    os.makedirs(os.path.dirname(c["out_png"]), exist_ok=True)
    fig.savefig(c["out_png"], dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(c["out_pdf"], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"解析到 {len(found)} 个候选数(用到 {n_parsed} 个)；图 -> {c['out_png']}、{c['out_pdf']}")
    print("画进图里的数(显示名：down 点估计[CI] / up 点估计[CI]；来源行前 90 字符在 CSV 之外，需要时用 --- 下面的核对)：")
    for yy, label, dn, up in rows:
        print(f"  {label:14s} dn {dn[0]:+.3f} [{dn[1]:+.3f},{dn[2]:+.3f}]   up {up[0]:+.3f} [{up[1]:+.3f},{up[2]:+.3f}]")
    print("---")
    for yy, label, dn, up in rows:
        print(f"  {label:14s} <- {dn[3]}")


if __name__ == "__main__":
    run_fig_within_tf()