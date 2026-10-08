# scripts/tflayout/05_phase0_stats.py

import argparse
import os

import numpy as np
import pandas as pd


def run_phase0_stats(sites="out/sites.parquet", layout="out/tf_layout.parquet",
                     pairs="out/pair_spacing.parquet", choice="out/motif_choice.tsv",
                     outdir="out/fig"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    st = pd.read_parquet(sites)
    lay = pd.read_parquet(layout)
    pr = pd.read_parquet(pairs)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (1) motif − 峰顶偏移 = 定位分辨率；按 replicate 支持度分层
    mo = st[st.res == "motif"]
    off = mo["offset"].dropna()
    ax[0, 0].hist(off, bins=np.arange(-100, 101, 4), color="#4C72B0", alpha=.6,
                  label="all")
    if "rep_support" in mo:
        hi = mo.loc[mo.rep_support >= 2, "offset"].dropna()
        ax[0, 0].hist(hi, bins=np.arange(-100, 101, 4), histtype="step",
                      color="k", label=f"rep≥2 (SD={hi.std():.1f})")
    ax[0, 0].axvline(0, c="k", lw=.8)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set(title=f"motif − summit  SD={off.std():.1f} bp",
                 xlabel="offset (bp)", ylabel="sites")

    # (2) 每启动子位点数
    npg = lay.groupby("gene_id").size()
    ax[0, 1].hist(npg, bins=np.arange(0, npg.max() + 2) - .5, color="#55A868")
    ax[0, 1].set(title=f"sites/promoter  median={npg.median():.0f}",
                 xlabel="n sites", ylabel="genes")

    # (3) 位点相对锚点(ATG)的分布，按分辨率分层
    for r, lab, c in ((1, "motif-anchored", "#C44E52"), (0, "peak-only", "#8172B2")):
        sub = lay.loc[lay.res_id == r, "site_pos"]
        ax[0, 2].hist(sub, bins=np.arange(-1000, 501, 20), histtype="step",
                      label=f"{lab} (n={len(sub)})", color=c, lw=1.5)
    ax[0, 2].axvline(0, c="k", lw=.8)
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].set(title="position vs anchor (ORF start)", xlabel="bp")

    # (4) Δd 分布
    ax[1, 0].hist(pr["dd"].abs(), bins=np.arange(0, 401, 5), color="#CCB974")
    ax[1, 0].set(title=f"|Δd| all pairs (n={len(pr)})", xlabel="bp")

    # (5) 双 motif 锚定子集 —— 周期性只能在这里谈
    sub = pr.loc[pr.both_motif == 1, "dd"].abs()
    h, edges = np.histogram(sub, bins=np.arange(0, 301, 1))
    ax[1, 1].plot(edges[:-1], h, lw=.8, color="#4C72B0")
    ax[1, 1].set(title=f"|Δd| both-motif (n={len(sub)})", xlabel="bp")

    # (6) 去趋势后的功率谱（窗口 0–300 bp，~167 bp 周期只有不到 2 个周期，仅作参考）
    if h.sum() > 0:
        sm = np.convolve(h.astype(float), np.ones(21) / 21, mode="same")
        res = h - sm
        freq = np.fft.rfftfreq(len(res), d=1.0)
        pw = np.abs(np.fft.rfft(res)) ** 2
        per = 1 / np.maximum(freq, 1e-9)
        sel = (per > 5) & (per < 250)
        ax[1, 2].plot(per[sel], pw[sel], lw=.9, color="#C44E52")
        ax[1, 2].set(xscale="log")
        for p0 in (10.5, 167):
            ax[1, 2].axvline(p0, ls="--", c="gray", lw=.8)
    ax[1, 2].set(title="periodogram of detrended |Δd| (both-motif)",
                 xlabel="period (bp)")

    plt.tight_layout()
    fp = os.path.join(outdir, "phase0_layout_qc.png")
    plt.savefig(fp, dpi=150)
    plt.close(fig)
    print("图 ->", fp)

    # ---- 判据小结
    n_tf = st["tf"].nunique()
    n_tf_m = st.loc[st.res == "motif", "tf"].nunique()
    print("\n===== Phase 0.1 四个数字 =====")
    print(f"1. 有 motif 锚定位点的 TF 比例  {n_tf_m}/{n_tf} = {n_tf_m / max(n_tf, 1):.1%}")
    print(f"   bp 级 token 占比            {lay.res_id.mean():.1%}")
    if os.path.exists(choice):
        ch = pd.read_csv(choice, sep="\t")
        if len(ch):
            src = ch.loc[ch.chosen, "motif"].str.split(":").str[0].value_counts()
            print(f"   选中 motif 来源            {src.to_dict()}")
    print(f"2. 每启动子位点数(中位/90%)    {npg.median():.0f} / {npg.quantile(.9):.0f}")
    if len(pr):
        print(f"3. |Δd| 中位数                 {pr.dd.abs().median():.0f} bp;  "
              f"<50bp 占 {(pr.dd.abs() < 50).mean():.1%}")
    print(f"4. 定位分辨率 SD(全部 motif 位点) {off.std():.1f} bp")
    if "rep_support" in mo:
        print(f"   其中 rep_support≥2 的子集    "
              f"{mo.loc[mo.rep_support >= 2, 'offset'].std():.1f} bp")
    print("\n判读（阈值为经验设定，不是文献标准）：")
    print("  SD ≲ 8 bp  → 螺旋相位可查，Fig 5 完整方案可行")
    print("  8–20 bp    → 只做距离衰减 + 核小体尺度，螺旋周期改为探索性")
    print("  > 20 bp    → 间距分辨率不足，需换锚定策略（如只用高质量 motif TF 子集）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="out/sites.parquet")
    ap.add_argument("--layout", default="out/tf_layout.parquet")
    ap.add_argument("--pairs", default="out/pair_spacing.parquet")
    ap.add_argument("--choice", default="out/motif_choice.tsv")
    ap.add_argument("--outdir", default="out/fig")
    a = ap.parse_args()
    run_phase0_stats(a.sites, a.layout, a.pairs, a.choice, a.outdir)
