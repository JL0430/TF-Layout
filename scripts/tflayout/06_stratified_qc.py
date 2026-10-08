# scripts/tflayout/06_stratified_qc.py

import argparse
import os

import numpy as np
import pandas as pd


def run_stratified_qc(sites="out/sites.parquet", choice="out/motif_choice.tsv",
                      pairs="out/pair_spacing.parquet", tiers=None,
                      top_tier_min=0.3, max_dd=400, outdir="out/fig"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if tiers is None:
        tiers = [0.0, 0.1, 0.2, top_tier_min, 1.0]
    os.makedirs(outdir, exist_ok=True)

    st = pd.read_parquet(sites)
    ch = pd.read_csv(choice, sep="\t")
    chosen = ch[ch["chosen"]].set_index("tf")["enrich"]

    mo = st[st["res"] == "motif"].copy()
    mo["enrich"] = mo["tf"].map(chosen)
    n_unmatched = int(mo["enrich"].isna().sum())
    mo = mo.dropna(subset=["enrich"])
    if n_unmatched:
        print(f"警告：{n_unmatched} 个 motif 位点在 motif_choice.tsv 里找不到对应 TF/enrich"
              "（sites.parquet 和 motif_choice.tsv 是不是同一次 03 跑出来的？），已跳过")

    print("=== 0. 按 motif 富集度(enrich)分层看定位分辨率(motif-峰顶偏移 SD) ===")
    mo["tier"] = pd.cut(mo["enrich"], tiers, include_lowest=True)
    summary = mo.groupby("tier", observed=True)["offset"].agg(
        n="count", sd="std", median="median")
    print(summary.to_string())
    print("（第一轮结果这里几乎看不出分层效果——说明主要矛盾大概率不在 motif 质量上，"
          "往下看按 rank 分层的结果）")

    print("\n=== 1. 按 rank（同一 TF 在同一启动子里第几强的峰，0=最强）分层 ===")
    if "rank" not in mo.columns:
        print("sites.parquet 里没有 rank 列，无法做这一步；用的 02/03 版本是不是太旧了？")
        return
    by_rank = mo.groupby("rank")["offset"].agg(n="count", sd="std", median="median")
    print(by_rank.to_string())
    print("如果 rank==0 的 SD 明显小于 rank>=1，说明 02 报出的'次要峰'(rank>=1)大概率"
          "不是真结合位点，是它们把噪音混进了整体的定位精度统计。")

    r0 = mo[mo["rank"] == 0]
    print(f"\n仅 rank==0: n={len(r0)}, SD={r0['offset'].std():.1f} bp，"
          f"median={r0['offset'].median():.1f} bp")
    best = r0[r0["enrich"] >= top_tier_min]
    if len(best):
        print(f"rank==0 且 enrich>={top_tier_min}（两个筛选叠加的最佳情形）："
              f"n={len(best)}, SD={best['offset'].std():.1f} bp，"
              f"median={best['offset'].median():.1f} bp")

    print("\n=== 2. rep_support 与定位精度的关系 ===")
    if "rep_support" in mo.columns:
        rs = mo.groupby("rep_support")["offset"].agg(n="count", sd="std")
        print(rs.to_string())
    else:
        print("sites.parquet 里没有 rep_support 列，跳过")

    print("\n=== 3. motif_score（匹配强度）与 offset 的关系 ===")
    print("背景：04_build_layout.py 实际拿去建 layout/算间距的坐标是 site_pos，对 motif")
    print("锚定的位点来说 site_pos 就是 motif 本身的位置，不是这里在测的峰顶(summit_abs)。")
    print("所以 offset 大不代表最终坐标错了 50bp，更可能说明：peak 定得不够准，导致")
    print("±100bp 的扫描窗口里找到的'最佳匹配'，未必是真正起作用的那个位点，可能是窗口里")
    print("刚好凑巧分数够高的巧合序列。如果这个猜测对，匹配分数低的应该 offset 更大、更")
    print("不可信；分数高的应该 offset 小、更可信。")
    if "motif_score" in mo.columns:
        from scipy.stats import spearmanr
        valid = mo["motif_score"].notna() & mo["offset"].notna()
        if valid.sum() >= 20:
            rho = spearmanr(mo.loc[valid, "motif_score"],
                            mo.loc[valid, "offset"].abs()).statistic
            print(f"Spearman ρ(motif_score, |offset|) = {rho:.3f}  (n={int(valid.sum())})")
            print("（负相关且幅度明显，说明分数越低offset越大，支持上面的猜测；"
                  "如果 ρ 接近 0，说明匹配强度跟 offset 没什么关系，得回去想别的原因）")
            try:
                q = pd.qcut(mo["motif_score"], 4, duplicates="drop")
                qs = mo.groupby(q, observed=True)["offset"].agg(
                    n="count", sd="std", median="median")
                print(qs.to_string())
            except ValueError:
                print("motif_score 取值种类太少，分不出 4 档")
        else:
            print(f"有效样本太少(n={int(valid.sum())})，跳过")
    else:
        print("sites.parquet 里没有 motif_score 列，跳过（是不是 03 的版本太旧了？）")

    print("\n=== 4. 峰的'有效宽度'(auc300/height，近似)与 offset 的关系 ===")
    print("背景：前面三个角度(TF整体富集度、peak排名、单点匹配分数)测的都是'这个位点"
          "可不可信'，都跟 offset 没关系。上一轮原始信号图看到的不是噪音，是稳定重现的"
          "不对称坡状/台阶状特征——如果峰本身是'宽'的，summit(平滑后最高点)大概率卡在"
          "坡最陡的那一侧，真正的 motif 可能在坡的任意位置，这跟'可信度'无关，跟'峰有多宽'"
          "有关。auc300(±150bp信号总和)/height(峰高)可以近似当成峰的有效宽度(如果峰是"
          "矩形，宽度=面积/高度)。这是这一轮最后一个要测的方向——如果这个也不相关，"
          "就没有再猜下去的必要了，直接把 ~50bp 当成这批数据的现实约束，回到中间档结论。")
    if {"auc300", "height"}.issubset(mo.columns):
        from scipy.stats import spearmanr
        width = mo["auc300"] / mo["height"].replace(0, np.nan)
        valid = width.notna() & mo["offset"].notna() & np.isfinite(width)
        if valid.sum() >= 20:
            rho = spearmanr(width[valid], mo.loc[valid, "offset"].abs()).statistic
            print(f"Spearman ρ(auc300/height, |offset|) = {rho:.3f}  (n={int(valid.sum())})")
            try:
                q = pd.qcut(width[valid], 4, duplicates="drop")
                qs = mo.loc[valid].groupby(q, observed=True)["offset"].agg(
                    n="count", sd="std", median="median")
                print(qs.to_string())
            except ValueError:
                print("宽度取值种类太少，分不出 4 档")
        else:
            print(f"有效样本太少(n={int(valid.sum())})，跳过")
    else:
        print("sites.parquet 里没有 auc300/height 列，跳过")

    # ---- 用 rank==0 的 motif 位点重建一遍间距对，直接看周期性是否恢复
    print(f"\n=== 5. 只用 rank==0 的 motif 位点重建间距对（不依赖 04 已经算好的"
          f" pair_spacing.parquet，因为那里面没留 rank 信息）===")
    pairs2 = []
    for gid, grp in r0.groupby("gene_id", sort=False):
        p = grp["site_pos"].to_numpy()
        t = grp["tf"].to_numpy()
        order = np.argsort(p)
        p, t = p[order], t[order]
        n = len(p)
        for i in range(n):
            for j in range(i + 1, n):
                dd = int(p[j] - p[i])
                if dd > max_dd:
                    break
                if t[i] == t[j]:
                    continue
                pairs2.append(dd)
    pairs2 = np.abs(np.array(pairs2))
    print(f"rank==0 限定后的双 motif 锚定间距记录: {len(pairs2)} 条")
    if len(pairs2) >= 50:
        print(f"|Δd| 中位数 {np.median(pairs2):.0f} bp；<50bp 占 {(pairs2 < 50).mean():.1%}")
    else:
        print("样本太少，周期性图不画了——rank==0 限定后覆盖的 TF-TF 组合可能不够多。")

    # ---- 画图：全英文，避免中文字体缺字形变方块
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    off_all = mo["offset"].dropna()
    ax[0, 0].hist(off_all, bins=np.arange(-150, 151, 4), alpha=.5, label="all ranks",
                 color="#4C72B0")
    ax[0, 0].hist(r0["offset"].dropna(), bins=np.arange(-150, 151, 4), histtype="step",
                 color="k", lw=1.3, label=f"rank=0 only (SD={r0['offset'].std():.1f})")
    ax[0, 0].axvline(0, c="k", lw=.7)
    ax[0, 0].axvline(50, c="gray", ls="--", lw=.7)
    ax[0, 0].axvline(-50, c="gray", ls="--", lw=.7)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set(title="motif-summit offset, all vs rank=0", xlabel="offset (bp)")

    by_rank["sd"].plot(kind="bar", ax=ax[0, 1], color="#55A868")
    ax[0, 1].set(title="offset SD by rank", xlabel="rank", ylabel="SD (bp)")

    if "motif_score" in mo.columns:
        ax[0, 2].hexbin(mo["motif_score"], mo["offset"].abs(), gridsize=40,
                        cmap="Blues", mincnt=1)
        ax[0, 2].set(title="|offset| vs motif_score", xlabel="motif_score",
                    ylabel="|offset| (bp)")
    else:
        ax[0, 2].axis("off")

    if len(pairs2) >= 50:
        h, edges = np.histogram(pairs2, bins=np.arange(0, min(301, int(pairs2.max()) + 2), 1))
        ax[1, 0].plot(edges[:-1], h, lw=.8, color="#4C72B0")
        ax[1, 0].set(title=f"|dd| rank=0 subset (n={len(pairs2)})", xlabel="bp")
        if h.sum() > 0 and len(h) > 21:
            sm = np.convolve(h.astype(float), np.ones(21) / 21, mode="same")
            res = h - sm
            freq = np.fft.rfftfreq(len(res), d=1.0)
            pw = np.abs(np.fft.rfft(res)) ** 2
            per = 1 / np.maximum(freq, 1e-9)
            sel = (per > 5) & (per < 250)
            ax[1, 1].plot(per[sel], pw[sel], lw=.9, color="#C44E52")
            ax[1, 1].set_xscale("log")
            for p0 in (10.5, 167):
                ax[1, 1].axvline(p0, ls="--", c="gray", lw=.8)
            ax[1, 1].set(title="periodogram, rank=0 subset", xlabel="period (bp)")
    else:
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
    if {"auc300", "height"}.issubset(mo.columns):
        width_plot = (mo["auc300"] / mo["height"].replace(0, np.nan)).clip(upper=600)
        ax[1, 2].hexbin(width_plot, mo["offset"].abs(), gridsize=40, cmap="Greens", mincnt=1)
        ax[1, 2].set(title="|offset| vs peak width proxy", xlabel="auc300/height (bp)",
                    ylabel="|offset| (bp)")
    else:
        ax[1, 2].axis("off")

    plt.tight_layout()
    fp = os.path.join(outdir, "rank_stratified_qc.png")
    plt.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"\n图 -> {fp}")
    print("看图重点：右上那张 bar 图，rank 越大 SD 是不是越大；左上黑色描边(rank=0)"
          "是不是比蓝色底图(全部)窄很多、更集中在 0 附近；第一行最右边那张，颜色深的区域"
          "是不是主要集中在'motif_score 高、|offset| 小'的左下角。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="out/sites.parquet")
    ap.add_argument("--choice", default="out/motif_choice.tsv")
    ap.add_argument("--pairs", default="out/pair_spacing.parquet")
    ap.add_argument("--top-tier-min", type=float, default=0.3,
                    help="划入'高富集组'的 enrich 下限，默认 0.3")
    ap.add_argument("--max-dd", type=int, default=400, help="重建间距对时的最大间距(bp)")
    ap.add_argument("--outdir", default="out/fig")
    a = ap.parse_args()
    run_stratified_qc(a.sites, a.choice, a.pairs, None, a.top_tier_min, a.max_dd, a.outdir)
