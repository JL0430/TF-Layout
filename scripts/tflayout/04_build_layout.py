# scripts/tflayout/04_build_layout.py

import argparse
import os

import numpy as np
import pandas as pd


def run_build_layout(sites="out/sites.parquet", outdir="out", dedup=30, max_dd=400):
    st = pd.read_parquet(sites)

    # 同一 (gene, TF) 内近距离位点去冗余：motif 锚定优先，其次峰高
    st["_is_peak"] = (st["res"] != "motif").astype(np.int8)
    st = st.sort_values(["gene_id", "tf", "_is_peak", "height"],
                        ascending=[True, True, True, False])
    keep_idx = []
    for _, grp in st.groupby(["gene_id", "tf"], sort=False):
        used = []
        for ix, sp in zip(grp.index, grp["site_pos"].to_numpy()):
            if all(abs(sp - u) > dedup for u in used):
                used.append(sp)
                keep_idx.append(ix)
    lay = st.loc[keep_idx].drop(columns="_is_peak").copy()

    # 占据量：每个 TF 内 log1p + 稳健 z（跨启动子），使不同 TF 的 a 可比
    lay["occ"] = np.log1p(lay["auc300"].clip(lower=0))
    g = lay.groupby("tf")["occ"]
    med = g.transform("median")
    mad = (lay["occ"] - med).abs().groupby(lay["tf"]).transform("median") * 1.4826
    lay["a"] = (lay["occ"] - med) / (mad + 1e-9)
    lay["m"] = lay["motif_score"].fillna(0.0)
    lay["res_id"] = (lay["res"] == "motif").astype(np.int8)      # 1 = bp 级, 0 = 峰级
    # 峰的"有效宽度"代理：auc300(±150bp信号总和)/height(峰高)，近似"如果峰是矩形，
    # 宽度=面积/高度"。06_stratified_qc.py 验证过这个量跟 motif-峰顶定位精度显著相关
    # (ρ≈0.3，越宽offset越大)，越窄的峰子集定位越准；这里只是把这个信息带出来存成一列，
    # 供下游按需筛选(比如只用最窄的一档做 bp 级分析)，不在这里强制过滤。
    lay["width_proxy"] = lay["auc300"] / lay["height"].replace(0, np.nan)
    lay = lay.sort_values(["gene_id", "site_pos"]).reset_index(drop=True)

    cols = ["gene_id", "tf", "site_pos", "motif_strand", "a", "m", "res_id",
            "height", "auc300", "width_proxy", "rep_support", "motif", "summit_abs",
            "site_abs", "chrom", "gene_strand"]
    cols = [c for c in cols if c in lay.columns]
    lay[cols].to_parquet(os.path.join(outdir, "tf_layout.parquet"))

    # ---- 成对间距：按位置排序后 i 在上游、j 在下游，dd = p_j - p_i ≥ 0
    #      (tf_i, tf_j) 与 (tf_j, tf_i) 代表不同的上下游顺序，保留方向信息
    pairs = []
    for gid, grp in lay.groupby("gene_id", sort=False):
        p = grp["site_pos"].to_numpy()
        t = grp["tf"].to_numpy()
        r = grp["res_id"].to_numpy()
        s = grp["motif_strand"].to_numpy()
        n = len(p)
        for i in range(n):
            for j in range(i + 1, n):
                dd = int(p[j] - p[i])
                if dd > max_dd:
                    break
                if t[i] == t[j]:
                    continue
                pairs.append((gid, t[i], t[j], dd, int(r[i] & r[j]),
                              int(s[i]), int(s[j])))
    pr = pd.DataFrame(pairs, columns=["gene_id", "tf_i", "tf_j", "dd", "both_motif",
                                      "strand_i", "strand_j"])
    pr.to_parquet(os.path.join(outdir, "pair_spacing.parquet"))

    npg = lay.groupby("gene_id").size()
    print(f"layout: {len(lay)} tokens / {lay.gene_id.nunique()} genes")
    print(f"每启动子位点数: 中位 {npg.median():.0f}, 均值 {npg.mean():.1f}, "
          f"90% 分位 {npg.quantile(.9):.0f}, 最大 {npg.max()}")
    print(f"bp 级(motif) token 占比 {lay.res_id.mean():.1%}")
    wp = lay.loc[lay.res_id == 1, "width_proxy"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(wp):
        print(f"motif token 的峰宽度代理(auc300/height): 中位 {wp.median():.0f}bp, "
              f"IQR {wp.quantile(.25):.0f}~{wp.quantile(.75):.0f}bp"
              "（越窄定位越准，按需在下游用这一列筛选更干净的子集）")
    if len(pr):
        gp = pr.groupby(["tf_i", "tf_j"]).size()
        print(f"成对间距记录 {len(pr)}，双 motif 锚定 {pr.both_motif.mean():.1%}")
        print(f"有序 TF 对数 {len(gp)}，每对中位样本数 {gp.median():.0f}，"
              f"样本数 ≥ 30 的对 {(gp >= 30).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="out/sites.parquet")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--dedup", type=int, default=30, help="同一 TF 内位点合并距离(bp)")
    ap.add_argument("--max-dd", type=int, default=400, help="成对间距上限(bp)")
    a = ap.parse_args()
    run_build_layout(a.sites, a.outdir, a.dedup, a.max_dd)
