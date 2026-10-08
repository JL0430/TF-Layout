# scripts/tflayout/07_inspect_examples.py

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def run_inspect_examples(sites="out/sites.parquet", bwdir="data/ChEC-seq",
                         sgd="data/SGD_features.tab", tfs=None, n_per_bucket=3,
                         flank=200, sigma=10.0, use_free=True, outdir="out/fig"):
    import pyBigWig
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    os.makedirs(outdir, exist_ok=True)
    if tfs is None:
        tfs = ["ABF1", "REB1"]

    # ------------------------------------------------ 染色体别名（跟 02/03 一致）
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV", "XVI"]
    chrom_alias = {}
    for i, r in enumerate(roman, 1):
        for k in (f"CHR{r}", r, str(i), f"{i:02d}", f"CHR{i}", f"CHR{i:02d}",
                  f"CHROMOSOME{r}", f"CHROMOSOME{i}", f"NC_0011{32 + i:02d}"):
            chrom_alias[k] = f"chr{r}"
    for k in ("CHRMITO", "MITO", "CHRM", "M", "MT", "CHRMT", "MITOCHONDRION",
              "NC_001224", "17", "CHR17"):
        chrom_alias[k] = "chrMito"

    # ------------------------------------------------ SGD 名称映射（跟 02/03 一致）
    sgd_df = pd.read_csv(sgd, sep="\t", header=None, dtype=str, quoting=3)
    keep = sgd_df[3].fillna("").str.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$") | \
        (sgd_df[1] == "ORF")
    sg = sgd_df[keep & sgd_df[3].notna()]
    name2sys, sys2disp = {}, {}
    for sysn, ali in zip(sg[3], sg[5]):
        for a_ in str(ali).split("|") if isinstance(ali, str) else []:
            if a_.strip():
                name2sys.setdefault(a_.strip().upper(), sysn.upper())
    for sysn, std in zip(sg[3], sg[4]):
        if isinstance(std, str) and std.strip():
            name2sys[std.strip().upper()] = sysn.upper()
            sys2disp[sysn.upper()] = std.strip().upper()
    for sysn in sg[3]:
        name2sys[sysn.upper()] = sysn.upper()
        sys2disp.setdefault(sysn.upper(), sysn.upper())

    # ------------------------------------------------ bigWig 索引（跟 02 一致）
    rx = re.compile(r"^GSM\d+_(.+?)_([A-Za-z0-9]+)\.(bw|bigwig)$", re.I)
    tf2paths, free_paths = {}, []
    for p in sorted(glob.glob(os.path.join(bwdir, "*.bw")) +
                    glob.glob(os.path.join(bwdir, "*.bigWig"))):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        u = m.group(1).upper()
        if re.sub(r"[^A-Z]", "", u) in ("FREEMNASE", "MNASE", "NOTF"):
            free_paths.append(p)
            continue
        tf2paths.setdefault(sys2disp.get(name2sys.get(u, u), u), []).append(p)

    def _open_group(paths):
        out = []
        for p in paths:
            try:
                bw = pyBigWig.open(p)
            except Exception:
                continue
            kmap = {}
            for k in bw.chroms():
                uu = str(k).upper()
                for u2 in (uu, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", uu)):
                    if u2 in chrom_alias:
                        kmap.setdefault(chrom_alias[u2], k)
                        break
            h = bw.header()
            out.append((bw, kmap, (h["sumData"] or 1.0) / (h["nBasesCovered"] or 1.0)))
        return out

    def _extract(handles, canon, s0, e0):
        w = e0 - s0
        vs = []
        for bw, kmap, nf in handles:
            key = kmap.get(canon)
            if key is None:
                continue
            n_chrom = bw.chroms()[key]
            ss, ee = max(0, s0), min(n_chrom, e0)
            v = np.zeros(w, np.float64)
            if ee > ss:
                v[ss - s0:ee - s0] = np.nan_to_num(
                    np.asarray(bw.values(key, ss, ee), np.float64))
            vs.append(v / nf)
        return np.mean(vs, axis=0) if vs else None

    free_h = _open_group(free_paths) if use_free else []

    # ------------------------------------------------ 挑例子：rank==0 的 motif 位点，
    # 按 offset 分桶(接近0 / 接近+50 / 接近-50)，每桶挑 height 最高的几个（信号强、不含糊）
    st = pd.read_parquet(sites)
    mo = st[(st["res"] == "motif") & (st["rank"] == 0)].copy()
    mo = mo[mo["tf"].isin([t.upper() for t in tfs]) & mo["tf"].isin(tf2paths.keys())]
    buckets = {
        "near_zero |off|<=10": mo[mo["offset"].abs() <= 10],
        "plus_50 off in [35,65]": mo[(mo["offset"] >= 35) & (mo["offset"] <= 65)],
        "minus_50 off in [-65,-35]": mo[(mo["offset"] >= -65) & (mo["offset"] <= -35)],
    }
    examples = []
    for label, d in buckets.items():
        picked = d.sort_values("height", ascending=False).head(n_per_bucket)
        for _, row in picked.iterrows():
            examples.append((label, row))
    print(f"共选了 {len(examples)} 个例子（每桶最多 {n_per_bucket} 个，按 height 从高到低挑）")
    for label, row in examples:
        print(f"  {label:25s} {row['tf']:8s} {row['gene_id']:12s} "
              f"offset={row['offset']:.0f}bp height={row['height']:.2g}")

    n = len(examples)
    if n == 0:
        print("一个例子都没挑到，检查 --tfs 或者数据是不是对得上")
        return
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3 * nrow), squeeze=False)

    tf_handles = {}
    for idx, (label, row) in enumerate(examples):
        ax = axes[idx // ncol][idx % ncol]
        tf = row["tf"]
        if tf not in tf_handles:
            tf_handles[tf] = _open_group(tf2paths[tf])
        canon = row["chrom"]
        s0, e0 = int(row["summit_abs"]) - flank, int(row["summit_abs"]) + flank
        v_raw = _extract(tf_handles[tf], canon, s0, e0)
        if v_raw is None:
            ax.set_title(f"{tf} @ {row['gene_id']}\n(no data)")
            continue
        b = _extract(free_h, canon, s0, e0) if free_h else None
        v_sub = np.maximum(v_raw - b, 0) if b is not None else v_raw
        v_sm = gaussian_filter1d(v_sub, sigma)

        x = np.arange(-flank, flank)
        flip = row["gene_strand"] == "-"
        x_plot = -x[::-1] if flip else x
        v_raw_plot = v_raw[::-1] if flip else v_raw
        v_sm_plot = v_sm[::-1] if flip else v_sm

        ax2 = ax.twinx()
        ax.plot(x_plot, v_raw_plot, color="#bbbbbb", lw=.6)
        ax2.plot(x_plot, v_sm_plot, color="#4C72B0", lw=1.4)
        ax.set_ylabel("raw", color="#999999", fontsize=7)
        ax2.set_ylabel("smoothed (sigma=%g, bg-sub)" % sigma, color="#4C72B0", fontsize=7)
        ax.axvline(0, color="k", lw=.9)
        motif_rel = int(row["site_abs"]) - int(row["summit_abs"])
        if flip:
            motif_rel = -motif_rel
        ax.axvline(motif_rel, color="#C44E52", ls="--", lw=1.3)
        ax.set_title(f"{tf} @ {row['gene_id']}  offset={row['offset']:.0f}bp\n{label}",
                     fontsize=8)
        ax.set_xlabel("bp from summit (black=summit, red dashed=motif)", fontsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for hs in list(tf_handles.values()) + [free_h]:
        for bw, _, _ in hs:
            bw.close()

    plt.tight_layout()
    fp = os.path.join(outdir, "raw_signal_examples.png")
    plt.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"\n图 -> {fp}")
    print("看图重点：灰色是原始信号，蓝色是按 02 同样参数(sigma、扣 free MNase 背景)平滑后"
          "的曲线；黑色竖线是 02 报出的峰顶，红色虚线是 03 匹配上的 motif 中心。")
    print("  - 如果灰色原始信号在红色虚线附近本来就有个清楚的小峰，但蓝色平滑曲线的最高点"
          "却在黑色竖线(别的地方)——说明是平滑/找峰步骤把位置带偏了，该调 02 的 sigma 或"
          "find_peaks 参数，是能修的。")
    print("  - 如果原始信号在两个位置看不出明显差别、或者红色虚线附近根本没有信号特征、"
          "或者原始信号本身就是宽宽的一大坨看不出锐利的中心——那更像是 ChEC-seq 在这个"
          "分辨率上本身就是这样，不是靠调参数能解决的，Fig 5 的螺旋相位分析大概率要放弃，"
          "退回距离衰减+核小体尺度这个中间档。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="out/sites.parquet")
    ap.add_argument("--bwdir", default="data/ChEC-seq")
    ap.add_argument("--sgd", default="data/SGD_features.tab")
    ap.add_argument("--tfs", nargs="*", default=None,
                    help="只看这些 TF 的例子，默认 ABF1 REB1（研究最透彻、该最干净的两个）")
    ap.add_argument("--n-per-bucket", type=int, default=3)
    ap.add_argument("--flank", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=10.0, help="跟 02 保持一致，别改")
    ap.add_argument("--no-free", action="store_true", help="不扣 free MNase 背景")
    ap.add_argument("--outdir", default="out/fig")
    a = ap.parse_args()
    run_inspect_examples(a.sites, a.bwdir, a.sgd, a.tfs, a.n_per_bucket, a.flank, a.sigma,
                         not a.no_free, a.outdir)
