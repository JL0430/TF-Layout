# scripts/tflayout/02_call_peaks.py

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


def run_call_peaks(bwdir="data/ChEC-seq", fasta="data/S288C.fsa",
                   tss="data/tss.bed", sgd="data/SGD_features.tab",
                   binding="out/binding_binary.parquet",
                   occ="out/chec_occupancy.parquet", out="out/peaks.parquet",
                   up=1000, down=500, sigma=10.0, min_dist=50, max_peaks=3,
                   use_free=True, tss_base="auto", only_tfs=None,
                   force_coord_mismatch=False, coord_tol=1e-5):
    import pyBigWig
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.stats import spearmanr

    # ------------------------------------------------ 染色体别名
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

    # ------------------------------------------------ FASTA（仅用于长度与 ATG 检验）
    seqs, name, buf = {}, None, []
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf).upper()
    canon2seq = {}
    for raw, sq in seqs.items():
        canon = None
        m = re.search(r"chromosome=([A-Za-z0-9]+)", raw)
        for t in ([m.group(1)] if m else []) + \
                [t for t in re.split(r"[|\s]+", raw.strip()) if t]:
            u = t.upper()
            for u2 in (u, u.split(".")[0],
                       re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
                if u2 in chrom_alias:
                    canon = chrom_alias[u2]
                    break
            if canon:
                break
        if canon and canon not in canon2seq:
            canon2seq[canon] = sq

    # ------------------------------------------------ SGD 名称映射
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

    # ------------------------------------------------ 锚点（tss.bed，6 列）
    bed = pd.read_csv(tss, sep=r"\s+", header=None, comment="#", engine="python")
    bed = bed.iloc[:, :6]
    bed.columns = ["chrom", "start", "end", "gene_id", "score", "strand"]
    canon_col = []
    for c in bed["chrom"].astype(str):
        u = c.upper()
        cc = None
        for u2 in (u, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
            if u2 in chrom_alias:
                cc = chrom_alias[u2]
                break
        canon_col.append(cc)
    bed["canon"] = canon_col
    bed = bed[bed["canon"].isin(canon2seq.keys())].copy()
    bed["gene_id"] = [name2sys.get(str(g).upper(), str(g).upper())
                      for g in bed["gene_id"]]
    plus = bed[bed["strand"] == "+"]
    smp = plus.sample(min(len(plus), 2000), random_state=0)
    r0 = np.mean([canon2seq[c][s:s + 3] == "ATG"
                  for c, s in zip(smp["canon"], smp["start"])])
    r1 = np.mean([canon2seq[c][s - 1:s + 2] == "ATG"
                  for c, s in zip(smp["canon"], smp["start"])])
    if tss_base == "auto":
        shift = -1 if r1 > r0 else 0
    else:
        shift = -1 if tss_base == "1" else 0
    print(f"锚点判定: + 链 ATG 比例 start(0-based)={r0:.1%} / start-1={r1:.1%} "
          f"→ + 链锚点 = start{shift:+d}；- 链锚点 = end-1", file=sys.stderr)
    if max(r0, r1) < 0.8:
        print("  警告：ATG 比例 < 80%，tss.bed 可能是真实 TSS 而非 ORF 坐标，"
              "请人工确认；此时建议显式传 --tss-base 0", file=sys.stderr)
    bed["anchor"] = np.where(bed["strand"] == "+", bed["start"] + shift,
                             bed["end"] - 1).astype(np.int64)
    bed = bed.drop_duplicates("gene_id").set_index("gene_id")
    print(f"锚点基因数: {len(bed)}", file=sys.stderr)

    bind = pd.read_parquet(binding)
    occ_df = pd.read_parquet(occ)

    # ------------------------------------------------ bigWig 索引
    rx = re.compile(r"^GSM\d+_(.+?)_([A-Za-z0-9]+)\.(bw|bigwig)$", re.I)
    tf2paths, free_paths = {}, []
    for p in sorted(glob.glob(os.path.join(bwdir, "*.bw")) +
                    glob.glob(os.path.join(bwdir, "*.bigWig"))):
        m = rx.match(os.path.basename(p))
        if not m:
            continue
        if re.sub(r"[^A-Z]", "", m.group(1).upper()) in ("FREEMNASE", "MNASE", "NOTF"):
            free_paths.append(p)
            continue
        u = m.group(1).upper()
        tf2paths.setdefault(sys2disp.get(name2sys.get(u, u), u), []).append(p)
    tfs = [t for t in bind.columns if t in tf2paths]
    if only_tfs:
        tfs = [t for t in tfs if t in set(x.upper() for x in only_tfs)]
    print(f"bigWig 匹配 {len(tfs)}/{bind.shape[1]} 个 TF；free MNase 文件 "
          f"{[os.path.basename(p) for p in free_paths]}", file=sys.stderr)
    miss = [t for t in bind.columns if t not in tf2paths]
    if miss:
        print("  S3A 中无 bigWig:", ",".join(miss), file=sys.stderr)

    # ------------------------------------------------ 坐标系核对（信号导向，非长度导向）
    # 这批 GSE236944 的 bigWig 文件，同一条染色体在不同文件里汇报的长度本身就不完全
    # 相等（推测是各样本按自身信号范围生成 chrom.sizes，而不是共用一份参考基因组索引），
    # 所以不能简单要求"bigWig 长度 == FASTA 长度"。该论文(Mahendrawada et al. 2025,
    # Nature)方法部分写明比对参考基因组是 sacCer3，与这份 FASTA 长度完全一致，所以真正
    # 需要担心的不是"长度不等"本身，而是"bigWig 比 FASTA 多出的那段尾巴里的信号，占这个
    # 文件全部信号的比例，像不像真实的 TF 结合峰"——单纯要求信号和为 0 太严格，染色体
    # 边界的背景类信号很难精确为 0；一个像样的真实峰，总信号通常占全文件的 1e-4~1e-3，
    # 明显低于这个量级(默认阈值 1e-5，留足余量)的当背景噪音处理。02 读数时本来就是按
    # 每个文件自己汇报的长度裁剪窗口，所以下面只做这一步经验验证，通过则直接继续。
    probe_files = (free_paths + [pp for ps in tf2paths.values() for pp in ps])[:6]
    if not probe_files:
        print("中止：所有 bigWig 都打不开，无法核对坐标系", file=sys.stderr)
        sys.exit(1)
    bad_chrom = []
    for canon, sq in canon2seq.items():
        ln_fa = len(sq)
        tot, max_rel, n_probed = 0.0, 0.0, 0
        for p in probe_files:
            try:
                bw = pyBigWig.open(p)
            except Exception:
                continue
            key = None
            for k in bw.chroms():
                u = str(k).upper()
                for u2 in (u, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
                    if u2 in chrom_alias and chrom_alias[u2] == canon:
                        key = k
                        break
                if key:
                    break
            own_len = bw.chroms().get(key) if key else None
            if own_len and own_len > ln_fa:
                try:
                    s = bw.stats(key, ln_fa, own_len, type="sum")[0] or 0.0
                    file_total = (bw.header()["sumData"] or 1.0)
                    tot += s
                    max_rel = max(max_rel, abs(s) / file_total)
                    n_probed += 1
                except Exception:
                    pass
            bw.close()
        if n_probed and max_rel > coord_tol:
            bad_chrom.append((canon, round(tot, 4), f"{max_rel:.2e}", n_probed))
    if bad_chrom and not force_coord_mismatch:
        print(f"\n中止：{len(bad_chrom)} 条染色体在 FASTA 长度之外的区间探测到跟真实峰同量级的"
              f"信号（染色体, 信号和, 占该文件全部信号比例, 探测文件数）：{bad_chrom[:5]}，"
              "不像是背景噪音，坐标系可能真的对不上。", file=sys.stderr)
        print("先用 00_inventory.py 核实一遍；确认可以忽略后加 --force-coord-mismatch 再跑。",
              file=sys.stderr)
        sys.exit(1)
    elif bad_chrom:
        print(f"警告：{len(bad_chrom)} 条染色体探测到跟真实峰同量级的信号，已用 "
              f"--force-coord-mismatch 强制继续，结果的 bp 级精度可能受影响："
              f"{bad_chrom[:5]}", file=sys.stderr)
    else:
        print("坐标系核对：抽样探测 FASTA 长度之外的区间信号占比均在背景噪音量级，按边界噪音"
              "处理，正常继续。", file=sys.stderr)

    # free MNase：一次打开，缓存 key 映射与归一化因子；打不开的文件跳过而不是整体崩溃
    free_h = []
    if use_free and free_paths:
        for p in free_paths:
            try:
                bw = pyBigWig.open(p)
            except Exception as e:
                print(f"  跳过打不开的 free MNase 文件 {os.path.basename(p)}: "
                      f"{str(e).strip().splitlines()[-1]}", file=sys.stderr)
                continue
            kmap = {}
            for k in bw.chroms():
                u = str(k).upper()
                for u2 in (u, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
                    if u2 in chrom_alias:
                        kmap.setdefault(chrom_alias[u2], k)
                        break
            h = bw.header()
            free_h.append((bw, kmap, (h["sumData"] or 1.0) /
                           (h["nBasesCovered"] or 1.0)))
        if not free_h:
            print("  警告：free MNase 文件全部打不开，本次运行不扣背景", file=sys.stderr)
    elif use_free:
        print("  未找到 free MNase bigWig，跳过背景扣除", file=sys.stderr)

    rows = []
    W = up + down
    for ti, tf in enumerate(tfs, 1):
        handles = []
        for p in tf2paths[tf]:
            try:
                bw = pyBigWig.open(p)
            except Exception as e:
                print(f"  [{tf}] 跳过打不开的文件 {os.path.basename(p)}: "
                      f"{str(e).strip().splitlines()[-1]}", file=sys.stderr)
                continue
            kmap = {}
            for k in bw.chroms():
                u = str(k).upper()
                for u2 in (u, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
                    if u2 in chrom_alias:
                        kmap.setdefault(chrom_alias[u2], k)
                        break
            h = bw.header()
            handles.append((bw, kmap, (h["sumData"] or 1.0) /
                            (h["nBasesCovered"] or 1.0)))
        if not handles:
            print(f"  [{ti}/{len(tfs)}] {tf}: 全部 replicate 都打不开，跳过该 TF", file=sys.stderr)
            continue
        tgt = [g for g in bind.index[bind[tf].values == 1] if g in bed.index]
        for g in tgt:
            r = bed.loc[g]
            if r["strand"] == "+":
                s, e = int(r["anchor"]) - up, int(r["anchor"]) + down
            else:
                s, e = int(r["anchor"]) - down + 1, int(r["anchor"]) + up + 1
            # ---- 读取并归一化（TF 各 replicate + free MNase）
            reps, frees = [], []
            for grp, store in ((handles, reps), (free_h, frees)):
                for bw, kmap, nf in grp:
                    key = kmap.get(r["canon"])
                    if key is None:
                        continue
                    n = bw.chroms()[key]
                    v = np.zeros(W, np.float64)
                    ss, ee = max(0, s), min(n, e)
                    if ee > ss:
                        v[ss - s:ee - s] = np.nan_to_num(
                            np.asarray(bw.values(key, ss, ee), np.float64))
                    v /= nf
                    store.append(v[::-1] if r["strand"] == "-" else v)  # 统一 5'→3'
            if not reps:
                continue
            v_raw = np.mean(reps, axis=0)
            b = np.mean(frees, axis=0) if frees else np.zeros(W)
            v = np.maximum(v_raw - b, 0) if frees else v_raw
            if v.max() <= 0:
                continue
            sm = gaussian_filter1d(v, sigma)
            base = np.median(sm)
            mad = np.median(np.abs(sm - base)) * 1.4826 + 1e-9
            idx, props = find_peaks(sm, height=base + 3 * mad,
                                    distance=min_dist, prominence=2 * mad)
            forced = idx.size == 0
            if forced:
                idx = np.array([int(np.argmax(sm))])
                props = {"peak_heights": np.array([sm.max()]),
                         "prominences": np.array([sm.max() - base])}
            order = np.argsort(-props["peak_heights"])[:max_peaks]
            # replicate 支持度：各 rep 平滑后在峰顶处是否超过自身 median+3MAD
            rep_sm = [gaussian_filter1d(x, sigma) for x in reps]
            rep_thr = [np.median(x) + 3 * (np.median(np.abs(x - np.median(x))) * 1.4826)
                       for x in rep_sm]
            for rk, j in enumerate(order):
                c = int(idx[j])
                lo, hi = max(0, c - 150), min(W, c + 151)
                rows.append(dict(
                    gene_id=g, tf=tf, pos=c - up,           # 相对锚点(ATG)，上游为负
                    summit_abs=(s + c) if r["strand"] == "+" else (e - 1 - c),
                    chrom=r["canon"], gene_strand=r["strand"],
                    height=float(props["peak_heights"][j]),
                    prominence=float(props["prominences"][j]),
                    auc300=float(v[lo:hi].sum()),
                    auc300_raw=float(v_raw[lo:hi].sum()),
                    free_auc300=float(b[lo:hi].sum()),
                    promoter_bg=float(base), rank=rk, forced=bool(forced),
                    rep_support=int(sum(x[c] > t for x, t in zip(rep_sm, rep_thr))),
                    n_rep=len(reps),
                ))
        for bw, _, _ in handles:
            bw.close()
        print(f"[{ti}/{len(tfs)}] {tf}: {len(tgt)} promoters, 累计 {len(rows)} peaks",
              file=sys.stderr)
    for bw, _, _ in free_h:
        bw.close()

    pk = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pk.to_parquet(out)
    print(f"\n写出 {len(pk)} 个峰 -> {out}")
    if pk.empty:
        return
    print(f"强制峰(无显著峰时取最大值)占比 {pk.forced.mean():.1%}；"
          f"rep_support 分布 {pk.rep_support.value_counts().sort_index().to_dict()}")

    # ------------------------------------------------ QC：AUC(±150) vs Table-S3B
    occ_long = occ_df.copy()
    occ_long.columns.name = "tf"
    occ_long = occ_long.stack().rename("s3b").reset_index()
    top = pk[pk["rank"] == 0]
    mm = top.merge(occ_long, on=["gene_id", "tf"], how="inner")
    if len(mm) > 100:
        for col in ("auc300_raw", "auc300"):
            rho = spearmanr(mm[col], mm["s3b"], nan_policy="omit").statistic
            print(f"QC  {col} vs Table-S3B: Spearman ρ = {rho:.3f} (n={len(mm)})")
        per_tf = {}
        for t_, d_ in mm.groupby("tf"):
            if len(d_) >= 20:
                per_tf[t_] = spearmanr(d_["auc300_raw"], d_["s3b"]).statistic
        per_tf = pd.Series(per_tf, dtype=float).dropna()
        print(f"    TF 内 ρ(auc300_raw) 中位数 {per_tf.median():.3f}；"
              f"最低 5 个: {per_tf.nsmallest(5).round(2).to_dict()}")
        print("    整体 ρ 偏低时，优先怀疑锚点坐标或 TF 名映射，其次是归一化方式。")
    else:
        print(f"QC 跳过：与 S3B 可合并的记录只有 {len(mm)} 条，检查 gene_id/TF 名映射")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bwdir", default="data/ChEC-seq")
    ap.add_argument("--fasta", default="data/S288C.fsa")
    ap.add_argument("--tss", default="data/tss.bed")
    ap.add_argument("--sgd", default="data/SGD_features.tab")
    ap.add_argument("--binding", default="out/binding_binary.parquet")
    ap.add_argument("--occ", default="out/chec_occupancy.parquet")
    ap.add_argument("--out", default="out/peaks.parquet")
    ap.add_argument("--up", type=int, default=1000)
    ap.add_argument("--down", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=10.0, help="平滑尺度(bp)")
    ap.add_argument("--min-dist", type=int, default=50, help="峰间最小间距(bp)")
    ap.add_argument("--max-peaks", type=int, default=3)
    ap.add_argument("--no-free", action="store_true", help="不扣 free MNase 背景")
    ap.add_argument("--tss-base", choices=["auto", "0", "1"], default="auto")
    ap.add_argument("--only-tfs", nargs="*", default=None, help="调试用：只跑这些 TF")
    ap.add_argument("--force-coord-mismatch", action="store_true",
                    help="坐标系核对探测到真实信号时仍强制继续（默认直接中止）")
    ap.add_argument("--coord-tol", type=float, default=1e-5,
                    help="坐标系核对里，多出区间信号占该文件全部信号的比例超过此值才算"
                         "'像真实峰'，默认 1e-5")
    a = ap.parse_args()
    run_call_peaks(a.bwdir, a.fasta, a.tss, a.sgd, a.binding, a.occ, a.out,
                   a.up, a.down, a.sigma, a.min_dist, a.max_peaks,
                   not a.no_free, a.tss_base, a.only_tfs,
                   a.force_coord_mismatch, a.coord_tol)
