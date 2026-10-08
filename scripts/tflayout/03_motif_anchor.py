# scripts/tflayout/03_motif_anchor.py

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def run_motif_anchor(peaks="out/peaks.parquet", fasta="data/S288C.fsa",
                     sgd="data/SGD_features.tab",
                     jaspar="data/motif/JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt",
                     yetfasco_dir="data/motif/ALIGNED_ENOLOGO_FORMAT_PWMS",
                     alias=None, flank=100, pval=1e-4, pseudo=1e-3,
                     trim_ic=0.25, min_len=5, select_n=400, min_enrich=0.05,
                     seed=0, chunk=1000, out="out/sites.parquet",
                     choice_out="out/motif_choice.tsv"):
    from numpy.lib.stride_tricks import sliding_window_view

    rng = np.random.default_rng(seed)

    # ------------------------------------------------ 染色体别名 + FASTA
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
    lut = np.full(256, -1, np.int8)
    for i_, ch in enumerate("ACGT"):
        lut[ord(ch)] = i_
    chrom_code = {}
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
        if canon and canon not in chrom_code:
            chrom_code[canon] = lut[np.frombuffer(sq.encode(), np.uint8)]
    cnt = np.zeros(4)
    for arr in chrom_code.values():
        cnt += np.bincount(arr[arr >= 0], minlength=4)
    at = (cnt[0] + cnt[3]) / 2 / cnt.sum()
    gc = (cnt[1] + cnt[2]) / 2 / cnt.sum()
    bg = np.array([at, gc, gc, at])
    print(f"背景碱基频率(由 FASTA 计算, 链对称) A/T={at:.4f} C/G={gc:.4f}")

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

    # ------------------------------------------------ 读 motif：cands[tf] = [(motif_id, prob矩阵 Lx4)]
    raw_motifs = []                                  # (motif_id, tf_raw_name, matrix)
    if jaspar and os.path.exists(jaspar):
        mid, tfraw, rows, in_mat = None, None, [], False
        for line in list(open(jaspar)) + ["MOTIF __END__\n"]:
            if line.startswith("MOTIF"):
                if mid and rows:
                    raw_motifs.append((f"JASPAR:{mid}", tfraw, np.asarray(rows, float)))
                parts = line.split()
                mid = parts[1]
                tfraw = parts[2] if len(parts) > 2 else parts[1]
                rows, in_mat = [], False
            elif line.lstrip().startswith("letter-probability"):
                in_mat = True
            elif in_mat:
                toks = line.split()
                try:
                    vals = [float(x) for x in toks]
                except ValueError:
                    vals = []
                if len(vals) == 4:
                    rows.append(vals)
                else:
                    in_mat = False
    n_j = len(raw_motifs)
    for p in sorted(glob.glob(os.path.join(yetfasco_dir or "", "*.pwm"))):
        stem = os.path.basename(p)[:-4]
        tfraw = stem.rsplit("_", 1)[0]
        rows = []
        for line in open(p):
            toks = [t for t in re.split(r"[\s,;:|\[\]]+", line.strip()) if t]
            if not toks:
                continue
            lab = None
            if toks[0].upper() in ("A", "C", "G", "T"):
                lab, toks = toks[0].upper(), toks[1:]
            try:
                nums = [float(t) for t in toks]
            except ValueError:
                continue
            if nums:
                rows.append((lab, nums))
        lens = {len(n) for _, n in rows}
        if len(rows) == 4 and len(lens) == 1 and (rows[0][0] or list(lens)[0] != 4):
            if all(lb for lb, _ in rows):
                rows = [x for b_ in "ACGT" for x in rows if x[0] == b_]
            mat = np.asarray([n for _, n in rows], float).T
        elif rows and lens == {4}:
            mat = np.asarray([n for _, n in rows], float)
        elif rows and lens == {5}:
            mat = np.asarray([n[1:] for _, n in rows], float)
        else:
            print(f"  跳过无法解析的 PWM: {stem}")
            continue
        raw_motifs.append((f"YeTFaSCo:{stem}", tfraw, mat))
    print(f"读入 motif: JASPAR {n_j} 个, YeTFaSCo {len(raw_motifs) - n_j} 个")

    cands = {}
    n_conv = 0
    for mid, tfraw, mat in raw_motifs:
        if "::" in tfraw or (("-" in tfraw) and not re.match(
                r"^Y[A-P][LR]\d{3}[WC]-[A-Z]$", tfraw.upper())):
            continue                                   # 异源二聚体/复合体
        rs = mat.sum(axis=1, keepdims=True)
        is_prob = bool((mat >= -1e-6).all() and (mat <= 1 + 1e-6).all()
                       and np.allclose(rs, 1, atol=0.05))
        if is_prob:
            mat = mat / np.where(rs > 0, rs, 1)
        else:
            # YeTFaSCo 的 ALIGNED_ENOLOGO_FORMAT 存的不是概率，是 enoLOGOS 风格的
            # 对数似然值(可正可负，量级可达几百，例如未观测到的碱基记为 -800)。
            # 按 enoLOGOS 论文(Workman et al. 2005, NAR, Eq.1)的定义反推概率：
            #   P(b,i) ∝ Pref(b) * base^value(b,i)
            # 这里 Pref 取论文默认的等概率背景 0.25，base 取 2（与本项目其余脚本
            # 统一用 bits 为单位的习惯一致）。具体底数/背景 YeTFaSCo 官方文档没有
            # 逐字确认过，是按 enoLOGOS 论文条款推断的最合理假设；数值上做过健全性
            # 检查（softmax 结果落在 [0,1] 且不会出现数值溢出），但强烈建议用已知
            # TF（如 ABF1/REB1）的输出共识序列跟 YeTFaSCo 网站上的 logo 图目视核对
            # 一次，确认这个假设成立。
            shifted = mat - mat.max(axis=1, keepdims=True)   # 数值稳定，平移不改变结果
            p = 0.25 * np.power(2.0, shifted)
            rs2 = p.sum(axis=1, keepdims=True)
            mat = np.where(rs2 > 0, p / np.where(rs2 > 0, rs2, 1), 0.25)
            n_conv += 1
        pm = np.clip(mat, 1e-12, 1)
        ic = 2 + (mat * np.log2(pm)).sum(axis=1)
        good = np.where(ic >= trim_ic)[0]
        if good.size == 0:
            continue
        mat = mat[good[0]:good[-1] + 1]
        if mat.shape[0] < min_len:
            continue
        u = tfraw.upper()
        tf = sys2disp.get(name2sys.get(u, u), u)
        cands.setdefault(tf, []).append((mid, mat))
    if n_conv:
        print(f"  按 enoLOGOS 公式重建概率的 YeTFaSCo(ENOLOGO格式)矩阵: {n_conv} 个"
              "——假设未经 YeTFaSCo 官方文档逐字确认，建议抽查已知 TF 的共识序列")

    forced = {}
    if alias and os.path.exists(alias):
        for ln in open(alias):
            if ln.strip() and not ln.startswith("#"):
                t, mm = ln.rstrip("\n").split("\t")[:2]
                u = t.strip().upper()
                forced[sys2disp.get(name2sys.get(u, u), u)] = mm.strip()

    # ------------------------------------------------ 逐 TF：选 motif + 扫描
    pk = pd.read_parquet(peaks)
    Wlen = 2 * flank + 1
    ar_cache = {}
    recs, choice_rows = [], []
    tfs = sorted(pk["tf"].unique())
    for ti, tf in enumerate(tfs, 1):
        sub = pk[pk["tf"] == tf].reset_index(drop=True)
        X = np.full((len(sub), Wlen), -1, np.int8)
        wstart = (sub["summit_abs"].to_numpy(np.int64) - flank)
        for k_, (ch, s0) in enumerate(zip(sub["chrom"], wstart)):
            arr = chrom_code.get(ch)
            if arr is None:
                continue
            ss, ee = max(0, s0), min(arr.size, s0 + Wlen)
            if ee > ss:
                X[k_, ss - s0:ee - s0] = arr[ss:ee]
        r0 = np.where(sub["rank"].to_numpy() == 0)[0]
        samp = rng.choice(r0, size=min(select_n, r0.size), replace=False) \
            if r0.size else np.array([], int)
        Xsh = rng.permuted(X[samp], axis=1) if samp.size else X[:0]

        results = {}
        for mid, mat in cands.get(tf, []):
            L = mat.shape[0]
            if L > Wlen:
                continue
            lo = np.log2(((mat + pseudo) / (1 + 4 * pseudo)) / bg)
            lo_rc = lo[::-1][:, [3, 2, 1, 0]]
            # 背景下精确 p 值阈值（分数量化后卷积，FIMO 同思路）
            q = np.rint(lo * 100).astype(np.int64)
            qmin = q.min(axis=1)
            dist = np.array([1.0])
            for i_ in range(L):
                ker = np.zeros(int(q[i_].max() - qmin[i_]) + 1)
                for b_ in range(4):
                    ker[int(q[i_, b_] - qmin[i_])] += bg[b_]
                dist = np.convolve(dist, ker)
            tail = np.cumsum(dist[::-1])[::-1]
            ix = min(int(np.searchsorted(-tail, -pval, side="left")), tail.size - 1)
            thr = (ix + qmin.sum()) / 100.0
            if L not in ar_cache:
                ar_cache[L] = np.arange(L)
            ar = ar_cache[L]
            outs = {}
            for tag, M in (("all", X), ("shuf", Xsh)):
                best = np.full(M.shape[0], -np.inf)
                bst = np.full(M.shape[0], -1, np.int64)
                bstr = np.zeros(M.shape[0], np.int8)
                for c0 in range(0, M.shape[0], chunk):
                    V = sliding_window_view(M[c0:c0 + chunk], L, axis=1)
                    bad = (V < 0).any(axis=-1)
                    Vc = np.where(V < 0, 0, V)
                    sf = lo[ar, Vc].sum(axis=-1)
                    sr = lo_rc[ar, Vc].sum(axis=-1)
                    sf[bad] = -np.inf
                    sr[bad] = -np.inf
                    jf, jr = sf.argmax(axis=1), sr.argmax(axis=1)
                    rr = np.arange(sf.shape[0])
                    vf, vr = sf[rr, jf], sr[rr, jr]
                    use_f = vf >= vr
                    best[c0:c0 + chunk] = np.where(use_f, vf, vr)
                    bst[c0:c0 + chunk] = np.where(use_f, jf, jr)
                    bstr[c0:c0 + chunk] = np.where(use_f, 1, -1)
                outs[tag] = (best, bst, bstr)
            hr = float(np.mean(outs["all"][0][samp] >= thr)) if samp.size else np.nan
            hs = float(np.mean(outs["shuf"][0] >= thr)) if samp.size else np.nan
            results[mid] = dict(L=L, thr=thr, hit_real=hr, hit_shuf=hs,
                                enrich=hr - hs, scan=outs["all"])
            choice_rows.append(dict(tf=tf, motif=mid, L=L, thr=round(thr, 3),
                                    hit_real=hr, hit_shuf=hs, enrich=hr - hs,
                                    n_sample=int(samp.size), chosen=False))

        chosen = None
        if tf in forced and forced[tf] in results:
            chosen = forced[tf]
        elif results:
            bestmid, bestval = None, -np.inf
            for k_, v_ in results.items():
                ev = v_["enrich"] if np.isfinite(v_["enrich"]) else -1.0
                if ev > bestval:
                    bestmid, bestval = k_, ev
            if bestval >= min_enrich:
                chosen = bestmid
        for row in choice_rows:
            if row["tf"] == tf and row["motif"] == chosen:
                row["chosen"] = True

        sign = np.where(sub["gene_strand"].to_numpy() == "+", 1, -1)
        base = dict(
            gene_id=sub["gene_id"].to_numpy(), tf=tf, pos=sub["pos"].to_numpy(),
            summit_abs=sub["summit_abs"].to_numpy(), chrom=sub["chrom"].to_numpy(),
            gene_strand=sub["gene_strand"].to_numpy(),
            height=sub["height"].to_numpy(), auc300=sub["auc300"].to_numpy(),
            rank=sub["rank"].to_numpy(), rep_support=sub["rep_support"].to_numpy()
            if "rep_support" in sub else np.full(len(sub), -1),
            motif=chosen or "", res="peak", motif_score=np.nan, motif_strand=0,
            offset=np.nan, site_abs=sub["summit_abs"].to_numpy(),
            site_pos=sub["pos"].to_numpy())
        df = pd.DataFrame(base)
        if chosen:
            R = results[chosen]
            best, bst, bstr = R["scan"]
            hit = best >= R["thr"]
            center = wstart + bst + R["L"] // 2
            off = center - sub["summit_abs"].to_numpy()
            df.loc[hit, "res"] = "motif"
            df.loc[hit, "motif_score"] = best[hit]
            df.loc[hit, "motif_strand"] = (bstr * sign)[hit]
            df.loc[hit, "offset"] = (off * sign)[hit]
            df.loc[hit, "site_abs"] = center[hit]
            df.loc[hit, "site_pos"] = (sub["pos"].to_numpy() + off * sign)[hit]
        recs.append(df)
        if ti % 20 == 0 or ti == len(tfs):
            print(f"  [{ti}/{len(tfs)}] {tf}: 候选 {len(results)} 个, 选中 {chosen}")

    st = pd.concat(recs, ignore_index=True)
    for c in ("motif_strand", "site_abs", "site_pos"):
        st[c] = st[c].astype(np.int64)
    st.to_parquet(out)
    ch = pd.DataFrame(choice_rows)
    ch.to_csv(choice_out, sep="\t", index=False)

    n_tf_mo = int(ch.loc[ch.chosen, "tf"].nunique()) if len(ch) else 0
    n_tf_cand = int(ch["tf"].nunique()) if len(ch) else 0
    print(f"\nTF 总数 {len(tfs)}；有候选 motif {n_tf_cand} 个；"
          f"通过富集门槛(≥{min_enrich}) {n_tf_mo} 个 ({n_tf_mo / max(len(tfs), 1):.1%})")
    print("有候选但未通过富集门槛:",
          ",".join(sorted(set(ch.tf) - set(ch.loc[ch.chosen, "tf"]))) if len(ch) else "无")
    print("完全无候选 motif:", ",".join(t for t in tfs if t not in cands))
    src = ch.loc[ch.chosen, "motif"].str.split(":").str[0].value_counts().to_dict() \
        if len(ch) else {}
    print("选中 motif 来源:", src)
    n_m = int((st["res"] == "motif").sum())
    print(f"写出 {len(st)} 个位点 -> {out}；motif 锚定 {n_m} ({n_m / max(len(st), 1):.1%})")
    off = st.loc[st.res == "motif", "offset"]
    if len(off):
        print(f"motif-峰顶偏移: 中位 {off.median():.1f} bp, SD {off.std():.1f} bp, "
              f"IQR {off.quantile(.25):.0f}~{off.quantile(.75):.0f}")
    print(f"motif 选择明细 -> {choice_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", default="out/peaks.parquet")
    ap.add_argument("--fasta", default="data/S288C.fsa")
    ap.add_argument("--sgd", default="data/SGD_features.tab")
    ap.add_argument("--jaspar",
                    default="data/motif/JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt")
    ap.add_argument("--yetfasco-dir", default="data/motif/ALIGNED_ENOLOGO_FORMAT_PWMS")
    ap.add_argument("--alias", default=None,
                    help="可选 TSV: tf<TAB>motif_id（如 YeTFaSCo:YGL073W_476），强制指定")
    ap.add_argument("--flank", type=int, default=100)
    ap.add_argument("--pval", type=float, default=1e-4)
    ap.add_argument("--trim-ic", type=float, default=0.25, help="两端低信息列裁剪阈值(bits)")
    ap.add_argument("--select-n", type=int, default=400)
    ap.add_argument("--min-enrich", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/sites.parquet")
    ap.add_argument("--choice-out", default="out/motif_choice.tsv")
    a = ap.parse_args()
    run_motif_anchor(a.peaks, a.fasta, a.sgd, a.jaspar, a.yetfasco_dir, a.alias,
                     a.flank, a.pval, 1e-3, a.trim_ic, 5, a.select_n, a.min_enrich,
                     a.seed, 1000, a.out, a.choice_out)
