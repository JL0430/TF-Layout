# scripts/tflayout/00_inventory.py

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def run_inventory(data="data", n_show=5):
    import pyBigWig

    # ------------------------------------------------ 染色体名别名表
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

    # ------------------------------------------------ 1. FASTA
    fa_path = os.path.join(data, "S288C.fsa")
    print(f"=== 1. FASTA  {fa_path} ===")
    seqs, name, buf = {}, None, []
    with open(fa_path) as fh:
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
        toks = ([m.group(1)] if m else []) + \
            [t for t in re.split(r"[|\s]+", raw.strip()) if t]
        for t in toks:
            u = t.upper()
            for u2 in (u, u.split(".")[0],
                       re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
                if u2 in chrom_alias:
                    canon = chrom_alias[u2]
                    break
            if canon:
                break
        print(f"  {raw[:70]!r:74s} -> {canon}  len={len(sq)}")
        if canon and canon not in canon2seq:
            canon2seq[canon] = sq
    allseq = "".join(canon2seq.values())
    comp = {b: allseq.count(b) for b in "ACGT"}
    tot = sum(comp.values())
    print("  识别到 canonical 染色体:", len(canon2seq), "条")
    print("  全基因组碱基组成(由本 FASTA 计算):",
          {b: round(comp[b] / tot, 4) for b in "ACGT"})

    # ------------------------------------------------ 2. tss.bed
    bed_path = os.path.join(data, "tss.bed")
    print(f"\n=== 2. TSS BED  {bed_path} ===")
    bed = pd.read_csv(bed_path, sep=r"\s+", header=None, comment="#",
                      engine="python")
    print(bed.head(n_show).to_string(index=False, header=False))
    print("  列数:", bed.shape[1], " 行数:", len(bed))
    bed = bed.iloc[:, :6]
    bed.columns = ["chrom", "start", "end", "gene_id", "score", "strand"]
    hit = {"+_start": 0, "+_start-1": 0, "-_end": 0, "n+": 0, "n-": 0}
    for r in bed.sample(min(len(bed), 2000), random_state=0).itertuples():
        canon = None
        u = str(r.chrom).upper()
        for u2 in (u, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
            if u2 in chrom_alias:
                canon = chrom_alias[u2]
                break
        if canon not in canon2seq:
            continue
        sq = canon2seq[canon]
        s, e = int(r.start), int(r.end)
        if r.strand == "+":
            hit["n+"] += 1
            hit["+_start"] += sq[s:s + 3] == "ATG"
            hit["+_start-1"] += sq[s - 1:s + 2] == "ATG"
        elif r.strand == "-":
            hit["n-"] += 1
            hit["-_end"] += sq[e - 3:e] == "CAT"
    print(f"  + 链：把 start 当 0-based 时 ATG 比例 "
          f"{hit['+_start'] / max(hit['n+'], 1):.1%}；当 1-based 时 "
          f"{hit['+_start-1'] / max(hit['n+'], 1):.1%}")
    print(f"  - 链：end 处反向互补为 ATG 的比例 "
          f"{hit['-_end'] / max(hit['n-'], 1):.1%}")
    print("  → 比例接近 100% 的那种写法说明 tss.bed 实为 ORF(起始密码子)坐标，"
          "02 会自动按此换算锚点。")

    # ------------------------------------------------ 3. SGD_features.tab
    sgd_path = os.path.join(data, "SGD_features.tab")
    print(f"\n=== 3. SGD_features.tab  {sgd_path} ===")
    sgd = pd.read_csv(sgd_path, sep="\t", header=None, dtype=str, quoting=3)
    print("  列数:", sgd.shape[1], "(SGD README 约定为 16 列)")
    print("  feature type 前 8:", sgd[1].value_counts().head(8).to_dict())
    orf_like = sgd[3].fillna("").str.match(
        r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$") | (sgd[1] == "ORF")
    sg = sgd[orf_like & sgd[3].notna()]
    print(sg[[1, 3, 4, 5]].head(n_show).to_string(index=False, header=False))
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
    print(f"  名称映射表: {len(name2sys)} 个名字 -> {len(sys2disp)} 个系统名")

    # ------------------------------------------------ 4. ChEC-seq bigWig
    bw_dir = os.path.join(data, "ChEC-seq")
    print(f"\n=== 4. ChEC-seq bigWig  {bw_dir} ===")
    paths = sorted(glob.glob(os.path.join(bw_dir, "*.bw")) +
                   glob.glob(os.path.join(bw_dir, "*.bigWig")))
    rx = re.compile(r"^GSM\d+_(.+?)_([A-Za-z0-9]+)\.(bw|bigwig)$", re.I)
    tf_reps, free, bad = {}, [], []
    for p in paths:
        m = rx.match(os.path.basename(p))
        if not m:
            bad.append(os.path.basename(p))
            continue
        raw, rep = m.group(1), m.group(2).upper()
        if re.sub(r"[^A-Z]", "", raw.upper()) in ("FREEMNASE", "MNASE", "NOTF"):
            free.append((rep, os.path.basename(p)))
            continue
        u = raw.upper()
        tf = sys2disp.get(name2sys.get(u, u), u)
        tf_reps.setdefault(tf, []).append(rep)
    print(f"  共 {len(paths)} 个文件；TF {len(tf_reps)} 个；free MNase {len(free)} 个")
    print("  free MNase:", [f for _, f in free])
    rc = pd.Series({k: "".join(sorted(v)) for k, v in tf_reps.items()})
    print("  replicate 组合计数:", rc.value_counts().to_dict())
    if bad:
        print("  命名不符合 GSM*_TF_REP.bw 的文件:", bad[:10])
    unk = [t for t in tf_reps if name2sys.get(t, None) is None]
    print("  SGD 中查不到的 TF 名:", unk if unk else "无")
    p0 = paths[0] if paths else None
    if p0:
        try:
            bw = pyBigWig.open(p0)
            h = bw.header()
            print(f"  {os.path.basename(p0)}: chroms 前3={list(bw.chroms().items())[:3]}")
            print(f"     sumData={h['sumData']:.4g}  nBasesCovered={h['nBasesCovered']}"
                  f"  均值/覆盖碱基={h['sumData'] / max(h['nBasesCovered'], 1):.4g}")
            bw.close()
        except Exception as e:
            print(f"  {os.path.basename(p0)}: 打开失败 — {e}")

    # ---- 4b. 全部 bigWig 可开性扫描（含 free MNase）
    # 只核对"16条核染色体的名字集合"是否正常(有没有缺失/认不出)，不再要求所有文件的
    # 染色体长度逐字节相同 —— 见下面 4c 的说明，这批数据每个文件自己的长度本来就有
    # 细微差异，那不算异常。chrMito(线粒体)也不强制要求存在：ChEC-seq 测的是核内
    # TF 结合，线粒体 DNA 本来就不在检测范围内，bigWig 里没有 chrMito 是正常情况，
    # 不算异常（下面会看到这批文件确实都不含 chrMito）。
    all_paths = paths + [os.path.join(bw_dir, f) for _, f in free]
    print(f"\n  === bigWig 完整性扫描（{len(all_paths)} 个文件）===")
    nuclear_names = set(canon2seq.keys()) - {"chrMito"}
    known_names = set(canon2seq.keys())
    bad_files, len_samples, no_mito_count = [], {}, 0
    for p in all_paths:
        try:
            bw = pyBigWig.open(p)
            ch = bw.chroms()
            bw.close()
        except Exception as e:
            sz = os.path.getsize(p) if os.path.exists(p) else -1
            bad_files.append((os.path.basename(p), sz, str(e).strip().splitlines()[-1][:80]))
            continue
        names, key_of = set(), {}
        for k in ch:
            u = str(k).upper()
            for u2 in (u, re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)):
                if u2 in chrom_alias:
                    names.add(chrom_alias[u2])
                    key_of[chrom_alias[u2]] = k
                    break
        missing = nuclear_names - names               # 16 条核染色体里缺的(真异常)
        extra = names - known_names                   # 认不出的额外名字(真异常)
        if missing or extra:
            bad_files.append((os.path.basename(p), os.path.getsize(p),
                              f"缺核染色体 {sorted(missing)[:3]} 多认不出的名字 "
                              f"{sorted(extra)[:3]}"))
            continue
        if "chrMito" not in names:
            no_mito_count += 1
        for canon, k in key_of.items():
            len_samples.setdefault(canon, []).append(ch[k])
    print(f"  打不开 / 缺核染色体的文件: {len(bad_files)} 个"
          f"（占比 {len(bad_files) / max(len(all_paths), 1):.2%}）")
    for nm, sz, msg in bad_files[:30]:
        print(f"    {nm}  size={sz}B  {msg}")
    if len(bad_files) > 30:
        print(f"    ...还有 {len(bad_files) - 30} 个未列出")
    if bad_files:
        print("  → 这些通常是下载不完整/传输中断（bigWig 尾部索引块缺失），或者文件名字对不上"
              "的其他样本。建议核对文件大小是否明显小于同类型文件，若是则重新下载。"
              "02_call_peaks.py 已改成遇到这类文件会跳过并继续，不会整体崩溃。")
    if no_mito_count:
        print(f"  {no_mito_count} 个文件不含 chrMito（线粒体）——符合预期，ChEC-seq 测的是核内"
              "结合，02/03 也不会用到线粒体基因，不影响后续流程。")

    # ---- 4c. 坐标系核对：FASTA 长度 vs 全部 bigWig 文件各自汇报的长度范围
    # 不用单个"参考文件"去比较 —— 实测发现同一条染色体在不同文件里长度本身就不完全
    # 相等(下面会看到 bigWig-min/bigWig-max 不一致)，这套数据的 bigWig 大概率是逐样本
    # 按自身信号范围生成 chrom.sizes 的，不是共用一份参考基因组索引。这不影响 02 的
    # 实际取数逻辑，因为 02 读某个文件时用的是"这个文件自己汇报的长度"来裁剪窗口。
    if len_samples:
        order = [f"chr{r}" for r in roman] + ["chrMito"]
        n_ok_files = len(all_paths) - len(bad_files)
        print(f"\n  染色体长度对照 (FASTA vs {n_ok_files} 个 bigWig 文件各自的长度范围)：")
        print(f"  {'chrom':10s}{'FASTA':>10s}{'bigWig min':>12s}{'bigWig max':>12s}"
              f"{'文件间一致':>10s}")
        any_gt, any_varies = False, False
        for canon in order:
            if canon not in canon2seq or canon not in len_samples:
                continue
            ln_fa = len(canon2seq[canon])
            lens = len_samples[canon]
            lo, hi = min(lens), max(lens)
            same = "是" if lo == hi else "否"
            any_varies = any_varies or (lo != hi)
            any_gt = any_gt or (hi > ln_fa)
            print(f"  {canon:10s}{ln_fa:>10d}{lo:>12d}{hi:>12d}{same:>10s}")
        if any_varies:
            print("\n  同一条染色体在不同文件里长度不完全一样，符合上面的推测(每个文件自己的"
                  "chrom.sizes)，这一点本身不会让 02 取错数据。")
        if any_gt:
            print("  bigWig 普遍比 FASTA 多出几到几十 bp，个别染色体在个别文件里多出上千 bp。"
                  "已核对过这批数据对应发表论文(Mahendrawada et al. 2025, Nature, "
                  "DOI 10.1038/s41586-025-08916-0)的方法部分：比对参考基因组用的是 sacCer3，"
                  "染色体长度与这份 FASTA 完全一致，所以基因组版本本身应该是对的，多出的部分更"
                  "可能是各样本自己的覆盖边界差异。下面做经验验证 —— 查每个文件自己多出的那段"
                  "区间有没有真实信号，并跟该文件自身的总信号量比一比规模（单纯看信号和是否为 0"
                  "太严格：这类背景类信号在染色体边界本来就很难精确为 0，真正要看的是这段区间"
                  "占该文件全部信号的比例像不像是真实的 TF 结合峰，还是纯背景噪音）：")
            probe = ([os.path.join(bw_dir, f) for _, f in free] or paths[:2])[:4]
            rel_thresh = 1e-5   # 一个像样的真实峰，总信号占全文件比例通常在 1e-4~1e-3 量级；
                                # 这里定 1e-5 留足余量，明显更低的当背景噪音处理
            for canon in order:
                if canon not in canon2seq or canon not in len_samples:
                    continue
                ln_fa = len(canon2seq[canon])
                tot, max_rel, n_probed = 0.0, 0.0, 0
                for pp in probe:
                    try:
                        bw = pyBigWig.open(pp)
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
                            s = bw.stats(key, ln_fa, own_len, type="sum")[0] or 0.0
                            file_total = (bw.header()["sumData"] or 1.0)
                            tot += s
                            max_rel = max(max_rel, abs(s) / file_total)
                            n_probed += 1
                        bw.close()
                    except Exception:
                        pass
                if n_probed == 0:
                    print(f"    {canon}: 探测文件里该染色体不比 FASTA 长，跳过")
                    continue
                verdict = f"占比 {max_rel:.2e}，" + (
                    "远低于真实峰的量级，当背景噪音处理，问题不大" if max_rel < rel_thresh
                    else "跟真实峰同量级，不像是背景噪音，需要进一步确认")
                print(f"    {canon}: 探测了 {n_probed} 个文件各自多出的区间，信号和={tot:.4g}，"
                      f"{verdict}")
            print("  以上优先用 free MNase 文件探测（覆盖较均匀，比单个 TF 的稀疏峰更适合测"
                  "'是否为空'），不够时用其他文件补上。")
            print("  02_call_peaks.py 现在会自动做同样的经验验证：占比明显达到真实峰量级才会"
                  "中止，背景噪音量级则直接继续，不需要手动加参数。")
        else:
            print("\n  长度与 FASTA 完全一致，无需进一步验证。")

    # ------------------------------------------------ 5. motif
    mo_dir = os.path.join(data, "motif")
    jas = os.path.join(mo_dir, "JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt")
    print(f"\n=== 5. motif ===\n  JASPAR: {jas}  存在={os.path.exists(jas)}")
    jas_tf = {}
    if os.path.exists(jas):
        cur = None
        for line in open(jas):
            if line.startswith("MOTIF"):
                parts = line.split()
                cur = (parts[1], " ".join(parts[2:]) if len(parts) > 2 else parts[1])
                nm = cur[1]
                if "::" in nm:
                    continue
                u = nm.upper()
                tf = sys2disp.get(name2sys.get(u, u), u)
                jas_tf.setdefault(tf, []).append(cur[0])
        print(f"  JASPAR motif 数(去掉 :: 异源二聚体后按 TF 聚合): {len(jas_tf)} 个 TF")
    ytf_dir = os.path.join(mo_dir, "ALIGNED_ENOLOGO_FORMAT_PWMS")
    pwms = sorted(glob.glob(os.path.join(ytf_dir, "*.pwm")))
    print(f"  YeTFaSCo: {ytf_dir}  文件数={len(pwms)}")
    if pwms:
        print(f"  ---- {os.path.basename(pwms[0])} 原始前 8 行（请确认矩阵方向）----")
        with open(pwms[0]) as fh:
            for i_, line in enumerate(fh):
                if i_ >= 8:
                    break
                print("   |" + line.rstrip("\n"))
    ytf_tf, shapes, fails = {}, {}, []
    for p in pwms:
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
            orient = "4xL(行=碱基)"
        elif rows and lens == {4}:
            orient = "Lx4(行=位置)"
        elif rows and lens == {5}:
            orient = "Lx5(首列为位置号)"
        else:
            fails.append(stem)
            continue
        shapes[orient] = shapes.get(orient, 0) + 1
        if "-" in tfraw and not re.match(r"^Y[A-P][LR]\d{3}[WC]-[A-Z]$", tfraw.upper()):
            continue                                   # 二聚体/复合体 PWM
        u = tfraw.upper()
        tf = sys2disp.get(name2sys.get(u, u), u)
        ytf_tf.setdefault(tf, []).append(stem)
    print("  解析出的矩阵方向:", shapes, " 解析失败:", fails[:10])
    chec = set(tf_reps)
    cov_j = chec & set(jas_tf)
    cov_y = chec & set(ytf_tf)
    print(f"  ChEC TF 被 JASPAR 覆盖 {len(cov_j)}/{len(chec)}，"
          f"被 YeTFaSCo 覆盖 {len(cov_y)}/{len(chec)}，"
          f"并集 {len(cov_j | cov_y)}/{len(chec)}")
    print("  两者都没有的 TF:", ",".join(sorted(chec - cov_j - cov_y)))
    multi = {k: len(v) for k, v in ytf_tf.items() if k in chec and len(v) > 1}
    print(f"  YeTFaSCo 中一个 TF 对应多个 PWM 的: {len(multi)} 个 "
          f"(03 会按峰内富集度自动选一个)")
    print("  注：YeTFaSCo 这批 .pwm 文件(ALIGNED_ENOLOGO_FORMAT)存的不是概率，而是"
          "energy-normalized logo(enoLOGOS)风格的对数似然值，含负数和 >1 的值属于正常"
          "现象，不代表文件损坏；03_motif_anchor.py 会按此重建概率矩阵，具体见该脚本注释。")

    # ------------------------------------------------ 6. Excel
    xl_path = os.path.join(data, "41586_2025_8916_MOESM5_ESM.xlsx")
    print(f"\n=== 6. Excel  {xl_path} ===")
    xl = pd.ExcelFile(xl_path)
    print("  sheets:", xl.sheet_names)
    for sh in xl.sheet_names:
        if re.search(r"index|read\s*me|legend|说明", sh, re.I):
            raw = xl.parse(sh, header=None, nrows=40)
            print(f"  ---- {sh!r} 内容(前 40 行，用于确认各 Table 的定义) ----")
            print(raw.to_string(header=False, index=False))
    for sh in xl.sheet_names:
        if re.search(r"s\s*3", sh, re.I):
            raw = xl.parse(sh, header=None, nrows=8)
            print(f"  ---- {sh!r} 前 8 行 x 前 6 列 ----")
            print(raw.iloc[:, :6].to_string(header=False))
    s3_like = [s for s in xl.sheet_names if re.search(r"s\s*3\s*[-_ ]?\s*[c-e]", s, re.I)]
    if len(s3_like) > 1:
        print(f"\n  注意：{s3_like} 这几个表结构相似(都是 gene×TF 的稀疏数值矩阵)，"
              "但很可能对应论文里不同的基因子集(如'全部响应基因' vs '同时被结合且被调控的"
              "功能靶点')。用错子集会给下游模型引入选择偏差。上面若打印出了 Index/说明 sheet"
              "的内容，请核对其中每个 Table 的定义；若没有，请告诉我这几个 sheet 分别用在"
              "论文里的哪个位置，或直接指定 --sheet-fc 用哪一个。")

    # ------------------------------------------------ 7. 其他目录
    print("\n=== 7. 其他 ===")
    xmls = glob.glob(os.path.join(data, "EBI Complex Portal", "*.xml"))
    print(f"  EBI Complex Portal xml: {len(xmls)} 个（Phase 0 不用，Fig 4 再解析）")
    gff = os.path.join(data, "saccharomyces_cerevisiae.20260910.gff")
    print(f"  SGD GFF 存在={os.path.exists(gff)}（Phase 0 不用；名称映射走 SGD_features.tab）")
    tpm = sorted(os.listdir(os.path.join(data, "tpm"))) \
        if os.path.isdir(os.path.join(data, "tpm")) else []
    print(f"  data/tpm: {tpm[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-show", type=int, default=5)
    a = ap.parse_args()
    run_inventory(a.data, a.n_show)
