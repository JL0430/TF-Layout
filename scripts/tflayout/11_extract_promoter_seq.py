# scripts/tflayout/11_extract_promoter_seq.py

import argparse
import os
import re

import numpy as np
import pandas as pd


def run_extract_promoter_seq(fasta="data/S288C.fsa", tss="data/tss.bed",
                             outdir="out", upstream=1000, downstream=500):
    os.makedirs(outdir, exist_ok=True)

    # ---- 读FASTA，chrom名归一化(跟00/02一致的罗马数字规则，另外FASTA header常见
    # NCBI RefSeq格式 "ref|NC_001133|..."，也要能识别，之前漏了这个分支)
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV", "XVI"]
    chrom_alias = {}
    for i, r in enumerate(roman, 1):
        for k in (f"CHR{r}", r, str(i), f"{i:02d}", f"CHR{i}", f"CHR{i:02d}"):
            chrom_alias[k] = f"chr{r}"
    chrom_alias["MITO"] = "chrMito"
    chrom_alias["M"] = "chrMito"
    chrom_alias["MT"] = "chrMito"
    refseq2roman = {f"NC_0011{33 + i:02d}": roman[i] for i in range(16)}
    refseq2roman["NC_001224"] = None  # 线粒体，RefSeq号跟核染色体不连续，单独查

    def to_canon(u):
        u = str(u).upper().strip().strip(">")
        for tok in u.split("|"):
            t = tok.strip()
            if t in refseq2roman:
                return "chr" + refseq2roman[t] if refseq2roman[t] else "chrMito"
            t0 = t.split(".")[0]  # 去掉版本号后缀，如 NC_001133.9 -> NC_001133
            if t0 in refseq2roman:
                return "chr" + refseq2roman[t0] if refseq2roman[t0] else "chrMito"
        u2 = re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)
        return chrom_alias.get(u, chrom_alias.get(u2))

    seqs = {}
    cur_name, cur_chunks = None, []
    with open(fasta) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_name is not None:
                    seqs[cur_name] = "".join(cur_chunks)
                cur_name = line[1:].split()[0]
                cur_chunks = []
            else:
                cur_chunks.append(line.strip())
    if cur_name is not None:
        seqs[cur_name] = "".join(cur_chunks)
    canon_seqs = {}
    for k, v in seqs.items():
        c = to_canon(k)
        if c:
            canon_seqs[c] = v
    print(f"读入 FASTA：{len(canon_seqs)} 条染色体")

    # ---- 读tss.bed，判定0/1-based(跟02一致)，算锚点
    bed = pd.read_csv(tss, sep=r"\s+", header=None, comment="#", engine="python")
    bed = bed.iloc[:, :6]
    bed.columns = ["chrom_raw", "start", "end", "gene_id", "score", "strand"]
    bed["chrom"] = bed["chrom_raw"].map(to_canon)

    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

    def revcomp(s):
        return "".join(comp.get(b, "N") for b in reversed(s.upper()))

    def atg_at(chrom, pos0):
        """pos0: 0-based起点，取3个碱基看是不是ATG"""
        seq = canon_seqs.get(chrom)
        if seq is None or pos0 < 0 or pos0 + 3 > len(seq):
            return False
        return seq[pos0:pos0 + 3].upper() == "ATG"

    plus = bed[bed["strand"] == "+"]
    n_check = min(500, len(plus))
    sample = plus.sample(n=n_check, random_state=0) if n_check else plus
    hit_0based = sum(atg_at(r.chrom, int(r.start)) for r in sample.itertuples())
    hit_1based = sum(atg_at(r.chrom, int(r.start) - 1) for r in sample.itertuples())
    one_based = hit_1based > hit_0based
    print(f"锚点判定：0-based命中{hit_0based}/{n_check}，1-based命中{hit_1based}/{n_check} "
          f"-> 判定为{'1-based(start-1做锚点)' if one_based else '0-based(start做锚点)'}")

    def anchor(row):
        if row.strand == "+":
            return (int(row.start) - 1) if one_based else int(row.start)
        else:
            return int(row.end) - 1

    bed["anchor"] = bed.apply(anchor, axis=1)

    # ---- 逐基因提取窗口，负链反向互补
    records = []
    n_clip, n_missing_chrom = 0, 0
    for row in bed.itertuples():
        chrom = row.chrom
        seq = canon_seqs.get(chrom)
        if seq is None:
            n_missing_chrom += 1
            continue
        a = row.anchor
        if row.strand == "+":
            lo, hi = a - upstream, a + downstream
        else:
            lo, hi = a - downstream, a + upstream
        lo_clip, hi_clip = max(0, lo), min(len(seq), hi)
        frag = seq[lo_clip:hi_clip]
        pad_left = lo_clip - lo
        pad_right = hi - hi_clip
        if pad_left > 0 or pad_right > 0:
            n_clip += 1
        frag = "N" * pad_left + frag + "N" * pad_right
        if row.strand == "-":
            frag = revcomp(frag)
        records.append((row.gene_id, frag))

    out = pd.DataFrame(records, columns=["gene_id", "seq"])
    expect_len = upstream + downstream
    bad_len = (out["seq"].str.len() != expect_len).sum()
    print(f"提取 {len(out)} 个基因的启动子序列，窗口长度应为 {expect_len}bp，"
          f"长度不对的有 {bad_len} 个")
    if n_missing_chrom:
        print(f"  {n_missing_chrom} 个基因所在染色体在FASTA里找不到，已跳过")
    if n_clip:
        print(f"  {n_clip} 个基因的窗口超出染色体边界，超出部分用N填充")
    n_frac = (out["seq"].str.count("N") / out["seq"].str.len()).mean()
    print(f"  N碱基平均占比: {n_frac:.4%}（主要是边界填充，正常应该很小）")

    out.to_parquet(os.path.join(outdir, "promoter_seq.parquet"))
    print(f"-> out/promoter_seq.parquet（列：gene_id, seq；每条长度{expect_len}bp，"
          f"上游{upstream}bp+下游{downstream}bp，已按链方向统一成5'->3'）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/S288C.fsa")
    ap.add_argument("--tss", default="data/tss.bed")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--upstream", type=int, default=1000)
    ap.add_argument("--downstream", type=int, default=500)
    a = ap.parse_args()
    run_extract_promoter_seq(a.fasta, a.tss, a.outdir, a.upstream, a.downstream)
