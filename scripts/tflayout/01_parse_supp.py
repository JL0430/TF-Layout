# scripts/tflayout/01_parse_supp.py

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def run_parse_supp(xlsx="data/41586_2025_8916_MOESM5_ESM.xlsx",
                   sgd="data/SGD_features.tab", bwdir="data/ChEC-seq",
                   outdir="out", sheet_bind=None, sheet_occ=None, sheet_fc=None):
    os.makedirs(outdir, exist_ok=True)

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

    # ------------------------------------------------ ChEC TF 名（用于核对）
    chec_tfs = set()
    for p in glob.glob(os.path.join(bwdir, "*.bw")):
        m = re.match(r"^GSM\d+_(.+?)_[A-Za-z0-9]+\.bw$", os.path.basename(p), re.I)
        if m and re.sub(r"[^A-Z]", "", m.group(1).upper()) not in ("FREEMNASE", "MNASE"):
            u = m.group(1).upper()
            chec_tfs.add(sys2disp.get(name2sys.get(u, u), u))

    xl = pd.ExcelFile(xlsx)
    tables, report = {}, []
    for tag, forced in (("a", sheet_bind), ("b", sheet_occ), ("c", sheet_fc)):
        sh = forced
        if sh is None:
            hits = [s for s in xl.sheet_names
                    if re.search(rf"s\s*3\s*[-_ ]?\s*{tag}(?![a-z])", s, re.I)]
            if len(hits) != 1:
                raise KeyError(f"找不到唯一的 S3{tag.upper()} sheet，候选={hits}；"
                               f"全部 sheet={xl.sheet_names}。请用 --sheet-* 显式指定")
            sh = hits[0]
        raw = xl.parse(sh, header=None)
        # 表头行：不是"非空单元格最多的那行"（对 S3b/S3c 这类大部分基因都是 NaN 的稀疏表，
        # 偶尔一行数据碰巧比表头更"满"，会把表头行挤掉），而是找前 30 行里第一行满足
        # "除第一列外的非空单元格里，大部分是文本(TF 名)而不是数字"——真表头是 TF 名
        # 字符串，数据行则不是空就是数值，靠这个区分更稳。
        hdr = 0
        for i in range(min(30, len(raw))):
            row = raw.iloc[i, 1:].dropna()
            if len(row) == 0:
                continue
            non_numeric = row[pd.to_numeric(row, errors="coerce").isna()]
            if len(non_numeric) / len(row) >= 0.8:
                hdr = i
                break
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hdr]]
        idcol = df.columns[0]
        # 去掉除第一列外的「非数值列」（如基因俗名、描述）
        dropped = []
        for c in df.columns[1:]:
            v = df[c].dropna()
            if len(v) == 0:
                # 这一列全是空的（比如某个 TF 在这张已经筛过的子集表里一个显著基因都没有）——
                # 之前这里直接 continue，会把该列原样留成 object dtype，混进最终矩阵后
                # 让 fc.values 整体变成 object 数组，后面 np.isfinite 就会报错。这里强制
                # 赋成干净的 float NaN。
                df[c] = np.nan
                continue
            if not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
                vv = v.astype(str).str.strip().str.upper()
                vv = vv.replace({"TRUE": "1", "FALSE": "0", "YES": "1", "NO": "0"})
                num = pd.to_numeric(vv, errors="coerce")
                if num.notna().mean() < 0.5:
                    dropped.append(c)
                    continue
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.strip().str.upper().replace(
                        {"TRUE": "1", "FALSE": "0", "YES": "1", "NO": "0",
                         "NAN": np.nan, "": np.nan}), errors="coerce")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop(columns=dropped)
        g = df[idcol].astype(str).str.strip().str.upper()
        mapped = [name2sys.get(x, x) for x in g]
        n_mapped = int(sum(x in name2sys for x in g))
        # 先按原始 idcol 删列，再插入新的 gene_id 列——如果 idcol 本身就叫 "gene_id"
        # （真实 Table-S3a 就是这样），反过来做（先赋值同名列、再 drop 那个名字）会把
        # 刚写好的映射结果连着 idcol 一起删掉，导致后面 groupby("gene_id") 找不到列。
        # 用 concat 而不是 insert，避免 pandas 在宽表上反复 insert 报的碎片化警告。
        df = pd.concat([pd.Series(mapped, index=df.index, name="gene_id"),
                        df.drop(columns=[idcol])], axis=1)
        df = df.groupby("gene_id").first()
        newcols = {}
        for c in df.columns:
            u = c.strip().upper()
            newcols[c] = sys2disp.get(name2sys.get(u, u), u)
        for c, t in newcols.items():
            report.append(dict(sheet=f"S3{tag.upper()}", column=c, tf=t,
                               in_sgd=t.upper() in name2sys,
                               has_chec_bw=t in chec_tfs))
        df = df.rename(columns=newcols)
        df = df.loc[:, ~df.columns.duplicated()]       # 重名列保留第一列
        df.index.name = "gene_id"
        df.columns.name = "tf"
        tables[tag] = df
        print(f"S3{tag.upper()}: sheet={sh!r} 表头行={hdr} 形状={df.shape} "
              f"基因名可映射到 SGD {n_mapped}/{len(g)}  去掉非数值列={dropped}")

    bind = tables["a"].fillna(0)
    bind = (bind > 0).astype(np.int8)
    occ = tables["b"]
    fc = tables["c"]
    sig = fc.notna() & (fc != 0)

    for nm, df in (("binding_binary", bind), ("chec_occupancy", occ),
                   ("log2fc_sig", fc), ("sig_mask", sig)):
        df.to_parquet(os.path.join(outdir, f"{nm}.parquet"))
    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(outdir, "supp_tf_name_map.tsv"), sep="\t", index=False)

    print(f"\nbinding   {bind.shape}  阳性率 {bind.values.mean():.4f}")
    print(f"occupancy {occ.shape}")
    # 用 to_numpy(dtype=float) 兜底：万一某列因为其他没预见到的原因仍留了非数值 dtype，
    # 这里会明确转换或报出可读的错误，而不是让 np.isfinite 在最后一步才神秘崩溃。
    fv = fc.to_numpy(dtype=float, na_value=np.nan)
    print(f"log2FC    {fc.shape}  非空比例 {np.isfinite(fv).mean():.4f}  "
          f"精确为 0 的比例 {(fv == 0).mean():.4f}")
    print("  → 若「精确为 0」比例很高，说明非显著基因被填成了 0 而不是空值；"
          "sig_mask 已把 0 视为不显著。")
    print(f"ChEC TF(bigWig) {len(chec_tfs)} 个；S3A 列 {bind.shape[1]} 个；"
          f"交集 {len(set(bind.columns) & chec_tfs)}")
    miss = sorted(set(bind.columns) - chec_tfs)
    print("S3A 中无 bigWig 的 TF:", ",".join(miss) if miss else "无")
    miss2 = sorted(chec_tfs - set(bind.columns))
    print("有 bigWig 但 S3A 中没有的 TF:", ",".join(miss2) if miss2 else "无")
    print(f"被耗竭 TF(S3C 列) {fc.shape[1]} 个，与 ChEC 交集 "
          f"{len(set(fc.columns) & chec_tfs)}")
    print(f"名称映射明细 -> {os.path.join(outdir, 'supp_tf_name_map.tsv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/41586_2025_8916_MOESM5_ESM.xlsx")
    ap.add_argument("--sgd", default="data/SGD_features.tab")
    ap.add_argument("--bwdir", default="data/ChEC-seq")
    ap.add_argument("--out", default="out")
    ap.add_argument("--sheet-bind", default=None)
    ap.add_argument("--sheet-occ", default=None)
    ap.add_argument("--sheet-fc", default=None)
    a = ap.parse_args()
    run_parse_supp(a.xlsx, a.sgd, a.bwdir, a.out,
                   a.sheet_bind, a.sheet_occ, a.sheet_fc)
