# scripts/tflayout/08_build_model_inputs.py

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def run_build_model_inputs(tpm_dir="data/tpm", layout="out/tf_layout.parquet",
                           log2fc="out/log2fc_sig.parquet", sig="out/sig_mask.parquet",
                           tss="data/tss.bed", outdir="out"):
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------ 0. L_g 验证：tf_layout.parquet
    # 本身就是 L_g，按 gene_id 分组即可，这里只做健全性检查+打印一个例子
    lay = pd.read_parquet(layout)
    g0 = lay["gene_id"].iloc[0]
    example = lay[lay["gene_id"] == g0].sort_values("site_pos")
    print("=== 0. L_g 健全性检查（tf_layout.parquet 按 gene_id 分组就是 L_g）===")
    print(f"共 {lay['gene_id'].nunique()} 个基因；例子 {g0} 有 {len(example)} 个 token:")
    print(example[["tf", "site_pos", "motif_strand", "a", "m", "res_id"]]
          .head(8).to_string(index=False))
    print("字段对应《方案.txt》的 (t,p,s,a,m,r)：tf=t, site_pos=p, motif_strand=s, "
          "a=a, m=m, res_id=r。不用额外重建，训练时 df.groupby('gene_id') 直接拿。")

    print(f"\n=== data/{os.path.basename(tpm_dir)}/ 文件一览（仅供参考，Head A 只用"
          "DMSO_expression.txt）===")
    for p in sorted(glob.glob(os.path.join(tpm_dir, "*.txt"))):
        try:
            df = pd.read_csv(p, sep=None, engine="python", nrows=3)
            print(f"  {os.path.basename(p)}: 列数 {df.shape[1]}，列名 "
                  f"{list(df.columns)[:4]}{'...' if df.shape[1] > 4 else ''}")
        except Exception as e:
            print(f"  {os.path.basename(p)}: 读取失败 - {e}")

    # ------------------------------------------------ 1. Head A: baseline log(TPM)
    # 精确复现 new55.py 的处理：DMSO_expression.txt 的 TPM_median 列，log1p 后
    # z-score，z-score 只在 train 基因上 fit（染色体 holdout：val=chrXIII/XIV，
    # test=chrXV/XVI，其余 train——跟 CHR_HOLDOUT_PRESETS['fixed_saccer3'] 一致）
    print("\n=== 1. Head A: baseline log(TPM)，精确复现 CITRA(new55.py) 的处理方式 ===")
    tpm_path = os.path.join(tpm_dir, "DMSO_expression.txt")
    if os.path.exists(tpm_path):
        raw = pd.read_csv(tpm_path, sep="\t")
        gene_col = raw.columns[0]
        raw["TPM_median"] = pd.to_numeric(raw["TPM_median"], errors="coerce")
        raw = raw.dropna(subset=[gene_col, "TPM_median"])
        print(f"  {tpm_path}：{len(raw)} 个基因，列=TPM_median"
              "（跟 new55.py 里 tpm_column='TPM_median' 一致）")

        roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                 "XI", "XII", "XIII", "XIV", "XV", "XVI"]
        chrom_alias = {}
        for i, r in enumerate(roman, 1):
            for k in (f"CHR{r}", r, str(i), f"{i:02d}", f"CHR{i}", f"CHR{i:02d}"):
                chrom_alias[k] = f"chr{r}"
        bed = pd.read_csv(tss, sep=r"\s+", header=None, comment="#", engine="python")
        bed = bed.iloc[:, :4]
        bed.columns = ["chrom_raw", "start", "end", "gene_id"]

        def to_canon(u):
            u = str(u).upper()
            u2 = re.sub(r"^(CHROMOSOME|CHR)[_\-.]?", "CHR", u)
            return chrom_alias.get(u, chrom_alias.get(u2))

        bed["chrom"] = bed["chrom_raw"].map(to_canon)
        gene2chrom = dict(zip(bed["gene_id"], bed["chrom"]))

        raw["chrom"] = raw[gene_col].map(gene2chrom)
        val_chrs, test_chrs = {"chrXIII", "chrXIV"}, {"chrXV", "chrXVI"}
        raw["split"] = np.where(raw["chrom"].isin(test_chrs), "test",
                                np.where(raw["chrom"].isin(val_chrs), "val", "train"))
        n_unmapped = int(raw["chrom"].isna().sum())
        if n_unmapped:
            print(f"  {n_unmapped} 个基因在 tss.bed 里找不到染色体，归入 train"
                  "（不进 val/test 评估，但仍参与训练）")

        y_log1p = np.log1p(raw["TPM_median"].to_numpy())
        train_mask = (raw["split"] == "train").to_numpy()
        mu, sd = float(y_log1p[train_mask].mean()), float(y_log1p[train_mask].std())
        y_z = (y_log1p - mu) / (sd if sd > 0 else 1.0)

        head_a = pd.DataFrame({
            "gene_id": raw[gene_col].to_numpy(),
            "log_tpm_baseline": y_z.astype(np.float32),
            "split": raw["split"].to_numpy(),
        }).set_index("gene_id")
        print(f"  切分：train={(raw['split'] == 'train').sum()}  "
              f"val(chrXIII/XIV)={(raw['split'] == 'val').sum()}  "
              f"test(chrXV/XVI)={(raw['split'] == 'test').sum()}")
        print(f"  z-score 只在 train 上拟合：mean={mu:.4f} std={sd:.4f}"
              "（跟 new55.py StandardScaler 在 train fit、val/test 只 transform 是"
              "同一套逻辑，没有信息泄漏）")
        head_a.to_parquet(os.path.join(outdir, "head_a_baseline_logtpm.parquet"))
        print("  -> out/head_a_baseline_logtpm.parquet（列：log_tpm_baseline, split）")
    else:
        print(f"  找不到 {tpm_path}，Head A 建不出来，检查 --tpm-dir 是否指对了")

    # ------------------------------------------------ 2. Head B/C: 耗竭响应标签
    print("\n=== 2. Head B(回归log2FC) / Head C(三分类down/ns/up) 标签 ===")
    fc = pd.read_parquet(log2fc)
    sg = pd.read_parquet(sig)
    long_fc = fc.reset_index().melt(id_vars=fc.index.name or "index",
                                    var_name="tf_depleted", value_name="log2fc")
    long_fc = long_fc.rename(columns={fc.index.name or "index": "gene_id"})
    long_sig = sg.reset_index().melt(id_vars=sg.index.name or "index",
                                     var_name="tf_depleted", value_name="is_sig")
    long_sig = long_sig.rename(columns={sg.index.name or "index": "gene_id"})
    lbl = long_fc.merge(long_sig, on=["gene_id", "tf_depleted"])
    lbl["is_sig"] = lbl["is_sig"].fillna(False)
    lbl["direction_3class"] = np.where(~lbl["is_sig"], "ns",
                                       np.where(lbl["log2fc"] > 0, "up", "down"))
    lbl.to_parquet(os.path.join(outdir, "head_bc_labels.parquet"))
    print(f"长表 {len(lbl)} 条 (gene_id × tf_depleted)")
    print(lbl["direction_3class"].value_counts(normalize=True).round(4).to_string())
    print("-> out/head_bc_labels.parquet（Head B 直接用 log2fc 列做回归目标；"
          "Head C 直接用 direction_3class 列做三分类目标，ns 占比应该 >90%，"
          "训练时记得用 class-balanced focal loss，不要用普通交叉熵）")

    # ------------------------------------------------ 3. condition 向量：默认方案
    print("\n=== 3. condition 向量(ctx_D) 的默认构造方案（跟09脚本实现一致）===")
    print("其余TF的'活性代理' = 该TF自己的基因在条件D下的log2FC，直接从上面第2节的"
          "head_bc_labels.parquet里查(gene_id=TF自己的系统名, tf_depleted=D)，不用"
          "另外拉数据；查不到记0。被耗竭TF D自己的槽位强制置0(图1b原文：因为degron"
          "降解的是蛋白，mRNA往往不随之下降，必须显式覆盖，不能依赖数据自然呈现)。"
          "这是按最省事、且复用已核实过的S3C数据的方案定的默认值，不是确认过的"
          "标准答案，如果想法不一样，改09_torch_dataset.py里的_build_ctx方法就行。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpm-dir", default="data/tpm")
    ap.add_argument("--layout", default="out/tf_layout.parquet")
    ap.add_argument("--log2fc", default="out/log2fc_sig.parquet")
    ap.add_argument("--sig", default="out/sig_mask.parquet")
    ap.add_argument("--tss", default="data/tss.bed")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    run_build_model_inputs(a.tpm_dir, a.layout, a.log2fc, a.sig, a.tss, a.outdir)

