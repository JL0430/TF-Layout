# scripts/tflayout/20_dense_log2fc_check.py

import os
import re

import numpy as np
import pandas as pd

CONFIG = dict(
    tpm_dir="data/tpm",
    dmso_file="DMSO_mean.txt",
    iaa_file="3IAA_mean.txt",
    labels="out/head_bc_labels.parquet",   # 08号产出：gene_id, tf_depleted, log2fc(显著才有), direction_3class
    pseudocount=1.0,                        # log2((IAA+pc)/(DMSO+pc))
    fc_threshold=1.3,                       # 论文显著性门槛之一(|FC|≥1.3)，只用来描述不显著对的数值分布
    outdir="out/results/_dense_check",
)


def run_dense_check(tpm_dir="data/tpm", dmso_file="DMSO_mean.txt", iaa_file="3IAA_mean.txt",
                    labels="out/head_bc_labels.parquet", pseudocount=1.0, fc_threshold=1.3,
                    outdir="out/results/_dense_check"):
    """唯一入口。"""
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def pearson(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 else float("nan")

    def read_table(path):
        for kw in (dict(sep="\t"), dict(sep=r"\s+", engine="python")):
            try:
                df = pd.read_csv(path, **kw)
            except Exception:  # noqa: BLE001 —— 换一种分隔符再试
                continue
            if df.shape[1] >= 3:
                return df
        return None

    def gene_col(df):
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.isna().mean() > 0.5:
                return c
        return None

    say("=" * 78)
    say("稠密 log2FC 可用性诊断(只读)")
    say("=" * 78)
    paths = {k: os.path.join(tpm_dir, f) for k, f in (("DMSO", dmso_file), ("3IAA", iaa_file))}
    raw = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}，先确认 CONFIG['tpm_dir'] 和文件名")
        with open(p, encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "").rstrip("\n")[:300] for _ in range(3)]
        say(f"[{k}] {p}  {os.path.getsize(p) / 2 ** 20:.1f}MB  前3行(截到300字符)：")
        for h in head:
            say(f"    {h}")
        raw[k] = read_table(p)
        if raw[k] is None:
            raise SystemExit(f"{p} 用 tab/空白分隔都读不出≥3列，把上面前3行贴回来")
    lbl = pd.read_parquet(labels)
    dep_names = sorted(set(lbl["tf_depleted"].astype(str).str.strip().str.upper()))
    dep_set = set(dep_names)
    say(f"标签文件：{len(lbl)} 行，{len(dep_names)} 个耗竭条件名(含 GCN4_SM 这类条件列)")

    drop_tok = {"DMSO", "3IAA", "IAA", "MEAN", "TPM", "AVG", "AVERAGE"}
    mats = {}
    for k, df in raw.items():
        gc = gene_col(df)
        if gc is None:  # 没有非数字列：基因ID在 index 里(表头比数据少一列的情况)
            df = df.reset_index()
            gc = df.columns[0]
        df = df.set_index(df[gc].astype(str).str.strip().str.upper())
        mapping, unmatched = {}, []
        for c in df.columns:
            if c == gc:
                continue
            toks = [t for t in re.split(r"[_\-\s\.]+", str(c).strip().upper()) if t and t not in drop_tok]
            hit = None
            for n in range(len(toks), 0, -1):  # 最长前缀匹配
                cand = "_".join(toks[:n])
                if cand in dep_set:
                    hit = cand
                    break
            if hit is None:
                unmatched.append(str(c))
            elif hit in mapping.values():
                unmatched.append(f"{c}(重复映射到{hit})")
            else:
                mapping[c] = hit
        say(f"[{k}] 基因列={gc!r}，{df.shape[0]} 个基因；{len(mapping)} 列匹配到耗竭条件名，"
            f"{len(unmatched)} 列没匹配上{(': ' + str(unmatched[:12])) if unmatched else ''}")
        m = df[list(mapping)].apply(pd.to_numeric, errors="coerce")
        m.columns = [mapping[c] for c in m.columns]
        mats[k] = m
    conds = sorted(set(mats["DMSO"].columns) & set(mats["3IAA"].columns))
    genes = mats["DMSO"].index.intersection(mats["3IAA"].index)
    if not conds or len(genes) == 0:
        raise SystemExit("两个文件没有共同的条件列或基因，看上面的列名/基因列诊断")
    say(f"两个文件共有 {len(conds)} 个条件、{len(genes)} 个基因")
    d = mats["DMSO"].loc[genes, conds].to_numpy(np.float64)
    i = mats["3IAA"].loc[genes, conds].to_numpy(np.float64)
    say(f"数值范围：DMSO 中位 {np.nanmedian(d):.2f}、最大 {np.nanmax(d):.0f}；3IAA 中位 {np.nanmedian(i):.2f}"
        f"(应该是 TPM 量级；如果中位数是负数或都在0~1之间，说明已经是对数/归一化过的，pseudocount 要改)")
    dense = pd.DataFrame(np.log2((i + pseudocount) / (d + pseudocount)), index=genes, columns=conds)
    long = dense.stack().rename("log2fc_dense").reset_index()
    long.columns = ["gene_id", "tf_depleted", "log2fc_dense"]

    lb = lbl.assign(gene_id=lbl["gene_id"].astype(str).str.strip().str.upper(),
                    tf_depleted=lbl["tf_depleted"].astype(str).str.strip().str.upper())
    j = lb.merge(long, on=["gene_id", "tf_depleted"], how="left")
    cover = float(j["log2fc_dense"].notna().mean())
    say(f"标签里的 (基因,条件) 对有 {cover:.1%} 能在 tpm 里找到稠密 log2FC"
        f"(低于~90%说明基因ID或条件名对不齐，看上面的匹配诊断)")
    sig = j["log2fc"].notna() & j["log2fc_dense"].notna()
    ns = j["log2fc"].isna() & j["log2fc_dense"].notna()
    thr = float(np.log2(fc_threshold))
    r_all = pearson(j.loc[sig, "log2fc_dense"], j.loc[sig, "log2fc"])
    sign_all = float(np.mean(np.sign(j.loc[sig, "log2fc_dense"]) == np.sign(j.loc[sig, "log2fc"])))
    slope = float(np.polyfit(j.loc[sig, "log2fc"], j.loc[sig, "log2fc_dense"], 1)[0]) if sig.sum() > 3 \
        else float("nan")
    a_ns = j.loc[ns, "log2fc_dense"].abs()
    a_sg = j.loc[sig, "log2fc_dense"].abs()
    lab = np.r_[np.ones(len(a_sg)), np.zeros(len(a_ns))]
    sc = np.r_[a_sg.to_numpy(), a_ns.to_numpy()]
    rk = pd.Series(sc).rank().to_numpy()
    auc = float((rk[lab == 1].sum() - len(a_sg) * (len(a_sg) + 1) / 2) / max(len(a_sg) * len(a_ns), 1))
    say(f"\n[1] 显著对(S3c 有 log2FC，n={int(sig.sum())})：稠密 vs S3c Pearson r={r_all:.3f}，符号一致率="
        f"{sign_all:.3f}，斜率(稠密对S3c)={slope:.2f}")
    say(f"[2] 不显著对(n={int(ns.sum())})：|稠密log2FC| 中位 {a_ns.median():.3f}、90%分位 "
        f"{a_ns.quantile(.9):.3f}、99%分位 {a_ns.quantile(.99):.3f}；超过 log2({fc_threshold})={thr:.3f} 的"
        f"比例 {float((a_ns >= thr).mean()):.3f}")
    say(f"[3] 用 |稠密log2FC| 区分显著/不显著的 AUROC = {auc:.3f}(越接近1，说明稠密值跟显著性判定越一致)")
    rows = []
    for tf, g in j[j["log2fc_dense"].notna()].groupby("tf_depleted"):
        s_ = g[g["log2fc"].notna()]
        n_ = g[g["log2fc"].isna()]
        rows.append(dict(tf_depleted=tf, n_sig=len(s_), r_sig=pearson(s_["log2fc_dense"], s_["log2fc"])
                         if len(s_) >= 5 else np.nan,
                         sign_acc_sig=float(np.mean(np.sign(s_["log2fc_dense"]) == np.sign(s_["log2fc"])))
                         if len(s_) else np.nan,
                         ns_abs_median=float(n_["log2fc_dense"].abs().median()) if len(n_) else np.nan,
                         ns_frac_above_thr=float((n_["log2fc_dense"].abs() >= thr).mean()) if len(n_)
                         else np.nan))
    pc = pd.DataFrame(rows).sort_values("n_sig", ascending=False)
    pc.to_csv(os.path.join(outdir, "per_condition.csv"), index=False)
    ok = pc[pc["n_sig"] >= 20]
    say(f"[4] 逐条件(≥20个显著对的 {len(ok)} 个)：r 中位 {ok['r_sig'].median():.3f}、四分位 "
        f"{ok['r_sig'].quantile(.25):.3f}~{ok['r_sig'].quantile(.75):.3f}；符号一致率中位 "
        f"{ok['sign_acc_sig'].median():.3f}；r<0.5 的条件 {int((ok['r_sig'] < 0.5).sum())} 个"
        f"(列在 per_condition.csv)")
    say("\n判读参考(建议，不是硬规则)：[1] r≳0.9 且符号一致率≳0.95 → 稠密值跟 S3c 同源、可信；"
        "[2] 不显著对的 |log2FC| 90%分位明显小于 log2(1.3)=0.38 → 噪声水平低，可以当弱监督；"
        "反之噪声大，只适合按精度加权或只当辅助任务。r 明显低(<0.7)则多半是列名/基因对齐出了问题，"
        "先看开头几行的列名/基因匹配诊断。")
    out_l = long.merge(lb[["gene_id", "tf_depleted"]], on=["gene_id", "tf_depleted"], how="inner")
    try:
        out_l.to_parquet(os.path.join(outdir, "dense_log2fc.parquet"), index=False)
    except ImportError:
        out_l.to_csv(os.path.join(outdir, "dense_log2fc.csv.gz"), index=False, compression="gzip")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(j.loc[sig, "log2fc"], j.loc[sig, "log2fc_dense"], s=3, alpha=.3)
        ax[0].axline((0, 0), slope=1, color="k", lw=.6)
        ax[0].set_xlabel("S3c log2FC (significant pairs)")
        ax[0].set_ylabel("dense log2FC from TPM means")
        ax[0].set_title(f"r={r_all:.3f}, sign agreement={sign_all:.3f}")
        bins = np.linspace(-3, 3, 121)
        ax[1].hist(j.loc[ns, "log2fc_dense"].clip(-3, 3), bins=bins, alpha=.6, density=True, label="not significant")
        ax[1].hist(j.loc[sig, "log2fc_dense"].clip(-3, 3), bins=bins, alpha=.6, density=True, label="significant")
        for t_ in (-thr, thr):
            ax[1].axvline(t_, color="k", lw=.6, ls="--")
        ax[1].legend(fontsize=8)
        ax[1].set_title("dense log2FC distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig", "dense_vs_s3c.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        say("(没有 matplotlib，跳过画图)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    say(f"\n写出 {outdir}/summary.txt、per_condition.csv、dense_log2fc.parquet、fig/dense_vs_s3c.png")
    return dict(per_condition=pc, r_sig=r_all)


if __name__ == "__main__":
    run_dense_check(**CONFIG)
