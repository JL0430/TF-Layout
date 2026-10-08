# scripts/tflayout/21_build_dense_target.py
"""
产出：
  out/head_b_dense_target.parquet   每个标签对一行：gene_id, tf_depleted(都是标签文件原文)、
                                    log2fc_dense(校准后，09 号默认读这一列)、log2fc_dense_raw、
                                    log2fc_wt(该基因 WT 3-IAA 响应)、log2fc_s3c、split
  out/results/_dense_target/summary.txt、per_tf_calibration.csv、fig/dense_target.png
"""
import os
import re

import numpy as np
import pandas as pd

CONFIG = dict(
    tpm_dir="data/tpm",
    dmso_file="DMSO_mean.txt",
    iaa_file="3IAA_mean.txt",
    dmso_rep_file="DMSO_only.txt",      # 重复级别(只用于第6条诊断)
    iaa_rep_file="3IAA_only.txt",
    labels="out/head_bc_labels.parquet",
    head_a="out/head_a_baseline_logtpm.parquet",  # 取 split 列(跟 09/16 号同源)
    layout="out/tf_layout.parquet",               # 取 TF 名，判断哪些条件进训练(跟 09 号同一规则)
    pseudocount=1.0,
    column_alias={},                    # 例如 {"HSF1": "HSF1_30"}；默认不猜，见文件头第2条
    calibration="per_tf",               # "per_tf" | "global" | "none"
    min_sig_per_tf=20,
    fc_threshold=1.3,
    replicate_check=True,
    out_path="out/head_b_dense_target.parquet",
    outdir="out/results/_dense_target",
)


def run_build_dense_target(tpm_dir="data/tpm", dmso_file="DMSO_mean.txt", iaa_file="3IAA_mean.txt",
                           dmso_rep_file="DMSO_only.txt", iaa_rep_file="3IAA_only.txt",
                           labels="out/head_bc_labels.parquet",
                           head_a="out/head_a_baseline_logtpm.parquet",
                           layout="out/tf_layout.parquet", pseudocount=1.0, column_alias=None,
                           calibration="per_tf", min_sig_per_tf=20, fc_threshold=1.3,
                           replicate_check=True, out_path="out/head_b_dense_target.parquet",
                           outdir="out/results/_dense_target"):
    """唯一入口，步骤见文件头。"""
    if calibration not in ("per_tf", "global", "none"):
        raise SystemExit(f"calibration 只能是 per_tf/global/none，收到 {calibration}")
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    alias = {str(k).upper(): str(v).upper() for k, v in (column_alias or {}).items()}
    drop_tok = {"DMSO", "3IAA", "IAA", "MEAN", "TPM", "AVG", "AVERAGE", "ONLY"}
    thr = float(np.log2(fc_threshold))
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def pearson(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
            return float("nan")
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    def read_table(path):
        for kw in (dict(sep="\t"), dict(sep=r"\s+", engine="python")):
            try:
                df_ = pd.read_csv(path, **kw)
            except Exception:  # noqa: BLE001 —— 换一种分隔符再试
                continue
            if df_.shape[1] >= 3:
                return df_
        return None

    def split_gene_col(df_):
        """返回 (以大写基因ID为索引、只剩数值列的表, 基因列名)。"""
        gc = None
        for c in df_.columns:
            if pd.to_numeric(df_[c], errors="coerce").isna().mean() > 0.5:
                gc = c
                break
        if gc is None:  # 表头比数据少一列：基因ID在 index 里
            df_ = df_.reset_index()
            gc = df_.columns[0]
        df_ = df_.set_index(df_[gc].astype(str).str.strip().str.upper())
        return df_.drop(columns=[gc]), gc

    def col_tokens(c):
        return [t for t in re.split(r"[_\-\s\.]+", str(c).strip().upper()) if t and t not in drop_tok]

    def match_cond(toks, dep_set):
        """最长前缀匹配(先查 alias)。返回 (条件名或None, 剩余的记号)。"""
        joined = "_".join(toks)
        if joined in alias and alias[joined] in dep_set:
            return alias[joined], []
        for n in range(len(toks), 0, -1):
            cand = "_".join(toks[:n])
            cand = alias.get(cand, cand)
            if cand in dep_set:
                return cand, toks[n:]
        return None, toks

    say("=" * 78)
    say("Head B 稠密 log2FC 辅助目标构建(21号，2026-09-25b)")
    say("=" * 78)

    # ---------------- 1. 标签、split、训练用的条件集合 ----------------
    lbl = pd.read_parquet(labels)
    lbl = lbl[["gene_id", "tf_depleted", "log2fc", "direction_3class"]].copy()
    lbl["_g"] = lbl["gene_id"].astype(str).str.strip().str.upper()
    lbl["_t"] = lbl["tf_depleted"].astype(str).str.strip().str.upper()
    dep_set = set(lbl["_t"])
    lay_tfs = set(pd.read_parquet(layout, columns=["tf"])["tf"].astype(str).str.strip().str.upper())
    train_conds = sorted(t for t in dep_set if t in lay_tfs)  # 跟 09 号\"以 layout 的 TF 名为准\"一致
    ha = pd.read_parquet(head_a)
    if "split" not in ha.columns:
        raise SystemExit(f"{head_a} 没有 split 列(08 号产出应该有)，没法只用训练基因拟合校准")
    split_of_gene = {str(k).strip().upper(): v for k, v in ha["split"].items()}
    lbl["split"] = lbl["_g"].map(split_of_gene)
    say(f"标签：{len(lbl)} 行、{len(dep_set)} 个条件；其中能对上 layout TF 名、会进训练的 "
        f"{len(train_conds)} 个(09号同一规则)；split 分布 "
        f"{dict(lbl.drop_duplicates('_g')['split'].value_counts(dropna=False))}(按基因计)")

    # ---------------- 2. 解析均值文件 ----------------
    mats, wt = {}, {}
    for k, fn in (("DMSO", dmso_file), ("3IAA", iaa_file)):
        p = os.path.join(tpm_dir, fn)
        if not os.path.exists(p):
            raise SystemExit(f"找不到 {p}")
        raw = read_table(p)
        if raw is None:
            raise SystemExit(f"{p} 用 tab/空白分隔都读不出≥3列")
        df, gc = split_gene_col(raw)
        mapping, unmatched, wt_col = {}, [], None
        for c in df.columns:
            toks = col_tokens(c)
            if toks == ["WT"]:
                wt_col = c
                continue
            hit, _ = match_cond(toks, dep_set)
            if hit is None:
                unmatched.append(str(c))
            elif hit in mapping.values():
                unmatched.append(f"{c}(重复映射到{hit})")
            else:
                mapping[c] = hit
        m = df[list(mapping)].apply(pd.to_numeric, errors="coerce")
        m.columns = [mapping[c] for c in m.columns]
        mats[k] = m
        if wt_col is not None:
            wt[k] = pd.to_numeric(df[wt_col], errors="coerce")
        say(f"[{k}] {p}：基因列={gc!r}，{df.shape[0]} 个基因；{len(mapping)} 列匹配到条件名"
            f"{'(含 alias)' if alias else ''}；WT 列={'有' if wt_col is not None else '没找到'}；"
            f"没匹配上 {len(unmatched)} 列{(': ' + str(unmatched[:12])) if unmatched else ''}")
    conds = sorted(set(mats["DMSO"].columns) & set(mats["3IAA"].columns))
    genes = mats["DMSO"].index.intersection(mats["3IAA"].index)
    if not conds or len(genes) == 0:
        raise SystemExit("两个文件没有共同的条件列或基因")
    d_ = mats["DMSO"].loc[genes, conds].to_numpy(np.float64)
    i_ = mats["3IAA"].loc[genes, conds].to_numpy(np.float64)
    dense = pd.DataFrame(np.log2((i_ + pseudocount) / (d_ + pseudocount)), index=genes, columns=conds)
    long = dense.stack().rename("log2fc_dense_raw").reset_index()
    long.columns = ["_g", "_t", "log2fc_dense_raw"]
    j = lbl.merge(long, on=["_g", "_t"], how="left")
    if len(j) != len(lbl):
        raise SystemExit("合并后行数变了(tpm 文件里有重复的基因ID？)")
    wt_lfc = None
    if "DMSO" in wt and "3IAA" in wt:
        wt_lfc = np.log2((wt["3IAA"].reindex(genes) + pseudocount) /
                         (wt["DMSO"].reindex(genes) + pseudocount))
        j["log2fc_wt"] = j["_g"].map(wt_lfc)
    else:
        j["log2fc_wt"] = np.nan
    tr_c = j["_t"].isin(train_conds)
    miss_tf = sorted(t for t in train_conds if t not in set(conds))
    cov_all = float(j["log2fc_dense_raw"].notna().mean())
    cov_tr = float(j.loc[tr_c, "log2fc_dense_raw"].notna().mean())
    say(f"覆盖率：全部标签对 {cov_all:.1%}；进训练的 {len(train_conds)} 个条件上 {cov_tr:.1%}；"
        f"这些条件里 tpm 文件完全没有的 {len(miss_tf)} 个: {miss_tf}"
        + ("  (需要的话在 CONFIG['column_alias'] 里补映射)" if miss_tf else ""))

    # ---------------- 3. 校准(只用训练基因的显著对) ----------------
    sig = j["log2fc"].notna() & j["log2fc_dense_raw"].notna()
    fit_m = sig & (j["split"] == "train")

    def ols(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        A = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(coef[0]), float(coef[1])

    if fit_m.sum() < 10:
        raise SystemExit(f"训练基因上只有 {int(fit_m.sum())} 个有稠密值的显著对，没法校准(split 对不上？)")
    a_g, b_g = ols(j.loc[fit_m, "log2fc_dense_raw"], j.loc[fit_m, "log2fc"])
    cal_rows = []
    coef_of = {}
    for t, g in j[fit_m].groupby("_t"):
        n_ = len(g)
        a_k, b_k = ols(g["log2fc_dense_raw"], g["log2fc"]) if n_ >= max(3, min_sig_per_tf) else (np.nan, np.nan)
        use_own = calibration == "per_tf" and np.isfinite(b_k) and b_k > 0
        coef_of[t] = (a_k, b_k) if use_own else (a_g, b_g)
        cal_rows.append(dict(tf_depleted=t, n_train_sig=n_, a=a_k, b=b_k, used=("own" if use_own else "global"),
                             r_train=pearson(g["log2fc_dense_raw"], g["log2fc"])))
    if calibration == "none":
        j["log2fc_dense"] = j["log2fc_dense_raw"]
    else:
        ab = np.array([coef_of.get(t, (a_g, b_g)) for t in j["_t"]], dtype=np.float64)
        if calibration == "global":
            ab[:] = (a_g, b_g)
        j["log2fc_dense"] = ab[:, 0] + ab[:, 1] * j["log2fc_dense_raw"].to_numpy(np.float64)
    cal = pd.DataFrame(cal_rows).sort_values("n_train_sig", ascending=False)
    own = cal[cal["used"] == "own"]
    say(f"\n[校准] 模式={calibration}；全局(全部训练显著对 n={int(fit_m.sum())})：S3c ≈ {a_g:+.3f} + "
        f"{b_g:.3f}×稠密值" + ("(只打印，none 模式不应用)" if calibration == "none" else ""))
    if calibration == "per_tf" and len(own):
        say(f"  用自己系数的条件 {len(own)} 个(训练显著对≥{min_sig_per_tf} 且 b>0)：b 中位 "
            f"{own['b'].median():.3f}、四分位 {own['b'].quantile(.25):.3f}~{own['b'].quantile(.75):.3f}；"
            f"a 中位 {own['a'].median():+.3f}、|a| 90%分位 {own['a'].abs().quantile(.9):.3f}；"
            f"其余 {len(cal) - len(own)} 个条件 + 训练集里没有显著对的条件用全局系数")
    for sp in ("val", "test"):
        m_ = sig & (j["split"] == sp)
        if m_.any():
            e_raw = j.loc[m_, "log2fc_dense_raw"] - j.loc[m_, "log2fc"]
            e_cal = j.loc[m_, "log2fc_dense"] - j.loc[m_, "log2fc"]
            say(f"  {sp:4s} 显著对(n={int(m_.sum())}，没参与拟合)：r 原始 "
                f"{pearson(j.loc[m_, 'log2fc_dense_raw'], j.loc[m_, 'log2fc']):.3f} -> 校准后 "
                f"{pearson(j.loc[m_, 'log2fc_dense'], j.loc[m_, 'log2fc']):.3f}；RMSE 原始 "
                f"{float(np.sqrt(np.mean(e_raw ** 2))):.3f} -> 校准后 {float(np.sqrt(np.mean(e_cal ** 2))):.3f}")

    # ---------------- 4. 不显著对的数值分布 + 监督量 ----------------
    ns = j["log2fc"].isna() & j["log2fc_dense"].notna()
    for col, nm in (("log2fc_dense_raw", "原始"), ("log2fc_dense", "校准后")):
        a_ = j.loc[ns & tr_c, col].abs()
        say(f"[不显著对，进训练的条件，{nm}] n={len(a_)}：|值| 中位 {a_.median():.3f}、90%分位 "
            f"{a_.quantile(.9):.3f}、99%分位 {a_.quantile(.99):.3f}；≥log2({fc_threshold})={thr:.3f} "
            f"的比例 {float((a_ >= thr).mean()):.3f}")
    trm = (j["split"] == "train") & tr_c
    n_gene_tr = j.loc[trm, "_g"].nunique()
    say(f"[监督量] 训练基因 {n_gene_tr} 个：每基因显著标签平均 "
        f"{j.loc[trm, 'log2fc'].notna().sum() / max(n_gene_tr, 1):.1f} 个 -> 稠密目标平均 "
        f"{j.loc[trm, 'log2fc_dense'].notna().sum() / max(n_gene_tr, 1):.1f} 个")

    # ---------------- 5. WT 通用 3-IAA 响应诊断 ----------------
    if wt_lfc is not None:
        m_ = sig & j["log2fc_wt"].notna()
        big = m_ & (j["log2fc_wt"].abs() >= thr)
        say(f"\n[WT 3-IAA 响应(诊断，见文件头第5条)] 基因的 WT log2FC：|值| 中位 "
            f"{float(np.nanmedian(np.abs(wt_lfc))):.3f}、≥{thr:.3f} 的基因 "
            f"{int((np.abs(wt_lfc) >= thr).sum())}/{int(np.isfinite(wt_lfc).sum())}")
        say(f"  显著对上 S3c 跟该基因 WT 响应的 Pearson r={pearson(j.loc[m_, 'log2fc'], j.loc[m_, 'log2fc_wt']):.3f}"
            f"(n={int(m_.sum())})；其中 WT 自己就变化≥阈值的 {int(big.sum())} 对，符号一致率 "
            f"{float(np.mean(np.sign(j.loc[big, 'log2fc']) == np.sign(j.loc[big, 'log2fc_wt']))) if big.any() else float('nan'):.3f}")
        gm = j[sig].groupby("_g").agg(n=("log2fc", "size"), mean_s3c=("log2fc", "mean"),
                                        wt=("log2fc_wt", "first"))
        gm = gm[gm["n"] >= 3]
        say(f"  基因级(≥3 个显著对的 {len(gm)} 个基因)：显著 log2FC 均值 vs WT 响应 r="
            f"{pearson(gm['mean_s3c'], gm['wt']):.3f}  ← 越高，说明\"基因自身爱不爱变\"里通用 3-IAA "
            "响应的成分越大(解读信心5/10)")
    else:
        say("\n[WT 3-IAA 响应] 两个文件里没都找到 WT 列，跳过")

    # ---------------- 6. 重复间一致性(可选，见文件头第6条) ----------------
    if replicate_check:
        try:
            gsm_re = re.compile(r"^GSM\d+$")

            def rep_tokens(c):
                """重复级别文件的列名前面多一段 GEO 样本号(如 GSM7586800_ABF1_DMSO_A_S1)，
                跟条件名匹配无关，均值文件没有这一段——先剥掉再走跟20/21号其余部分一样的
                最长前缀匹配(2026-09-25c 修复，见文件头第6条追记；真实数据暴露的问题，不是猜测)。"""
                toks = col_tokens(c)
                while toks and gsm_re.match(toks[0]):
                    toks = toks[1:]
                return toks

            reps = {}
            for k, fn in (("DMSO", dmso_rep_file), ("3IAA", iaa_rep_file)):
                p = os.path.join(tpm_dir, fn)
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                raw = read_table(p)
                if raw is None:
                    raise ValueError(f"{p} 读不出≥3列")
                df, _ = split_gene_col(raw)
                by_cond = {}
                for c in df.columns:
                    hit, rest = match_cond(rep_tokens(c), dep_set)
                    if hit is not None:
                        by_cond.setdefault(hit, []).append(("_".join(rest), c))
                reps[k] = (df, {t: [c for _, c in sorted(v)] for t, v in by_cond.items()})
                n2 = sum(1 for v in reps[k][1].values() if len(v) >= 2)
                say(f"\n[重复间一致性] {p}：{df.shape[1]} 列，认出 {len(reps[k][1])} 个条件、其中 {n2} 个有≥2个"
                    f"重复；列名样例 {list(map(str, df.columns[:6]))}")
            dd, dmap = reps["DMSO"]
            ii, imap = reps["3IAA"]
            gg = dd.index.intersection(ii.index)
            rr, rr_sig, n_used = [], [], 0
            lab_ns = set(zip(j.loc[j["log2fc"].isna(), "_g"], j.loc[j["log2fc"].isna(), "_t"]))
            lab_sig = set(zip(j.loc[j["log2fc"].notna(), "_g"], j.loc[j["log2fc"].notna(), "_t"]))
            for t in sorted(set(dmap) & set(imap)):
                if len(dmap[t]) < 2 or len(imap[t]) < 2:
                    continue
                x = [np.log2((pd.to_numeric(ii.loc[gg, imap[t][r]], errors="coerce") + pseudocount) /
                             (pd.to_numeric(dd.loc[gg, dmap[t][r]], errors="coerce") + pseudocount))
                     for r in (0, 1)]
                is_ns = np.array([(g_, t) in lab_ns for g_ in gg])
                is_sg = np.array([(g_, t) in lab_sig for g_ in gg])
                if is_ns.sum() >= 50:
                    rr.append(pearson(x[0][is_ns], x[1][is_ns]))
                if is_sg.sum() >= 20:
                    rr_sig.append(pearson(x[0][is_sg], x[1][is_sg]))
                n_used += 1
            if not n_used:
                raise ValueError("没有哪个条件在两个文件里都认出≥2个重复")
            say(f"  用了 {n_used} 个条件(各取前两个重复)：不显著对上两份 log2FC 的 split-half r 中位 "
                f"{np.nanmedian(rr):.3f}(四分位 {np.nanpercentile(rr, 25):.3f}~{np.nanpercentile(rr, 75):.3f})；"
                f"显著对上 {np.nanmedian(rr_sig) if rr_sig else float('nan'):.3f}  "
                "← 不显著对上明显>0 说明亚阈值的稠密值里有可重复的信号，接近0 说明它们主要只是告诉模型\"≈0\"")
        except Exception as e:  # noqa: BLE001 —— 可选诊断，失败不影响主产出
            say(f"\n[重复间一致性] 跳过({type(e).__name__}: {e})——重复级别文件的列名格式我是猜的(4/10)，"
                "把上面打印的列名样例贴回来即可")

    # ---------------- 7. 写文件 ----------------
    out = j[["gene_id", "tf_depleted", "log2fc_dense", "log2fc_dense_raw", "log2fc_wt", "split"]].copy()
    out["log2fc_s3c"] = j["log2fc"].to_numpy()
    out["in_training_tfs"] = tr_c.to_numpy()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_parquet(out_path, index=False)
    cal.to_csv(os.path.join(outdir, "per_tf_calibration.csv"), index=False)
    say(f"\n写出 {out_path}({len(out)} 行；09 号 set_dense_target 默认读 log2fc_dense 列)、"
        f"{outdir}/per_tf_calibration.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        mv = sig & j["split"].isin(["val", "test"])
        ax[0].scatter(j.loc[mv, "log2fc"], j.loc[mv, "log2fc_dense"], s=3, alpha=.3)
        ax[0].axline((0, 0), slope=1, color="k", lw=.6)
        ax[0].set_xlabel("S3c log2FC (val/test significant pairs)")
        ax[0].set_ylabel("calibrated dense log2FC")
        ax[0].set_title(f"r={pearson(j.loc[mv, 'log2fc'], j.loc[mv, 'log2fc_dense']):.3f} (not used in fit)")
        bins = np.linspace(-2, 2, 121)
        ax[1].hist(j.loc[ns & tr_c, "log2fc_dense"].clip(-2, 2), bins=bins, density=True, alpha=.7)
        for t_ in (-thr, thr):
            ax[1].axvline(t_, color="k", lw=.6, ls="--")
        ax[1].set_title("calibrated dense log2FC, non-significant pairs")
        if len(own):
            ax[2].hist(own["b"].clip(0, 3), bins=40)
            ax[2].axvline(b_g, color="r", lw=.8)
            ax[2].set_title("per-condition slope b_k (red = global)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig", "dense_target.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        say("(没有 matplotlib，跳过画图)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    say(f"把 {outdir}/summary.txt 贴回来即可")
    return dict(table=out, calibration=cal)


if __name__ == "__main__":
    run_build_dense_target(**CONFIG)
