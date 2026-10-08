# scripts/tflayout/19_compare_runs.py

import glob
import json
import os
import time

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 配置：全部写死在这里
# ------------------------------------------------------------------------------
CONFIG = dict(
    results_root="out/results",
    # 2026-10-01a 第11批：显式两个实验(见文件头第11批第1条；第10批是 "auto")
    runs=("v8_headA_all", "v10_abl_no_knockout"),  # "auto"=results_root 下所有有 predictions_test.parquet 的实验
    # 2026-09-30a 第10批：参照=新主线 v8(第9批是 ("v3_ce_marker_dense", "v2_ce"))
    references=("v8_headA_all",),
    # 2026-09-26a：runs="auto" 时不参与(是参照时例外)；2026-09-28b 加上已结案的实验；2026-09-30a 再加 v2_ce/主线 v3/v4 no_cis
    # (第9批的 skip_runs 是下面这串去掉最后三个)
    skip_runs=("run1", "v2_full", "v2_ctx_marker", "v3_ce_dense", "v3_ce_marker", "v4_abl_no_knockout",
               "v4_abl_no_layout", "v4_hcgene", "v5_abl_no_position", "v5_abl_shuffle_position",
               "v6_abl_shift_position", "v2_ce", "v3_ce_marker_dense", "v4_abl_no_cis"),
    headline_runs="auto",   # 2026-09-26a：[4] 段用全部 seed 报告的实验："auto"=seed 数多于共有 seed 的；"all"；或名字列表
    verdict_variants=("perhead", "selected"),  # [5] 判读表覆盖的权重套数
    noninf_margin=dict(A_r_gene=0.03, B_r=0.02, B_r_within_tf=0.02, B_sign_acc=0.02,
                       C_auroc_down=0.01, C_auroc_up=0.01, C_auprc_down=0.02, C_auprc_up=0.02,
                       C_macro_f1_offset=0.01),  # 非劣阈值(建议值，信心5/10)
    variants=("selected", "perhead"),  # 各实验有哪套就比哪套
    seeds="common",         # "common"=所有参与比较的实验共有的 seed；或显式 (42, 123)
    offset_grid=(-4.0, 4.0, 0.25),
    n_boot=500,             # 配对整群 bootstrap 次数(见文件头2026-09-25第3条)
    stack=True,             # 有先验列时另算"+stack"变体(见文件头2026-09-25第2条)
    outdir="out/results/_compare_b11",  # 2026-10-01a 第11批(第10批是 _compare_b10，第9批是 _compare，都不覆盖)
    dense_target="out/head_b_dense_target.parquet",  # 2026-09-25b：21号产出，不存在就不算 B_dr/B_drns
    print_variants=("selected", "perhead"),  # [3] 段只打印这几套(+stack 仍写进 csv)
    match_variant=True,     # 2026-09-26b：每套权重只跟参照的同一套比(False=旧行为：都跟参照的 selected 比)
    pairwise_extra=True,    # 2026-09-26b：[3b] 两两共有 seed 多于全局共有 seed 时另算一组差值
    strata=True,            # 2026-09-26b：[6] 按 D∈L_g / D∉L_g 分层的差值(需要 17 号导出的 D_in_Lg 列)
    strata_runs="auto",     # "auto"=名字里带 "_abl_" 的实验；或名字列表
    strata_variant="perhead",  # [6] 用哪套权重(参照也用同一套)
    # 2026-09-27a：[3b] 另外指定的(基准, 实验)配对，见文件头第6批
    # 第6批的三对(位置消融两两)已结案；2026-09-28b 换成 no_cis 对 v8(见文件头第8批)
    # 2026-09-29a 第9批：v8 补到 5 seed 后，[3b] 本来就会自动给出"v8 对主线"的 5 对 5(两两共有 seed 5 > 全局共有 2，
    # 全局被只有 2 个 seed 的 v2_ce 卡住)；这里再显式列一次当保险(代码里对重复配对有去重，不会打印两遍)。
    # 2026-09-30a 第10批：清空(第9批是 (("v4_abl_no_cis","v8_headA_all"), ("v3_ce_marker_dense","v8_headA_all")))
    extra_pairs=(),
)


def run_compare(results_root="out/results", runs="auto", references=("v3_ce_marker_dense", "v2_ce"),
                variants=("selected", "perhead"), seeds="common", offset_grid=(-4.0, 4.0, 0.25),
                n_boot=500, stack=True, outdir="out/results/_compare", reference=None,
                dense_target="out/head_b_dense_target.parquet",
                print_variants=("selected", "perhead"), skip_runs=("run1", "v2_full", "v2_ctx_marker"),
                headline_runs="auto", verdict_variants=("perhead", "selected"),
                noninf_margin=None, match_variant=True, pairwise_extra=True, strata=True,
                strata_runs="auto", strata_variant="perhead", extra_pairs=()):
    """唯一入口。reference=单个名字(旧写法)时等价于 references=[reference]。"""
    t0 = time.time()
    noninf_margin = noninf_margin or dict(A_r_gene=0.03, B_r=0.02, B_r_within_tf=0.02, B_sign_acc=0.02,
                                          C_auroc_down=0.01, C_auroc_up=0.01, C_auprc_down=0.02,
                                          C_auprc_up=0.02, C_macro_f1_offset=0.01)
    if reference is not None:
        references = [reference]
    references = [r for r in (references if not isinstance(references, str) else [references])]
    os.makedirs(outdir, exist_ok=True)
    lines = []
    cls = {"down": 0, "ns": 1, "up": 2}

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def pearson(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def auroc_auprc(score, pos):
        pos = np.asarray(pos, bool)
        n1 = int(pos.sum())
        n0 = len(pos) - n1
        if n1 == 0 or n0 == 0:
            return float("nan"), float("nan")
        sc = np.asarray(score, np.float64)
        _, inv_, cnt_ = np.unique(sc, return_inverse=True, return_counts=True)
        avg_rank = np.cumsum(cnt_) - (cnt_ - 1) / 2.0  # 跟 pandas rank(average) 相同
        r = avg_rank[inv_.reshape(-1)]
        auc = float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
        p = pos[np.argsort(-np.asarray(score, np.float64), kind="mergesort")]
        prec = np.cumsum(p) / np.arange(1, len(p) + 1)
        return auc, float(prec[p].sum() / n1)

    def macro_f1(pred, true):
        cm = np.bincount(true * 3 + pred, minlength=9).reshape(3, 3)
        f1 = []
        for c in range(3):
            tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
            f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return float(np.mean(f1))

    lo_, hi_, st_ = offset_grid
    grid = np.arange(lo_, hi_ + 1e-9, st_)

    def tune(prob_val, yv):
        lp = np.log(np.clip(prob_val, 1e-12, 1.0))
        best = (-1.0, 0.0, 0.0)
        for od in grid:
            for ou in grid:
                f = macro_f1((lp + np.array([od, 0.0, ou])).argmax(1), yv)
                if f > best[0] + 1e-12:
                    best = (f, float(od), float(ou))
        return best

    def apply(prob, best):
        return (np.log(np.clip(prob, 1e-12, 1.0)) + np.array([best[1], 0.0, best[2]])).argmax(1)

    # ---- 1. 发现实验、读预测 ----
    if runs == "auto":
        runs = sorted(os.path.basename(os.path.dirname(p)) for p in
                      glob.glob(os.path.join(results_root, "*", "predictions_test.parquet"))
                      if not os.path.basename(os.path.dirname(p)).startswith("_"))
        runs = [r for r in runs if r not in skip_runs or r in references]  # 2026-09-26a
    references = [r for r in references if r in runs]
    if not references:
        raise SystemExit(f"CONFIG['references'] 里的实验都没有导出结果(现有: {runs})")
    reference = references[0]  # 对齐样本顺序、取标签用第一个参照
    runs = references + [r for r in runs if r not in references]
    say("=" * 78)
    say(f"实验对比  {time.strftime('%Y-%m-%d %H:%M:%S')}  参照={references}  实验={runs}")
    say("=" * 78)
    data = {}
    for r in runs:
        d = os.path.join(results_root, r)
        pv, pt = os.path.join(d, "predictions_val.parquet"), os.path.join(d, "predictions_test.parquet")
        if not (os.path.exists(pv) and os.path.exists(pt)):
            say(f"⚠ {r}: 缺 predictions_val/test.parquet，跳过")
            continue
        dv, dt = pd.read_parquet(pv), pd.read_parquet(pt)
        pg_path = os.path.join(d, "per_gene_head_a.csv")
        pg = pd.read_csv(pg_path) if os.path.exists(pg_path) else None
        sd = {}
        for var, suf in (("selected", ""), ("perhead", "_ph")):
            ss = sorted(int(c[len("y_b_seed"):-len(suf)] if suf else c[len("y_b_seed"):])
                        for c in dt.columns if c.startswith("y_b_seed")
                        and (c.endswith(suf) if suf else not c.endswith("_ph")))
            if ss and var in variants:
                sd[var] = ss
        # 老版 17 号(run1)的 predictions 没有 y_a_seed 列：从 per_gene_head_a.csv 补
        for var, suf in (("selected", ""), ("perhead", "_ph")):
            for s in sd.get(var, []):
                col = f"y_a_seed{s}{suf}"
                for df_, sp in ((dv, "val"), (dt, "test")):
                    if col not in df_.columns and pg is not None and col in pg.columns:
                        m = pg[pg["split"] == sp].set_index("gene_id")[col]
                        df_[col] = df_["gene_id"].map(m).to_numpy()
        data[r] = dict(val=dv, test=dt, seeds=sd)
        say(f"  {r}: val {len(dv)} 条 / test {len(dt)} 条；seed: "
            + "；".join(f"{v}={s}" for v, s in sd.items()))
    if reference not in data:
        raise SystemExit(f"参照实验 {reference} 没有可用的导出结果(先跑 17 号)")

    # 对齐：所有实验按 (gene_id, tf_depleted) 用参照实验的行顺序
    key_ref = {sp: data[reference][sp][["gene_id", "tf_depleted"]] for sp in ("val", "test")}
    for r in list(data):
        for sp in ("val", "test"):
            df_ = data[r][sp].set_index(["gene_id", "tf_depleted"])
            k_ = pd.MultiIndex.from_frame(key_ref[sp])
            if len(df_) != len(k_) or not k_.isin(df_.index).all():
                say(f"⚠ {r}: {sp} 的样本集合跟参照实验不同(数据文件变过？)，这个实验不参与对比")
                data.pop(r)
                break
            data[r][sp] = df_.loc[k_].reset_index()
    if seeds == "common":
        sets = [set(s) for r in data for s in data[r]["seeds"].values()]
        seeds = sorted(set.intersection(*sets)) if sets else []
    seeds = [int(s) for s in seeds]
    for r in data:
        for var, ss in data[r]["seeds"].items():
            miss = sorted(set(seeds) - set(ss))
            if miss:
                raise SystemExit(f"{r}/{var} 缺 seed {miss}，没法在这组 seed 上比较；改 CONFIG['seeds']")
    if not seeds:
        raise SystemExit("参与比较的实验没有共有的 seed")
    say(f"比较用的 seed(各实验都在这组 seed 上重算集成)：{seeds}")

    ref_t = data[reference]["test"]
    yv = data[reference]["val"]["y_c_true"].map(cls).to_numpy().astype(np.int64)
    yt = ref_t["y_c_true"].map(cls).to_numpy().astype(np.int64)
    yb_t = ref_t["y_b_true"].to_numpy(np.float64)
    ya_t = ref_t["y_a_true"].to_numpy(np.float64)
    genes_t = ref_t["gene_id"].to_numpy()
    tfs_t = ref_t["tf_depleted"].to_numpy()
    ybd_t = None  # 2026-09-25b：稠密 log2FC(见文件头第3批第1条)
    if dense_target and os.path.exists(dense_target):
        dd_ = (pd.read_csv(dense_target) if str(dense_target).endswith((".csv", ".csv.gz"))
               else pd.read_parquet(dense_target, columns=["gene_id", "tf_depleted", "log2fc_dense"]))
        dd_ = pd.DataFrame({"_g": dd_["gene_id"].astype(str).str.strip().str.upper(),
                            "_t": dd_["tf_depleted"].astype(str).str.strip().str.upper(),
                            "_v": pd.to_numeric(dd_["log2fc_dense"], errors="coerce")}
                           ).drop_duplicates(subset=["_g", "_t"])
        kk_ = pd.DataFrame({"_g": pd.Series(genes_t).astype(str).str.strip().str.upper(),
                            "_t": pd.Series(tfs_t).astype(str).str.strip().str.upper()})
        ybd_t = kk_.merge(dd_, on=["_g", "_t"], how="left")["_v"].to_numpy(np.float64)
        say(f"稠密 log2FC：{dense_target} -> test {int(np.isfinite(ybd_t).sum())}/{len(ybd_t)} 条有值"
            f"(不显著样本里 {np.isfinite(ybd_t[~np.isfinite(ref_t['y_b_true'].to_numpy(np.float64))]).mean():.1%})")
    elif dense_target:
        say(f"提示：没找到 {dense_target}，不算 B_dr/B_drns(先跑 21 号)")
    tf_code = pd.factorize(tfs_t)[0]
    n_tf_codes = int(tf_code.max()) + 1 if len(tf_code) else 0
    has_prior = all(c in data[reference][sp].columns for sp in ("val", "test")
                    for c in ("base_TFxbound_p_down", "base_TFxbound_p_up"))
    if stack and not has_prior:
        say(f"提示：{reference} 的导出结果里没有先验概率列(老格式)，跳过 +stack 变体——"
            "重新跑一次 17 号(老格式会自动重导)即可")
    prior = {}
    if has_prior:
        for sp in ("val", "test"):
            q = data[reference][sp]
            qd, qu = q["base_TFxbound_p_down"].to_numpy(np.float64), q["base_TFxbound_p_up"].to_numpy(np.float64)
            prior[sp] = np.column_stack([qd, np.clip(1.0 - qd - qu, 1e-12, 1.0), qu])
    _, first_gene = np.unique(genes_t, return_index=True)
    uniq, inv = np.unique(genes_t, return_inverse=True)
    rows_of_gene = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])

    def probs_of(df_, s, suf):
        pdn = df_[f"p_down_seed{s}{suf}"].to_numpy(np.float64)
        pup = df_[f"p_up_seed{s}{suf}"].to_numpy(np.float64)
        return np.column_stack([pdn, np.clip(1.0 - pdn - pup, 1e-12, 1.0), pup])

    def metrics(ya, yb, pr_t, pred_off, rows=None, first=None, heads="ABC"):
        """rows=None：全 test；否则是 bootstrap 抽出的行下标(first 是抽中基因的首行)。
        heads="C"：只算 Head C(堆叠变体用，A/B 跟不堆叠那行相同)。"""
        if rows is None:
            rows, first = slice(None), first_gene
        out = {}
        if "A" in heads:
            out["A_r_gene"] = pearson(ya[first], ya_t[first])
        if "B" in heads:
            ybt, ybp, tfc = yb_t[rows], yb[rows], tf_code[rows]
            sg = np.isfinite(ybt)
            p_, t_, c_ = ybp[sg], ybt[sg], tfc[sg]
            out["B_r"] = pearson(p_, t_)
            out["B_sign_acc"] = float(np.mean(np.sign(p_) == np.sign(t_))) if len(p_) else float("nan")
            cnt = np.maximum(np.bincount(c_, minlength=n_tf_codes), 1)
            mp = np.bincount(c_, weights=p_, minlength=n_tf_codes) / cnt
            mt = np.bincount(c_, weights=t_, minlength=n_tf_codes) / cnt
            out["B_r_within_tf"] = pearson(p_ - mp[c_], t_ - mt[c_])
            if ybd_t is not None:  # 2026-09-25b
                ybd_r, ybp_all = ybd_t[rows], yb[rows]
                okd = np.isfinite(ybd_r)
                out["B_dr"] = pearson(ybp_all[okd], ybd_r[okd])
                okn = okd & ~sg
                out["B_dr_ns"] = pearson(ybp_all[okn], ybd_r[okn])
        yct, prt = yt[rows], pr_t[rows]
        out["C_auroc_down"], out["C_auprc_down"] = auroc_auprc(prt[:, 0], yct == 0)
        out["C_auroc_up"], out["C_auprc_up"] = auroc_auprc(prt[:, 2], yct == 2)
        out["C_macro_f1_offset"] = macro_f1(pred_off[rows], yct)
        return out

    def stack_feats(pm, pq):
        lm, lq = np.log(np.clip(pm, 1e-12, 1.0)), np.log(np.clip(pq, 1e-12, 1.0))
        return np.column_stack([lm[:, 0] - lm[:, 1], lm[:, 2] - lm[:, 1],
                                lq[:, 0] - lq[:, 1], lq[:, 2] - lq[:, 1]])

    def fit_apply_stack(pm_v, pm_t, lam=1e-3, n_iter=50):
        """跟 17 号 fit_stack/apply_stack 同一做法：val 上 Newton 法拟合多类 logistic(ns=参照)，
        返回 (val 堆叠概率, test 堆叠概率)。"""
        X = stack_feats(pm_v, prior["val"])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        Y = np.column_stack([yv == 0, yv == 2]).astype(np.float64)
        d = Z.shape[1]
        W = np.zeros((2, d))
        for _ in range(n_iter):
            eta = Z @ W.T
            m = np.maximum(eta.max(1, keepdims=True), 0.0)
            ex = np.exp(eta - m)
            P = ex / (np.exp(-m) + ex.sum(1, keepdims=True))
            G = ((Y - P).T @ Z - lam * W).reshape(-1)
            H = np.zeros((2 * d, 2 * d))
            for a_ in range(2):
                for b_ in range(2):
                    wab = P[:, a_] * ((a_ == b_) - P[:, b_])
                    H[a_ * d:(a_ + 1) * d, b_ * d:(b_ + 1) * d] = (Z * wab[:, None]).T @ Z
            step = np.linalg.solve(H + lam * np.eye(2 * d), G)
            W += step.reshape(2, d)
            if np.abs(step).max() < 1e-8:
                break
        out = []
        for pm, pq in ((pm_v, prior["val"]), (pm_t, prior["test"])):
            Z_ = np.column_stack([np.ones(len(pm)), (stack_feats(pm, pq) - mu) / sd])
            eta = Z_ @ W.T
            lg = np.column_stack([eta[:, 0], np.zeros(len(eta)), eta[:, 1]])
            lg = lg - lg.max(1, keepdims=True)
            out.append(np.exp(lg) / np.exp(lg).sum(1, keepdims=True))
        return out[0], out[1]

    # ---- 2. 逐实验、逐套、逐 seed + 集成 ----
    def ens_for(r, var, ss):
        """2026-09-26b：在 seed 集合 ss 上给实验 r 的某套权重算逐 seed 指标 + 集成(第2段和 [3b] 共用，逻辑跟
        原来第2段逐行相同)。返回 (ya_e, yb_e, pt_e, off_e, per_seed, pv_e, b_e)。"""
        suf = "_ph" if var == "perhead" else ""
        dv_, dt_ = data[r]["val"], data[r]["test"]
        per_seed = {}
        for s in ss:
            pr_v, pr_t = probs_of(dv_, s, suf), probs_of(dt_, s, suf)
            b = tune(pr_v, yv)
            per_seed[s] = metrics(dt_[f"y_a_seed{s}{suf}"].to_numpy(np.float64),
                                  dt_[f"y_b_seed{s}{suf}"].to_numpy(np.float64), pr_t,
                                  apply(pr_t, b))
        ya_e = np.mean([dt_[f"y_a_seed{s}{suf}"].to_numpy(np.float64) for s in ss], 0)
        yb_e = np.mean([dt_[f"y_b_seed{s}{suf}"].to_numpy(np.float64) for s in ss], 0)
        pv_e = np.mean([probs_of(dv_, s, suf) for s in ss], 0)
        pt_e = np.mean([probs_of(dt_, s, suf) for s in ss], 0)
        b_e = tune(pv_e, yv)
        off_e = apply(pt_e, b_e)
        return ya_e, yb_e, pt_e, off_e, per_seed, pv_e, b_e

    res_rows, ens_cache = [], {}
    for r in data:
        for var, suf in (("selected", ""), ("perhead", "_ph")):
            if var not in data[r]["seeds"]:
                continue
            dv_, dt_ = data[r]["val"], data[r]["test"]
            ya_e, yb_e, pt_e, off_e, per_seed, pv_e, b_e = ens_for(r, var, seeds)
            for s in seeds:
                res_rows += [dict(run=r, variant=var, scope=f"seed{s}", metric=k, value=v)
                             for k, v in per_seed[s].items()]
            m_e = metrics(ya_e, yb_e, pt_e, off_e)
            ens_cache[(r, var)] = (ya_e, yb_e, pt_e, off_e, per_seed)
            res_rows += [dict(run=r, variant=var, scope=f"ensemble{len(seeds)}", metric=k, value=v)
                         for k, v in m_e.items()]
            res_rows.append(dict(run=r, variant=var, scope=f"ensemble{len(seeds)}",
                                 metric="C_offset_down_up", value=f"{b_e[1]:+.2f}/{b_e[2]:+.2f}"))
            if stack and has_prior:  # 2026-09-25：模型集成 + TF×结合先验 堆叠(只看 Head C)
                sv, stt = fit_apply_stack(pv_e, pt_e)
                b_s = tune(sv, yv)
                off_s = apply(stt, b_s)
                ps_s = {}
                for s in seeds:  # 逐 seed 也各自堆叠，给"逐seed配对差"用
                    sv1, st1 = fit_apply_stack(probs_of(dv_, s, suf), probs_of(dt_, s, suf))
                    ps_s[s] = metrics(None, None, st1, apply(st1, tune(sv1, yv)), heads="C")
                m_s = metrics(None, None, stt, off_s, heads="C")
                ens_cache[(r, var + "+stack")] = (None, None, stt, off_s, ps_s)
                res_rows += [dict(run=r, variant=var + "+stack", scope=f"ensemble{len(seeds)}",
                                  metric=k, value=v) for k, v in m_s.items()]
    res = pd.DataFrame(res_rows)
    res.to_csv(os.path.join(outdir, "comparison_long.csv"), index=False)

    # ---- 3. 相对参照实验的差值：配对整群 bootstrap(每个参照各一组) ----
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(uniq), size=len(uniq)) for _ in range(int(n_boot))]
    boot_rows = [(np.concatenate([rows_of_gene[c] for c in ch]), first_gene[ch]) for ch in boots]
    boot_cache = {}

    def boot_metrics(key):
        if key not in boot_cache:
            ya_x, yb_x, pt_x, off_x, _ = ens_cache[key]
            hd = "C" if key[1].endswith("+stack") else "ABC"
            boot_cache[key] = [metrics(ya_x, yb_x, pt_x, off_x, rw, fg, heads=hd)
                               for rw, fg in boot_rows]
        return boot_cache[key]

    delta_rows = []
    # 2026-09-26b：match_variant=True 时参照的每一套权重都当基准、只跟同名那套比(见文件头第5批第1条)；
    # False 时是旧行为(参照只取 selected / selected+stack，别的实验每一套都跟它比)
    ref_vars_all = (("selected", "perhead", "selected+stack", "perhead+stack") if match_variant
                    else ("selected", "selected+stack"))
    for ref in references:
        for ref_var in ref_vars_all:
            ref_key = (ref, ref_var)
            if ref_key not in ens_cache:
                continue
            ref_boot = boot_metrics(ref_key)
            ya_r, yb_r, pt_r, off_r, ps_r = ens_cache[ref_key]
            ref_full = metrics(ya_r, yb_r, pt_r, off_r, heads="C" if ref_var.endswith("+stack") else "ABC")
            for key in ens_cache:
                if key[0] in references[:references.index(ref)]:
                    continue  # 前面的参照已经跟这个参照比过了(反向差值只是相反数)
                if key == ref_key or key[0] == ref or key[1].endswith("+stack") != ref_var.endswith("+stack"):
                    continue  # 堆叠变体只跟堆叠的参照比，不堆叠的只跟不堆叠的比
                if match_variant and key[1] != ref_var:
                    continue  # 2026-09-26b：同口径(selected↔selected、perhead↔perhead……)
                ya_x, yb_x, pt_x, off_x, ps_x = ens_cache[key]
                hd = "C" if key[1].endswith("+stack") else "ABC"
                full_x = metrics(ya_x, yb_x, pt_x, off_x, heads=hd)
                xb = boot_metrics(key)
                for k in full_x:
                    arr = np.asarray([mb[k] - rb[k] for mb, rb in zip(xb, ref_boot)], np.float64)
                    arr = arr[np.isfinite(arr)]
                    paired = [ps_x[s][k] - ps_r[s][k] for s in seeds]
                    delta_rows.append(dict(reference=f"{ref}/{ref_var}", run=key[0], variant=key[1],
                                           metric=k, ref=ref_full[k], value=full_x[k],
                                           delta=full_x[k] - ref_full[k],
                                           ci_lo=float(np.quantile(arr, .025)) if len(arr) else np.nan,
                                           ci_hi=float(np.quantile(arr, .975)) if len(arr) else np.nan,
                                           paired_seed_delta_mean=float(np.nanmean(paired)),
                                           n_seeds_better=int(sum(1 for x in paired if x > 0))))
    dlt = pd.DataFrame(delta_rows)
    dlt.to_csv(os.path.join(outdir, "delta_vs_reference.csv"), index=False)

    # ---- 4. summary ----
    short = dict(A_r_gene="A_r", B_r="B_r", B_sign_acc="B_sign", B_r_within_tf="B_r|TF",
                 C_auroc_down="C_AUCdn", C_auroc_up="C_AUCup", C_auprc_down="C_APdn",
                 C_auprc_up="C_APup", C_macro_f1_offset="C_mF1")
    if ybd_t is not None:  # 2026-09-25b
        short.update(B_dr="B_dr", B_dr_ns="B_drns")
    say(f"\n[1] 共有 seed {seeds} 上的集成(ensemble{len(seeds)})，C_mF1=在 val 上调偏置后的 test macro-F1")
    say("      " + "run/variant".ljust(28) + "".join(v.rjust(9) for v in short.values()))
    ens_scope = f"ensemble{len(seeds)}"
    for (r, var) in ens_cache:
        sub = res[(res["run"] == r) & (res["variant"] == var) & (res["scope"] == ens_scope)]
        mv = dict(zip(sub["metric"], sub["value"]))
        say("      " + f"{r}/{var}".ljust(28) + "".join(
            (f"{mv[k]:.3f}" if isinstance(mv.get(k), float) and np.isfinite(mv[k]) else "nan")
            .rjust(9) for k in short))
    say("      (+stack 行=模型集成与 TF×结合先验在 val 上堆叠，只算 Head C；A/B 列为 nan)"
        + ("\n      (B_dr/B_drns=Head B 预测 vs 21号稠密log2FC：全部有值样本/只看不显著样本)"
           if ybd_t is not None else ""))
    say(f"\n[2] 逐 seed 均值±std(同一组 seed)")
    for (r, var), (_, _, _, _, ps) in ens_cache.items():
        say(f"  {r}/{var}: " + "  ".join(
            f"{short[k]}={np.nanmean([ps[s][k] for s in seeds]):.3f}±"
            f"{np.nanstd([ps[s][k] for s in seeds]):.3f}" for k in short if k in ps[seeds[0]]))
    say(f"\n[3] 差值(集成对集成；只打印 {list(print_variants) if print_variants else '全部'} 这几套，全部差值见 "
        f"delta_vs_reference.csv)；95% CI=配对按基因整群 bootstrap({n_boot}次)；逐seed配对差=同一 seed 两个"
        f"实验相减再平均，括号里是 {len(seeds)} 个 seed 中变好的个数。CI 不含0才算超出 test 抽样噪声"
        "(只反映 test 基因抽样的不确定性，不含训练随机性——只有2个seed时训练随机性要看逐seed配对差)")
    for ref_name in dlt["reference"].unique() if len(dlt) else []:
        dr = dlt[dlt["reference"] == ref_name]
        to_print = [k for k in ens_cache if (k[0], k[1]) in set(zip(dr["run"], dr["variant"]))
                    and (not print_variants or k[1] in print_variants)]
        if not to_print:
            continue  # 2026-09-26b：这个参照(比如 +stack)下没有要打印的套数，不打空标题
        say(f"  ── 参照 {ref_name} ──")
        for (r, var) in to_print:
            sub = dr[(dr["run"] == r) & (dr["variant"] == var)]
            say(f"  {r}/{var}:")
            for row in sub.itertuples():
                flag = " ↑" if row.ci_lo > 0 else (" ↓" if row.ci_hi < 0 else "")
                say(f"      {short.get(row.metric, row.metric):8s} {row.ref:.4f} -> {row.value:.4f}  Δ={row.delta:+.4f}  "
                    f"CI=({row.ci_lo:+.4f}, {row.ci_hi:+.4f}){flag}  逐seed配对差 "
                    f"{row.paired_seed_delta_mean:+.4f}({row.n_seeds_better}/{len(seeds)})")
    # ---- [3b] 两两共有 seed 多于全局共有 seed 的配对(2026-09-26b，见文件头第5批第2条) ----
    pair_rows = []
    if pairwise_extra:
        pair_jobs = []
        for ref in references:
            for r in data:
                if r == ref or r in references[:references.index(ref)]:
                    continue
                for var in ("selected", "perhead"):
                    if var not in data[ref]["seeds"] or var not in data[r]["seeds"]:
                        continue
                    ss = tuple(sorted(set(data[ref]["seeds"][var]) & set(data[r]["seeds"][var])))
                    if len(ss) > len(seeds):
                        pair_jobs.append((ref, r, var, ss))
        for ref, r in (extra_pairs or ()):  # 2026-09-27a：指定配对(基准不必是参照)
            if ref not in data or r not in data:
                say(f"提示：extra_pairs 里的 {ref} -> {r} 缺导出结果(或被 skip_runs 排除)，跳过")
                continue
            for var in ("selected", "perhead"):
                if var not in data[ref]["seeds"] or var not in data[r]["seeds"]:
                    continue
                ss = tuple(sorted(set(data[ref]["seeds"][var]) & set(data[r]["seeds"][var])))
                if ss and (ref, r, var, ss) not in pair_jobs:
                    pair_jobs.append((ref, r, var, ss))
        ens2 = {}

        def ens_boot(r, var, ss):
            k = (r, var, ss)
            if k not in ens2:
                ya_e, yb_e, pt_e, off_e, ps, _, _ = ens_for(r, var, list(ss))
                ens2[k] = (metrics(ya_e, yb_e, pt_e, off_e),
                           [metrics(ya_e, yb_e, pt_e, off_e, rw, fg) for rw, fg in boot_rows], ps)
            return ens2[k]

        if pair_jobs:
            say(f"\n[3b] 两两共有 seed 多于全局共有 seed {seeds} 的配对 + CONFIG['extra_pairs'] 指定的配对：在这两个实验自己的"
                "共有 seed 上重算集成再比较"
                f"(同口径；95% CI=配对按基因整群 bootstrap {n_boot} 次；逐seed配对差括号里=变好的 seed 数)")
        for ref, r, var, ss in pair_jobs:
            full_r, bt_r, ps_r = ens_boot(ref, var, ss)
            full_x, bt_x, ps_x = ens_boot(r, var, ss)
            say(f"  {r}/{var} 对 {ref}/{var}  ({len(ss)} 个 seed {list(ss)})")
            for k in full_x:
                arr = np.asarray([bx[k] - br[k] for bx, br in zip(bt_x, bt_r)], np.float64)
                arr = arr[np.isfinite(arr)]
                lo, hi = ((float(np.quantile(arr, .025)), float(np.quantile(arr, .975))) if len(arr)
                          else (np.nan, np.nan))
                paired = [ps_x[s][k] - ps_r[s][k] for s in ss]
                n_better = int(sum(1 for x in paired if x > 0))
                pair_rows.append(dict(reference=f"{ref}/{var}", run=r, variant=var, seeds=str(list(ss)),
                                      metric=k, ref=full_r[k], value=full_x[k], delta=full_x[k] - full_r[k],
                                      ci_lo=lo, ci_hi=hi, paired_seed_delta_mean=float(np.nanmean(paired)),
                                      n_seeds_better=n_better))
                if k not in short:
                    continue
                flag = " ↑" if lo > 0 else (" ↓" if hi < 0 else "")
                say(f"      {short[k]:8s} {full_r[k]:.4f} -> {full_x[k]:.4f}  Δ={full_x[k] - full_r[k]:+.4f}  "
                    f"CI=({lo:+.4f}, {hi:+.4f}){flag}  逐seed配对差 {float(np.nanmean(paired)):+.4f}"
                    f"({n_better}/{len(ss)})")
        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(os.path.join(outdir, "pairwise_delta.csv"), index=False)

    # ---- [4] 各实验用自己的全部 seed 做集成(报告用) ----
    hl_sel = []
    for r in data:
        for var, suf in (("selected", ""), ("perhead", "_ph")):
            ss = data[r]["seeds"].get(var)
            if not ss:
                continue
            if headline_runs == "auto":
                if len(ss) <= len(seeds):
                    continue
            elif headline_runs != "all" and r not in headline_runs:
                continue
            hl_sel.append((r, var, suf, list(ss)))
    hl_rows = []
    if hl_sel:
        say(f"\n[4] 各实验用自己的全部 seed 做集成(报告用；seed 数不同的实验之间不可直接比较)；方括号=按基因整群 "
            f"bootstrap({n_boot}次)的 95% CI，不含训练随机性")
    for (r, var, suf, ss) in hl_sel:
        dv_, dt_ = data[r]["val"], data[r]["test"]
        ps_all = {}
        for s_ in ss:
            pr_v, pr_t = probs_of(dv_, s_, suf), probs_of(dt_, s_, suf)
            ps_all[s_] = metrics(dt_[f"y_a_seed{s_}{suf}"].to_numpy(np.float64),
                                 dt_[f"y_b_seed{s_}{suf}"].to_numpy(np.float64), pr_t,
                                 apply(pr_t, tune(pr_v, yv)))
        ya_e = np.mean([dt_[f"y_a_seed{s_}{suf}"].to_numpy(np.float64) for s_ in ss], 0)
        yb_e = np.mean([dt_[f"y_b_seed{s_}{suf}"].to_numpy(np.float64) for s_ in ss], 0)
        pv_e = np.mean([probs_of(dv_, s_, suf) for s_ in ss], 0)
        pt_e = np.mean([probs_of(dt_, s_, suf) for s_ in ss], 0)
        off_e = apply(pt_e, tune(pv_e, yv))
        variants_hl = [(var, "ABC", ya_e, yb_e, pt_e, off_e)]
        if stack and has_prior:
            sv_, st_ = fit_apply_stack(pv_e, pt_e)
            variants_hl.append((var + "+stack", "C", None, None, st_, apply(st_, tune(sv_, yv))))
        for (vname, hd, ya_x, yb_x, pt_x, off_x) in variants_hl:
            full = metrics(ya_x, yb_x, pt_x, off_x, heads=hd)
            bt = [metrics(ya_x, yb_x, pt_x, off_x, rw, fg, heads=hd) for rw, fg in boot_rows]
            parts, sd_txt = [], []
            for k in full:
                arr = np.asarray([b[k] for b in bt], np.float64)
                arr = arr[np.isfinite(arr)]
                lo, hi = (float(np.quantile(arr, .025)), float(np.quantile(arr, .975))) if len(arr) else (np.nan, np.nan)
                sv_k = ([ps_all[s_][k] for s_ in ss if k in ps_all[s_]]
                        if not vname.endswith("+stack") else [])  # +stack 行没有逐 seed 对应值
                hl_rows.append(dict(run=r, variant=vname, n_seeds=len(ss), seeds=str(ss), metric=k,
                                    ensemble=full[k], ci_lo=lo, ci_hi=hi,
                                    seed_mean=float(np.nanmean(sv_k)) if sv_k else np.nan,
                                    seed_std=float(np.nanstd(sv_k)) if sv_k else np.nan))
                parts.append(f"{short.get(k, k)}={full[k]:.3f}[{lo:.3f},{hi:.3f}]")
                if hd == "ABC":
                    sd_txt.append(f"{short.get(k, k)}={np.nanmean(sv_k):.3f}±{np.nanstd(sv_k):.3f}")
            say(f"  {r}/{vname}  ({len(ss)} seeds {ss})")
            say("      集成: " + "  ".join(parts))
            if sd_txt:
                say("      逐seed均值±std: " + "  ".join(sd_txt))
    if hl_rows:
        pd.DataFrame(hl_rows).to_csv(os.path.join(outdir, "headline_all_seeds.csv"), index=False)

    # ---- [5] 判读表：把 [3] 段的差值压成每个(参照,实验,权重)一行 ----
    key_metrics = tuple(noninf_margin)
    verdict_rows = []
    if len(dlt):
        say("\n[5] 判读表(基于 [3] 段的差值；更好=CI>0，更差=CI<0，非劣=CI含0且下界>−margin，"
            "无法排除明显变差=CI含0且下界≤−margin；margin 见 CONFIG['noninf_margin']，建议值)")
    for ref_name in (dlt["reference"].unique() if len(dlt) else []):
        dr = dlt[dlt["reference"] == ref_name]
        for (r, var) in dict.fromkeys(zip(dr["run"], dr["variant"])):
            if var not in verdict_variants:
                continue
            sub = dr[(dr["run"] == r) & (dr["variant"] == var) & (dr["metric"].isin(key_metrics))]
            cat = dict(better=[], worse=[], noninf=[], unsure=[])
            for row in sub.itertuples():
                mg = noninf_margin[row.metric]
                c_ = ("better" if row.ci_lo > 0 else "worse" if row.ci_hi < 0
                      else "noninf" if row.ci_lo > -mg else "unsure")
                cat[c_].append(short.get(row.metric, row.metric))
                verdict_rows.append(dict(reference=ref_name, run=r, variant=var, metric=row.metric,
                                         delta=row.delta, ci_lo=row.ci_lo, ci_hi=row.ci_hi,
                                         margin=mg, verdict=c_))
            say(f"  {r}/{var} 对 {ref_name}: 更好[{','.join(cat['better'])}] 更差[{','.join(cat['worse'])}] "
                f"非劣[{','.join(cat['noninf'])}] 无法排除明显变差[{','.join(cat['unsure'])}]")
    if verdict_rows:
        pd.DataFrame(verdict_rows).to_csv(os.path.join(outdir, "verdict.csv"), index=False)

    # ---- [6] 按 D∈L_g / D∉L_g 分层的差值(2026-09-26b，见文件头第5批第3条) ----
    strata_rows = []
    if strata:
        ref0, sv = references[0], strata_variant
        if "D_in_Lg" not in ref_t.columns:
            say(f"\n[6] 跳过分层：{ref0} 的 predictions_test 没有 D_in_Lg 列(老格式导出)")
        elif (ref0, sv) not in ens_cache:
            say(f"\n[6] 跳过分层：参照 {ref0} 没有 {sv} 这套权重")
        else:
            in_lg = ref_t["D_in_Lg"].to_numpy().astype(bool)
            sruns = ([r for r in data if "_abl_" in r] if strata_runs == "auto"
                     else [r for r in strata_runs if r in data])
            sruns = [r for r in sruns if r != ref0 and (r, sv) in ens_cache]
            layers = (("D∈L_g", in_lg), ("D∉L_g", ~in_lg))
            s_cache = {}

            def strat_eval(key, lname, mask):
                if (key, lname) not in s_cache:
                    ya_x, yb_x, pt_x, off_x, _ = ens_cache[key]
                    idx = np.flatnonzero(mask)
                    s_cache[(key, lname)] = (
                        metrics(ya_x, yb_x, pt_x, off_x, idx, None, heads="B"),
                        [metrics(ya_x, yb_x, pt_x, off_x, rw[mask[rw]], None, heads="B") for rw, _ in boot_rows])
                return s_cache[(key, lname)]

            if sruns:
                cols6 = [k for k in ("B_r", "B_r_within_tf", "B_sign_acc", "B_dr_ns", "C_auroc_down",
                                     "C_auprc_down", "C_auroc_up", "C_auprc_up", "C_macro_f1_offset") if k in short]
                say(f"\n[6] 按 D∈L_g 分层的差值(参照 {ref0}/{sv}，全局共有 seed {seeds} 的集成；Δ=实验−参照，"
                    "*=配对整群 bootstrap 95% CI 不含0；只算集成)。D∈L_g=被耗竭 TF 在该基因启动子上有位点(位点删除只在\n"
                    "      这一层改变输入)；D∉L_g 那层对 no_knockout 来说输入完全相同，差值只反映训练随机性(阴性对照)")
                say("      " + "".ljust(26) + "".join(short[k].rjust(10) for k in cols6))
            for lname, mask in (layers if sruns else ()):
                ref_full, ref_bt = strat_eval((ref0, sv), lname, mask)
                n_sig = int((mask & np.isfinite(yb_t)).sum())
                say(f"  {lname}：{int(mask.sum())} 条样本({mask.mean():.1%})，显著 {n_sig}、down "
                    f"{int((mask & (yt == 0)).sum())}、up {int((mask & (yt == 2)).sum())}")
                say("      " + f"参照 {ref0}".ljust(26)[:26] + "".join(
                    (f"{ref_full[k]:.3f}" if np.isfinite(ref_full[k]) else "nan").rjust(10) for k in cols6))
                for r in sruns:
                    full_x, bt_x = strat_eval((r, sv), lname, mask)
                    cells = []
                    for k in full_x:
                        arr = np.asarray([bx[k] - br[k] for bx, br in zip(bt_x, ref_bt)], np.float64)
                        arr = arr[np.isfinite(arr)]
                        lo, hi = ((float(np.quantile(arr, .025)), float(np.quantile(arr, .975))) if len(arr)
                                  else (np.nan, np.nan))
                        d_ = full_x[k] - ref_full[k]
                        strata_rows.append(dict(reference=f"{ref0}/{sv}", run=r, variant=sv, stratum=lname,
                                                n_rows=int(mask.sum()), n_sig=n_sig, metric=k, ref=ref_full[k],
                                                value=full_x[k], delta=d_, ci_lo=lo, ci_hi=hi))
                        if k in cols6:
                            star = "*" if (lo > 0 or hi < 0) else " "
                            cells.append((f"{d_:+.3f}{star}" if np.isfinite(d_) else "nan").rjust(10))
                    say("      " + f"Δ {r}".ljust(26)[:26] + "".join(cells))
            if strata_rows:
                pd.DataFrame(strata_rows).to_csv(os.path.join(outdir, "strata_delta.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = list(ens_cache)
        fig, ax = plt.subplots(figsize=(13, 4.2))
        w = .8 / max(len(keys), 1)
        for i, (r, var) in enumerate(keys):
            sub = res[(res["run"] == r) & (res["variant"] == var) & (res["scope"] == ens_scope)]
            mv = dict(zip(sub["metric"], sub["value"]))
            ax.bar(np.arange(len(short)) + i * w, [float(mv.get(k, np.nan)) for k in short],
                   width=w, label=f"{r}/{var}")
        ax.set_xticks(np.arange(len(short)) + .4 - w / 2)
        ax.set_xticklabels(list(short.values()))
        ax.set_title(f"Test, {len(seeds)}-seed ensembles on common seeds {seeds}")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "comparison_test.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        say("  (没有 matplotlib，跳过画图)")
    say(f"\n写出: {outdir}/comparison_long.csv、delta_vs_reference.csv、headline_all_seeds.csv、verdict.csv、"
        + ("pairwise_delta.csv、" if pair_rows else "") + ("strata_delta.csv、" if strata_rows else "")
        + "comparison_test.png、summary.txt"
        f"  (用时 {time.time() - t0:.0f} 秒)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(outdir, "compare_facts.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(seeds=seeds, references=references, runs=list(data), stack=bool(stack and has_prior),
                       dense_target=dense_target if ybd_t is not None else None, match_variant=bool(match_variant),
                       pairwise=sorted({f"{r_['run']}/{r_['variant']} vs {r_['reference']}" for r_ in pair_rows}),
                       strata=bool(strata_rows)),
                  fh, ensure_ascii=False)
    return dict(long=res, delta=dlt, pairwise=pd.DataFrame(pair_rows), strata=pd.DataFrame(strata_rows))


if __name__ == "__main__":
    run_compare(**CONFIG)