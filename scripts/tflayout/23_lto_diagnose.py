# scripts/tflayout/23_lto_diagnose.py

import glob
import os
import time

import numpy as np
import pandas as pd

CONFIG = dict(
    lto_dirs=("out/results/_lto2",),   # 2026-09-28a：默认只看 v2(v1="out/results/_lto" 要看就加回来)
    main_results_dir="out/results/v3_ce_marker_dense",     # 同配置标准模型的 17 号导出(标签、D_in_Lg、对照预测都从这里取)
    seed=42,                                               # 对照取哪个 seed 的标准模型(跟 22 号一致)
    std_ckpt="out/checkpoints/v3_ce_marker_dense/seed42_best.pt",  # [4] 用
    lto_ckpt_dir="out/checkpoints/_lto2",                  # [4] 用(22 号 v2 才有)
    n_boot=300,
    min_sig_tf=20,                                         # 逐 TF 汇总只看显著样本 ≥ 这么多的 TF
    labels="out/head_bc_labels.parquet",                   # 2026-09-28a [5]：TF 类型(激活/抑制)用训练染色体基因上的标签定
    head_a="out/head_a_baseline_logtpm.parquet",           # 2026-09-28a [5]：split 列
    tf_type_thr=0.65,                                      # 2026-09-28a [5]：down 占显著响应 ≥ 这个比例=激活型，≤1−它=抑制型
    outdir="out/results/_lto_diag",
)


def run_lto_diag(lto_dirs, main_results_dir, seed, std_ckpt, lto_ckpt_dir, n_boot, min_sig_tf, outdir,
                 labels="out/head_bc_labels.parquet", head_a="out/head_a_baseline_logtpm.parquet", tf_type_thr=0.65):
    """唯一入口，各段见文件头。"""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    lines = []

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

    def spearman(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        return pearson(pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy())

    def auroc_auprc(score, pos):
        pos = np.asarray(pos, bool)
        sc = np.asarray(score, np.float64)
        ok = np.isfinite(sc)
        pos, sc = pos[ok], sc[ok]
        n1 = int(pos.sum())
        n0 = len(pos) - n1
        if n1 < 3 or n0 < 3:
            return float("nan"), float("nan")
        _, inv_, cnt_ = np.unique(sc, return_inverse=True, return_counts=True)
        r = (np.cumsum(cnt_) - (cnt_ - 1) / 2.0)[inv_.reshape(-1)]
        auc = float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
        p = pos[np.argsort(-sc, kind="mergesort")]
        return auc, float((np.cumsum(p) / np.arange(1, len(p) + 1))[p].sum() / n1)

    def ci_of(arr):
        arr = np.asarray(arr, np.float64)
        arr = arr[np.isfinite(arr)]
        return (float(np.quantile(arr, .025)), float(np.quantile(arr, .975))) if len(arr) else (np.nan, np.nan)

    say("=" * 78)
    say(f"leave-TF-out 诊断(23 号 v2，2026-09-28a)  {time.strftime('%Y-%m-%d %H:%M:%S')}  标准模型导出={main_results_dir}  "
        f"seed={seed}")
    say("=" * 78)

    # ---- 0. 标准模型导出：标签(全部 124 个 TF 的 val+test 样本)、D_in_Lg、对照预测 ----
    parts = [pd.read_parquet(os.path.join(main_results_dir, f"predictions_{sp}.parquet"))
             for sp in ("val", "test") if os.path.exists(os.path.join(main_results_dir, f"predictions_{sp}.parquet"))]
    if len(parts) != 2:
        raise SystemExit(f"找不到 {main_results_dir}/predictions_val|test.parquet(先 --export)")
    L = pd.concat(parts, ignore_index=True).drop_duplicates(["gene_id", "tf_depleted"])
    yc_raw = L["y_c_true"]
    L["yc"] = (yc_raw if pd.api.types.is_numeric_dtype(yc_raw)
               else yc_raw.astype(str).map({"down": 0, "ns": 1, "up": 2})).astype(np.int64)
    seeds_all = sorted(int(cn[len("y_b_seed"):]) for cn in L.columns
                       if cn.startswith("y_b_seed") and cn[len("y_b_seed"):].isdigit())
    std = L[["gene_id", "tf_depleted"]].copy()
    if f"y_b_seed{seed}" in L.columns:
        std["yb_id"], std["pdn_id"], std["pup_id"] = (L[f"y_b_seed{seed}"], L[f"p_down_seed{seed}"],
                                                      L[f"p_up_seed{seed}"])
    if seeds_all:
        std["yb_ens"] = np.mean([L[f"y_b_seed{s_}"].to_numpy(np.float64) for s_ in seeds_all], 0)
        std["pdn_ens"] = np.mean([L[f"p_down_seed{s_}"].to_numpy(np.float64) for s_ in seeds_all], 0)
        std["pup_ens"] = np.mean([L[f"p_up_seed{s_}"].to_numpy(np.float64) for s_ in seeds_all], 0)
    n_bound = (L.groupby("gene_id")["D_in_Lg"].sum().astype(np.float64) if "D_in_Lg" in L.columns else None)
    # 2026-09-28a [5]：TF 类型(激活/抑制)——只用训练染色体基因上的标签，只用来分组描述
    frac_dn = None
    if labels and head_a and os.path.exists(labels) and os.path.exists(head_a):
        _lb = pd.read_parquet(labels)
        _sp = pd.read_parquet(head_a)
        if "split" in _sp.columns:
            _sp_map = {str(k).upper(): v for k, v in _sp["split"].items()}
            _lb = _lb[_lb["gene_id"].astype(str).str.upper().map(_sp_map) == "train"]
            _tf = _lb["tf_depleted"].astype(str).str.strip().str.upper()
            _dir = _lb["direction_3class"].astype(str)
            _ndn = (_dir == "down").groupby(_tf).sum()
            _nup = (_dir == "up").groupby(_tf).sum()
            _tot = _ndn + _nup
            frac_dn = (_ndn / _tot)[_tot >= 10]
    say(f"标准模型导出：{len(L)} 条(val+test)、{L['tf_depleted'].nunique()} 个 TF、{L['gene_id'].nunique()} 个基因；"
        f"seed {seeds_all}；D_in_Lg 列{'有' if n_bound is not None else '没有(老格式，跳过 layout 富度)'}")

    # ---- 1. 读 LTO 结果(v1/v2 自动识别) ----
    sets = []
    for d in lto_dirs:
        f2 = sorted(glob.glob(os.path.join(d, "fold*_pred.csv.gz")))
        f1 = sorted(glob.glob(os.path.join(d, "fold*_predictions.csv.gz")))
        if f2:
            P = pd.concat([pd.read_csv(p) for p in f2], ignore_index=True)
            tag = "v2"
            var = {}
            for cn in P.columns:
                if cn.startswith("yb__"):
                    _, s_, t_ = cn.split("__")
                    var[f"v2:{s_}/{'选中' if t_ == 'sel' else '各头最优'}"] = (cn, f"pdn__{s_}__{t_}", f"pup__{s_}__{t_}")
            fs = sorted(glob.glob(os.path.join(d, "fold*_seen.csv.gz")))
            S = pd.concat([pd.read_csv(p) for p in fs], ignore_index=True) if fs else None
        elif f1:
            P = pd.concat([pd.read_csv(p) for p in f1], ignore_index=True)
            tag, S = "v1", None
            var = {"v1:raw/选中": ("y_b_lto", "p_down_lto", "p_up_lto")}
            if "y_b_lto_ph" in P.columns:
                var["v1:raw/各头最优"] = ("y_b_lto_ph", "p_down_lto_ph", "p_up_lto_ph")
        else:
            continue
        sets.append((tag, d, P, var, S))
        say(f"{tag}: {d} -> {len(P)} 条被留出样本、{P['tf_depleted'].nunique()} 个 TF、{P['fold'].nunique()} 折；"
            f"{len(var)} 套预测" + (f"；已见 TF 对照 {len(S)} 条" if S is not None else ""))
    if not sets:
        raise SystemExit(f"{lto_dirs} 里都没有 LTO 预测文件(先跑 22 号)")

    mkeys = ("B_r", "B_sign", "B_r|TF", "C_AUCdn", "C_APdn", "C_AUCup", "C_APup", "C_AUCany")
    rows_out, tf_out = [], []
    for tag, d, P, var, S in sets:
        say(f"\n{'#' * 78}\n# {tag}：{d}\n{'#' * 78}")
        P = P.merge(std, on=["gene_id", "tf_depleted"], how="left")
        # 基因先验(其它 TF 实测标签)：逐折，只用这一折训练过的 TF
        P["gp_dn"], P["gp_up"], P["gp_b"], P["gp_n"] = np.nan, np.nan, np.nan, 0.0
        P["mp_dn"], P["mp_up"], P["mp_b"] = np.nan, np.nan, np.nan
        for f, Pf in P.groupby("fold"):
            held = set(Pf["tf_depleted"])
            O = L[~L["tf_depleted"].isin(held) & L["gene_id"].isin(set(Pf["gene_id"]))]
            g = O.groupby("gene_id")
            gp = pd.DataFrame(dict(gp_dn=g["yc"].apply(lambda v: float(np.mean(v == 0))),
                                   gp_up=g["yc"].apply(lambda v: float(np.mean(v == 2))),
                                   gp_b=g["y_b_true"].mean(), gp_n=g.size().astype(np.float64)))
            idx = Pf.index
            for cn in ("gp_dn", "gp_up", "gp_b", "gp_n"):
                P.loc[idx, cn] = Pf["gene_id"].map(gp[cn]).to_numpy(np.float64)
            if S is not None:
                Sf = S[S["fold"] == f]
                gs = Sf.groupby("gene_id")
                for cn, src in (("mp_dn", "pdn"), ("mp_up", "pup"), ("mp_b", "yb")):
                    P.loc[idx, cn] = Pf["gene_id"].map(gs[src].mean()).to_numpy(np.float64)
        P["gp_b"] = P["gp_b"].fillna(0.0)  # 其它 TF 下从没显著过 -> 先验 log2FC 取 0
        if n_bound is not None:
            P["n_bound"] = P["gene_id"].map(n_bound).to_numpy(np.float64)
        say(f"基因先验：每个被留出样本平均用到 {P['gp_n'].mean():.0f} 个其它 TF 的实测结果"
            f"(该基因在其它 TF 下显著的比例 中位 {(P['gp_dn'] + P['gp_up']).median():.3f})")
        V = dict(var)
        if "yb_id" in P.columns:
            V[f"标准模型 seed{seed}"] = ("yb_id", "pdn_id", "pup_id")
        if "yb_ens" in P.columns:
            V[f"标准模型 {len(seeds_all)}-seed 集成"] = ("yb_ens", "pdn_ens", "pup_ens")
        V["基因先验(其它TF实测标签；参照)"] = ("gp_b", "gp_dn", "gp_up")
        if S is not None:
            V["模型基因先验(见过TF的预测均值)"] = ("mp_b", "mp_dn", "mp_up")
        if n_bound is not None:
            P["_nan"] = np.nan
            V["layout富度(n_bound；只看C)"] = ("_nan", "n_bound", "n_bound")
        arr = {k: tuple(P[c_].to_numpy(np.float64) for c_ in cols) for k, cols in V.items()}
        yb_t, yc_t = P["y_b_true"].to_numpy(np.float64), P["y_c_true"].to_numpy(np.int64)
        tcode = pd.factorize(P["tf_depleted"])[0]
        nt = int(tcode.max()) + 1

        def mets(rows, key):
            yp, pdn, pup = arr[key]
            ok = np.isfinite(yb_t[rows]) & np.isfinite(yp[rows])
            p_, t_, tc_ = yp[rows][ok], yb_t[rows][ok], tcode[rows][ok]
            cnt = np.maximum(np.bincount(tc_, minlength=nt), 1)
            mp_ = np.bincount(tc_, weights=p_, minlength=nt) / cnt
            mt_ = np.bincount(tc_, weights=t_, minlength=nt) / cnt
            out = {"B_r": pearson(p_, t_),
                   "B_sign": float(np.mean(np.sign(p_) == np.sign(t_))) if len(p_) else float("nan"),
                   "B_r|TF": pearson(p_ - mp_[tc_], t_ - mt_[tc_])}
            out["C_AUCdn"], out["C_APdn"] = auroc_auprc(pdn[rows], yc_t[rows] == 0)
            out["C_AUCup"], out["C_APup"] = auroc_auprc(pup[rows], yc_t[rows] == 2)
            any_sc = pdn[rows] + pup[rows] if not np.array_equal(pdn, pup) else pdn[rows]
            out["C_AUCany"] = auroc_auprc(any_sc, yc_t[rows] != 1)[0]
            return out

        genes = P["gene_id"].to_numpy()
        uniq, inv = np.unique(genes, return_inverse=True)
        rows_of_gene = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
        brng = np.random.default_rng(0)
        boot = [np.concatenate([rows_of_gene[g] for g in brng.integers(0, len(uniq), len(uniq))])
                for _ in range(int(n_boot))]
        lto_sel = [k for k in var if k.endswith("/选中")]
        raw_key = next((k for k in lto_sel if ":raw/" in k), lto_sel[0] if lto_sel else None)
        mean_key = next((k for k in lto_sel if ":mean/" in k), None)
        zero_key = next((k for k in lto_sel if ":zero/" in k), None)  # 2026-09-28a
        ci_keys = [k for k in lto_sel] + [k for k in V if k.startswith(("基因先验", "模型基因先验"))]
        gp_key = "基因先验(其它TF实测标签；参照)"
        valid = (np.isfinite(P["yb_id"].to_numpy(np.float64)) if "yb_id" in P.columns else np.ones(len(P), bool))
        strata = {"全部": valid, "D∈L_g": valid & P["in_lg"].to_numpy(bool), "D∉L_g": valid & ~P["in_lg"].to_numpy(bool)}
        say(f"\n[1] 同一批被留出样本上并排比(点估计；带 [CI] 的是按基因整群 bootstrap {n_boot} 次)。C_AUCany=用 P(dn)+P(up) 区分"
            "显著/ns；layout富度只有 C 列有意义")
        for sname, smask in strata.items():
            rows = np.where(smask)[0]
            if not len(rows):
                continue
            say(f"  ── 分层 {sname}：{len(rows)} 条，显著 {int(np.isfinite(yb_t[rows]).sum())} 条 ──")
            pt = {k: mets(rows, k) for k in V}
            bt = {k: [] for k in ci_keys}
            for br in boot:
                rr = br[smask[br]]
                for k in ci_keys:
                    bt[k].append(mets(rr, k))
            for k in V:
                txt = []
                for mk in mkeys:
                    v = pt[k][mk]
                    lo, hi = ci_of([b[mk] for b in bt[k]]) if k in bt else (np.nan, np.nan)
                    txt.append(f"{mk}={v:.3f}" + (f"[{lo:.3f},{hi:.3f}]" if k in bt else ""))
                    rows_out.append(dict(set=tag, stratum=sname, variant=k, metric=mk, value=v, ci_lo=lo, ci_hi=hi))
                say(f"    {k}: " + "  ".join(txt))
            mgp_key = "模型基因先验(见过TF的预测均值)"
            pairs = [(k, gp_key) for k in lto_sel]
            if mgp_key in V:  # 2026-09-28a：公平基线(不用标签)
                pairs += [(k, mgp_key) for k in lto_sel]
            if mean_key:
                pairs += [(mean_key, raw_key)] + [(k, mean_key) for k in lto_sel if ":knn_" in k]
                if zero_key:  # 2026-09-28a：只删位点 vs 平均 TF
                    pairs.append((zero_key, mean_key))
            for a_, b_ in pairs:
                txt = []
                for mk in mkeys:
                    d0 = pt[a_][mk] - pt[b_][mk]
                    lo, hi = ci_of([x[mk] - y[mk] for x, y in zip(bt[a_], bt[b_])])
                    flag = "↑" if lo > 0 else ("↓" if hi < 0 else "")
                    txt.append(f"{mk}={d0:+.3f}[{lo:+.3f},{hi:+.3f}]{flag}")
                    rows_out.append(dict(set=tag, stratum=sname, variant=f"{a_} − {b_}", metric=mk, value=d0,
                                         ci_lo=lo, ci_hi=hi))
                say(f"    Δ {a_} − {b_}: " + "  ".join(txt))

        # ---- [2] 反转诊断 ----
        say("\n[2] 反转诊断(逐 TF；只看显著样本 ≥%d 的 TF)：C_AUCany<0.5=把会变的基因判得更像 ns；ρ(模型变化概率, 基因先验变化"
            "比例) 在 TF 内算 Spearman，<0=模型恰好把平时易变的基因判成不变" % min_sig_tf)
        gp_any = P["gp_dn"].to_numpy(np.float64) + P["gp_up"].to_numpy(np.float64)
        diag_keys = lto_sel + [k for k in V if k.startswith("标准模型 seed")]
        for k in diag_keys:
            yp, pdn, pup = arr[k]
            s_any = pdn + pup
            auc_tf, rho_tf = [], []
            for tf, g in P.groupby("tf_depleted"):
                ix = g.index.to_numpy()
                if int(np.isfinite(yb_t[ix]).sum()) < min_sig_tf:
                    continue
                auc_tf.append(auroc_auprc(s_any[ix], yc_t[ix] != 1)[0])
                rho_tf.append(spearman(s_any[ix], gp_any[ix]))
                tf_out.append(dict(set=tag, variant=k, tf_depleted=tf, n_sig=int(np.isfinite(yb_t[ix]).sum()),
                                   frac_in_lg=float(P.loc[ix, "in_lg"].mean()), auc_any=auc_tf[-1], rho_prior=rho_tf[-1],
                                   aucdn=auroc_auprc(pdn[ix], yc_t[ix] == 0)[0],
                                   aucup=auroc_auprc(pup[ix], yc_t[ix] == 2)[0],
                                   r_b=pearson(yp[ix], yb_t[ix]),
                                   aucany_prior=auroc_auprc(gp_any[ix], yc_t[ix] != 1)[0]))
            a_, r_ = np.asarray(auc_tf, np.float64), np.asarray(rho_tf, np.float64)
            extra = ""
            if n_bound is not None:
                extra = f"；全体 ρ(变化概率, n_bound)={spearman(s_any, P['n_bound'].to_numpy(np.float64)):+.3f}"
            say(f"  {k}: {len(a_)} 个 TF，C_AUCany 中位 {np.nanmedian(a_):.3f}(<0.5 的 {int(np.sum(a_ < 0.5))} 个)；"
                f"TF 内 ρ(变化概率, 基因先验) 中位 {np.nanmedian(r_):+.3f}(<0 的 {int(np.sum(r_ < 0))} 个)"
                f"；全体 ρ={spearman(s_any, gp_any):+.3f}{extra}")
        a_p = [r_["aucany_prior"] for r_ in tf_out if r_["set"] == tag and r_["variant"] == diag_keys[0]]
        if a_p:
            say(f"  (参照) 基因先验自己的逐 TF C_AUCany 中位 {np.nanmedian(a_p):.3f}")

        # ---- [3] Head B 偏相关 ----
        say("\n[3] Head B 是不是就是基因先验(显著样本；偏相关=分别对先验做线性回归取残差后的 r)。GP=基因先验(其它TF实测标签)，"
            "MGP=模型基因先验(见过TF的预测均值，v2 才有)")
        sig = np.isfinite(yb_t)
        gpb = P["gp_b"].to_numpy(np.float64)
        mpb = P["mp_b"].to_numpy(np.float64)
        has_mp = bool(np.isfinite(mpb[sig]).sum() >= 10)

        def partial(yp, ok, cols):
            X = np.column_stack([np.ones(ok.sum())] + [c_[ok] for c_ in cols])
            res_p = yp[ok] - X @ np.linalg.lstsq(X, yp[ok], rcond=None)[0]
            res_t = yb_t[ok] - X @ np.linalg.lstsq(X, yb_t[ok], rcond=None)[0]
            return pearson(res_p, res_t)

        for k in lto_sel + [k2 for k2 in V if k2.startswith("标准模型")]:
            yp = arr[k][0]
            ok = sig & np.isfinite(yp) & np.isfinite(gpb)
            if ok.sum() < 10:
                continue
            txt = (f"  {k}: r(ŷ_B, 真值)={pearson(yp[ok], yb_t[ok]):.3f}  r(ŷ_B, GP)={pearson(yp[ok], gpb[ok]):.3f}  "
                   f"偏相关|GP={partial(yp, ok, [gpb]):.3f}")
            if has_mp:  # 2026-09-28a
                ok2 = ok & np.isfinite(mpb)
                txt += (f"  r(ŷ_B, MGP)={pearson(yp[ok2], mpb[ok2]):.3f}  偏相关|MGP={partial(yp, ok2, [mpb]):.3f}  "
                        f"偏相关|GP+MGP={partial(yp, ok2, [gpb, mpb]):.3f}")
            say(txt + f"  (n={int(ok.sum())})")
        say(f"  (参照) r(GP, 真值)={pearson(gpb[sig], yb_t[sig]):.3f}"
            + (f"  r(MGP, 真值)={pearson(mpb[sig], yb_t[sig]):.3f}" if has_mp else ""))

        # ---- [5] 只删位点(zero)在哪类 TF 上有效(2026-09-28a) ----
        if zero_key is not None and frac_dn is not None:
            say(f"\n[5] \"只删位点\"(zero)按 TF 类型分组(D∈L_g 层)：类型=该 TF 在训练染色体基因上的显著响应里 down 的比例，"
                f"≥{tf_type_thr:.2f} 激活型、≤{1 - tf_type_thr:.2f} 抑制型，其余混合；只用来分组描述")
            fd = P["tf_depleted"].astype(str).str.strip().str.upper().map(frac_dn).to_numpy(np.float64)
            grp = np.where(~np.isfinite(fd), "未知(显著<10)",
                           np.where(fd >= tf_type_thr, "激活型", np.where(fd <= 1 - tf_type_thr, "抑制型", "混合")))
            inlg = P["in_lg"].to_numpy(bool)
            gk = [k for k in (raw_key, zero_key, mean_key) if k] + \
                 [k for k in V if k.startswith("标准模型 seed") or k.startswith("模型基因先验")]
            for gname in ("激活型", "抑制型", "混合", "未知(显著<10)"):
                rows = np.where(inlg & (grp == gname))[0]
                n_tf_g = int(pd.Series(P["tf_depleted"].to_numpy()[grp == gname]).nunique())
                if len(rows) == 0:
                    continue
                yc_g = yc_t[rows]
                say(f"  {gname}：{n_tf_g} 个 TF，D∈L_g {len(rows)} 条(down {int((yc_g == 0).sum())}、up {int((yc_g == 2).sum())})")
                for k in gk:
                    _, pdn, pup = arr[k]
                    a_dn = auroc_auprc(pdn[rows], yc_g == 0)[0]
                    a_up = auroc_auprc(pup[rows], yc_g == 2)[0]
                    say(f"    {k}: AUROC dn/up = {a_dn:.3f}/{a_up:.3f}")
            zdn = arr[zero_key][1]
            per = []
            for tf, g in P.groupby("tf_depleted"):
                ix = g.index.to_numpy()
                ix = ix[inlg[ix]]
                if int((yc_t[ix] == 0).sum()) < 5:
                    continue
                f_ = frac_dn.get(str(tf).strip().upper(), np.nan)
                per.append((f_, auroc_auprc(zdn[ix], yc_t[ix] == 0)[0]))
            per = np.asarray(per, np.float64)
            if len(per) >= 5:
                say(f"  逐 TF(D∈L_g 里 down ≥5 条的 {len(per)} 个)：Spearman ρ(TF 的 down 比例, zero 的 AUROC_dn) = "
                    f"{spearman(per[:, 0], per[:, 1]):+.3f}(>0 = 越偏激活型，删位点越能预测 down)")

    pd.DataFrame(rows_out).to_csv(os.path.join(outdir, "metrics_diag.csv"), index=False)
    pd.DataFrame(tf_out).to_csv(os.path.join(outdir, "per_tf_diag.csv"), index=False)

    # ---- [4] 机制核对(需要 torch) ----
    say(f"\n{'#' * 78}\n[4] 机制核对：没见过的 TF 那一维 marker 列是不是\"跟偏置同步漂移\"\n{'#' * 78}")
    try:
        import torch
    except ImportError:
        torch = None
        say("  没有 torch，跳过")
    if torch is not None:
        def load_ck(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(p, map_location="cpu")

        def as_np(t):
            return np.asarray(t.detach().to(torch.float64).cpu().numpy(), np.float64)

        def key_of(sd, suf):
            ks = [k for k in sd if k.endswith(suf)]
            return ks[0] if ks else None

        def mean_pair_cos(M):
            if M.shape[1] < 2:
                return float("nan")
            Mn = M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)
            C = Mn.T @ Mn
            n = C.shape[0]
            return float((C.sum() - np.trace(C)) / (n * (n - 1)))

        def cos(a, b):
            return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

        def report(label, sd, untrained, trained, extra_untrained=None):
            kW, kB = key_of(sd, "fusion.condition.net.0.weight"), key_of(sd, "fusion.condition.net.0.bias")
            if kW is None or kB is None:
                say(f"  {label}: state_dict 里没有条件 MLP 第一层，跳过")
                return
            W, b = as_np(sd[kW]), as_np(sd[kB])
            H, n_tf = W.shape
            init_norm = float(np.sqrt(H / (3.0 * n_tf)))
            U, T = W[:, untrained], W[:, trained]
            sh = U.mean(1)
            resid_u = np.linalg.norm(U - sh[:, None], axis=0)
            resid_t = np.linalg.norm(T - sh[:, None], axis=0)
            say(f"  {label}: 第一层 {H}×{n_tf}；初始化时每列范数期望≈{init_norm:.3f}")
            # 2026-09-28a：数量核对(见文件头 v2 第4条)
            nu = U.shape[1]
            r2 = float(np.mean(resid_u ** 2))
            d2 = float(sh @ sh) - r2 / nu
            pred_cos = d2 / (d2 + r2) if d2 > 0 else float("nan")
            obs_cos = mean_pair_cos(U)
            say(f"    未训练列 {nu} 个：范数中位 {np.median(np.linalg.norm(U, axis=0)):.3f}；共同分量范数 {np.linalg.norm(sh):.3f}"
                f"(扣掉初始化平均残差后估计漂移 |d|≈{np.sqrt(max(d2, 0.0)):.3f})；残差范数中位 {np.median(resid_u):.3f}"
                f"(=初始化期望的 {np.median(resid_u) / init_norm:.2f} 倍，AdamW 衰减预期略 <1)")
            say(f"    两两余弦：实测 {obs_cos:.3f}，机制预测 d²/(d²+r²)={pred_cos:.3f}(差 {obs_cos - pred_cos:+.3f})")
            say(f"    训练过的列 {T.shape[1]} 个：范数中位 {np.median(np.linalg.norm(T, axis=0)):.3f}，两两余弦均值 "
                f"{mean_pair_cos(T):.3f}；跟未训练列共同分量的差 范数中位 {np.median(resid_t):.3f}")
            nb = float(np.linalg.norm(b))
            say(f"    cos(未训练列共同分量, 第一层偏置)={cos(sh, b):.3f}(近似预测 d²/(|共同分量|·|b|)="
                f"{d2 / max(np.linalg.norm(sh) * nb, 1e-12):.3f}，假设偏置漂移≈列漂移、各自初始化跟漂移正交；|b|={nb:.3f})")
            tm = T.mean(1)
            say(f"    翻转方向：没见过的 TF 翻转≈−共同分量，训练过的 TF 平均翻转=−(训练过的列均值)；两者余弦 {cos(sh, tm):+.3f}、"
                f"范数比 {np.linalg.norm(sh) / max(np.linalg.norm(tm), 1e-12):.2f}(明显为负 = 把第一层推向\"平均耗竭\"的反方向)")
            if extra_untrained is not None and len(extra_untrained):
                X = W[:, extra_untrained]
                say(f"    另一组未训练列(从没被耗竭过的 TF，{X.shape[1]} 个)的均值跟上面共同分量的余弦 {cos(X.mean(1), sh):.3f}"
                    f"(≈1 说明两组未训练列是同一个东西)")
            for suf, init_std in (("tf_embed_corr.weight", 0.02), ("tf_bias_c.weight", 0.0)):
                kE = key_of(sd, suf)
                if kE is None:
                    continue
                E = as_np(sd[kE])
                nu, ntr = np.linalg.norm(E[untrained], axis=1), np.linalg.norm(E[trained], axis=1)
                say(f"    {suf} 行范数：未训练 中位 {np.median(nu):.4f}(初始化期望≈{init_std * np.sqrt(E.shape[1]):.4f})、"
                    f"训练过 中位 {np.median(ntr):.4f}")

        if std_ckpt and os.path.exists(std_ckpt):
            ck = load_ck(std_ckpt)
            sd = ck["model_state"]
            kbc = key_of(sd, "tf_bias_c.weight")
            if kbc is None:
                say(f"  {std_ckpt}: 没有 tf_bias_c(head_c_mode=delta)，没法识别从没被耗竭过的 TF，跳过")
            else:
                zero_rows = np.all(as_np(sd[kbc]) == 0.0, axis=1)
                never, dep = np.where(zero_rows)[0], np.where(~zero_rows)[0]
                say(f"  标准模型 {std_ckpt}：tf_bias_c 全零行 {len(never)} 个(=从没被耗竭过的 TF；预期 178−124=54)")
                if len(never) >= 2 and len(dep) >= 2:
                    report("标准模型", sd, never, dep)
        else:
            say(f"  找不到 {std_ckpt}，跳过标准模型")
        for p in sorted(glob.glob(os.path.join(lto_ckpt_dir, "fold*_seed*.pt"))):
            ck = load_ck(p)
            held, trained = np.asarray(ck.get("held_idx", []), np.int64), np.asarray(ck.get("trained_idx", []), np.int64)
            sd = ck["model_state"]
            n_tf = next(v.shape[1] for k, v in sd.items() if k.endswith("fusion.condition.net.0.weight"))
            never = np.setdiff1d(np.arange(n_tf), np.concatenate([held, trained]))
            if len(held) >= 2 and len(trained) >= 2:
                report(f"LTO {os.path.basename(p)}(未训练=本折被留出的 {len(held)} 个 TF)", sd, held, trained, never)
        say("  判读(2026-09-28a 改写；v1 版\"两两余弦 >0.5\"的标准写错了)：两两余弦实测跟 d²/(d²+r²) 的预测差 <0.03、残差≈初始化\n"
            "  期望×(0.9~1)、偏置余弦跟近似预测同号同量级 -> \"没见过的列 = 衰减后的初始化 + 跟偏置同步的漂移\"成立；翻转方向余弦明显\n"
            "  为负 -> 反向的直接原因(翻转没见过的 TF 把第一层推向训练过的 TF 平均翻转的反方向)。LTO 折里被留出 TF 的列跟从没被耗竭\n"
            "  过的 TF 的列共同分量余弦≈1 -> 两者在模型眼里是同一种\"没见过\"。")

    say("\n总体判读(建议，不是硬规则；2026-09-28a 更新)：\n"
        "  - [1] mean − 模型基因先验 的配对 CI 含0(或 |Δ|<0.015)-> 修好表示之后，LTO 就是\"按平均 TF 猜\"，没有 TF 特异的泛化；\n"
        "    zero − mean 在 D∈L_g 层 C_AUCdn 的 CI >0 -> \"删位点\"是唯一能迁移的 TF 特异信号；\n"
        "  - [3] 偏相关|MGP≈0(<0.1)-> Head B 的 LTO 表现就是模型自己的\"平均 TF\"预测；偏相关|GP 仍 >0.1 而 |MGP≈0 -> 模型的基因倾向\n"
        "    跟实测基因倾向不完全一样，但都不是 TF 特异的；\n"
        "  - [4] 数量核对通过 -> 22 号文件头的机制(9/10)和\"反向\"的原因(8/10)可以写进论文方法/讨论；\n"
        "  - [5] 激活型 AUROC_dn 明显高于抑制型/混合 -> 删位点的可迁移性来自\"激活因子离开 -> 下调\"这条通用规则。")
    say(f"\n写出 {outdir}/summary.txt、metrics_diag.csv、per_tf_diag.csv  (用时 {time.time() - t0:.0f} 秒)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(metrics=pd.DataFrame(rows_out), per_tf=pd.DataFrame(tf_out))


if __name__ == "__main__":
    run_lto_diag(**CONFIG)
