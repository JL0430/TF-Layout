# scripts/tflayout/24_position_scan.py

import glob
import importlib.util
import os
import re
import sys
import time

import numpy as np
import pandas as pd

CONFIG = dict(
    paths=dict(layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
               head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
               promoter_tokens="out/promoter_token_ids.parquet",
               bpe_tokenizer="out/bpe_tokenizer.json"),
    ckpt_root="out/checkpoints",
    results_root="out/results",
    main_run="v3_ce_marker_dense",          # 被扫描的主线(全部 seed)
    control_run="v6_abl_shift_position",    # 对照：训练时就随机平移过坐标的模型(全部 seed)；None=不做对照
    dist_models=("v3_ce_marker_dense", "v5_abl_no_position", "v5_abl_shuffle_position", "v6_abl_shift_position"),  # [1] 用 17 号导出
    dist_bins=(0, 100, 200, 300, 500, 1000),  # [1] |D 的位点到 ATG| 最小值的分箱边界(bp)，最后一箱到无穷
    near_bp=200, far_bp=300,                  # [1] 近端(<near_bp) vs 远端(≥far_bp)
    offsets=(-400, -200, -100, -50, -25, 25, 50, 100, 200, 400),  # [3] 整体平移 Δ(bp；负=往上游)
    jitters=(10, 25, 50, 100, 200, 400),      # [4] 局部抖动 σ(bp)
    variant="perhead",   # 用哪套权重："perhead"(报告口径，status 7.9)或 "selected"
    head_a_all=True,     # [2]
    n_boot=200,          # 配对整群 bootstrap 次数([1] 的 log2R、[2]、[3][4] 的\"全部\"分层差值)
    device=None,         # None=有 GPU 用 GPU
    num_workers=4, eval_batch_size=512,
    smoke_first=True, smoke_genes=40,   # 冒烟：主线第一个 seed、40 个 test 基因，走一遍未扰动/offset/jitter
    repro_tol=1e-3,      # 未扰动时跟 17 号导出的逐样本预测最大差的容忍度(超过 0.01 直接停)
    outdir="out/results/_pos_scan",
)


def run_pos_scan(paths, ckpt_root, results_root, main_run, control_run, dist_models, dist_bins, near_bp,
                 far_bp, offsets, jitters, variant, head_a_all, n_boot, device, num_workers, eval_batch_size,
                 smoke_first, smoke_genes, repro_tol, outdir):
    """唯一入口，各段见文件头。"""
    t_all = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def load(fn):  # 跟 18/22 号同一种按路径动态加载(文件名以数字开头，不能 import)
        name = f"_tflayout_{fn[:-3]}"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, fn))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    def pearson(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def auroc_auprc(score, pos):
        pos = np.asarray(pos, bool)
        sc = np.asarray(score, np.float64)
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

    def to_np(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "float") and hasattr(t, "cpu"):
            t = t.float().cpu()
        return np.asarray(t.numpy() if hasattr(t, "numpy") else t, np.float64)

    def softmax_np(lg):
        lg = np.asarray(lg, np.float64)
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(1, keepdims=True)

    tl = load("16_train_loop.py")
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    say("=" * 78)
    say(f"位置信息扫描(24 号，第7批 2026-09-28a)  {time.strftime('%Y-%m-%d %H:%M:%S')}  16号代码版本 {tl.CODE_VERSION}  "
        f"设备={device}  权重={variant}")
    say("=" * 78)

    # ---- 0. Dataset、切分、D 位点离 ATG 的距离 ----
    ds = tl.TFLayoutDataset(paths["layout"], paths["labels"], paths["head_a"], paths["sgd"],
                            paths["promoter_tokens"], paths["bpe_tokenizer"])
    if not hasattr(ds, "set_position_probe"):
        raise SystemExit("09_torch_dataset.py 没有 set_position_probe(第7批)，先替换 09 号")
    split_idx = tl.build_split_indices(ds, paths["head_a"])
    smp = ds.samples
    genes_all = smp["gene_id"].to_numpy()
    tfs_all = smp["tf_depleted"].to_numpy()
    yb_all = smp["log2fc"].to_numpy(np.float64)
    yc_all = smp["direction_3class"].map(ds.class2idx).to_numpy().astype(np.int64)
    exact = smp["direction_3class"].value_counts()
    class_counts = torch.tensor([float(exact.get(k, 0.0)) for k in ("down", "ns", "up")])
    dmin_map, nsite_map, all_pos = {}, {}, []
    for g, lg in ds.layout_by_gene.items():
        ti = np.asarray(lg["tf_idx"], np.int64)
        if not ti.size:
            continue
        ps = np.asarray(lg["pos"], np.float64)
        all_pos.append(ps)
        for t in np.unique(ti):
            m = ti == t
            dmin_map[(g, int(t))] = float(np.abs(ps[m]).min())
            nsite_map[(g, int(t))] = int(m.sum())
    dep_idx = np.array([ds.tf2idx.get(t, -1) for t in tfs_all], np.int64)
    dmin_all = np.array([dmin_map.get((g, int(d)), np.nan) for g, d in zip(genes_all, dep_idx)], np.float64)
    nds_all = np.array([nsite_map.get((g, int(d)), 0) for g, d in zip(genes_all, dep_idx)], np.int64)
    in_lg_all = np.isfinite(dmin_all)
    ap = np.concatenate(all_pos) if all_pos else np.zeros(0)
    say(f"样本 {len(smp)}，test {len(split_idx['test'])}；全部位点坐标(相对 ATG) 分位数 5/25/50/75/95% = "
        + "/".join(f"{v:.0f}" for v in (np.quantile(ap, [.05, .25, .5, .75, .95]) if ap.size else [np.nan] * 5))
        + f"bp，>0(ATG 下游)的占 {float((ap > 0).mean()) if ap.size else float('nan'):.1%}")
    say(f"D∈L_g 的样本 {int(in_lg_all.sum())}({in_lg_all.mean():.1%})；其中 D 的位点离 ATG 的最小距离 中位 "
        f"{np.nanmedian(dmin_all):.0f}bp，D 在该启动子上有 ≥2 个位点的占 {float((nds_all[in_lg_all] >= 2).mean()):.1%}")

    te = np.asarray(split_idx["test"], np.int64)
    g_te = genes_all[te]
    uniq_g, inv_g = np.unique(g_te, return_inverse=True)
    rows_of_gene = np.split(np.argsort(inv_g, kind="stable"), np.cumsum(np.bincount(inv_g))[:-1])
    _, first_row = np.unique(inv_g, return_index=True)  # 每个 test 基因第一次出现的行(Head A 按基因算)
    yb_te, yc_te = yb_all[te], yc_all[te]
    tcode_te = pd.factorize(tfs_all[te])[0]
    n_tf_te = int(tcode_te.max()) + 1
    in_lg_te = in_lg_all[te]
    gene2ya = ds.head_a.to_dict()
    ya_g_true = np.array([gene2ya.get(g, np.nan) for g in uniq_g], np.float64)
    brng = np.random.default_rng(0)
    draws = [brng.integers(0, len(uniq_g), len(uniq_g)) for _ in range(int(n_boot))]
    boot_rows = [np.concatenate([rows_of_gene[j] for j in dr]) for dr in draws]

    def mets(rows, gsel, P):
        """P: ya_g(按 uniq_g 排的基因级 Head A 预测，可为 None)、yb/pdn/pup(test 行)。rows=test 行下标(可重复)。"""
        out = {}
        if gsel is not None and P.get("ya_g") is not None:
            out["A_r"] = pearson(P["ya_g"][gsel], ya_g_true[gsel])
        yb_t, yb_p = yb_te[rows], P["yb"][rows]
        sig = np.isfinite(yb_t) & np.isfinite(yb_p)
        p_, t_, tc_ = yb_p[sig], yb_t[sig], tcode_te[rows][sig]
        cnt = np.maximum(np.bincount(tc_, minlength=n_tf_te), 1)
        mp_ = np.bincount(tc_, weights=p_, minlength=n_tf_te) / cnt
        mt_ = np.bincount(tc_, weights=t_, minlength=n_tf_te) / cnt
        out["B_r"] = pearson(p_, t_)
        out["B_sign"] = float(np.mean(np.sign(p_) == np.sign(t_))) if len(p_) else float("nan")
        out["B_r|TF"] = pearson(p_ - mp_[tc_], t_ - mt_[tc_])
        yc_ = yc_te[rows]
        out["C_AUCdn"], out["C_APdn"] = auroc_auprc(P["pdn"][rows], yc_ == 0)
        out["C_AUCup"], out["C_APup"] = auroc_auprc(P["pup"][rows], yc_ == 2)
        out["C_AUCany"] = auroc_auprc(P["pdn"][rows] + P["pup"][rows], yc_ != 1)[0]
        return out

    # ---- [1] 数据：D 位点离 ATG 的距离 vs 响应率；各模型能否复现 ----
    say(f"\n[1] D 的位点离 ATG 的最小距离 vs 基因响应(只看 D∈L_g 的样本；\"TF调整比\"=Σ观测/Σ期望，期望=该 TF 自己的平均响应率，"
        "控制不同 TF 靶基因多少不同)")
    edges = list(dist_bins) + [np.inf]
    sig_all = (yc_all != 1).astype(np.float64)

    def bin_label(i):
        return f"[{edges[i]:.0f},{edges[i + 1]:.0f})" if np.isfinite(edges[i + 1]) else f"≥{edges[i]:.0f}"

    def tf_expect(idx, x):
        tc = pd.factorize(tfs_all[idx])[0]
        m = np.bincount(tc, weights=x) / np.maximum(np.bincount(tc), 1)
        return m[tc]

    rows_d = []
    all_i = np.arange(len(smp))
    e_all = tf_expect(all_i, sig_all)
    say("  数据(全部 split 的标签，不涉及模型)：")
    for i in range(len(edges) - 1):
        m = in_lg_all & (dmin_all >= edges[i]) & (dmin_all < edges[i + 1])
        if not m.any():
            continue
        rd = dict(scope="all_splits", bin=bin_label(i), n=int(m.sum()), rate_sig=float(sig_all[m].mean()),
                  rate_dn=float((yc_all[m] == 0).mean()), rate_up=float((yc_all[m] == 2).mean()),
                  tf_adj_ratio=float(sig_all[m].sum() / max(e_all[m].sum(), 1e-12)))
        rows_d.append(rd)
        say(f"    {rd['bin']:>12s}bp  n={rd['n']:6d}  响应率 {rd['rate_sig']:.3f}(down {rd['rate_dn']:.3f}/up "
            f"{rd['rate_up']:.3f})  TF调整比 {rd['tf_adj_ratio']:.2f}")
    m = ~in_lg_all
    say(f"    {'D∉L_g':>12s}    n={int(m.sum()):6d}  响应率 {sig_all[m].mean():.3f}(down {(yc_all[m] == 0).mean():.3f}/up "
        f"{(yc_all[m] == 2).mean():.3f})  TF调整比 {sig_all[m].sum() / max(e_all[m].sum(), 1e-12):.2f}")
    rows_d.append(dict(scope="all_splits", bin="D_not_in_Lg", n=int(m.sum()), rate_sig=float(sig_all[m].mean()),
                       rate_dn=float((yc_all[m] == 0).mean()), rate_up=float((yc_all[m] == 2).mean()),
                       tf_adj_ratio=float(sig_all[m].sum() / max(e_all[m].sum(), 1e-12))))

    # test 行上：数据 vs 各模型(17 号导出的集成预测；各模型 seed 数不同，只比形状)
    key_te = pd.Index(pd.Series(g_te).astype(str) + "|" + pd.Series(tfs_all[te]).astype(str))
    src = {"数据(标签)": sig_all[te]}
    exp_cache = {}
    for run in dist_models:
        pth = os.path.join(results_root, run, "predictions_test.parquet")
        if not os.path.exists(pth):
            say(f"  (没有 {pth}，[1] 跳过 {run})")
            continue
        E = pd.read_parquet(pth)
        exp_cache[run] = E
        cdn, cup = (("ens_ph_p_down", "ens_ph_p_up") if variant == "perhead" and "ens_ph_p_down" in E.columns
                    else ("ens_p_down", "ens_p_up"))
        pos = pd.Index(E["gene_id"].astype(str) + "|" + E["tf_depleted"].astype(str)).get_indexer(key_te)
        if (pos < 0).any():
            say(f"  ⚠ {run} 的导出有 {int((pos < 0).sum())} 条 test 样本对不上，[1] 跳过它")
            continue
        src[f"{run}[{cdn[:-7]}]"] = E[cdn].to_numpy(np.float64)[pos] + E[cup].to_numpy(np.float64)[pos]
    near_te = in_lg_te & (dmin_all[te] < near_bp)
    far_te = in_lg_te & (dmin_all[te] >= far_bp)
    say(f"  test 行(D∈L_g {int(in_lg_te.sum())} 条)：每箱 = 数据的 TF调整比 | 各模型\"预测变化概率 P(dn)+P(up)\"的 TF调整比")
    hdr = "    " + f"{'分箱':>12s}  {'n':>5s}  " + "  ".join(f"{k[:28]:>28s}" for k in src)
    say(hdr)
    e_src = {k: tf_expect(te, v) for k, v in src.items()}
    for i in range(len(edges) - 1):
        mm = in_lg_te & (dmin_all[te] >= edges[i]) & (dmin_all[te] < edges[i + 1])
        if not mm.any():
            continue
        vals = {k: float(v[mm].sum() / max(e_src[k][mm].sum(), 1e-12)) for k, v in src.items()}
        rows_d.append(dict(scope="test", bin=bin_label(i), n=int(mm.sum()), **{f"ratio::{k}": v for k, v in vals.items()}))
        say("    " + f"{bin_label(i):>12s}  {int(mm.sum()):5d}  " + "  ".join(f"{vals[k]:28.2f}" for k in src))

    def log2r(rows, x):
        tc = tcode_te[rows]
        xr = x[rows]
        e = (np.bincount(tc, weights=xr, minlength=n_tf_te) / np.maximum(np.bincount(tc, minlength=n_tf_te), 1))[tc]
        nr, fr = near_te[rows], far_te[rows]
        a_, b_ = xr[nr].sum() / max(e[nr].sum(), 1e-12), xr[fr].sum() / max(e[fr].sum(), 1e-12)
        return float(np.log2(a_ / b_)) if a_ > 0 and b_ > 0 else float("nan")

    say(f"  近端(<{near_bp}bp，{int(near_te.sum())} 条) vs 远端(≥{far_bp}bp，{int(far_te.sum())} 条)的 log2(TF调整比之比) "
        f"[按基因整群 bootstrap {n_boot} 次]；>0 = 近端更容易响应")
    all_rows = np.arange(len(te))
    lr_pt = {k: log2r(all_rows, v) for k, v in src.items()}
    lr_bt = {k: [log2r(br, v) for br in boot_rows] for k, v in src.items()}
    nf_rows = []
    for k in src:
        lo, hi = ci_of(lr_bt[k])
        nf_rows.append(dict(source=k, log2R=lr_pt[k], ci_lo=lo, ci_hi=hi))
        say(f"    {k}: log2R={lr_pt[k]:+.3f}[{lo:+.3f},{hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
    main_key = next((k for k in src if k.startswith(main_run + "[")), None)
    if main_key:
        for k in src:
            if k in ("数据(标签)", main_key):
                continue
            d_ = [a - b for a, b in zip(lr_bt[k], lr_bt[main_key])]
            lo, hi = ci_of(d_)
            nf_rows.append(dict(source=f"{k} − {main_key}", log2R=lr_pt[k] - lr_pt[main_key], ci_lo=lo, ci_hi=hi))
            say(f"    Δ {k} − 主线: {lr_pt[k] - lr_pt[main_key]:+.3f}[{lo:+.3f},{hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
    pd.DataFrame(rows_d).to_csv(os.path.join(outdir, "dist_response.csv"), index=False)
    pd.DataFrame(nf_rows).to_csv(os.path.join(outdir, "dist_nearfar.csv"), index=False)

    # ---- 模型读取/评估的公共部分 ----
    def load_ck(p):
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:  # 很老的 torch 没有 weights_only 参数
            return torch.load(p, map_location="cpu")

    def seeds_of(run):
        out = []
        for p in sorted(glob.glob(os.path.join(ckpt_root, run, "seed*_best.pt"))):
            mt = re.match(r"seed(\d+)_best\.pt$", os.path.basename(p))
            if mt:
                out.append((int(mt.group(1)), p))
        return sorted(out)

    def build(ck, dset):
        """按 checkpoint 重建模型并把 Dataset 切到训练时的条件编码/消融；返回 (model, head_states 或 None, eval_kw)。"""
        cfg = ck.get("train_config") or {}
        if not ck.get("model_kwargs"):
            raise RuntimeError("checkpoint 没有 model_kwargs(第2批之前的老格式)，24 号不支持")
        model = tl.SiameseHeadsModel(**ck["model_kwargs"])
        model.load_state_dict(ck["model_state"], strict=True)
        model = model.to(device)
        model.eval()
        dset.set_ctx_mode(cfg.get("ctx_mode", "legacy"), float(cfg.get("ctx_clip", 3.0)), verbose=False)
        if hasattr(dset, "set_ablation"):
            dset.set_ablation(cfg.get("ablation", "none"), verbose=False)
        hs = ck.get("head_states") if variant == "perhead" else None
        ekw = dict(device=device, num_workers=int(num_workers), amp=cfg.get("amp", "bf16"),
                   forward_mode=cfg.get("forward_mode", "grouped"), eval_batch_size=int(eval_batch_size),
                   layout_buckets=1, lambda_b=cfg.get("lambda_b", 1.0), lambda_c=cfg.get("lambda_c", 1.0),
                   lambda_sign=cfg.get("lambda_sign", 0.1), gamma=cfg.get("gamma", 2.0), beta=cfg.get("beta", 0.999),
                   lambda_a=cfg.get("lambda_a", 1.0), loss_b=cfg.get("loss_b", "mse"),
                   huber_delta=cfg.get("huber_delta", 1.0), lambda_bd=0.0)  # loss 不看，稠密项关掉免得打印提示
        return model, hs, ekw

    def run_eval(model, hs, dset, rows, ekw):
        if hs:
            ev = tl.evaluate_per_head(model, hs, dset, list(rows), class_counts, batch_size=128, **ekw)
        else:
            ev = tl.evaluate(model, dset, list(rows), class_counts, batch_size=128, **ekw)
        pr = softmax_np(to_np(ev["logits_c"]))
        return to_np(ev["y_a_pred"]), to_np(ev["y_b_pred"]), pr[:, 0], pr[:, 2]

    def free(model):
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    sfx = "_ph" if variant == "perhead" else ""
    runs = [(main_run, "主线")] + ([(control_run, "对照(训练时整体平移)")] if control_run else [])
    run_seeds = {r: seeds_of(r) for r, _ in runs}
    for r, lab in runs:
        say(f"{lab} {r}: seed {[s for s, _ in run_seeds[r]]}")
    if not run_seeds[main_run]:
        raise SystemExit(f"{ckpt_root}/{main_run}/ 下没有 seed*_best.pt")

    # ---- [2] Head A 在全部 test 基因上 ----
    ha_rows = []
    if head_a_all:
        say(f"\n[2] Head A 在全部 Head A test 基因上(不只是进 B/C 网格的那些)；主线 {len(run_seeds[main_run])} 个 seed，"
            f"{variant} 权重的 A 头")
        ha = pd.read_parquet(paths["head_a"])
        if "split" not in ha.columns:
            say("  head_a 文件没有 split 列，跳过")
        else:
            test_genes = [g for g, sp in ha["split"].items() if sp == "test"]
            d0 = ds.tf_list[0]
            pseudo = pd.DataFrame(dict(gene_id=test_genes, tf_depleted=d0, log2fc=np.nan, direction_3class="ns"))
            p_path = os.path.join(outdir, "_headA_pseudo_labels.parquet")
            pseudo.to_parquet(p_path, index=False)
            ds_a = tl.TFLayoutDataset(paths["layout"], p_path, paths["head_a"], paths["sgd"],
                                      paths["promoter_tokens"], paths["bpe_tokenizer"])
            ga = ds_a.samples["gene_id"].to_numpy()
            grid = set(g_te.tolist())
            is_grid = np.array([g in grid for g in ga], bool)
            has_lay = np.array([len(ds_a.layout_by_gene.get(g, {"tf_idx": []})["tf_idx"]) > 0 for g in ga], bool)
            ya_t = np.array([ds_a.head_a.get(g, np.nan) for g in ga], np.float64)
            say(f"  Head A test 基因 {len(ga)} 个：进 B/C 网格的 {int(is_grid.sum())} 个、网格外 {int((~is_grid).sum())} 个"
                f"(网格外里启动子上没有任何位点的 {int((~is_grid & ~has_lay).sum())} 个)；伪样本的 D 固定为 {d0}"
                "(Head A=ŷ_WT，跟 D 无关)")
            E_main = exp_cache.get(main_run)
            if E_main is None and os.path.exists(os.path.join(results_root, main_run, "predictions_test.parquet")):
                E_main = pd.read_parquet(os.path.join(results_root, main_run, "predictions_test.parquet"))
            preds_a = {}
            for s, p in run_seeds[main_run]:
                ck = load_ck(p)
                model, hs, ekw = build(ck, ds_a)
                # 每个伪样本是一个不同的基因：分组评估一个 batch 里的基因数=eval_batch_size(平时约4个)，cis 分支的显存会
                # 大很多，所以这里把每 batch 条数压到 64
                ya_p, _, _, _ = run_eval(model, hs, ds_a, range(len(ds_a)), dict(ekw, eval_batch_size=min(64, int(eval_batch_size))))
                free(model)
                preds_a[s] = ya_p
                col = f"y_a_seed{s}{sfx if hs else ''}"
                rep = ""
                if E_main is not None and col in E_main.columns:
                    ref = E_main.drop_duplicates("gene_id").set_index("gene_id")[col]
                    mm = is_grid & np.isin(ga, ref.index.to_numpy())
                    if mm.any():
                        dmax = float(np.max(np.abs(ya_p[mm] - ref.reindex(ga[mm]).to_numpy(np.float64))))
                        rep = f"；网格基因跟 17 号导出 {col} 最大差 {dmax:.2e}"
                        if dmax > 10 * repro_tol:
                            raise SystemExit(f"Head A 复现核对失败(seed{s} 最大差 {dmax:.3g})：模型重建或 Dataset 设置跟 17 号不一致")
                say(f"  seed{s}: r 全部={pearson(ya_p, ya_t):.4f}  网格={pearson(ya_p[is_grid], ya_t[is_grid]):.4f}  "
                    f"网格外={pearson(ya_p[~is_grid], ya_t[~is_grid]):.4f}{rep}")
            ens = np.mean([v for v in preds_a.values()], 0)
            hrng = np.random.default_rng(1)
            parts = {"全部": np.ones(len(ga), bool), "网格(=820基因口径)": is_grid, "网格外": ~is_grid,
                     "网格外且有位点": ~is_grid & has_lay, "网格外且无位点": ~is_grid & ~has_lay}
            for nm, msk in parts.items():
                ix = np.where(msk & np.isfinite(ya_t))[0]
                if len(ix) < 5:
                    continue
                bt = [pearson(ens[b_], ya_t[b_]) for b_ in (hrng.choice(ix, len(ix)) for _ in range(int(n_boot)))]
                lo, hi = ci_of(bt)
                per_s = [pearson(v[ix], ya_t[ix]) for v in preds_a.values()]
                say(f"  {len(preds_a)}-seed 集成 {nm}({len(ix)} 个基因): r={pearson(ens[ix], ya_t[ix]):.4f}[{lo:.4f},{hi:.4f}]"
                    f"  逐seed {np.mean(per_s):.4f}±{np.std(per_s):.4f}")
            ha_rows = [dict(gene_id=g, in_grid=bool(ig), has_sites=bool(hl), y_a_true=t_, y_a_ens=e_,
                            **{f"y_a_seed{s}": float(v[i]) for s, v in preds_a.items()})
                       for i, (g, ig, hl, t_, e_) in enumerate(zip(ga, is_grid, has_lay, ya_t, ens))]
            pd.DataFrame(ha_rows).to_csv(os.path.join(outdir, "head_a_all_genes.csv"), index=False)
            del ds_a

    # ---- 冒烟：主线第一个 seed、少量 test 基因，确认探针真的改变了预测，并估计用时 ----
    probes = [("none", 0.0)] + [("offset", float(v)) for v in offsets if v != 0] + [("jitter", float(v)) for v in jitters]
    n_evals = sum(len(run_seeds[r]) for r, _ in runs) * len(probes)
    if smoke_first:
        s0, p0 = run_seeds[main_run][0]
        sm_genes = set(uniq_g[:int(smoke_genes)].tolist())
        sm_rows = te[np.isin(g_te, list(sm_genes))]
        ck = load_ck(p0)
        model, hs, ekw = build(ck, ds)
        say(f"\n===== 冒烟：{main_run} seed{s0}、{len(sm_genes)} 个 test 基因({len(sm_rows)} 条) =====")
        res = {}
        for kind, v in (("none", 0.0), ("offset", 100.0), ("jitter", 50.0)):
            ds.set_position_probe(kind, v, verbose=False)
            t0 = time.time()
            res[(kind, v)] = run_eval(model, hs, ds, sm_rows, ekw)
            res[(kind, v)] = res[(kind, v)] + (time.time() - t0,)
        ds.set_position_probe("none", verbose=False)
        free(model)
        base = res[("none", 0.0)]
        for key in (("offset", 100.0), ("jitter", 50.0)):
            r_ = res[key]
            dA, dB = float(np.max(np.abs(r_[0] - base[0]))), float(np.max(np.abs(r_[1] - base[1])))
            dC = float(max(np.max(np.abs(r_[2] - base[2])), np.max(np.abs(r_[3] - base[3]))))
            say(f"  {key[0]}={key[1]:g}: 相对未扰动 最大|Δ| Head A {dA:.3g}、Head B {dB:.3g}、P(dn/up) {dC:.3g}")
            if key[0] == "offset" and max(dA, dB, dC) == 0.0:
                raise SystemExit("冒烟失败：offset 探针没有改变任何预测——多半是 DataLoader worker 没拿到探针后的 Dataset"
                                 "(检查 16 号 evaluate 是否每次现建 loader)，先把上面的日志贴回来")
        sec_per_row = max(np.mean([res[k][4] for k in res]) / max(len(sm_rows), 1), 1e-7)
        say(f"冒烟通过：每次评估约 {sec_per_row * len(te):.0f} 秒(按行数线性外推，含 worker 启动，偏保守)；"
            f"[3][4] 共 {n_evals} 次评估，预计约 {sec_per_row * len(te) * n_evals / 60:.0f} 分钟")

    # ---- [3][4] 探针扫描 ----
    say(f"\n[3][4] 探针扫描：{len(probes)} 个探针(未扰动 + {len(offsets)} 个 offset + {len(jitters)} 个 jitter) × "
        f"{sum(len(run_seeds[r]) for r, _ in runs)} 个模型，test {len(te)} 条")
    acc, per_seed = {}, {}
    for run, lab in runs:
        E_run = exp_cache.get(run)
        if E_run is None and os.path.exists(os.path.join(results_root, run, "predictions_test.parquet")):
            E_run = pd.read_parquet(os.path.join(results_root, run, "predictions_test.parquet"))
        pos_e = (pd.Index(E_run["gene_id"].astype(str) + "|" + E_run["tf_depleted"].astype(str)).get_indexer(key_te)
                 if E_run is not None else None)
        for s, p in run_seeds[run]:
            t_s = time.time()
            ck = load_ck(p)
            try:
                model, hs, ekw = build(ck, ds)
            except Exception as e:  # noqa: BLE001 —— 一个 seed 坏了不连累别的
                say(f"  ⚠ {run} seed{s}: 重建失败({type(e).__name__}: {e})，跳过")
                continue
            abl = (ck.get("train_config") or {}).get("ablation", "none")
            if abl not in ("none", "shift_position"):
                say(f"  ⚠ {run} seed{s}: ablation={abl}，位置探针只支持 none/shift_position，跳过")
                free(model)
                continue
            if variant == "perhead" and not hs:
                say(f"  ⚠ {run} seed{s}: 没有 head_states，退回选中权重")
            for kind, v in probes:
                ds.set_position_probe(kind, v, verbose=False)
                ya_p, yb_p, pdn, pup = run_eval(model, hs, ds, te, ekw)
                if kind == "none" and (pos_e is None or (pos_e < 0).any()):
                    say(f"  {run} seed{s}: 没有 17 号导出或样本对不上，跳过复现核对")
                elif kind == "none":
                    chk = []
                    for nm, arr_, col in (("y_b", yb_p, f"y_b_seed{s}{sfx if hs else ''}"),
                                          ("p_down", pdn, f"p_down_seed{s}{sfx if hs else ''}"),
                                          ("p_up", pup, f"p_up_seed{s}{sfx if hs else ''}")):
                        if col in E_run.columns:
                            chk.append(float(np.max(np.abs(arr_ - E_run[col].to_numpy(np.float64)[pos_e]))))
                    if chk:
                        mx = max(chk)
                        say(f"  {run} seed{s}: 未扰动时跟 17 号导出逐样本最大差 {mx:.2e}"
                            + ("  ✓" if mx <= repro_tol else "  ⚠ 超过容忍度"))
                        if mx > 10 * repro_tol:
                            raise SystemExit(f"复现核对失败({run} seed{s} 最大差 {mx:.3g})：重建/Dataset 设置跟 17 号不一致，"
                                             "探针结果不可信，先把日志贴回来")
                ya_g = ya_p[first_row]
                P_s = dict(ya_g=ya_g, yb=yb_p, pdn=pdn, pup=pup)
                per_seed.setdefault((run, kind, v), []).append(mets(np.arange(len(te)), np.arange(len(uniq_g)), P_s))
                a_ = acc.setdefault((run, kind, v), dict(ya_g=0.0, yb=0.0, pdn=0.0, pup=0.0, n=0))
                for k_ in ("ya_g", "yb", "pdn", "pup"):
                    a_[k_] = a_[k_] + P_s[k_]
                a_["n"] += 1
            ds.set_position_probe("none", verbose=False)
            free(model)
            say(f"  {run} seed{s}: {len(probes)} 个探针完成，{(time.time() - t_s) / 60:.1f} 分钟")

    # ---- 集成指标、分层、配对 CI(同一批 test 行、同一组模型，探针 − 未扰动) ----
    say("\n汇总(集成=同一个实验全部 seed 的预测取平均；Δ=探针 − 未扰动；CI=按基因整群 bootstrap，只算\"全部\"分层)")
    ens = {}
    for key, a_ in acc.items():
        ens[key] = {k_: a_[k_] / a_["n"] for k_ in ("ya_g", "yb", "pdn", "pup")}
        ens[key]["n"] = a_["n"]
    strata = {"全部": np.ones(len(te), bool), "D∈L_g": in_lg_te, "D∉L_g": ~in_lg_te}
    all_g = np.arange(len(uniq_g))
    mkeys = ("A_r", "B_r", "B_sign", "B_r|TF", "C_AUCdn", "C_APdn", "C_AUCup", "C_APup", "C_AUCany")
    point, boot_m = {}, {}
    for key, P in ens.items():
        for sn, sm in strata.items():
            point[key + (sn,)] = mets(np.where(sm)[0], all_g if sn == "全部" else None, P)
        boot_m[key] = [mets(br, dr, P) for br, dr in zip(boot_rows, draws)]
    out_rows = []
    for run, lab in runs:
        if (run, "none", 0.0) not in ens:
            continue
        b0 = boot_m[(run, "none", 0.0)]
        p0 = point[(run, "none", 0.0, "全部")]
        say(f"\n  ── {lab} {run}({ens[(run, 'none', 0.0)]['n']} 个 seed 集成)；未扰动：" +
            "  ".join(f"{mk}={p0.get(mk, np.nan):.3f}" for mk in mkeys))
        for kind in ("offset", "jitter"):
            say(f"    {'[3] 整体平移 Δ(bp)' if kind == 'offset' else '[4] 局部抖动 σ(bp)'}：" +
                ("A_r / B_r|TF / C_APdn 带 CI，其余只标 *(CI 不含0)" if kind == "offset" else "同上"))
            for (k_run, k_kind, v), _ in sorted(ens.items(), key=lambda kv: kv[0][2]):
                if k_run != run or k_kind != kind:
                    continue
                pt = point[(run, kind, v, "全部")]
                bt = boot_m[(run, kind, v)]
                txt = []
                for mk in mkeys:
                    d0 = pt.get(mk, np.nan) - p0.get(mk, np.nan)
                    lo, hi = ci_of([x.get(mk, np.nan) - y.get(mk, np.nan) for x, y in zip(bt, b0)])
                    star = "*" if (lo > 0 or hi < 0) else ""
                    txt.append(f"{mk}={d0:+.3f}[{lo:+.3f},{hi:+.3f}]{star}" if mk in ("A_r", "B_r|TF", "C_APdn")
                               else f"{mk}={d0:+.3f}{star}")
                    ps_ = per_seed.get((run, kind, v), [])
                    ps0 = per_seed.get((run, "none", 0.0), [])
                    ps_d = [a.get(mk, np.nan) - b.get(mk, np.nan) for a, b in zip(ps_, ps0)]
                    out_rows.append(dict(run=run, probe=kind, value=v, stratum="全部", metric=mk,
                                         value_ens=pt.get(mk, np.nan), delta=d0, ci_lo=lo, ci_hi=hi,
                                         delta_seed_mean=float(np.nanmean(ps_d)) if ps_d else np.nan,
                                         n_seed_worse=int(np.sum(np.asarray(ps_d) < 0)) if ps_d else 0))
                say(f"      {v:+6.0f}: " + "  ".join(txt))
                for sn in ("D∈L_g", "D∉L_g"):
                    ps_ = point[(run, kind, v, sn)]
                    b_ = point[(run, "none", 0.0, sn)]
                    for mk in mkeys[1:]:
                        out_rows.append(dict(run=run, probe=kind, value=v, stratum=sn, metric=mk,
                                             value_ens=ps_.get(mk, np.nan),
                                             delta=ps_.get(mk, np.nan) - b_.get(mk, np.nan)))
        say("    分层(点估计 Δ：B_r|TF / C_APdn / C_AUCany)：")
        for kind in ("offset", "jitter"):
            for (k_run, k_kind, v), _ in sorted(ens.items(), key=lambda kv: kv[0][2]):
                if k_run != run or k_kind != kind:
                    continue
                seg = []
                for sn in ("D∈L_g", "D∉L_g"):
                    ps_, b_ = point[(run, kind, v, sn)], point[(run, "none", 0.0, sn)]
                    seg.append(f"{sn} " + "/".join(f"{ps_.get(mk, np.nan) - b_.get(mk, np.nan):+.3f}"
                                                   for mk in ("B_r|TF", "C_APdn", "C_AUCany")))
                say(f"      {kind} {v:+6.0f}: " + "   ".join(seg))
    M = pd.DataFrame(out_rows)
    M.to_csv(os.path.join(outdir, "scan_metrics.csv"), index=False)

    # ---- 敏感尺度摘要 ----
    say("\n敏感尺度(集成、\"全部\"分层；明显=Δ≤−0.01 且 CI 上界 <0)：")
    for run, lab in runs:
        if M.empty or run not in set(M["run"]):
            continue
        for mk in ("A_r", "B_r|TF", "C_APdn", "C_AUCany"):
            sub = M[(M["run"] == run) & (M["stratum"] == "全部") & (M["metric"] == mk)]
            bad = sub[(sub["delta"] <= -0.01) & (sub["ci_hi"] < 0)]
            up = bad[(bad["probe"] == "offset") & (bad["value"] < 0)]["value"]
            dn = bad[(bad["probe"] == "offset") & (bad["value"] > 0)]["value"]
            jt = bad[bad["probe"] == "jitter"]["value"]
            off = sub[sub["probe"] == "offset"]["delta"]
            say(f"  {lab} {mk}: 往上游平移最小明显 |Δ|={(f'{-up.max():.0f}bp' if len(up) else '无')}  往下游={(f'{dn.min():.0f}bp' if len(dn) else '无')}"
                f"  抖动最小明显 σ={(f'{jt.min():.0f}bp' if len(jt) else '无')}  offset 范围内最大|Δ|="
                f"{(f'{off.abs().max():.3f}' if len(off) else 'nan')}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, kind in zip(axes, ("offset", "jitter")):
            for run, lab in runs:
                for mk, col in (("A_r", "C0"), ("B_r|TF", "C1"), ("C_APdn", "C2"), ("C_AUCany", "C3")):
                    sub = M[(M["run"] == run) & (M["stratum"] == "全部") & (M["metric"] == mk) & (M["probe"] == kind)]
                    if sub.empty:
                        continue
                    xs = [0.0] + sub["value"].tolist() if kind == "offset" else sub["value"].tolist()
                    ys = [0.0] + sub["delta"].tolist() if kind == "offset" else sub["delta"].tolist()
                    o = np.argsort(xs)
                    ax.plot(np.asarray(xs)[o], np.asarray(ys)[o], color=col, ls="-" if run == main_run else "--",
                            marker="o", ms=3, label=f"{mk} ({'main' if run == main_run else 'shift-trained'})")
            ax.axhline(0, color="k", lw=.6)
            ax.set_xlabel("global offset Δ (bp; <0 = upstream)" if kind == "offset" else "jitter σ (bp)")
            if kind == "jitter":
                ax.set_xscale("log")
            ax.set_ylabel("Δ metric vs unperturbed (ensemble)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig", "pos_scan.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        say("(没有 matplotlib，跳过画图)")

    say("\n判读(预先写下，建议，不是硬规则；详见文件头)：\n"
        "  (a) 对照(shift 训练)模型 offset 范围内最大|Δ|<0.01 -> 探针干净；主线\"最小明显 |Δ|\"= 绝对位置的敏感尺度；\n"
        "  (b) 主线\"抖动最小明显 σ\"= 有效位置分辨率；≥100bp -> 模型没用 <50bp 的精细间距；\n"
        "  (c) [1] 数据 log2R 的 CI>0 且主线同号、no_position/shift 相对主线的差 CI<0 -> 主线学到了\"D 离 ATG 近才更可能起作用\"；\n"
        "  (d) [2] 1108 基因 r 跟 820 基因 r 差 <0.03 -> 可直接跟 CITRA 比；网格外基因明显差 -> 第8批全基因训练要分开报告。")
    say(f"\n写出 {outdir}/summary.txt、dist_response.csv、dist_nearfar.csv、head_a_all_genes.csv、scan_metrics.csv、fig/pos_scan.png"
        f"  (总用时 {(time.time() - t_all) / 60:.1f} 分钟)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(metrics=M, head_a=pd.DataFrame(ha_rows))


if __name__ == "__main__":
    run_pos_scan(**CONFIG)
