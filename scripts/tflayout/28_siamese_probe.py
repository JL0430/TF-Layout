# scripts/tflayout/28_siamese_probe.py

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
    run_with="v8_headA_all",            # W：训练时删位点(论文主模型)
    run_without="v10_abl_no_knockout",  # N：训练时不删位点(第11批补满 5 个 seed)
    variant="perhead",                  # B/C 头用哪套权重："perhead"(报告口径)或 "selected"
    n_boot=500,                         # 按基因整群 bootstrap 次数(跟 19 号一样)
    min_rows_tf=20,                     # [3] 该 TF 在 test 里 D∈L_g 的行数下限
    min_sig_tf=8,                       # [3] 算 role_D 需要的 D∈L_g 显著行(down+up)下限
    null_csv="out/results/_paper/null_seed_variability.csv",  # 27 号零分布；没有就只给 bootstrap CI
    compare_dir="out/results/_compare_b11",                    # 19 号第11批(自检 s3)
    device=None,                        # None=有 GPU 用 GPU
    num_workers=4,
    eval_batch_size=512,
    smoke_first=True, smoke_genes=40,   # 冒烟：W 第一个 seed、40 个 test 基因，两种推理各走一遍
    repro_tol=1e-3,                     # 原生格跟 17 号导出的容忍度(bf16 + batch 组成，24/25 号实测 1e-3 量级)；>10 倍直接停
    outdir="out/results/_siamese_probe",
)


def run_siamese_probe(paths, ckpt_root, results_root, run_with, run_without, variant, n_boot, min_rows_tf,
                      min_sig_tf, null_csv, compare_dir, device, num_workers, eval_batch_size, smoke_first,
                      smoke_genes, repro_tol, outdir):
    """唯一入口，各段见文件头。"""
    t_all = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    lines, md, tex = [], [], []
    METS = ("B_r", "B_sign_acc", "B_r_within_tf", "C_auroc_down", "C_auprc_down", "C_auroc_up", "C_auprc_up")
    SHOW = ("C_auprc_down", "C_auprc_up", "C_auroc_down", "C_auroc_up", "B_r_within_tf", "B_r", "B_sign_acc")
    SHORT = dict(B_r="B_r", B_sign_acc="B_sign", B_r_within_tf="B_r|TF", C_auroc_down="C_AUCdn",
                 C_auprc_down="C_APdn", C_auroc_up="C_AUCup", C_auprc_up="C_APup")
    TEXM = dict(B_r=r"$r_B$", B_sign_acc="Sign acc.", B_r_within_tf=r"$r_B$ within TF",
                C_auroc_down=r"AUROC$_\downarrow$", C_auprc_down=r"AUPRC$_\downarrow$",
                C_auroc_up=r"AUROC$_\uparrow$", C_auprc_up=r"AUPRC$_\uparrow$")
    SCOPES = ("全部", "D∈L_g", "D∉L_g")
    NULL_SCOPE = {"全部": "test", "D∈L_g": "D∈L_g", "D∉L_g": "D∉L_g"}
    CELLS = ("W_del", "W_nodel", "N_nodel", "N_del")
    CELL_DESC = {"W_del": "主模型原样(训练删/推理删)", "W_nodel": "反事实(训练删/推理不删)",
                 "N_nodel": "训练消融原样(训练不删/推理不删)", "N_del": "零样本(训练不删/推理删)"}
    # (名字, 变体, 参照, 是否涉及两组不同权重 = 有训练随机性)
    CONTRASTS = (("Δ_infer", "W_nodel", "W_del", False), ("Δ_train", "N_nodel", "W_nodel", True),
                 ("Δ_total", "N_nodel", "W_del", True), ("Δ_zero", "N_del", "N_nodel", False))
    MODE = {"del": "none", "nodel": "no_knockout"}

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def load(fn):  # 跟 18/22/24/25 号同一种按路径动态加载(文件名以数字开头，不能 import)
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
        if len(x) < 3 or x.std() <= 1e-12 * max(1.0, abs(float(x.mean()))) or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def spearman(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return float("nan")
        return pearson(pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy())

    def auroc_auprc(score, pos):  # 跟 19/27 号逐行相同
        pos = np.asarray(pos, bool)
        n1 = int(pos.sum())
        n0 = len(pos) - n1
        if n1 == 0 or n0 == 0:
            return float("nan"), float("nan")
        sc = np.asarray(score, np.float64)
        _, inv_, cnt_ = np.unique(sc, return_inverse=True, return_counts=True)
        r = (np.cumsum(cnt_) - (cnt_ - 1) / 2.0)[inv_.reshape(-1)]
        auc = float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
        p = pos[np.argsort(-sc, kind="mergesort")]
        return auc, float((np.cumsum(p) / np.arange(1, len(p) + 1))[p].sum() / n1)

    def pooled_wtf(score, pos, tcode, min_each=2):
        """TF 内合并 AUROC = Σ_t U_t / Σ_t n1·n0(同一 TF 内随机一对正/负例排对的概率)；每个 TF 正负例各 ≥min_each。"""
        num = den = 0.0
        for t_ in np.unique(tcode):
            m_ = tcode == t_
            p_ = pos[m_]
            n1 = int(p_.sum())
            n0 = len(p_) - n1
            if n1 < min_each or n0 < min_each:
                continue
            a_, _ = auroc_auprc(score[m_], p_)
            num += a_ * n1 * n0
            den += n1 * n0
        return float(num / den) if den else float("nan")

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

    def f3(v, sign=False):
        if v is None or not np.isfinite(v):
            return "nan"
        return f"{v:+.3f}" if sign else f"{v:.3f}"

    def rd(path):
        if path and os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:  # noqa: BLE001 —— 读不出来就当缺失
                say(f"  ⚠ 读 {path} 失败：{type(e).__name__}: {e}")
        return None

    def texesc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

    def tex_table(caption, label, header, rows, colspec, notes="", wide=False):
        """booktabs 表，跟 27 号同一写法(\\resizebox 缩到栏宽)。"""
        env = "table*" if wide else "table"
        out = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
               r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%",
               rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
        for r_ in rows:
            out.append(r_ if (r_.strip() in ("\\midrule", "\\addlinespace") or r_.rstrip().endswith("\\\\"))
                       else (r_ + r" \\"))
        out += [r"\bottomrule", r"\end{tabular}}"]
        if notes:
            out.append(rf"\par\smallskip\parbox{{0.97\linewidth}}{{\scriptsize {notes}}}")
        out.append(rf"\end{{{env}}}")
        tex.append("\n".join(out) + "\n")

    tl = load("16_train_loop.py")
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "CPU"
    say("=" * 100)
    say(f"删位点推理期探针(28 号，第12批 2026-10-02a)  {time.strftime('%Y-%m-%d %H:%M:%S')}  16号代码版本 {tl.CODE_VERSION}  "
        f"设备={device}({gpu_name})  权重={variant}")
    say("=" * 100)

    # ------------------------------------------------------------------ [0] Dataset、切分、D∈L_g
    ds = tl.TFLayoutDataset(paths["layout"], paths["labels"], paths["head_a"], paths["sgd"],
                            paths["promoter_tokens"], paths["bpe_tokenizer"])
    if not hasattr(ds, "set_ablation"):
        raise SystemExit("09_torch_dataset.py 没有 set_ablation(第4批)，先替换 09 号")
    split_idx = tl.build_split_indices(ds, paths["head_a"])
    smp = ds.samples
    genes_all = smp["gene_id"].to_numpy()
    tfs_all = smp["tf_depleted"].to_numpy()
    yb_all = pd.to_numeric(smp["log2fc"], errors="coerce").to_numpy(np.float64)
    yc_all = smp["direction_3class"].map(ds.class2idx).to_numpy().astype(np.int64)
    vc = smp["direction_3class"].value_counts()
    class_counts = torch.tensor([float(vc.get(k_, 0.0)) for k_ in ("down", "ns", "up")])
    site_cnt = {}
    for g, lg in ds.layout_by_gene.items():
        ti = np.asarray(lg["tf_idx"], np.int64)
        if ti.size:
            u_, c_ = np.unique(ti, return_counts=True)
            for t_, n_ in zip(u_.tolist(), c_.tolist()):
                site_cnt[(g, int(t_))] = int(n_)
    dep_idx = np.array([ds.tf2idx.get(t, -1) for t in tfs_all], np.int64)
    nds_all = np.array([site_cnt.get((g, int(d)), 0) for g, d in zip(genes_all, dep_idx)], np.int64)
    in_lg_all = nds_all > 0
    te = np.asarray(split_idx["test"], np.int64)
    tr = np.asarray(split_idx["train"], np.int64)
    g_te, t_te = genes_all[te], tfs_all[te]
    yb_te, yc_te, in_lg_te, nds_te = yb_all[te], yc_all[te], in_lg_all[te], nds_all[te]
    tcode_te = pd.factorize(t_te)[0]
    n_tf_te = int(tcode_te.max()) + 1
    key_te = pd.Index((pd.Series(g_te).astype(str) + "|" + pd.Series(t_te).astype(str)).to_numpy())
    uniq_g, inv_g = np.unique(g_te, return_inverse=True)
    rows_of_gene = np.split(np.argsort(inv_g, kind="stable"), np.cumsum(np.bincount(inv_g))[:-1])
    brng = np.random.default_rng(0)
    boot_rows = [np.concatenate([rows_of_gene[j] for j in brng.integers(0, len(uniq_g), len(uniq_g))])
                 for _ in range(int(n_boot))]
    scope_mask = {"全部": np.ones(len(te), bool), "D∈L_g": in_lg_te, "D∉L_g": ~in_lg_te}
    scope_rows = {sc: np.flatnonzero(m_) for sc, m_ in scope_mask.items()}
    boot_scope = {sc: [br[m_[br]] for br in boot_rows] for sc, m_ in scope_mask.items()}
    sig_te = np.isfinite(yb_te)
    say(f"\n[0] test：{len(te)} 行 = {len(uniq_g)} 基因 × {n_tf_te} 个被耗竭 TF；显著 {int(sig_te.sum())}(down {int((yc_te == 0).sum())}"
        f" / up {int((yc_te == 2).sum())})")
    say(f"    D∈L_g {int(in_lg_te.sum())} 行({in_lg_te.mean():.1%})，显著 {int((in_lg_te & sig_te).sum())}(down "
        f"{int((in_lg_te & (yc_te == 0)).sum())} / up {int((in_lg_te & (yc_te == 2)).sum())})；其中 D 在启动子上 ≥2 个位点的占 "
        f"{float((nds_te[in_lg_te] >= 2).mean()) if in_lg_te.any() else float('nan'):.1%}")
    exports = {}
    for run in (run_with, run_without):
        p_ = os.path.join(results_root, run, "predictions_test.parquet")
        if not os.path.exists(p_):
            say(f"    (没有 {p_}：(s1) 对 {run} 跳过)")
            continue
        E = pd.read_parquet(p_)
        pos = pd.Index((E["gene_id"].astype(str) + "|" + E["tf_depleted"].astype(str)).to_numpy()).get_indexer(key_te)
        if (pos < 0).any():
            say(f"    ⚠ {run} 的 17 号导出有 {int((pos < 0).sum())} 条 test 行对不上，(s1) 对它跳过")
            continue
        exports[run] = E.iloc[pos].reset_index(drop=True)
    checks = []
    if run_with in exports and "D_in_Lg" in exports[run_with].columns:
        n_bad = int((exports[run_with]["D_in_Lg"].to_numpy().astype(bool) != in_lg_te).sum())
        checks.append(("s4", n_bad == 0, f"自算 D∈L_g vs 17 号导出 D_in_Lg：不一致 {n_bad} 行"))
        say(f"    (s4) {checks[-1][2]} " + ("✓" if n_bad == 0 else "⚠(先查 09 号 TF 名对齐)"))

    # ------------------------------------------------------------------ [1] 推理(2 个训练方式 × 共有 seed × 2 种推理)
    def seeds_of(run):
        out = {}
        for p_ in sorted(glob.glob(os.path.join(ckpt_root, run, "seed*_best.pt"))):
            mt = re.match(r"seed(\d+)_best\.pt$", os.path.basename(p_))
            if mt:
                out[int(mt.group(1))] = p_
        return out

    def load_ck(p_):
        try:
            return torch.load(p_, map_location="cpu", weights_only=False)
        except TypeError:  # 很老的 torch 没有 weights_only 参数
            return torch.load(p_, map_location="cpu")

    def build(ck):
        cfg = ck.get("train_config") or {}
        if not ck.get("model_kwargs"):
            raise SystemExit("checkpoint 没有 model_kwargs(第2批之前的老格式)，28 号不支持")
        model = tl.SiameseHeadsModel(**ck["model_kwargs"])
        model.load_state_dict(ck["model_state"], strict=True)
        model = model.to(device)
        model.eval()
        ds.set_ctx_mode(cfg.get("ctx_mode", "legacy"), float(cfg.get("ctx_clip", 3.0)), verbose=False)
        ekw = dict(device=device, num_workers=int(num_workers), amp=cfg.get("amp", "bf16"),
                   forward_mode=cfg.get("forward_mode", "grouped"), eval_batch_size=int(eval_batch_size),
                   layout_buckets=1, lambda_b=cfg.get("lambda_b", 1.0), lambda_c=cfg.get("lambda_c", 1.0),
                   lambda_sign=cfg.get("lambda_sign", 0.1), gamma=cfg.get("gamma", 2.0), beta=cfg.get("beta", 0.999),
                   lambda_a=cfg.get("lambda_a", 1.0), loss_b=cfg.get("loss_b", "mse"),
                   huber_delta=cfg.get("huber_delta", 1.0), lambda_bd=0.0)  # loss 不看，稠密项关掉
        return model, cfg, ekw

    def infer(model, ck, rows, ekw):
        """B 头、C 头各用自己的权重(perhead)跑一次前向；跑完恢复成选中的权重。"""
        hs = (ck.get("head_states") or {}) if variant == "perhead" else {}
        out = {}
        for h in ("B", "C"):
            st = hs.get(h) if hs else None
            model.load_state_dict(st if st is not None else ck["model_state"], strict=True)
            ev = tl.evaluate(model, ds, list(rows), class_counts, batch_size=128, **ekw)
            if h == "B":
                out["yb"] = to_np(ev["y_b_pred"])
            else:
                pr = softmax_np(to_np(ev["logits_c"]))
                out["pdn"], out["pup"] = pr[:, 0], pr[:, 2]
        model.load_state_dict(ck["model_state"], strict=True)
        out["used_ph"] = bool(hs) and ("B" in hs) and ("C" in hs)
        return out

    def free(model):
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    sw, sn = seeds_of(run_with), seeds_of(run_without)
    seeds = sorted(set(sw) & set(sn))
    say(f"\n[1] 推理：W={run_with} seed {sorted(sw)}；N={run_without} seed {sorted(sn)}；共有 seed {seeds}(下面只用共有 seed)")
    if not seeds:
        raise SystemExit(f"{ckpt_root}/{run_with}/ 与 {ckpt_root}/{run_without}/ 没有共有的 seed*_best.pt")
    k = len(seeds)
    if smoke_first:
        ck = load_ck(sw[seeds[0]])
        model, _, ekw = build(ck)
        sm_rows = te[np.isin(g_te, uniq_g[:int(smoke_genes)])]
        t0 = time.time()
        for m_ in ("del", "nodel"):
            ds.set_ablation(MODE[m_], verbose=False)
            o = infer(model, ck, sm_rows, ekw)
            if not all(np.isfinite(o[x]).all() for x in ("yb", "pdn", "pup")):
                raise SystemExit(f"冒烟：推理模式 {m_} 的预测里有非有限值")
        dt = time.time() - t0
        free(model)
        est = dt / max(len(sm_rows), 1) * len(te) * 2 * k / 60
        say(f"  冒烟通过：{len(sm_rows)} 行 × 2 种推理 × 2 个头 用 {dt:.1f} 秒；全量预计 {est:.1f} 分钟(不含载入，冒烟有预热开销，偏高)")
    preds = {c: {} for c in CELLS}
    t_inf, hist_rows, info = {}, [], {}
    for tag, run, smap in (("W", run_with, sw), ("N", run_without, sn)):
        for s in seeds:
            ck = load_ck(smap[s])
            model, cfg, ekw = build(ck)
            if tag == "W" and "cfg" not in info:
                info["cfg"], info["mk"] = dict(cfg), dict(ck.get("model_kwargs") or {})
                info["n_params"] = int(sum(int(p.numel()) for p in model.parameters()))
                info["n_trainable"] = int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))
                info["by_child"] = [(nm, int(sum(int(p.numel()) for p in ch.parameters())))
                                    for nm, ch in model.named_children()]
            hist = ck.get("history") or []
            hist_rows.append(dict(run=run, seed=s, n_val=len(hist),
                                  train_min=float(sum(float(h.get("train_sec", 0.0) or 0.0) for h in hist)) / 60,
                                  val_min=float(sum(float(h.get("val_sec", 0.0) or 0.0) for h in hist)) / 60,
                                  epochs=float(max([float(h.get("epoch_progress", np.nan)) for h in hist] or [np.nan])),
                                  best_epoch=float((ck.get("best") or {}).get("epoch_progress", ck.get("best_epoch"))
                                                   if (ck.get("best") or {}).get("epoch_progress", ck.get("best_epoch"))
                                                   is not None else np.nan),
                                  code_version=ck.get("code_version"), ablation_trained=cfg.get("ablation", "none")))
            for m_ in ("del", "nodel"):
                ds.set_ablation(MODE[m_], verbose=False)
                t0 = time.time()
                preds[f"{tag}_{m_}"][s] = infer(model, ck, te, ekw)
                t_inf[(tag, s, m_)] = time.time() - t0
            free(model)
            native = "W_del" if tag == "W" else "N_nodel"
            o = preds[native][s]
            E = exports.get(run)
            sfx = "_ph" if o["used_ph"] else ""
            rep = ""
            if E is not None and f"y_b_seed{s}{sfx}" in E.columns and f"p_down_seed{s}{sfx}" in E.columns:
                dmax = max(float(np.nanmax(np.abs(o["yb"] - E[f"y_b_seed{s}{sfx}"].to_numpy(np.float64)))),
                           float(np.nanmax(np.abs(o["pdn"] - E[f"p_down_seed{s}{sfx}"].to_numpy(np.float64)))),
                           float(np.nanmax(np.abs(o["pup"] - E[f"p_up_seed{s}{sfx}"].to_numpy(np.float64)))))
                checks.append(("s1", dmax <= 10 * repro_tol, f"{run} seed{s} 原生格 {native} vs 17 号导出 最大差 {dmax:.2e}"))
                rep = f"；(s1) 原生格跟 17 号导出最大差 {dmax:.2e}"
                if dmax > 10 * repro_tol:
                    raise SystemExit(f"(s1) 复现核对失败({run} seed{s} 最大差 {dmax:.3g})：模型重建或 Dataset 设置跟 17 号不一致")
            nd = ~in_lg_te
            d2 = max(float(np.max(np.abs(preds[f"{tag}_del"][s][x][nd] - preds[f"{tag}_nodel"][s][x][nd])))
                     for x in ("yb", "pdn", "pup")) if nd.any() else 0.0
            checks.append(("s2", d2 <= repro_tol, f"{run} seed{s} D∉L_g 行 del vs nodel 最大差 {d2:.2e}"))
            say(f"  {tag} {run} seed{s}：推理 删位点 {t_inf[(tag, s, 'del')]:.0f} 秒 / 不删 {t_inf[(tag, s, 'nodel')]:.0f} 秒"
                f"(权重={'各头最优' if o['used_ph'] else '选中'}){rep}；(s2) D∉L_g 行两种推理最大差 {d2:.2e}"
                + ("" if d2 <= repro_tol else " ⚠"))
    ens = {c: {x: np.mean([preds[c][s][x] for s in seeds], 0) for x in ("yb", "pdn", "pup")} for c in CELLS}

    # ------------------------------------------------------------------ [2] 指标与 2×2 分解
    def mets(P, rows):
        """跟 19 号 metrics 同定义(只算 B/C)；rows=test 内的行下标(可重复)。"""
        out = {}
        ybt, ybp, tc = yb_te[rows], P["yb"][rows], tcode_te[rows]
        sg = np.isfinite(ybt)
        p_, t_, c_ = ybp[sg], ybt[sg], tc[sg]
        out["B_r"] = pearson(p_, t_)
        out["B_sign_acc"] = float(np.mean(np.sign(p_) == np.sign(t_))) if len(p_) else float("nan")
        cnt = np.maximum(np.bincount(c_, minlength=n_tf_te), 1)
        mp = np.bincount(c_, weights=p_, minlength=n_tf_te) / cnt
        mt = np.bincount(c_, weights=t_, minlength=n_tf_te) / cnt
        out["B_r_within_tf"] = pearson(p_ - mp[c_], t_ - mt[c_])
        yct = yc_te[rows]
        out["C_auroc_down"], out["C_auprc_down"] = auroc_auprc(P["pdn"][rows], yct == 0)
        out["C_auroc_up"], out["C_auprc_up"] = auroc_auprc(P["pup"][rows], yct == 2)
        return out

    t2 = time.time()
    point, boots = {}, {}
    for c in CELLS:
        for sc in SCOPES:
            point[(c, sc)] = mets(ens[c], scope_rows[sc])
            boots[(c, sc)] = [mets(ens[c], br) for br in boot_scope[sc]]
    per_seed = {(c, sc, s): mets(preds[c][s], scope_rows[sc]) for c in CELLS for sc in SCOPES for s in seeds}
    say(f"\n[2] 2×2 分解({k}-seed 集成；指标定义跟 19 号相同；bootstrap {n_boot} 次按基因整群，用时 {time.time() - t2:.0f} 秒)")
    cell_rows = []
    for c in CELLS:
        for sc in SCOPES:
            for m in METS:
                cell_rows.append(dict(cell=c, desc=CELL_DESC[c], scope=sc, metric=m, ensemble=point[(c, sc)][m],
                                      **{f"seed{s}": per_seed[(c, sc, s)][m] for s in seeds}))
    pd.DataFrame(cell_rows).to_csv(os.path.join(outdir, "cells.csv"), index=False)

    null_df = rd(null_csv)
    if null_df is None:
        say(f"  (没有 {null_csv}：Δ_train/Δ_total 只给 bootstrap CI；先 --tables 一次再跑 28 号)")

    def null_rms(scope, metric, cons):
        """k=1/2 实测；k≥3 时 cons=False 按 rms_2·sqrt(2/k) 外推(=27 号 ‡)，cons=True 直接用实测 k=2(=27 号 v2 §，更保守)。"""
        if null_df is None:
            return np.nan, "无零分布"
        q = null_df[(null_df["variant"] == variant) & (null_df["scope"] == NULL_SCOPE[scope]) & (null_df["metric"] == metric)]
        if k in (1, 2):
            v = q[q["k"] == k]["rms"]
            return (float(v.iloc[0]), f"实测k={k}") if len(v) else (np.nan, "无")
        v = q[q["k"] == 2]["rms"]
        if not len(v):
            return np.nan, "无"
        return (float(v.iloc[0]), f"保守k={k}") if cons else (float(v.iloc[0]) * np.sqrt(2.0 / k), f"外推k={k}")

    crow_list = []
    for cname, a, b, trn in CONTRASTS:
        for sc in SCOPES:
            for m in METS:
                d = point[(a, sc)][m] - point[(b, sc)][m]
                arr = np.asarray([x[m] - y[m] for x, y in zip(boots[(a, sc)], boots[(b, sc)])], np.float64)
                lo, hi = ci_of(arr)
                sd_ = np.asarray([per_seed[(a, sc, s)][m] - per_seed[(b, sc, s)][m] for s in seeds], np.float64)
                row = dict(contrast=cname, variant_cell=a, reference_cell=b, scope=sc, metric=m,
                           value_variant=point[(a, sc)][m], value_reference=point[(b, sc)][m], delta=d, ci_lo=lo, ci_hi=hi,
                           test_sig=bool(np.isfinite(lo) and (lo > 0 or hi < 0)), k=k,
                           seed_delta_mean=float(np.nanmean(sd_)) if np.isfinite(sd_).any() else np.nan,
                           seed_delta_sd=float(np.nanstd(sd_, ddof=1)) if np.isfinite(sd_).sum() > 1 else np.nan,
                           seeds_better=int((sd_ > 0).sum()), seeds_worse=int((sd_ < 0).sum()), training_noise=trn,
                           tot_lo=np.nan, tot_hi=np.nan, tot_rms=np.nan, tot_how="", tot_sig=False,
                           cons_lo=np.nan, cons_hi=np.nan, cons_rms=np.nan, cons_how="", cons_sig=False)
                if trn:
                    se = (hi - lo) / 3.92 if np.isfinite(lo) and np.isfinite(hi) else np.nan
                    for cons, tg in ((False, "tot"), (True, "cons")):
                        rms, how = null_rms(sc, m, cons)
                        if np.isfinite(se) and np.isfinite(rms):
                            sdv = float(np.sqrt(se ** 2 + rms ** 2))
                            row.update({f"{tg}_lo": d - 1.96 * sdv, f"{tg}_hi": d + 1.96 * sdv,
                                        f"{tg}_sig": bool(d - 1.96 * sdv > 0 or d + 1.96 * sdv < 0)})
                        row.update({f"{tg}_rms": rms, f"{tg}_how": how})
                crow_list.append(row)
    C = pd.DataFrame(crow_list)
    C.to_csv(os.path.join(outdir, "contrasts.csv"), index=False)

    def crow(cname, sc, m):
        q = C[(C["contrast"] == cname) & (C["scope"] == sc) & (C["metric"] == m)]
        return q.iloc[0] if len(q) else None

    def mark(r_):
        return ("†" if r_["test_sig"] else "") + ("‡" if r_["tot_sig"] else "") + ("§" if r_["cons_sig"] else "")

    say("  Δ = 变体 − 参照，负 = 去掉删位点后变差；†=bootstrap CI 不含0(只含 test 抽样)；‡=合成区间不含0(训练随机性按 rms_2·√(2/k)"
        "外推)；§=保守合成区间不含0(直接用实测 k=2 的 rms)。‡/§ 只对 Δ_train、Δ_total(两组不同权重)有意义")
    for sc in SCOPES:
        m_ = scope_mask[sc]
        say(f"  ── {sc}：{int(m_.sum())} 行，显著 {int((m_ & sig_te).sum())}(down {int((m_ & (yc_te == 0)).sum())} / up "
            f"{int((m_ & (yc_te == 2)).sum())})")
        say("      " + "".ljust(34) + "".join(f"{SHORT[x]:>10}" for x in SHOW))
        for c in CELLS:
            say("      " + f"{c} {CELL_DESC[c]}"[:34].ljust(34) + "".join(f"{f3(point[(c, sc)][x]):>10}" for x in SHOW))
        for cname, a, b, _ in CONTRASTS:
            cells = []
            for x in SHOW:
                r_ = crow(cname, sc, x)
                cells.append(f"{f3(r_['delta'], True)}{mark(r_)}".rjust(10))
            say("      " + f"{cname} = {a} − {b}"[:34].ljust(34) + "".join(cells))
        for cname, _, _, _ in CONTRASTS:
            parts = []
            for x in ("C_auprc_down", "C_auprc_up", "B_r_within_tf"):
                r_ = crow(cname, sc, x)
                ci_ = f"[{f3(r_['ci_lo'], True)},{f3(r_['ci_hi'], True)}]"
                tot_ = (f" 合成({f3(r_['tot_lo'], True)},{f3(r_['tot_hi'], True)}) 保守({f3(r_['cons_lo'], True)},"
                        f"{f3(r_['cons_hi'], True)})" if r_["training_noise"] and np.isfinite(r_["tot_lo"]) else "")
                parts.append(f"{SHORT[x]} {f3(r_['delta'], True)}{ci_}{tot_} 逐seed {f3(r_['seed_delta_mean'], True)}"
                             f"±{f3(r_['seed_delta_sd'])}({int(r_['seeds_better'])}/{k} 变好)")
            say(f"      {cname}: " + "；".join(parts))

    # 自检 s3：Δ_total 跟 19 号
    diffs = []
    dvr = rd(os.path.join(compare_dir, "delta_vs_reference.csv"))
    sdl = rd(os.path.join(compare_dir, "strata_delta.csv"))
    ref_name = f"{run_with}/{variant}"
    if dvr is not None:
        q = dvr[(dvr["reference"] == ref_name) & (dvr["run"] == run_without) & (dvr["variant"] == variant)]
        for m in METS:
            qm = q[q["metric"] == m]
            if len(qm):
                diffs.append(("全部", m, float(crow("Δ_total", "全部", m)["delta"]) - float(qm["delta"].iloc[0])))
    if sdl is not None:
        q = sdl[(sdl["reference"] == ref_name) & (sdl["run"] == run_without)]
        for sc in ("D∈L_g", "D∉L_g"):
            for m in METS:
                qm = q[(q["stratum"] == sc) & (q["metric"] == m)]
                if len(qm):
                    diffs.append((sc, m, float(crow("Δ_total", sc, m)["delta"]) - float(qm["delta"].iloc[0])))
    if diffs:
        mx = max(abs(x[2]) for x in diffs)
        checks.append(("s3", mx < 5e-3, f"Δ_total vs 19 号 {compare_dir}：{len(diffs)} 项，最大差 {mx:.1e}"))
        say(f"  (s3) {checks[-1][2]} " + ("✓" if mx < 5e-3 else "⚠ " + str(sorted(diffs, key=lambda x: -abs(x[2]))[:3])))
    else:
        say(f"  (s3) {compare_dir} 里没有 {run_without} 对 {ref_name} 的差值，跳过")

    # ------------------------------------------------------------------ [3] 逐 TF 方向一致性
    say(f"\n[3] 逐 TF 方向一致性(只看 D∈L_g 的 test 行；e_C=删位点让 P(dn)−P(up) 增加多少，>0=推向 down；e_B=删位点让 ŷ_B 增加多少，"
        "<0=推向 down；role_D=(n_down−n_up)/(n_down+n_up)，D∈L_g 显著行)")
    lg_rows = scope_rows["D∈L_g"]
    eff = {}
    for tag in ("W", "N"):
        Pd, Pn = ens[f"{tag}_del"], ens[f"{tag}_nodel"]
        eff[tag] = ((Pd["pdn"] - Pd["pup"]) - (Pn["pdn"] - Pn["pup"]), Pd["yb"] - Pn["yb"])

    def role_of(idx):
        m_ = in_lg_all[idx] & (yc_all[idx] != 1)
        q = pd.DataFrame(dict(tf=tfs_all[idx][m_], dn=(yc_all[idx][m_] == 0).astype(int),
                              up=(yc_all[idx][m_] == 2).astype(int))).groupby("tf").sum()
        q["n_sig"] = q["dn"] + q["up"]
        q["role"] = (q["dn"] - q["up"]) / q["n_sig"].clip(lower=1)
        return q

    role_tr, role_te = role_of(tr), role_of(te)
    pt_tf = pd.DataFrame(dict(tf=t_te[lg_rows], eC_W=eff["W"][0][lg_rows], eB_W=eff["W"][1][lg_rows],
                              eC_N=eff["N"][0][lg_rows], eB_N=eff["N"][1][lg_rows]))
    per_tf = pt_tf.groupby("tf").agg(n_rows=("eC_W", "size"), eC_W=("eC_W", "mean"), eB_W=("eB_W", "mean"),
                                     eC_N=("eC_N", "mean"), eB_N=("eB_N", "mean"))
    per_tf = per_tf.join(role_tr[["role", "n_sig"]].rename(columns={"role": "role_train", "n_sig": "n_sig_train"}))
    per_tf = per_tf.join(role_te[["role", "n_sig"]].rename(columns={"role": "role_test", "n_sig": "n_sig_test"}))
    per_tf.reset_index().to_csv(os.path.join(outdir, "per_tf_effect.csv"), index=False)
    trng = np.random.default_rng(1)
    rho = {}
    for src in ("train", "test"):
        q = per_tf[(per_tf["n_rows"] >= min_rows_tf) & (per_tf[f"n_sig_{src}"].fillna(0) >= min_sig_tf)]
        n_q = len(q)
        bix = [trng.integers(0, n_q, n_q) for _ in range(int(n_boot))] if n_q >= 5 else []
        cells = []
        for tag in ("W", "N"):
            for kind, sgn in (("C", 1.0), ("B", -1.0)):
                xv, yv = sgn * q[f"e{kind}_{tag}"].to_numpy(np.float64), q[f"role_{src}"].to_numpy(np.float64)
                r0 = spearman(xv, yv)
                lo, hi = ci_of([spearman(xv[ix], yv[ix]) for ix in bix])
                rho[(src, tag, kind)] = (r0, lo, hi, n_q)
                cells.append(f"{tag}:{'ε_C' if kind == 'C' else '−ε_B'} {f3(r0, True)}[{f3(lo, True)},{f3(hi, True)}]")
        say(f"  role 来自 {src}({n_q} 个 TF：test D∈L_g 行 ≥{min_rows_tf}、{src} 显著 ≥{min_sig_tf})Spearman：" + "  ".join(cells))
    sig_lg = lg_rows[yc_te[lg_rows] != 1]
    pos_dn = yc_te[sig_lg] == 0
    prior_row = pd.Series(t_te[sig_lg]).map(role_tr["role"]).fillna(0.0).to_numpy(np.float64)
    boot_sig = [br[(in_lg_te[br]) & (yc_te[br] != 1)] for br in boot_rows]
    row_auc = {}
    for tag in ("W", "N"):
        for kind, sgn in (("C", 1.0), ("B", -1.0)):
            sc_ = sgn * eff[tag][0 if kind == "C" else 1]
            a0 = auroc_auprc(sc_[sig_lg], pos_dn)[0]
            w0 = pooled_wtf(sc_[sig_lg], pos_dn, tcode_te[sig_lg])
            ab = [auroc_auprc(sc_[br], yc_te[br] == 0)[0] for br in boot_sig]
            wb = [pooled_wtf(sc_[br], yc_te[br] == 0, tcode_te[br]) for br in boot_sig]
            row_auc[(tag, kind)] = (a0, *ci_of(ab), w0, *ci_of(wb))
    a_pr = auroc_auprc(prior_row, pos_dn)[0]
    a_prb = ci_of([auroc_auprc(pd.Series(t_te[br]).map(role_tr["role"]).fillna(0.0).to_numpy(np.float64),
                               yc_te[br] == 0)[0] for br in boot_sig])
    say(f"  行级(D∈L_g 显著 {len(sig_lg)} 行，down {int(pos_dn.sum())} / up {int((~pos_dn).sum())})：分数区分 down/up 的 AUROC "
        "[按基因整群 bootstrap]；TF 内合并 = 只比同一被耗竭 TF 内部(每个 TF down、up 各 ≥2)")
    for tag in ("W", "N"):
        for kind in ("C", "B"):
            a0, alo, ahi, w0, wlo, whi = row_auc[(tag, kind)]
            say(f"    {tag} {'e_C' if kind == 'C' else '−e_B'}：合并 {f3(a0)}[{f3(alo)},{f3(ahi)}]  TF 内合并 {f3(w0)}[{f3(wlo)},{f3(whi)}]")
    say(f"    参照：train 标签的 role_D 当分数(TF 层面的平均角色，TF 内恒定) 合并 {f3(a_pr)}[{f3(a_prb[0])},{f3(a_prb[1])}]")
    re_df = pd.DataFrame(dict(gene_id=g_te[lg_rows], tf_depleted=t_te[lg_rows], n_D_sites=nds_te[lg_rows],
                              y_c=yc_te[lg_rows], y_b=yb_te[lg_rows], eC_W=eff["W"][0][lg_rows], eB_W=eff["W"][1][lg_rows],
                              eC_N=eff["N"][0][lg_rows], eB_N=eff["N"][1][lg_rows]))
    re_df.to_csv(os.path.join(outdir, "row_effect_DinLg.csv.gz"), index=False, compression="gzip")

    # ------------------------------------------------------------------ [4] 参数量、算力、超参
    say("\n[4] 模型规模、算力与超参(Methods 用)")
    H = pd.DataFrame(hist_rows)
    H.to_csv(os.path.join(outdir, "compute.csv"), index=False)
    cfg0, mk0 = info.get("cfg", {}), info.get("mk", {})
    say(f"  参数量(W，{run_with}) {info.get('n_params', 0):,}(可训练 {info.get('n_trainable', 0):,})；按子模块："
        + "、".join(f"{nm} {n_:,}" for nm, n_ in info.get("by_child", [])))
    for run in (run_with, run_without):
        q = H[H["run"] == run]
        if len(q):
            say(f"  {run}：每 seed 训练 {q['train_min'].mean():.1f}±{q['train_min'].std(ddof=1) if len(q) > 1 else 0:.1f} 分钟"
                f"(另验证 {q['val_min'].mean():.1f} 分钟)；跑到 epoch {q['epochs'].min():.2f}~{q['epochs'].max():.2f}；"
                f"选中 epoch {', '.join(f3(v) for v in q['best_epoch'])}；训练时 ablation={q['ablation_trained'].iloc[0]}")
    tt = np.asarray(list(t_inf.values()), np.float64)
    say(f"  推理：{len(te)} 行 × 2 个头的前向，每次中位 {np.median(tt):.1f} 秒(= {2 * len(te) / max(np.median(tt), 1e-9):,.0f} 行·头/秒)；"
        f"设备 {gpu_name}")
    hp_rows = [dict(source="train_config", key=k_, value=repr(v_)) for k_, v_ in sorted(cfg0.items(), key=lambda x: str(x[0]))]
    hp_rows += [dict(source="model_kwargs", key=k_, value=repr(v_)) for k_, v_ in sorted(mk0.items(), key=lambda x: str(x[0]))]
    pd.DataFrame(hp_rows).to_csv(os.path.join(outdir, "hyperparams.csv"), index=False)

    def hp(key):
        v_ = cfg0.get(key, mk0.get(key, None))
        if v_ is None:
            return "--"
        if isinstance(v_, bool):
            return "yes" if v_ else "no"
        if isinstance(v_, float):
            return f"{v_:g}"
        return texesc(v_)

    HP = ((r"Embedding dim.\ / attention heads", ("d_model", "n_heads")), ("cis Transformer layers", ("cis_layers",)),
          ("Layout Transformer layers", ("lay_layers",)), ("BPE vocabulary size", ("vocab_size",)), ("Dropout", ("dropout",)),
          (r"Optimizer / weight decay", ("optimizer", "weight_decay")), ("Learning rate / schedule", ("lr", "lr_schedule")),
          ("Warm-up steps / min.\\ LR ratio", ("warmup_steps", "min_lr_ratio")),
          ("Batch size (pairs) / TFs per gene", ("batch_size", "tfs_per_gene")),
          ("Max.\\ epochs / validation interval (epoch) / patience", ("n_epochs", "val_every", "patience")),
          ("Gradient clipping / mixed precision", ("grad_clip", "amp")),
          (r"Loss weights $\lambda_A,\lambda_B,\lambda_C,\lambda_{\mathrm{sign}},\lambda_{\mathrm{dense}}$",
           ("lambda_a", "lambda_b", "lambda_c", "lambda_sign", "lambda_bd")),
          (r"Head B loss / Huber $\delta$", ("loss_b", "huber_delta")),
          (r"Head C focal $\gamma$ / class-balance $\beta$", ("gamma", "beta")),
          ("Condition encoding / Head C mode", ("ctx_mode", "head_c_mode")),
          ("Model selection / per-head best weights", ("select_metric", "save_per_head_best")),
          ("Head A trained on all genes", ("head_a_all",)))
    hrow = [lab + " & " + " / ".join(hp(k_) for k_ in keys) for lab, keys in HP]
    qW = H[H["run"] == run_with]
    hrow += [r"\midrule", f"Parameters & {info.get('n_params', 0):,}".replace(",", "{,}"),
             f"Training time per seed (min) & {qW['train_min'].mean():.0f} $\\pm$ "
             f"{qW['train_min'].std(ddof=1) if len(qW) > 1 else 0:.0f}",
             f"Epochs trained / selected (mean) & {qW['epochs'].mean():.1f} / {qW['best_epoch'].mean():.1f}",
             f"Seeds in ensemble & {k}", f"Hardware & {texesc(gpu_name)}"]
    tex_table(r"Hyper-parameters and compute of the final model (read from the saved checkpoints; full list in "
              r"hyperparams.csv).", "tab:hparams", ["Setting", "Value"], hrow, "lc")

    # ------------------------------------------------------------------ [5] 规则核对 + 表 + 数字清单
    say("\n[5] 规则核对(预先写在文件头；\"明显\"=|Δ|≥0.02 且 CI 不含0)")

    def clear(r_, thr=0.02):
        return r_ is not None and np.isfinite(r_["delta"]) and abs(r_["delta"]) >= thr and bool(r_["test_sig"])

    rules = []
    ri = [crow("Δ_infer", "D∈L_g", x) for x in ("C_auprc_down", "C_auprc_up")]
    rules.append(("(p1) D∈L_g 层 Δ_infer 的 C_APdn 或 C_APup 明显<0(推理时读删位点对比)",
                  any(clear(r_) and r_["delta"] < 0 for r_ in ri),
                  "  ".join(f"{SHORT[r_['metric']]} {f3(r_['delta'], True)}[{f3(r_['ci_lo'], True)},{f3(r_['ci_hi'], True)}]"
                            f" 逐seed {int(r_['seeds_better'])}/{k} 变好" for r_ in ri)))
    rules.append(("(p2) D∈L_g 层 Δ_infer 的 |C_APdn|、|C_APup| 都<0.01(收益几乎全在训练期)",
                  all(abs(r_["delta"]) < 0.01 for r_ in ri), "  ".join(f"{SHORT[r_['metric']]} {f3(r_['delta'], True)}" for r_ in ri)))
    r3 = crow("Δ_train", "D∉L_g", "C_auprc_down")
    rules.append(("(p3) D∉L_g 层 Δ_train 的 C_APdn<0 且保守合成区间(§)不含0(删位点训练对间接响应也有用)",
                  bool(r3["delta"] < 0 and r3["cons_sig"]),
                  f"C_APdn {f3(r3['delta'], True)} 合成({f3(r3['tot_lo'], True)},{f3(r3['tot_hi'], True)}) 保守("
                  f"{f3(r3['cons_lo'], True)},{f3(r3['cons_hi'], True)}) 逐seed {int(r3['seeds_better'])}/{k} 变好"))
    rz = [crow("Δ_zero", "D∈L_g", x) for x in ("C_auprc_down", "C_auprc_up")]
    rules.append(("(p4) D∈L_g 层 Δ_zero 的 C_APdn、C_APup 都在 ±0.01 内(删 token 本身无害)",
                  all(abs(r_["delta"]) < 0.01 for r_ in rz),
                  "  ".join(f"{SHORT[r_['metric']]} {f3(r_['delta'], True)}{mark(r_)}" for r_ in rz)))
    r5 = rho.get(("test", "W", "C"), (np.nan,) * 4)
    rules.append(("(p5) Spearman(ε_C, role_test)>0 且 TF bootstrap CI 不含0(删位点效应带着 TF 方向)",
                  bool(np.isfinite(r5[1]) and r5[0] > 0 and r5[1] > 0),
                  f"W {f3(r5[0], True)}[{f3(r5[1], True)},{f3(r5[2], True)}]({r5[3]} 个 TF)；N 对照 "
                  f"{f3(rho.get(('test', 'N', 'C'), (np.nan,))[0], True)}"))
    r6 = row_auc.get(("W", "C"), (np.nan,) * 6)
    rules.append(("(p6) e_C 的 TF 内合并 AUROC(down vs up)>0.5 且 CI 不含0.5(基因特异的方向信息)",
                  bool(np.isfinite(r6[4]) and r6[3] > 0.5 and r6[4] > 0.5),
                  f"W {f3(r6[3])}[{f3(r6[4])},{f3(r6[5])}]；合并 {f3(r6[0])}；N 对照 TF 内 {f3(row_auc.get(('N', 'C'), (np.nan,) * 6)[3])}"))
    for nm, ok_, txt in rules:
        say(f"  {'成立' if ok_ else '不成立'}  {nm}：{txt}")
    chk_rows = []
    for tag in ("s1", "s2", "s3", "s4"):
        q = [x for x in checks if x[0] == tag]
        if q:
            bad = [x for x in q if not x[1]]
            txt = (q[0][2] if not bad else "；".join(x[2] for x in bad)) + (f"(共 {len(q)} 项)" if len(q) > 1 else "")
            chk_rows.append(dict(rule=f"自检 ({tag})", holds=not bad, numbers=txt))
            say(f"  自检 ({tag}) {'✓' if not bad else '⚠'} {txt}")
    pd.DataFrame([dict(rule=nm, holds=ok_, numbers=txt) for nm, ok_, txt in rules] + chk_rows).to_csv(
        os.path.join(outdir, "rule_checks_probe.csv"), index=False)

    # Table 6
    cols6 = ("C_auprc_down", "C_auprc_up", "C_auroc_down", "C_auroc_up", "B_r_within_tf")
    trows = []
    lab6 = {"Δ_infer": r"$\Delta$ inference only (train $+$; infer $-$ vs.\ $+$)",
            "Δ_train": r"$\Delta$ training only (infer $-$; train $-$ vs.\ $+$)",
            "Δ_total": r"$\Delta$ total (= training ablation)",
            "Δ_zero": r"$\Delta$ zero-shot (train $-$; infer $+$ vs.\ $-$)"}
    for sc, sc_tex in (("D∈L_g", r"$D\in L_g$"), ("D∉L_g", r"$D\notin L_g$"), ("全部", "All test pairs")):
        m_ = scope_mask[sc]
        trows.append(rf"\multicolumn{{{len(cols6) + 1}}}{{l}}{{\textit{{{sc_tex}}} ({int(m_.sum())} pairs, "
                     rf"{int((m_ & sig_te).sum())} significant)}} \\")
        trows.append(r"\quad Reference (train $+$, infer $+$) & " + " & ".join(f3(point[("W_del", sc)][x]) for x in cols6))
        for cname, _, _, _ in CONTRASTS:
            cells = []
            for x in cols6:
                r_ = crow(cname, sc, x)
                if sc == "D∉L_g" and cname in ("Δ_infer", "Δ_zero") and abs(r_["delta"]) < 1e-12:
                    cells.append(r"$\equiv 0$")
                    continue
                mk = (r"\dagger" if r_["test_sig"] else "") + (r"\ddagger" if r_["tot_sig"] else "") + \
                     (r"\S" if r_["cons_sig"] else "")
                cells.append(f"${r_['delta']:+.3f}" + (f"^{{{mk}}}$" if mk else "$"))
            trows.append(r"\quad " + lab6[cname] + " & " + " & ".join(cells))
    tex_table(rf"Where the benefit of explicit site deletion comes from ({k}-seed ensembles, test set). ``train $\pm$'': "
              r"model trained with/without deleting the depleted TF's sites ($L_g\setminus D$); ``infer $\pm$'': the same "
              r"at inference. $\Delta$ = variant $-$ comparator (negative = worse without deletion). For $D\notin L_g$ the "
              r"two inference modes receive identical inputs, so the inference-only and zero-shot differences are exactly 0.",
              "tab:siamese", ["Comparison"] + [TEXM[x] for x in cols6], trows, "l" + "c" * len(cols6), wide=True,
              notes=r"$\dagger$: 95\% gene-cluster bootstrap CI excludes 0. $\ddagger$: interval additionally including "
                    r"seed-to-seed training variability (from the reference model's seeds, extrapolated to $k$ seeds) "
                    r"excludes 0. $\S$: same with the measured 2-seed variability (no extrapolation; conservative). "
                    r"$\ddagger$/$\S$ apply only where two different sets of weights are compared.")

    md.append(f"\n## Table 6 删位点 2×2 分解(28 号，{time.strftime('%Y-%m-%d %H:%M')}；{k}-seed 集成，{variant})— "
              f"{outdir}/contrasts.csv\n")
    md.append("Δ_infer=同一组权重只改推理输入(CI 只含 test 抽样)；Δ_train/Δ_total 有训练随机性(‡ 外推、§ 保守)。\n")
    for sc in SCOPES:
        for cname, _, _, _ in CONTRASTS:
            if sc == "D∉L_g" and cname in ("Δ_infer", "Δ_zero"):
                continue
            q = [crow(cname, sc, x) for x in ("C_auprc_down", "C_auprc_up", "B_r_within_tf")]
            md.append(f"- {sc} {cname}: " + "；".join(
                f"{SHORT[r_['metric']]} {f3(r_['delta'], True)} [{f3(r_['ci_lo'], True)}, {f3(r_['ci_hi'], True)}]{mark(r_)}"
                + (f" 合成 [{f3(r_['tot_lo'], True)}, {f3(r_['tot_hi'], True)}] 保守 [{f3(r_['cons_lo'], True)}, "
                   f"{f3(r_['cons_hi'], True)}]" if r_["training_noise"] and np.isfinite(r_["tot_lo"]) else "")
                + f" 逐seed {int(r_['seeds_better'])}/{k} 变好" for r_ in q))
    md.append(f"\n## 逐 TF 方向一致性(28 号 [3]) — {outdir}/per_tf_effect.csv\n")
    for (src, tag, kind), (r0, lo, hi, n_q) in rho.items():
        md.append(f"- Spearman({'ε_C' if kind == 'C' else '−ε_B'}, role_{src}) {tag}: {f3(r0, True)} [{f3(lo, True)}, {f3(hi, True)}]"
                  f"({n_q} 个 TF)")
    for (tag, kind), (a0, alo, ahi, w0, wlo, whi) in row_auc.items():
        md.append(f"- 行级 AUROC {tag} {'e_C' if kind == 'C' else '−e_B'}: 合并 {f3(a0)} [{f3(alo)}, {f3(ahi)}]，TF 内合并 {f3(w0)} "
                  f"[{f3(wlo)}, {f3(whi)}]")
    md.append(f"- 参照(train role_D 当分数): {f3(a_pr)} [{f3(a_prb[0])}, {f3(a_prb[1])}]")
    md.append(f"\n## 模型规模与算力(28 号 [4]) — {outdir}/compute.csv、hyperparams.csv\n")
    md.append(f"- 参数量 {info.get('n_params', 0):,}；每 seed 训练 {qW['train_min'].mean():.1f}±"
              f"{qW['train_min'].std(ddof=1) if len(qW) > 1 else 0:.1f} 分钟；设备 {gpu_name}；集成 {k} 个 seed")
    md.append("\n## 28 号规则核对 — rule_checks_probe.csv\n")
    for nm, ok_, txt in rules:
        md.append(f"- {'成立' if ok_ else '不成立'}：{nm} —— {txt}")
    with open(os.path.join(outdir, "paper_numbers_probe.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(outdir, "tables_probe.tex"), "w", encoding="utf-8") as fh:
        fh.write("% 28_siamese_probe.py 自动生成(27 号 --tables 时并入 _paper/tables.tex)\n\n" + "\n".join(tex))

    # ------------------------------------------------------------------ [6] 图(图内全 ASCII)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mets_f = ("C_auprc_down", "C_auprc_up", "C_auroc_down", "B_r_within_tf")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for j, (sc, ttl) in enumerate((("D∈L_g", "D in L_g (depleted TF has a site)"), ("D∉L_g", "D not in L_g"))):
            w = 0.2
            for i, (cname, lab) in enumerate((("Δ_infer", "infer only"), ("Δ_train", "train only"),
                                              ("Δ_total", "total (ablation)"), ("Δ_zero", "zero-shot"))):
                z = [crow(cname, sc, x) for x in mets_f]
                v = np.array([r_["delta"] for r_ in z], np.float64)
                lo_ = np.array([r_["ci_lo"] for r_ in z], np.float64)
                hi_ = np.array([r_["ci_hi"] for r_ in z], np.float64)
                ax[j].bar(np.arange(len(mets_f)) + i * w, v, w, label=lab,
                          yerr=[np.clip(v - lo_, 0, None), np.clip(hi_ - v, 0, None)], capsize=2)
            ax[j].axhline(0, color="k", lw=.6)
            ax[j].set_xticks(np.arange(len(mets_f)) + 1.5 * w)
            ax[j].set_xticklabels([SHORT[x] for x in mets_f], fontsize=8)
            ax[j].set_title(ttl, fontsize=9)
        ax[0].set_ylabel("Delta (variant - comparator)")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig", "probe_decomposition.png"), dpi=200)
        plt.close(fig)
        q = per_tf[(per_tf["n_rows"] >= min_rows_tf) & (per_tf["n_sig_test"].fillna(0) >= min_sig_tf)]
        if len(q) >= 3:
            fig, ax = plt.subplots(figsize=(4.8, 4.0))
            ax.scatter(q["role_test"], q["eC_N"], s=12, c="0.7", label="trained w/o deletion (zero-shot)")
            ax.scatter(q["role_test"], q["eC_W"], s=14, c="C0", label="trained with deletion")
            ax.axhline(0, color="k", lw=.5)
            ax.axvline(0, color="k", lw=.5)
            ax.set_xlabel("observed direction of direct targets (test): (down-up)/(down+up)", fontsize=8)
            ax.set_ylabel("mean shift of P(down)-P(up) caused by deletion", fontsize=8)
            r_w = rho.get(("test", "W", "C"), (np.nan,) * 4)
            ax.set_title(f"per depleted TF (n={len(q)}): Spearman {r_w[0]:+.2f} [{r_w[1]:+.2f},{r_w[2]:+.2f}]", fontsize=8)
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig", "per_tf_effect.png"), dpi=200)
            plt.close(fig)
    except Exception as e:  # noqa: BLE001 —— 画图失败不影响表格
        say(f"  (画图跳过: {type(e).__name__}: {e})")

    say(f"\n写出 {outdir}/：summary.txt、cells.csv、contrasts.csv、per_tf_effect.csv、row_effect_DinLg.csv.gz、compute.csv、"
        f"hyperparams.csv、rule_checks_probe.csv、tables_probe.tex、paper_numbers_probe.md、fig/*.png  "
        f"(用时 {(time.time() - t_all) / 60:.1f} 分钟)")
    say("下一步：./run_all.sh --tables --skip-arch-selftest(27 号把 tables_probe.tex / paper_numbers_probe.md 并进 _paper/)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(contrasts=C, per_tf=per_tf, rules=rules, checks=checks)


if __name__ == "__main__":
    run_siamese_probe(**CONFIG)
