# scripts/tflayout/17_export_results.py

EXPORT_FORMAT = "2026-09-25"  # 导出格式版本：__main__ 自动发现时，结果比这个旧就重导
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------------------
# 配置：全部写死在这里，改完直接运行(不走命令行参数)
# ------------------------------------------------------------------------------
CONFIG = dict(
    # 【第2批】"auto"=自动发现(见文件头第2批第1条)；或显式列表，如
    #   [("run1", "out/checkpoints"), ("v2_full", "out/checkpoints/v2_full")]
    runs="auto",
    ckpt_root="out/checkpoints",
    results_root="out/results",
    force=False,              # False=该实验 summary.txt 比它全部 checkpoint 都新就跳过
    seeds=None,               # None=目录里有哪些 seed*_best.pt 就导哪些
    layout="out/tf_layout.parquet",
    labels="out/head_bc_labels.parquet",
    head_a="out/head_a_baseline_logtpm.parquet",
    sgd="data/SGD_features.tab",
    promoter_tokens="out/promoter_token_ids.parquet",
    bpe_tokenizer="out/bpe_tokenizer.json",
    device=None,              # None=有GPU用GPU
    num_workers=4,            # DataLoader后台进程数，只影响速度
    eval_batch_size=512,      # 分组评估每batch约多少条样本，只影响速度/显存
    n_boot=1000,              # 集成test指标的按基因整群bootstrap次数
    min_sig_per_tf=20,        # 逐TF表里算TF内r_b的最少显著样本数
    offset_grid=(-4.0, 4.0, 0.25),  # Head C logit偏置网格(起,止,步长)，只在val上搜
    write_csv_gz=True,        # test逐样本预测额外写一份csv.gz(Excel/R能直接读)
    make_figures=True,
    dense_target="out/head_b_dense_target.parquet",  # 2026-09-25b：21号产出；不存在就跳过稠密指标
)


def run_export(ckpt_dir="out/checkpoints", ckpt_paths=None,
               seeds=None, outdir="out/results/run1", run_name="run1", ds=None,
               layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
               head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
               promoter_tokens="out/promoter_token_ids.parquet",
               bpe_tokenizer="out/bpe_tokenizer.json", device=None, num_workers=4,
               eval_batch_size=512, n_boot=1000, min_sig_per_tf=20,
               offset_grid=(-4.0, 4.0, 0.25), write_csv_gz=True, make_figures=True,
               dense_target=None):
    """唯一入口(导出一个实验；自动发现多个实验的循环在 __main__ 里)。ds 可传入已建好的
    TFLayoutDataset(18号脚本复用)，seeds=None 时按 ckpt_dir 里的 seed*_best.pt 自动定。步骤：0加载16号脚本 → 1读checkpoint → 2建Dataset+切分+数据事实 →
    3条件不可见诊断 → 4基线 → 5逐seed推理+复现核对+loss分项 → 6集成+Head C偏置调整+
    bootstrap → 7逐TF/逐基因表 → 8写文件 → 9画图 → 10 summary。"""
    t_start = time.time()
    os.makedirs(outdir, exist_ok=True)
    fig_dir = os.path.join(outdir, "fig")
    summary = []
    cls_names = ["down", "ns", "up"]

    def say(msg=""):
        print(msg, flush=True)
        summary.append(str(msg))

    def to_np(x, dtype=None):
        if hasattr(x, "detach"):
            x = x.detach().cpu()
            if x.is_floating_point():
                x = x.float()
            x = x.numpy()
        x = np.asarray(x)
        return x.astype(dtype) if dtype is not None else x

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
        return pearson(pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy())

    def rmse(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2))) if ok.any() else float("nan")

    def auroc(score, pos):
        pos = np.asarray(pos, bool)
        n1 = int(pos.sum())
        n0 = len(pos) - n1
        if n1 == 0 or n0 == 0:
            return float("nan")
        r = pd.Series(np.asarray(score, np.float64)).rank().to_numpy()
        return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    def auprc(score, pos):
        pos = np.asarray(pos, bool)
        if pos.sum() == 0:
            return float("nan")
        p = pos[np.argsort(-np.asarray(score, np.float64), kind="mergesort")]
        prec = np.cumsum(p) / np.arange(1, len(p) + 1)
        return float(prec[p].sum() / p.sum())

    def cls_report(pred, true):
        pred, true = np.asarray(pred, np.int64), np.asarray(true, np.int64)
        cm = np.bincount(true * 3 + pred, minlength=9).reshape(3, 3)  # 行=真值，列=预测
        prec, rec, f1 = [], [], []
        for c in range(3):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            prec.append(float(p))
            rec.append(float(r))
            f1.append(float(2 * p * r / (p + r)) if p + r else 0.0)
        return dict(acc=float(np.trace(cm) / max(cm.sum(), 1)), macro_f1=float(np.mean(f1)),
                    prec=prec, rec=rec, f1=f1, cm=cm)

    def head_metrics(genes, tfs, ya_p=None, ya_t=None, yb_p=None, yb_t=None,
                     probs=None, yc_t=None, pred_c=None, ybd_t=None):
        """一个"模型"在一个split上的全部指标；ya_p/yb_p/probs为None时跳过对应的头
        (基线只覆盖部分头)。Head A按基因算(每基因只取一次)。
        ybd_t(2026-09-25b)：稠密 log2FC，给了就多算 B_dr/B_dr_ns(见文件头第3批第1条)。"""
        out = {}
        if yb_p is not None and ybd_t is not None:
            okd = np.isfinite(ybd_t) & np.isfinite(yb_p)
            okn = okd & ~np.isfinite(yb_t)
            out.update(B_dr=pearson(yb_p[okd], ybd_t[okd]), B_dr_ns=pearson(yb_p[okn], ybd_t[okn]),
                       B_dr_rmse_ns=rmse(yb_p[okn], ybd_t[okn]), B_n_dense=int(okd.sum()))
        if ya_p is not None:
            _, first = np.unique(genes, return_index=True)
            a_p, a_t = ya_p[first], ya_t[first]
            ok = np.isfinite(a_t)
            out.update(A_r_gene=pearson(a_p[ok], a_t[ok]),
                       A_spearman_gene=spearman(a_p[ok], a_t[ok]),
                       A_rmse_gene=rmse(a_p[ok], a_t[ok]), A_n_genes=int(ok.sum()))
        if yb_p is not None:
            sig = np.isfinite(yb_t)
            p, t, tf_s = yb_p[sig], yb_t[sig], tfs[sig]
            d = pd.DataFrame(dict(tf=tf_s, p=p, t=t))
            pc = (d["p"] - d.groupby("tf")["p"].transform("mean")).to_numpy()
            tc = (d["t"] - d.groupby("tf")["t"].transform("mean")).to_numpy()
            out.update(B_r=pearson(p, t), B_spearman=spearman(p, t), B_rmse=rmse(p, t),
                       B_sign_acc=float(np.mean(np.sign(p) == np.sign(t))) if len(p) else
                       float("nan"),
                       B_r_within_true_down=pearson(p[t < 0], t[t < 0]),
                       B_r_within_true_up=pearson(p[t > 0], t[t > 0]),
                       B_r_within_tf=pearson(pc, tc), B_n_sig=int(sig.sum()))
        if probs is not None or pred_c is not None:
            if pred_c is None:
                pred_c = probs.argmax(1)
            rep = cls_report(pred_c, yc_t)
            out.update(C_acc=rep["acc"], C_macro_f1=rep["macro_f1"],
                       C_f1_down=rep["f1"][0], C_f1_ns=rep["f1"][1], C_f1_up=rep["f1"][2],
                       C_prec_down=rep["prec"][0], C_rec_down=rep["rec"][0],
                       C_prec_up=rep["prec"][2], C_rec_up=rep["rec"][2])
            if probs is not None:
                out.update(C_auroc_down=auroc(probs[:, 0], yc_t == 0),
                           C_auroc_up=auroc(probs[:, 2], yc_t == 2),
                           C_auprc_down=auprc(probs[:, 0], yc_t == 0),
                           C_auprc_up=auprc(probs[:, 2], yc_t == 2))
        return out

    def fmt(v, nd=4):
        return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
            else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    # --------------------------------------------------------------------------
    # 0. 动态加载16号脚本(它会连带加载09/15号，15号再加载10/12/14号)
    # --------------------------------------------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    mod_name = "_tflayout_16_train_loop"
    if mod_name in sys.modules:
        tl = sys.modules[mod_name]
    else:
        spec = importlib.util.spec_from_file_location(mod_name,
                                                      os.path.join(here, "16_train_loop.py"))
        tl = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = tl  # 跟16号 _load_module 同样的注册顺序
        spec.loader.exec_module(tl)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    say("=" * 78)
    say(f"TF-Layout 训练结果导出【{run_name}】  ({time.strftime('%Y-%m-%d %H:%M:%S')})  "
        f"checkpoint 目录: {ckpt_dir}")
    say(f"设备: {device}  PyTorch {torch.__version__}  当前16号代码版本: {tl.CODE_VERSION}")
    say("=" * 78)

    # --------------------------------------------------------------------------
    # 1. 读 checkpoint
    # --------------------------------------------------------------------------
    ckpts, inv_rows, lc_rows = {}, [], []
    if seeds is None:  # 第2批：按目录里实际有的 seed*_best.pt(只认整数 seed，不认 .bak/.tmp)
        seeds = sorted(int(os.path.basename(f)[4:-8]) for f in
                       glob.glob(os.path.join(ckpt_dir, "seed*_best.pt"))
                       if os.path.basename(f)[4:-8].isdigit())
        seeds = seeds + sorted(k for k in (ckpt_paths or {}) if k not in seeds)
    for s in seeds:
        p = (ckpt_paths or {}).get(s) or os.path.join(ckpt_dir, f"seed{s}_best.pt")
        if not os.path.exists(p):
            say(f"⚠ seed{s}: 找不到 {p}，跳过这个seed")
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(p, map_location="cpu")
        ckpts[s] = ck
        cfg = ck.get("train_config") or {}
        n_par = int(sum(int(np.prod(v.shape)) for v in ck["model_state"].values()))
        inv_rows.append(dict(seed=s, path=p, size_mb=round(os.path.getsize(p) / 2 ** 20, 1),
                             code_version=ck.get("code_version"), best_epoch=ck.get("best_epoch"),
                             n_epochs_run=(max(int(h.get("epoch", 0)) for h in ck["history"]) + 1)
                             if ck.get("history") else 0,
                             n_validations=len(ck.get("history") or []), n_params=n_par,
                             **{k: cfg.get(k) for k in ("d_model", "n_heads", "cis_layers",
                                                        "lay_layers", "batch_size", "lr",
                                                        "weight_decay", "tfs_per_gene", "amp",
                                                        "forward_mode", "patience",
                                                        "lambda_b", "lambda_c",
                                                        "lambda_sign", "ctx_mode",
                                                        "head_c_mode", "select_metric",
                                                        "val_every", "loss_b", "optimizer",
                                                        "dropout", "gamma", "lambda_bd",
                                                        "head_a_all")},
                             has_head_states=bool(ck.get("head_states")),
                             best_val_index=(ck.get("best") or {}).get("val_index"),
                             best_epoch_progress=(ck.get("best") or {}).get("epoch_progress")))
        for h in ck.get("history") or []:
            lc_rows.append(dict(seed=s, **h))
        if ck.get("code_version") != tl.CODE_VERSION:
            say(f"⚠ seed{s}: checkpoint代码版本 {ck.get('code_version')} != 当前16号 "
                f"{tl.CODE_VERSION}。只要网络结构没变仍能加载(下面strict加载会验证)，"
                "但复现核对的差值要看仔细")
    if not ckpts:
        raise SystemExit("一个checkpoint都没读到，先确认 CONFIG['ckpt_dir'] 路径")
    inv = pd.DataFrame(inv_rows)
    lc = pd.DataFrame(lc_rows)
    say(f"\n[1] 读到 {len(ckpts)} 个checkpoint：")
    for r in inv.itertuples():
        say(f"  seed{r.seed}: {r.size_mb}MB  参数量{r.n_params:,}  版本{r.code_version}  "
            f"最优epoch={r.best_epoch}/共跑{r.n_epochs_run}个epoch(验证{r.n_validations}次)  "
            f"(d={r.d_model},heads={r.n_heads},cis={r.cis_layers},lay={r.lay_layers},"
            f"bs={r.batch_size},lr={r.lr},K={r.tfs_per_gene},{r.amp})"
            + (f"  [第2批] ctx={r.ctx_mode} headC={r.head_c_mode} 选模型={r.select_metric} "
               f"val_every={r.val_every} loss_b={r.loss_b} {r.optimizer} dropout={r.dropout} "
               f"选中@epoch{r.best_epoch_progress} 各头最优权重={'有' if r.has_head_states else '无'}"
               + (f" focalγ={r.gamma}" if r.gamma is not None and r.gamma == r.gamma else "")
               + (f" λ_Bd={r.lambda_bd}" if r.lambda_bd and r.lambda_bd == r.lambda_bd else "")
               + (" HeadA全基因=是" if str(getattr(r, "head_a_all", None)) == "True" else "")
               if isinstance(r.ctx_mode, str) else ""))

    # --------------------------------------------------------------------------
    # 2. Dataset + 切分 + 数据事实
    # --------------------------------------------------------------------------
    say("\n[2] 构建 Dataset(跟训练时同一套09号代码/同一批文件)")
    if ds is None:
        ds = tl.TFLayoutDataset(layout, labels, head_a, sgd, promoter_tokens, bpe_tokenizer)
    if hasattr(ds, "set_head_a_all"):  # 2026-09-28b：共享 Dataset 可能还挂着第8批伪样本行，导出只看真实行
        ds.set_head_a_all(False, verbose=False)
    split_idx = tl.build_split_indices(ds, head_a)
    cfg0 = ckpts[min(ckpts)].get("train_config") or {}
    run_ctx = (cfg0.get("ctx_mode", "legacy"), float(cfg0.get("ctx_clip", 3.0)))
    modes_seen = {(c.get("train_config") or {}).get("ctx_mode", "legacy") for c in ckpts.values()}
    if len(modes_seen) > 1:
        say(f"⚠ 这个实验里不同 seed 的 ctx_mode 不一致 {modes_seen}，第[3]段诊断按 {run_ctx[0]} 算")
    if hasattr(ds, "set_ctx_mode"):
        ds.set_ctx_mode(*run_ctx, verbose=False)
    say(f"  条件编码 ctx_mode={run_ctx[0]}(ctx_clip={run_ctx[1]})")
    _abl = cfg0.get("ablation", "none")
    if _abl != "none":
        say(f"  输入消融 ablation={_abl}(每个 seed 推理前按 checkpoint 记录的值切换；基线/诊断不受影响)")
    smp = ds.samples
    genes_all = smp["gene_id"].to_numpy()
    tfs_all = smp["tf_depleted"].to_numpy()
    y_c_all = smp["direction_3class"].map(ds.class2idx).to_numpy().astype(np.int64)
    y_b_all = smp["log2fc"].to_numpy(dtype=np.float64)
    split_of = np.empty(len(smp), dtype=object)
    for k, idx in split_idx.items():
        split_of[np.asarray(idx, dtype=np.int64)] = k
    gene2ya = ds.head_a.to_dict()
    y_a_all = np.array([gene2ya.get(g, np.nan) for g in genes_all], dtype=np.float64)
    exact = smp["direction_3class"].value_counts()
    class_counts = torch.tensor([float(exact.get(c, 0.0)) for c in cls_names])
    say(f"  样本 {len(smp)}，TF词表 {ds.n_tf}，类别计数(down/ns/up) "
        f"{[int(v) for v in to_np(class_counts)]}")
    facts = {}
    for k in ("train", "val", "test"):
        idx = np.asarray(split_idx[k], dtype=np.int64)
        g_u = pd.unique(genes_all[idx])
        per_gene = pd.Series(genes_all[idx]).value_counts()
        cc = np.bincount(y_c_all[idx], minlength=3)
        ns_frac = cc[1] / max(cc.sum(), 1)
        triv_f1_ns = 2 * ns_frac / (1 + ns_frac)
        facts[k] = dict(n_samples=int(len(idx)), n_genes=int(len(g_u)),
                        samples_per_gene_min=int(per_gene.min()),
                        samples_per_gene_max=int(per_gene.max()),
                        n_down=int(cc[0]), n_ns=int(cc[1]), n_up=int(cc[2]),
                        n_sig=int(np.isfinite(y_b_all[idx]).sum()), ns_frac=float(ns_frac),
                        always_ns_acc=float(ns_frac), always_ns_macro_f1=float(triv_f1_ns / 3),
                        sd_log2fc_sig=float(np.nanstd(y_b_all[idx])))
        f = facts[k]
        say(f"  {k:5s}: {f['n_samples']}条/{f['n_genes']}个基因(每基因{f['samples_per_gene_min']}"
            f"~{f['samples_per_gene_max']}条)  down/ns/up={f['n_down']}/{f['n_ns']}/{f['n_up']}"
            f"  显著(y_b非NaN)={f['n_sig']}  ns占比={f['ns_frac']:.4f}"
            f"(=永远猜ns的accuracy；此时macro-F1={f['always_ns_macro_f1']:.4f})"
            f"  显著log2FC标准差={f['sd_log2fc_sig']:.3f}")
    # 2026-09-25b：稠密 log2FC(只用于评估，见文件头第3批)
    ybd_eval = None
    if dense_target and os.path.exists(dense_target) and hasattr(ds, "set_dense_target"):
        st_d = ds.set_dense_target(dense_target, verbose=False)
        ybd_eval = np.asarray(ds._y_bd_arr, dtype=np.float64).copy()
        say(f"  稠密 log2FC(评估用)：{dense_target} -> {st_d['n_finite']}/{st_d['n']} 条样本有值"
            f"({st_d['frac']:.1%}；不显著样本里 {st_d.get('frac_ns', float('nan')):.1%})")
    elif dense_target:
        say(f"  (没找到 {dense_target} 或 Dataset 不支持稠密目标，跳过 B_dr/B_dr_ns 指标)")

    def ybd_of(ix):
        return None if ybd_eval is None else ybd_eval[ix]
    n_ya_genes = int(ds.head_a.notna().sum())
    n_used_genes = len(pd.unique(genes_all))
    _ha_all = any(bool((c.get("train_config") or {}).get("head_a_all")) for c in ckpts.values())
    if _ha_all:  # 2026-09-28b
        say(f"  Head A 标签文件里有 {n_ya_genes} 个基因；本实验训练/选模型时用了全部 Head A 基因(网格外基因是伪样本，16 号第17条)，"
            f"这里的评估仍只看 B/C 网格里的 {n_used_genes} 个(跟历次实验同口径)；全部 Head A test 基因的成绩看 25 号")
    else:
        say(f"  Head A 标签文件里有 {n_ya_genes} 个基因，进入训练/评估的只有 {n_used_genes} 个"
            f"(=Head B/C 标签里的基因；其余基因只有Head A标签，目前没被用上)")

    # --------------------------------------------------------------------------
    # 3. 条件不可见诊断(docstring第3条)
    # --------------------------------------------------------------------------
    say("\n[3] 条件编码诊断(ctx_wt/ctx_d，见docstring第3条)")
    ctx_wt_ref = to_np(ds[0]["ctx_wt"], np.float32)
    cache = getattr(ds, "_ctx_d_cache", None)
    ctx_rows, ctx_same = [], {}
    for tf in ds.tf_list:
        if cache is not None and tf in cache:
            v = np.asarray(cache[tf], np.float32)
        else:
            v = np.asarray(ds._build_ctx(tf), np.float32)
            v[ds.tf2idx[tf]] = 0.0
        ctx_same[tf] = bool(np.array_equal(v, ctx_wt_ref))
        ctx_rows.append(dict(tf=tf, ctx_d_n_nonzero=int((v != 0).sum()),
                             ctx_d_equals_ctx_wt=ctx_same[tf],
                             ctx_d_abs_sum=float(np.abs(v).sum())))
    tfsets = {g: set(v["tf_idx"].tolist()) for g, v in ds.layout_by_gene.items()}
    dep_idx_all = smp["tf_depleted"].map(ds.tf2idx).to_numpy()
    bound_all = np.fromiter((d in tfsets.get(g, ()) for g, d in zip(genes_all, dep_idx_all)),
                            dtype=bool, count=len(smp))
    invisible_all = (~bound_all) & np.array([ctx_same.get(t, False) for t in tfs_all])
    ctx_df = pd.DataFrame(ctx_rows)
    tf_in_samples = set(pd.unique(tfs_all))
    ctx_df["is_depleted_in_data"] = ctx_df["tf"].isin(tf_in_samples)
    n_dep = int(ctx_df["is_depleted_in_data"].sum())
    n_dep_same = int((ctx_df["ctx_d_equals_ctx_wt"] & ctx_df["is_depleted_in_data"]).sum())
    say(f"  ctx_wt 参考向量: 全0={bool(np.all(ctx_wt_ref == 0))}  非零个数={int((ctx_wt_ref != 0).sum())}")
    say(f"  数据里被耗竭过的TF {n_dep} 个，其中 ctx_d 跟 ctx_wt 完全相同(耗竭没让任何TF基因"
        f"显著变化)的有 {n_dep_same} 个")
    diag = {}
    for k in ("all", "train", "val", "test"):
        m = np.ones(len(smp), bool) if k == "all" else (split_of == k)
        sig = m & (y_c_all != 1)
        diag[k] = dict(frac_bound=float(bound_all[m].mean()),
                       frac_invisible=float(invisible_all[m].mean()),
                       frac_invisible_among_sig=float(invisible_all[sig].mean())
                       if sig.any() else float("nan"),
                       n_invisible_sig=int((invisible_all & sig).sum()), n_sig=int(sig.sum()))
        d = diag[k]
        say(f"  {k:5s}: D∈L_g {d['frac_bound']:.2%}；条件不可见(WT/D输入完全相同) "
            f"{d['frac_invisible']:.2%}；在down/up样本里占 {d['frac_invisible_among_sig']:.2%}"
            f"({d['n_invisible_sig']}/{d['n_sig']})")
    say("  → 条件不可见的样本上 Head C 只能输出同一个常数 logits=ψ_C(0)，Head B 只剩 ψ_corr(D,z_wt)；"
        "占比越大，越需要改 ctx 编码(status 第8节)")

    # --------------------------------------------------------------------------
    # 4. 基线(只用train拟合)
    # --------------------------------------------------------------------------
    say("\n[4] 基线(只用训练集拟合，见docstring第4条)")
    tr = split_of == "train"
    sig_all = np.isfinite(y_b_all)
    glob_mean_b = float(y_b_all[tr & sig_all].mean())
    tr_b = pd.DataFrame(dict(tf=tfs_all[tr & sig_all], b=bound_all[tr & sig_all],
                             y=y_b_all[tr & sig_all]))
    tf_mean = tr_b.groupby("tf")["y"].mean().to_dict()
    tf_n = tr_b.groupby("tf")["y"].size().to_dict()
    kb = 5.0
    tfb = tr_b.groupby(["tf", "b"])["y"].agg(["sum", "size"])
    tfb_mean = {key: (row["sum"] + kb * tf_mean.get(key[0], glob_mean_b)) / (row["size"] + kb)
                for key, row in tfb.iterrows()}
    base_b_tf = np.array([tf_mean.get(t, glob_mean_b) for t in tfs_all])
    base_b_tfb = np.array([tfb_mean.get((t, b), tf_mean.get(t, glob_mean_b))
                           for t, b in zip(tfs_all, bound_all)])
    base_bd_tf = None  # 2026-09-25b：TF 稠密均值基线(训练集该 TF 全部有稠密值样本的均值)
    if ybd_eval is not None:
        _okd = tr & np.isfinite(ybd_eval)
        _m = pd.Series(ybd_eval[_okd]).groupby(tfs_all[_okd]).mean().to_dict()
        _g0 = float(np.mean(ybd_eval[_okd])) if _okd.any() else 0.0
        base_bd_tf = np.array([_m.get(t, _g0) for t in tfs_all])
    tr_c = pd.DataFrame(dict(tf=tfs_all[tr], b=bound_all[tr], c=y_c_all[tr]))
    glob_p = np.bincount(tr_c["c"], minlength=3) / len(tr_c)
    cnt_tf = tr_c.groupby(["tf", "c"]).size().unstack(fill_value=0).reindex(
        columns=[0, 1, 2], fill_value=0)
    p_tf = {t: (row.to_numpy() + 1.0) / (row.sum() + 3.0) for t, row in cnt_tf.iterrows()}
    cnt_tfb = tr_c.groupby(["tf", "b", "c"]).size().unstack(fill_value=0).reindex(
        columns=[0, 1, 2], fill_value=0)
    alpha = 10.0
    p_tfb = {key: (row.to_numpy() + alpha * p_tf.get(key[0], glob_p)) / (row.sum() + alpha)
             for key, row in cnt_tfb.iterrows()}
    base_p_tf = np.stack([p_tf.get(t, glob_p) for t in tfs_all])
    base_p_tfb = np.stack([p_tfb.get((t, b), p_tf.get(t, glob_p))
                           for t, b in zip(tfs_all, bound_all)])
    # Head A：layout 摘要特征的线性回归
    feat_names = ["n_sites", "n_unique_tf", "sum_a", "mean_a", "sum_m", "mean_m",
                  "frac_motif_anchored"]
    gene_feat = {}
    for g in pd.unique(genes_all):
        v = ds.layout_by_gene.get(g)
        if v is None or len(v["tf_idx"]) == 0:
            gene_feat[g] = np.zeros(len(feat_names))
            continue
        a, mm, rr = np.asarray(v["a"], float), np.asarray(v["m"], float), np.asarray(v["res"], float)
        gene_feat[g] = np.array([len(v["tf_idx"]), len(set(v["tf_idx"].tolist())), a.sum(),
                                 a.mean(), mm.sum(), mm.mean(), (rr > 0.5).mean()])
    genes_u = np.array(list(gene_feat.keys()), dtype=object)
    X = np.stack([gene_feat[g] for g in genes_u])
    X = np.nan_to_num(X)
    gsplit = pd.Series(split_of).groupby(genes_all).first().reindex(genes_u).to_numpy()
    gya = np.array([gene2ya.get(g, np.nan) for g in genes_u])
    fit = (gsplit == "train") & np.isfinite(gya)
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-9
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(Xs[fit], gya[fit], rcond=None)
    base_a_gene = dict(zip(genes_u, Xs @ coef))
    base_a = np.array([base_a_gene[g] for g in genes_all])
    # ORACLE：同split内、同基因、其它TF的标签(见docstring第4条，只做分解分析)
    orc_b = np.full(len(smp), glob_mean_b)
    orc_p = np.tile(glob_p, (len(smp), 1))
    for k in ("val", "test"):
        ix = np.asarray(split_idx[k], dtype=np.int64)
        d = pd.DataFrame(dict(g=genes_all[ix], y=np.nan_to_num(y_b_all[ix]),
                              sg=np.isfinite(y_b_all[ix]).astype(float), c=y_c_all[ix]))
        gs = d.groupby("g")[["y", "sg"]].transform("sum")
        n_other = gs["sg"].to_numpy() - d["sg"].to_numpy()
        s_other = gs["y"].to_numpy() - d["y"].to_numpy()
        orc_b[ix] = np.where(n_other > 0, s_other / np.maximum(n_other, 1), glob_mean_b)
        oh = np.eye(3)[d["c"].to_numpy()]
        cnt = pd.DataFrame(oh).groupby(d["g"].to_numpy()).transform("sum").to_numpy() - oh
        orc_p[ix] = (cnt + 10.0 * glob_p) / (cnt.sum(1, keepdims=True) + 10.0)
    say(f"  Head A layout计数回归：{int(fit.sum())} 个训练基因拟合，特征={feat_names}")
    say(f"  Head B TF均值：{len(tf_mean)} 个TF(训练显著样本数中位 "
        f"{int(np.median(list(tf_n.values())))})，全局均值 {glob_mean_b:.3f}")

    # --------------------------------------------------------------------------
    # 5. 逐seed推理 + 复现核对 + loss 分项
    # --------------------------------------------------------------------------
    say("\n[5] 逐seed重新推理(val+test)")
    preds = {"val": {}, "test": {}}
    preds_ph = {"val": {}, "test": {}}  # 第2批：各头各取最优权重

    def ev_to_pred(ev_):
        ya_, yb_ = to_np(ev_["y_a_pred"], np.float64), to_np(ev_["y_b_pred"], np.float64)
        lg_ = to_np(ev_["logits_c"], np.float64)
        z_ = lg_ - lg_.max(1, keepdims=True)
        return dict(ya=ya_, yb=yb_, logits=lg_, probs=np.exp(z_) / np.exp(z_).sum(1, keepdims=True))
    loss_rows, metric_rows, recheck_rows = [], [], []
    sidx = {k: np.asarray(split_idx[k], dtype=np.int64) for k in ("val", "test")}
    for s, ck in ckpts.items():
        cfg = ck.get("train_config") or {}
        if int(cfg.get("n_tf", ds.n_tf)) != ds.n_tf:
            say(f"⚠ seed{s}: checkpoint 的 n_tf={cfg.get('n_tf')} 跟当前 Dataset 的 {ds.n_tf} "
                "不一致(数据文件变了？)，跳过")
            continue
        if ck.get("model_kwargs"):  # 第2批新 checkpoint：按记录的构造参数重建
            model = tl.SiameseHeadsModel(**ck["model_kwargs"])
        else:
            model = tl.SiameseHeadsModel(ds.n_tf, int(cfg.get("vocab_size", 4000)),
                                         int(cfg.get("d_model", 256)), int(cfg.get("n_heads", 8)),
                                         int(cfg.get("cis_layers", 6)), int(cfg.get("lay_layers", 4)),
                                         pad_token_id=int(cfg.get("pad_token_id", ds.pad_id)))
        if hasattr(ds, "set_ctx_mode"):
            ds.set_ctx_mode(cfg.get("ctx_mode", "legacy"), float(cfg.get("ctx_clip", 3.0)),
                            verbose=False)
        if hasattr(ds, "set_ablation"):  # 2026-09-26a：按 checkpoint 训练时的输入消融推理(默认 none)
            ds.set_ablation(cfg.get("ablation", "none"), verbose=False)
        model.load_state_dict(ck["model_state"], strict=True)
        model = model.to(device)
        model.eval()
        loss_kw = {k: cfg.get(k, dflt) for k, dflt in (("lambda_b", 1.0), ("lambda_c", 1.0),
                                                        ("lambda_sign", 0.1), ("gamma", 2.0),
                                                        ("beta", 0.999), ("lambda_a", 1.0),
                                                        ("loss_b", "mse"), ("huber_delta", 1.0),
                                                        ("lambda_bd", 0.0), ("dense_on", "ns"),
                                                        ("dense_delta", None))}
        # 2026-09-25b：用稠密目标训练过的 checkpoint，推理/loss 分项用它训练时那份稠密目标；
        # 其余 checkpoint 用评估用的那份(λ_bd=0 时它只影响 l_bd 这个监控项，不影响 total)
        if hasattr(ds, "set_dense_target"):
            want_d = cfg.get("dense_target") if loss_kw["lambda_bd"] else (dense_target if ybd_eval is not None else None)
            if want_d and not os.path.exists(want_d):
                say(f"  ⚠ seed{s}: 训练时的稠密目标 {want_d} 找不到，改用评估用的那份")
                want_d = dense_target if ybd_eval is not None else None
            ds.set_dense_target(want_d, verbose=False)
        ybd_loss = (np.asarray(ds._y_bd_arr, dtype=np.float64) if hasattr(ds, "_y_bd_arr") else None)
        if ybd_loss is not None and not np.isfinite(ybd_loss).any():
            ybd_loss = None
        if loss_kw["lambda_bd"] and ybd_loss is None:
            say(f"  ⚠ seed{s}: 这个 checkpoint 用 λ_Bd={loss_kw['lambda_bd']} 训练，但没有可用的稠密目标，"
                "loss 分项按 λ_Bd=0 算(复现核对里 val_loss 会对不上，不影响其它指标)")
            loss_kw["lambda_bd"] = 0.0
        t_s = time.time()
        for k in ("val", "test"):
            chunk = int(cfg.get("batch_size", 192)) if k == "val" else 128  # 跟训练时两处loss口径一致
            ev = tl.evaluate(model, ds, split_idx[k], class_counts, batch_size=chunk,
                             device=device, num_workers=num_workers,
                             amp=cfg.get("amp", "bf16"),
                             forward_mode=cfg.get("forward_mode", "grouped"),
                             eval_batch_size=eval_batch_size, layout_buckets=1, **loss_kw)
            preds[k][s] = ev_to_pred(ev)
            ya_p, yb_p = preds[k][s]["ya"], preds[k][s]["yb"]
            logits, probs = preds[k][s]["logits"], preds[k][s]["probs"]
            if ck.get("head_states"):  # 第2批：各头各取最优权重(16号 evaluate_per_head)
                ev_ph = tl.evaluate_per_head(model, ck["head_states"], ds, split_idx[k],
                                             class_counts, batch_size=chunk, device=device,
                                             num_workers=num_workers, amp=cfg.get("amp", "bf16"),
                                             forward_mode=cfg.get("forward_mode", "grouped"),
                                             eval_batch_size=eval_batch_size, layout_buckets=1,
                                             **loss_kw)
                preds_ph[k][s] = ev_to_pred(ev_ph)
                pp = preds_ph[k][s]
                met_ph = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], pp["ya"],
                                      y_a_all[sidx[k]], pp["yb"], y_b_all[sidx[k]], pp["probs"],
                                      y_c_all[sidx[k]], ybd_t=ybd_of(sidx[k]))
                metric_rows += [dict(model=f"seed{s}_perhead", split=k, metric=mk, value=mv)
                                for mk, mv in met_ph.items()]
                stored_ph = ck.get("test_eval_per_head") or {}
                if k == "test":
                    for mk_st, mk_new in (("r_a_gene", "A_r_gene"), ("r_b", "B_r"),
                                          ("macro_f1_c", "C_macro_f1")):
                        if mk_st in stored_ph:
                            recheck_rows.append(dict(seed=s, metric=f"perhead_{mk_st}",
                                                     stored=float(stored_ph[mk_st]),
                                                     recomputed=float(met_ph[mk_new]),
                                                     diff=float(met_ph[mk_new]) - float(stored_ph[mk_st])))
            if k == "test":
                inv_t = invisible_all[sidx[k]]
                if inv_t.any():
                    sd_inv = float(logits[inv_t].std(0).max())
                    sd_vis = float(logits[~inv_t].std(0).max()) if (~inv_t).any() else float("nan")
                    cls_inv = np.bincount(logits[inv_t].argmax(1), minlength=3)
                    say(f"  seed{s} test 实测：条件不可见样本的Head C logits 最大标准差 {sd_inv:.2e}"
                        f"(可见样本 {sd_vis:.2e})；不可见样本被判成 down/ns/up = "
                        f"{cls_inv.tolist()}  ← 前者≈0 即证实第3条机制")
            # loss分项(同一个分块口径)
            sums = dict(l_a=0.0, l_b=0.0, l_c=0.0, l_sign=0.0, l_bd=0.0, total=0.0)
            n_ch = 0
            ya_t32 = y_a_all[sidx[k]].astype(np.float32)
            yb_t32 = y_b_all[sidx[k]].astype(np.float32)
            yc_t64 = y_c_all[sidx[k]]
            ybd32 = ybd_loss[sidx[k]].astype(np.float32) if ybd_loss is not None else None
            for st in range(0, len(ya_p), chunk):
                sl = slice(st, st + chunk)
                _, parts = tl.compute_total_loss(
                    {"y_a_pred": torch.from_numpy(np.ascontiguousarray(ya_p[sl], np.float32)),
                     "y_b_pred": torch.from_numpy(np.ascontiguousarray(yb_p[sl], np.float32)),
                     "logits_c": torch.from_numpy(np.ascontiguousarray(logits[sl], np.float32))},
                    torch.from_numpy(np.ascontiguousarray(ya_t32[sl])),
                    torch.from_numpy(np.ascontiguousarray(yb_t32[sl])),
                    torch.from_numpy(np.ascontiguousarray(yc_t64[sl])), class_counts,
                    return_tensors=False,
                    y_bd=torch.from_numpy(np.ascontiguousarray(ybd32[sl])) if ybd32 is not None else None,
                    **loss_kw)
                for kk in sums:
                    sums[kk] += float(parts[kk])
                n_ch += 1
            parts_mean = {kk: v / max(n_ch, 1) for kk, v in sums.items()}
            w = dict(l_a=loss_kw["lambda_a"], l_b=loss_kw["lambda_b"], l_c=loss_kw["lambda_c"],
                     l_sign=loss_kw["lambda_sign"], l_bd=loss_kw["lambda_bd"])
            loss_rows.append(dict(seed=s, split=k, chunk=chunk, evaluate_loss=float(ev["loss"]),
                                  **parts_mean,
                                  **{f"share_{kk}": w[kk] * parts_mean[kk] /
                                     max(parts_mean["total"], 1e-12) for kk in w}))
            met = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], ya_p, y_a_all[sidx[k]],
                               yb_p, y_b_all[sidx[k]], probs, yc_t64, ybd_t=ybd_of(sidx[k]))
            metric_rows += [dict(model=f"seed{s}", split=k, metric=mk, value=mv)
                            for mk, mv in met.items()]
            if k == "val":
                hist_l = ck.get("history") or []
                bvi = (ck.get("best") or {}).get("val_index")  # 第2批：每次验证一条 history
                if bvi is not None and 0 <= int(bvi) < len(hist_l):
                    hb = hist_l[int(bvi)]
                else:
                    hb = {h.get("epoch"): h for h in hist_l}.get(ck.get("best_epoch"))
                if hb is not None and cfg.get("head_a_all"):  # 2026-09-28b：训练时 val 含伪样本，定义不同，跳过
                    say(f"  seed{s}: head_a_all 实验训练时的 val_loss 含伪样本分块，复现核对跳过 val_loss 这一项(test 各项照常)")
                elif hb is not None:
                    recheck_rows.append(dict(seed=s, metric="val_loss@best_epoch",
                                             stored=float(hb["val_loss"]),
                                             recomputed=float(ev["loss"]),
                                             diff=float(ev["loss"]) - float(hb["val_loss"])))
            if k == "test":
                stored = ck.get("test_eval") or {}
                for mk_st, mk_new in (("r_a_gene", "A_r_gene"), ("r_b", "B_r"),
                                      ("macro_f1_c", "C_macro_f1"), ("acc_c", "C_acc")):
                    if mk_st in stored:
                        recheck_rows.append(dict(seed=s, metric=mk_st,
                                                 stored=float(stored[mk_st]),
                                                 recomputed=float(met[mk_new]),
                                                 diff=float(met[mk_new]) - float(stored[mk_st])))
                if "f1_per_class" in stored:
                    recheck_rows.append(dict(seed=s, metric="stored_f1_per_class(down/ns/up)",
                                             stored=str([round(x, 4) for x in stored["f1_per_class"]]),
                                             recomputed=str([round(met["C_f1_down"], 4),
                                                             round(met["C_f1_ns"], 4),
                                                             round(met["C_f1_up"], 4)]),
                                             diff=float("nan")))
        say(f"  seed{s}: 完成({time.time() - t_s:.0f}秒)")
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    seeds_ok = sorted(preds["test"].keys())
    if not seeds_ok:
        raise SystemExit("没有任何seed推理成功")
    rc = pd.DataFrame(recheck_rows)
    num_rc = rc[rc["metric"] != "stored_f1_per_class(down/ns/up)"].copy()
    max_diff = float(np.nanmax(np.abs(num_rc["diff"].astype(float)))) if len(num_rc) else float("nan")
    say(f"  复现核对(重新推理 vs checkpoint里存的test指标+最优epoch的val_loss)：最大绝对差 {max_diff:.2e}"
        + ("  ✓" if np.isfinite(max_diff) and max_diff < 5e-3 else
           "  ⚠ 差得偏大，数据文件/代码可能在训练后改过，下面的结论要打折扣"))

    # --------------------------------------------------------------------------
    # 6. 集成 + Head C 偏置调整(只在val上搜) + bootstrap
    # --------------------------------------------------------------------------
    say(f"\n[6] {len(seeds_ok)}-seed集成 + Head C logit偏置(val上搜、原样套到test)")
    ens = {}
    for k in ("val", "test"):
        ens[k] = dict(ya=np.mean([preds[k][s]["ya"] for s in seeds_ok], 0),
                      yb=np.mean([preds[k][s]["yb"] for s in seeds_ok], 0),
                      probs=np.mean([preds[k][s]["probs"] for s in seeds_ok], 0))
        met = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], ens[k]["ya"], y_a_all[sidx[k]],
                           ens[k]["yb"], y_b_all[sidx[k]], ens[k]["probs"], y_c_all[sidx[k]],
                           ybd_t=ybd_of(sidx[k]))
        metric_rows += [dict(model="ensemble", split=k, metric=mk, value=mv)
                        for mk, mv in met.items()]
    lo, hi, step = offset_grid
    grid = np.arange(lo, hi + 1e-9, step)
    yv = y_c_all[sidx["val"]]

    def tune_offsets(prob_val):
        """在val上网格搜down/up两类的logit加性偏置(ns固定0)，返回(val macro-F1, od, ou)。"""
        lp = np.log(np.clip(prob_val, 1e-12, 1.0))
        bst = (-1.0, 0.0, 0.0)
        for od in grid:
            for ou in grid:
                f1m = cls_report((lp + np.array([od, 0.0, ou])).argmax(1), yv)["macro_f1"]
                if f1m > bst[0] + 1e-12:
                    bst = (f1m, float(od), float(ou))
        return bst

    def apply_offsets(prob, bst):
        return (np.log(np.clip(prob, 1e-12, 1.0)) + np.array([bst[1], 0.0, bst[2]])).argmax(1)

    best = tune_offsets(ens["val"]["probs"])
    say(f"  val上最优偏置: down {best[1]:+.2f}, up {best[2]:+.2f}  → val macro-F1 {best[0]:.4f}")
    seeds_ph = sorted(s_ for s_ in preds_ph["test"] if s_ in preds_ph["val"])
    ens_ph, best_ph, tuned_ph = None, None, None
    if seeds_ph:  # 第2批：各头各取最优权重那一套的集成 + 同样只在 val 上调偏置
        ens_ph = {k: dict(ya=np.mean([preds_ph[k][s_]["ya"] for s_ in seeds_ph], 0),
                          yb=np.mean([preds_ph[k][s_]["yb"] for s_ in seeds_ph], 0),
                          probs=np.mean([preds_ph[k][s_]["probs"] for s_ in seeds_ph], 0))
                  for k in ("val", "test")}
        for k in ("val", "test"):
            met = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], ens_ph[k]["ya"],
                               y_a_all[sidx[k]], ens_ph[k]["yb"], y_b_all[sidx[k]],
                               ens_ph[k]["probs"], y_c_all[sidx[k]], ybd_t=ybd_of(sidx[k]))
            metric_rows += [dict(model="ensemble_perhead", split=k, metric=mk, value=mv)
                            for mk, mv in met.items()]
        best_ph = tune_offsets(ens_ph["val"]["probs"])
        tuned_ph = {k: apply_offsets(ens_ph[k]["probs"], best_ph) for k in ("val", "test")}
        for k in ("val", "test"):
            met = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], yc_t=y_c_all[sidx[k]],
                               pred_c=tuned_ph[k])
            metric_rows += [dict(model="ensemble_perhead+C_offset", split=k, metric=mk, value=mv)
                            for mk, mv in met.items()]
        say(f"  【各头各取最优权重】集成 {len(seeds_ph)} 个seed；val上最优偏置: down "
            f"{best_ph[1]:+.2f}, up {best_ph[2]:+.2f}  → val macro-F1 {best_ph[0]:.4f}")
    best_tf = tune_offsets(base_p_tf[sidx["val"]])
    best_tfb = tune_offsets(base_p_tfb[sidx["val"]])
    best_orc = tune_offsets(orc_p[sidx["val"]])
    say(f"  (基线也在val上调同样的偏置，公平对比：TF先验 val macro-F1 {best_tf[0]:.4f}；"
        f"TF×结合先验 {best_tfb[0]:.4f})")
    tuned_pred = {k: apply_offsets(ens[k]["probs"], best) for k in ("val", "test")}
    for k in ("val", "test"):
        met = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], yc_t=y_c_all[sidx[k]],
                           pred_c=tuned_pred[k])
        metric_rows += [dict(model="ensemble+C_offset", split=k, metric=mk, value=mv)
                        for mk, mv in met.items()]
    # ---- 2026-09-25：Head C "模型+TF×结合先验"堆叠(文件头2026-09-25第1条)，只在 val 上拟合 ----
    def stack_feats(pm, pq):
        lm, lq = np.log(np.clip(pm, 1e-12, 1.0)), np.log(np.clip(pq, 1e-12, 1.0))
        return np.column_stack([lm[:, 0] - lm[:, 1], lm[:, 2] - lm[:, 1],
                                lq[:, 0] - lq[:, 1], lq[:, 2] - lq[:, 1]])

    def fit_stack(pm, pq, yy, lam=1e-3, n_iter=50):
        """多类 logistic(ns=参照类)，Newton 法；特征先标准化。返回 (W(2×5), mu, sd)。"""
        X = stack_feats(pm, pq)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        Y = np.column_stack([yy == 0, yy == 2]).astype(np.float64)
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
        return W, mu, sd

    def apply_stack(pm, pq, st):
        W, mu, sd = st
        Z = np.column_stack([np.ones(len(pm)), (stack_feats(pm, pq) - mu) / sd])
        eta = Z @ W.T
        lg = np.column_stack([eta[:, 0], np.zeros(len(eta)), eta[:, 1]])
        lg = lg - lg.max(1, keepdims=True)
        return np.exp(lg) / np.exp(lg).sum(1, keepdims=True)

    yv_ = y_c_all[sidx["val"]]
    stack_info = {}
    for tag_, ens_x in (("ensemble", ens), ("ensemble_perhead", ens_ph)):
        if ens_x is None:
            continue
        st_ = fit_stack(ens_x["val"]["probs"], base_p_tfb[sidx["val"]], yv_)
        sp_ = {k: apply_stack(ens_x[k]["probs"], base_p_tfb[sidx[k]], st_) for k in ("val", "test")}
        bst_ = tune_offsets(sp_["val"])
        for k in ("val", "test"):
            met = head_metrics(genes_all[sidx[k]], tfs_all[sidx[k]], probs=sp_[k],
                               yc_t=y_c_all[sidx[k]], pred_c=apply_offsets(sp_[k], bst_))
            metric_rows += [dict(model=f"{tag_}+prior_stack", split=k, metric=mk, value=mv)
                            for mk, mv in met.items()]
        stack_info[tag_] = dict(coef_down=st_[0][0].round(3).tolist(),
                                coef_up=st_[0][1].round(3).tolist(), offsets=bst_)
        say(f"  【{tag_}+TF×结合先验 堆叠】val上拟合的标准化系数(截距, 模型dn, 模型up, 先验dn, 先验up)："
            f"down类 {st_[0][0].round(2).tolist()}  up类 {st_[0][1].round(2).tolist()}；"
            f"val macro-F1(调偏置后) {bst_[0]:.4f}")

    # 基线指标
    for k in ("val", "test"):
        ix = sidx[k]
        g_, t_ = genes_all[ix], tfs_all[ix]
        for name, kw in (
                ("base_layout_linreg(A)", dict(ya_p=base_a[ix], ya_t=y_a_all[ix])),
                ("base_TF_mean(B)", dict(yb_p=base_b_tf[ix], yb_t=y_b_all[ix], ybd_t=ybd_of(ix))),
                ("base_TFxbound(B)", dict(yb_p=base_b_tfb[ix], yb_t=y_b_all[ix], ybd_t=ybd_of(ix))),
                ("base_TF_prior(C)", dict(probs=base_p_tf[ix], yc_t=y_c_all[ix],
                                          pred_c=apply_offsets(base_p_tf[ix], best_tf))),
                ("base_TFxbound_prior(C)", dict(probs=base_p_tfb[ix], yc_t=y_c_all[ix],
                                                pred_c=apply_offsets(base_p_tfb[ix], best_tfb))),
                ("base_always_ns(C)", dict(yc_t=y_c_all[ix], pred_c=np.ones(len(ix), np.int64))),
                ("ORACLE_gene_otherTF(B)", dict(yb_p=orc_b[ix], yb_t=y_b_all[ix])),
                ("ORACLE_gene_otherTF(C)", dict(probs=orc_p[ix], yc_t=y_c_all[ix],
                                                pred_c=apply_offsets(orc_p[ix], best_orc)))):
            met = head_metrics(g_, t_, **kw)
            metric_rows += [dict(model=name, split=k, metric=mk, value=mv)
                            for mk, mv in met.items()]
        if base_bd_tf is not None:  # 2026-09-25b
            met = head_metrics(g_, t_, yb_p=base_bd_tf[ix], yb_t=y_b_all[ix], ybd_t=ybd_of(ix))
            metric_rows += [dict(model="base_TF_dense_mean(B)", split=k, metric=mk, value=mv)
                            for mk, mv in met.items()]
    ml = pd.DataFrame(metric_rows)
    # 集成 test 的按基因整群 bootstrap(复用16号 bootstrap_ci)
    say(f"  集成test指标的按基因整群bootstrap({n_boot}次)...")
    ev_ens = dict(y_a_pred=torch.from_numpy(ens["test"]["ya"].astype(np.float32)),
                  y_a_true=torch.from_numpy(y_a_all[sidx["test"]].astype(np.float32)),
                  y_b_pred=torch.from_numpy(ens["test"]["yb"].astype(np.float32)),
                  y_b_true=torch.from_numpy(y_b_all[sidx["test"]].astype(np.float32)),
                  logits_c=torch.from_numpy(np.log(np.clip(ens["test"]["probs"], 1e-12, 1.0))
                                            .astype(np.float32)),
                  y_c_true=torch.from_numpy(y_c_all[sidx["test"]]))
    ci_ens = tl.bootstrap_ci(ev_ens, n_boot=n_boot, seed=0, groups=genes_all[sidx["test"]])
    ci_ph = None
    if ens_ph is not None:
        ev_ph_ens = dict(ev_ens, y_a_pred=torch.from_numpy(ens_ph["test"]["ya"].astype(np.float32)),
                         y_b_pred=torch.from_numpy(ens_ph["test"]["yb"].astype(np.float32)),
                         logits_c=torch.from_numpy(np.log(np.clip(ens_ph["test"]["probs"], 1e-12,
                                                                  1.0)).astype(np.float32)))
        ci_ph = tl.bootstrap_ci(ev_ph_ens, n_boot=n_boot, seed=0, groups=genes_all[sidx["test"]])

    # --------------------------------------------------------------------------
    # 7. 逐TF / 逐基因表
    # --------------------------------------------------------------------------
    ixt = sidx["test"]
    dt = pd.DataFrame(dict(gene_id=genes_all[ixt], tf_depleted=tfs_all[ixt], bound=bound_all[ixt],
                           invisible=invisible_all[ixt], y_b=y_b_all[ixt], y_c=y_c_all[ixt],
                           p_b=ens["test"]["yb"], pred_c=ens["test"]["probs"].argmax(1),
                           pred_c_tuned=tuned_pred["test"],
                           p_down=ens["test"]["probs"][:, 0], p_up=ens["test"]["probs"][:, 2],
                           q_down=base_p_tfb[ixt][:, 0], q_up=base_p_tfb[ixt][:, 2]))
    ctx_nz = dict(zip(ctx_df["tf"], ctx_df["ctx_d_n_nonzero"]))
    tf_rows = []
    for tf, g in dt.groupby("tf_depleted", sort=True):
        s_ = g[np.isfinite(g["y_b"])]
        row = dict(tf=tf, n_test=len(g), n_sig=len(s_), n_down=int((g["y_c"] == 0).sum()),
                   n_up=int((g["y_c"] == 2).sum()), frac_bound=float(g["bound"].mean()),
                   frac_invisible=float(g["invisible"].mean()), ctx_d_n_nonzero=ctx_nz.get(tf),
                   train_tf_mean_log2fc=tf_mean.get(tf, np.nan),
                   mean_true_log2fc_sig=float(s_["y_b"].mean()) if len(s_) else np.nan,
                   mean_pred_log2fc_sig=float(s_["p_b"].mean()) if len(s_) else np.nan,
                   sign_acc=float(np.mean(np.sign(s_["p_b"]) == np.sign(s_["y_b"])))
                   if len(s_) else np.nan,
                   r_b_within_tf=pearson(s_["p_b"], s_["y_b"]) if len(s_) >= min_sig_per_tf
                   else np.nan,
                   spearman_b_within_tf=spearman(s_["p_b"], s_["y_b"])
                   if len(s_) >= min_sig_per_tf else np.nan,
                   auroc_down_within_tf=auroc(g["p_down"], g["y_c"] == 0),
                   auroc_up_within_tf=auroc(g["p_up"], g["y_c"] == 2),
                   prior_auroc_down_within_tf=auroc(g["q_down"], g["y_c"] == 0),
                   prior_auroc_up_within_tf=auroc(g["q_up"], g["y_c"] == 2))
        for c, nm in ((0, "down"), (2, "up")):
            for col, tag in (("pred_c", ""), ("pred_c_tuned", "_tuned")):
                tp = int(((g[col] == c) & (g["y_c"] == c)).sum())
                fp = int(((g[col] == c) & (g["y_c"] != c)).sum())
                fn = int(((g[col] != c) & (g["y_c"] == c)).sum())
                row[f"f1_{nm}{tag}"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan
        tf_rows.append(row)
    per_tf = pd.DataFrame(tf_rows).sort_values("n_sig", ascending=False)
    rb_ok = per_tf["r_b_within_tf"].dropna()
    # 逐基因 Head A
    gene_rows = []
    for k in ("val", "test"):
        ix = sidx[k]
        _, first = np.unique(genes_all[ix], return_index=True)
        for j in first:
            g = genes_all[ix][j]
            row = dict(gene_id=g, split=k, y_a_true=y_a_all[ix][j], ens_y_a=ens[k]["ya"][j],
                       base_layout_linreg=base_a[ix][j],
                       n_sites=int(gene_feat.get(g, np.zeros(1))[0]))
            for s in seeds_ok:
                row[f"y_a_seed{s}"] = preds[k][s]["ya"][j]
            for s in seeds_ph:
                row[f"y_a_seed{s}_ph"] = preds_ph[k][s]["ya"][j]
            if ens_ph is not None:
                row["ens_y_a_ph"] = ens_ph[k]["ya"][j]
            gene_rows.append(row)
    per_gene = pd.DataFrame(gene_rows)
    per_gene["abs_err_ens"] = (per_gene["ens_y_a"] - per_gene["y_a_true"]).abs()

    # --------------------------------------------------------------------------
    # 8. 写文件
    # --------------------------------------------------------------------------
    say("\n[8] 写文件")
    written = []

    def save(df, name, **kw):
        path = os.path.join(outdir, name)
        if name.endswith(".parquet"):
            try:
                df.to_parquet(path, index=False)
            except ImportError:  # 没有pyarrow/fastparquet时退回csv.gz，不让导出整个失败
                name = name[:-len(".parquet")] + ".csv.gz"
                df.to_csv(os.path.join(outdir, name), index=False, compression="gzip")
                say(f"  (没有parquet引擎，{name} 改存成 csv.gz)")
        elif name.endswith(".csv.gz"):
            df.to_csv(path, index=False, compression="gzip", **kw)
        else:
            df.to_csv(path, index=False, **kw)
        written.append(name)

    save(inv, "checkpoint_inventory.csv")
    save(lc, "learning_curves.csv")
    save(ml, "metrics_long.csv")
    wide = ml[ml["split"] == "test"].pivot_table(index="model", columns="metric", values="value",
                                                 aggfunc="first")
    order = [f"seed{s}" for s in seeds_ok] + [f"seed{s}_perhead" for s in seeds_ph] + \
        ["ensemble", "ensemble+C_offset", "ensemble+prior_stack", "ensemble_perhead",
         "ensemble_perhead+C_offset", "ensemble_perhead+prior_stack"] + \
        sorted(m for m in wide.index if m.startswith("base_")) + \
        sorted(m for m in wide.index if m.startswith("ORACLE_"))
    wide = wide.reindex([m for m in order if m in wide.index])
    save(wide.reset_index(), "metrics_test_wide.csv")
    save(pd.DataFrame(loss_rows), "head_loss_parts.csv")
    save(rc, "recheck_vs_stored.csv")
    save(ctx_df, "ctx_diagnostic_per_tf.csv")
    save(per_tf, "per_tf_test.csv")
    save(per_gene, "per_gene_head_a.csv")
    pc_rows = []
    for mdl in ["ensemble", "ensemble+C_offset", "ensemble_perhead", "ensemble_perhead+C_offset"] + \
            [f"seed{s}" for s in seeds_ok]:
        for c in ("down", "ns", "up"):
            sub = ml[(ml["model"] == mdl) & (ml["split"] == "test")].set_index("metric")["value"]
            pc_rows.append(dict(model=mdl, cls=c, f1=sub.get(f"C_f1_{c}"),
                                precision=sub.get(f"C_prec_{c}"), recall=sub.get(f"C_rec_{c}")))
    save(pd.DataFrame(pc_rows), "head_c_per_class.csv")
    cm_e = cls_report(ens["test"]["probs"].argmax(1), y_c_all[ixt])["cm"]
    cm_t = cls_report(tuned_pred["test"], y_c_all[ixt])["cm"]
    for cm_, nm in ((cm_e, "confusion_test_ensemble.csv"), (cm_t, "confusion_test_ensemble_offset.csv")):
        save(pd.DataFrame(cm_, index=[f"true_{c}" for c in cls_names],
                          columns=[f"pred_{c}" for c in cls_names]).reset_index(), nm)
    for k in ("test", "val"):
        ix = sidx[k]
        cols = dict(gene_id=genes_all[ix], tf_depleted=tfs_all[ix], split=k,
                    D_in_Lg=bound_all[ix], condition_invisible=invisible_all[ix],
                    y_a_true=y_a_all[ix], y_b_true=y_b_all[ix],
                    y_c_true=np.array(cls_names, dtype=object)[y_c_all[ix]],
                    ens_y_a=ens[k]["ya"], ens_y_b=ens[k]["yb"],
                    ens_p_down=ens[k]["probs"][:, 0], ens_p_ns=ens[k]["probs"][:, 1],
                    ens_p_up=ens[k]["probs"][:, 2],
                    ens_pred_c=np.array(cls_names, dtype=object)[ens[k]["probs"].argmax(1)],
                    ens_pred_c_offset=np.array(cls_names, dtype=object)[tuned_pred[k]],
                    base_TF_mean_y_b=base_b_tf[ix], base_TFxbound_y_b=base_b_tfb[ix],
                    base_TF_p_down=base_p_tf[ix][:, 0], base_TF_p_up=base_p_tf[ix][:, 2],
                    base_TFxbound_p_down=base_p_tfb[ix][:, 0],
                    base_TFxbound_p_up=base_p_tfb[ix][:, 2],
                    ORACLE_gene_otherTF_y_b=orc_b[ix])
        if ybd_eval is not None:  # 2026-09-25b
            cols["y_bd_dense"] = ybd_eval[ix]
        for s in seeds_ok:
            cols[f"y_a_seed{s}"] = preds[k][s]["ya"]
            cols[f"y_b_seed{s}"] = preds[k][s]["yb"]
            cols[f"p_down_seed{s}"] = preds[k][s]["probs"][:, 0]
            cols[f"p_up_seed{s}"] = preds[k][s]["probs"][:, 2]
        for s in seeds_ph:  # 第2批：各头各取最优权重
            cols[f"y_a_seed{s}_ph"] = preds_ph[k][s]["ya"]
            cols[f"y_b_seed{s}_ph"] = preds_ph[k][s]["yb"]
            cols[f"p_down_seed{s}_ph"] = preds_ph[k][s]["probs"][:, 0]
            cols[f"p_up_seed{s}_ph"] = preds_ph[k][s]["probs"][:, 2]
        if ens_ph is not None:
            cols.update(ens_ph_y_a=ens_ph[k]["ya"], ens_ph_y_b=ens_ph[k]["yb"],
                        ens_ph_p_down=ens_ph[k]["probs"][:, 0], ens_ph_p_up=ens_ph[k]["probs"][:, 2],
                        ens_ph_pred_c_offset=np.array(cls_names, dtype=object)[tuned_ph[k]])
        pdf = pd.DataFrame(cols)
        save(pdf, f"predictions_{k}.parquet")
        if k == "test" and write_csv_gz and "predictions_test.csv.gz" not in written:
            save(pdf, "predictions_test.csv.gz", float_format="%.5g")

    # --------------------------------------------------------------------------
    # 9. 画图(英文标注)
    # --------------------------------------------------------------------------
    if make_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(fig_dir, exist_ok=True)
            # 9a 学习曲线
            if len(lc):
                fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
                for s in seeds_ok:
                    d = lc[lc["seed"] == s]
                    xs = (d["epoch_progress"].fillna(d["epoch"] + 1) if "epoch_progress" in d
                          else d["epoch"] + 1)
                    ax[0].plot(xs, d["train_loss"], "-o", ms=3, label=f"seed{s}")
                    ax[1].plot(xs, d["val_loss"], "-o", ms=3, label=f"seed{s}")
                ax[0].set_title("train loss (mean since previous validation)")
                ax[1].set_title("val total loss")
                for a in ax:
                    a.set_xlabel("epochs completed")
                    a.grid(alpha=.3)
                    a.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=150)
                plt.close(fig)
            # 9a' 第2批：每次验证的 val 指标曲线(老 checkpoint 没有这些字段就不画)
            if len(lc) and "val_composite" in lc.columns and lc["val_composite"].notna().any():
                cols_m = [("val_r_a_gene", "val Head A r (gene)"), ("val_r_b", "val Head B r"),
                          ("val_auprc_mean", "val Head C AUPRC (mean dn/up)"),
                          ("val_composite", "val composite"), ("val_l_a", "val l_a"),
                          ("val_l_b", "val l_b")]
                fig, axs = plt.subplots(2, 3, figsize=(14, 7))
                for a, (cm_, tt) in zip(axs.ravel(), cols_m):
                    for s in seeds_ok:
                        d = lc[lc["seed"] == s]
                        if cm_ in d:
                            a.plot(d["epoch_progress"], d[cm_], "-o", ms=2, label=f"seed{s}")
                    a.set_title(tt, fontsize=9)
                    a.set_xlabel("epochs completed")
                    a.grid(alpha=.3)
                axs[0, 0].legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, "val_metric_curves.png"), dpi=150)
                plt.close(fig)
            # 9b Head A 散点
            pg = per_gene[per_gene["split"] == "test"]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(pg["y_a_true"], pg["ens_y_a"], s=6, alpha=.5)
            ax.set_xlabel("true baseline log TPM (z-score)")
            ax.set_ylabel("predicted (5-seed ensemble)")
            ax.set_title(f"Head A, test genes n={len(pg)}, r={pearson(pg['ens_y_a'], pg['y_a_true']):.3f}")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "head_a_scatter_test.png"), dpi=150)
            plt.close(fig)
            # 9c Head B 散点
            sg = dt[np.isfinite(dt["y_b"])]
            fig, ax = plt.subplots(figsize=(5.5, 5))
            for flag, col, lab in ((False, "C0", "condition visible"),
                                   (True, "C3", "condition INVISIBLE")):
                q = sg[sg["invisible"] == flag]
                ax.scatter(q["y_b"], q["p_b"], s=5, alpha=.4, c=col, label=f"{lab} (n={len(q)})")
            ax.axhline(0, color="k", lw=.5)
            ax.axvline(0, color="k", lw=.5)
            ax.set_xlabel("true log2FC (significant only)")
            ax.set_ylabel("predicted log2FC (ensemble)")
            ax.set_title(f"Head B test, r={pearson(sg['p_b'], sg['y_b']):.3f}")
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "head_b_scatter_test.png"), dpi=150)
            plt.close(fig)
            # 9d 混淆矩阵
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            for a, cm_, tt in ((ax[0], cm_e, "argmax"), (ax[1], cm_t, "val-tuned logit offset")):
                rn = cm_ / np.maximum(cm_.sum(1, keepdims=True), 1)
                a.imshow(rn, cmap="Blues", vmin=0, vmax=1)
                for i in range(3):
                    for j in range(3):
                        a.text(j, i, f"{cm_[i, j]}\n{rn[i, j]:.2f}", ha="center", va="center",
                               fontsize=8, color="k" if rn[i, j] < .6 else "w")
                a.set_xticks(range(3))
                a.set_xticklabels(cls_names)
                a.set_yticks(range(3))
                a.set_yticklabels(cls_names)
                a.set_xlabel("predicted")
                a.set_ylabel("true")
                a.set_title(f"Head C test ({tt}), row-normalised")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "head_c_confusion_test.png"), dpi=150)
            plt.close(fig)
            # 9e 逐TF r_b
            ptf = per_tf.dropna(subset=["r_b_within_tf"]).sort_values("r_b_within_tf")
            if len(ptf):
                fig, ax = plt.subplots(figsize=(max(6, .18 * len(ptf)), 4))
                ax.bar(range(len(ptf)), ptf["r_b_within_tf"], color=np.where(
                    ptf["frac_invisible"] > .5, "C3", "C0"))
                ax.set_xticks(range(len(ptf)))
                ax.set_xticklabels(ptf["tf"], rotation=90, fontsize=6)
                ax.axhline(0, color="k", lw=.5)
                ax.set_ylabel("within-TF Pearson r (Head B)")
                ax.set_title(f"Head B within each depleted TF (>= {min_sig_per_tf} sig test samples); "
                             "red = >50% samples condition-invisible", fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, "head_b_within_tf_r.png"), dpi=150)
                plt.close(fig)
            # 9f 模型 vs 基线
            show = [("A_r_gene", "Head A r"), ("B_r", "Head B r"), ("B_sign_acc", "B sign acc"),
                    ("B_r_within_tf", "B within-TF r"), ("C_macro_f1", "C macro-F1"),
                    ("C_auprc_down", "C AUPRC down"), ("C_auprc_up", "C AUPRC up")]
            mdls = [m for m in ["ensemble", "ensemble+C_offset", "ensemble_perhead",
                                "ensemble_perhead+C_offset", "base_layout_linreg(A)",
                                "base_TF_mean(B)", "base_TFxbound(B)", "base_TF_prior(C)",
                                "base_TFxbound_prior(C)", "base_always_ns(C)"] if m in wide.index]
            fig, ax = plt.subplots(figsize=(12, 4.2))
            wbar = .8 / max(len(mdls), 1)
            for i, mdl in enumerate(mdls):
                vals = [wide.loc[mdl].get(c, np.nan) if c in wide.columns else np.nan for c, _ in show]
                ax.bar(np.arange(len(show)) + i * wbar, np.nan_to_num(np.asarray(vals, float)),
                       width=wbar, label=mdl)
            ax.set_xticks(np.arange(len(show)) + .4 - wbar / 2)
            ax.set_xticklabels([t for _, t in show])
            ax.set_title("Test: model (5-seed ensemble) vs train-fitted baselines "
                         "(missing bar = metric not defined for that baseline)")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "model_vs_baselines_test.png"), dpi=150)
            plt.close(fig)
            # 9g val loss 分项
            lr_ = pd.DataFrame(loss_rows)
            lv = lr_[lr_["split"] == "val"]
            if len(lv):
                fig, ax = plt.subplots(figsize=(7, 4))
                bottom = np.zeros(len(lv))
                cfg0 = ckpts[seeds_ok[0]].get("train_config") or {}
                lams = dict(l_a=cfg0.get("lambda_a", 1.0), l_b=cfg0.get("lambda_b", 1.0),
                            l_c=cfg0.get("lambda_c", 1.0),
                            l_sign=cfg0.get("lambda_sign", 0.1))
                for kk, lam in lams.items():
                    v = lam * lv[kk].to_numpy()
                    ax.bar([f"seed{s}" for s in lv["seed"]], v, bottom=bottom, label=f"{kk} x{lam}")
                    bottom += v
                ax.set_title("Val loss decomposition at the best epoch (same chunking as training)")
                ax.legend(fontsize=8)
                ax.grid(alpha=.3, axis="y")
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, "val_loss_parts.png"), dpi=150)
                plt.close(fig)
            written += [f"fig/{f}" for f in sorted(os.listdir(fig_dir))]
        except ImportError:
            say("  (没有matplotlib，跳过画图)")

    # --------------------------------------------------------------------------
    # 10. summary
    # --------------------------------------------------------------------------
    say("\n[10] 关键结果")
    lrdf = pd.DataFrame(loss_rows)
    say("  (a) 每个seed最优权重下的loss分项(λ加权前；share_*=该项×λ占total的比例)：")
    for r in lrdf.itertuples():
        say(f"      seed{r.seed} {r.split:4s}: l_a={r.l_a:.3f} l_b={r.l_b:.3f} l_c={r.l_c:.3f} "
            f"l_sign={r.l_sign:.3f}" + (f" l_bd={r.l_bd:.3f}" if getattr(r, "l_bd", 0.0) else "")
            + f" total={r.total:.3f}(evaluate报的{r.evaluate_loss:.3f})  "
            f"占比A/B/C={r.share_l_a:.0%}/{r.share_l_b:.0%}/{r.share_l_c:.0%}"
            + (f"/Bd={r.share_l_bd:.0%}" if getattr(r, "share_l_bd", 0.0) else ""))
    key = ["A_r_gene", "A_spearman_gene", "B_r", "B_spearman", "B_rmse", "B_sign_acc",
           "B_r_within_true_down", "B_r_within_true_up", "B_r_within_tf", "C_acc", "C_macro_f1",
           "C_f1_down", "C_f1_ns", "C_f1_up", "C_auroc_down", "C_auroc_up", "C_auprc_down",
           "C_auprc_up"]
    short = ["A_r", "A_rho", "B_r", "B_rho", "B_rmse", "B_sign", "B_r|dn", "B_r|up", "B_r|TF",
             "C_acc", "C_mF1", "C_F1dn", "C_F1ns", "C_F1up", "C_AUCdn", "C_AUCup", "C_APdn",
             "C_APup"]
    if ybd_eval is not None:  # 2026-09-25b
        key += ["B_dr", "B_dr_ns"]
        short += ["B_dr", "B_drns"]
    say("  (b) test 宽表(行=模型/基线，nan=该指标对这个基线没定义)。列名缩写：A_r/A_rho=Head A"
        "基因级Pearson/Spearman；B_sign=符号准确率；B_r|dn、B_r|up=只在真实下调/上调子集内的r；"
        "B_r|TF=TF内中心化r；C_mF1=macro-F1；AUC=AUROC；AP=AUPRC"
        + ("；B_dr/B_drns=Head B 预测 vs 稠密log2FC(全部有值样本/只看不显著样本)" if ybd_eval is not None else ""))
    say("      " + "model".ljust(24) + "".join(x.rjust(8) for x in short))
    for mdl, row in wide.iterrows():
        say("      " + str(mdl).ljust(24) + "".join(fmt(row.get(k, np.nan), 3).rjust(8)
                                                     for k in key))
    ft = facts["test"]
    say(f"      参照：AUROC随机=0.5；AUPRC随机=该类在test里的占比(down {ft['n_down'] / ft['n_samples']:.4f}"
        f"、up {ft['n_up'] / ft['n_samples']:.4f})；永远猜ns的acc={ft['always_ns_acc']:.4f}、"
        f"macro-F1={ft['always_ns_macro_f1']:.4f}")
    say(f"  (c) 集成test按基因整群95% CI: r_a={ci_ens['r_a_gene_ci']}  r_b={ci_ens['r_b_ci']}  "
        f"macro_f1(argmax)={ci_ens['macro_f1_c_ci']}")
    say(f"  (d) 逐TF：{len(rb_ok)} 个TF有≥{min_sig_per_tf}个显著test样本，TF内r_b 中位数 "
        f"{fmt(float(rb_ok.median()) if len(rb_ok) else float('nan'), 3)}、"
        f"四分位 {fmt(float(rb_ok.quantile(.25)) if len(rb_ok) else float('nan'), 3)}~"
        f"{fmt(float(rb_ok.quantile(.75)) if len(rb_ok) else float('nan'), 3)}；"
        f"r<0 的TF {int((rb_ok < 0).sum())} 个")
    say(f"  (e) val Head C 偏置: down {best[1]:+.2f}/up {best[2]:+.2f}")
    if ci_ph is not None:
        say(f"  (f)【各头各取最优权重】集成test按基因整群95% CI: r_a={ci_ph['r_a_gene_ci']}  "
            f"r_b={ci_ph['r_b_ci']}  macro_f1(argmax)={ci_ph['macro_f1_c_ci']}；val偏置 down "
            f"{best_ph[1]:+.2f}/up {best_ph[2]:+.2f}")
    for c_, nm_ in (("down", "down"), ("up", "up")):
        ok_ = per_tf[per_tf[f"n_{c_}"] >= 5]
        say(f"  (h) Head C TF内AUROC({nm_}，{len(ok_)}个TF有≥5个{nm_}样本；只比同一被耗竭TF内部的基因排序)："
            f"模型集成 中位 {fmt(float(ok_[f'auroc_{c_}_within_tf'].median()) if len(ok_) else float('nan'), 3)}"
            f"  vs TF×结合先验 {fmt(float(ok_[f'prior_auroc_{c_}_within_tf'].median()) if len(ok_) else float('nan'), 3)}")
    for tag_, inf_ in stack_info.items():
        sub_ = wide.loc[f"{tag_}+prior_stack"] if f"{tag_}+prior_stack" in wide.index else None
        if sub_ is not None:
            say(f"  (i) {tag_}+TF先验堆叠 test：AUROC dn/up={fmt(sub_.get('C_auroc_down'), 3)}/"
                f"{fmt(sub_.get('C_auroc_up'), 3)}  AUPRC dn/up={fmt(sub_.get('C_auprc_down'), 3)}/"
                f"{fmt(sub_.get('C_auprc_up'), 3)}  macro-F1(val调偏置)={fmt(sub_.get('C_macro_f1'), 3)}"
                f"；标准化系数 down {inf_['coef_down']} up {inf_['coef_up']}(截距,模型dn,模型up,先验dn,先验up)")
    if ybd_eval is not None and "ensemble" in wide.index:  # 2026-09-25b
        e_ = wide.loc["ensemble"]
        b_ = wide.loc["base_TF_dense_mean(B)"] if "base_TF_dense_mean(B)" in wide.index else {}
        say(f"  (j) Head B vs 稠密 log2FC(test)：集成 B_dr={fmt(e_.get('B_dr'), 3)}、只看不显著样本 "
            f"B_dr_ns={fmt(e_.get('B_dr_ns'), 3)}(RMSE {fmt(e_.get('B_dr_rmse_ns'), 3)})；TF稠密均值基线 "
            f"{fmt(b_.get('B_dr', np.nan) if len(b_) else np.nan, 3)}/"
            f"{fmt(b_.get('B_dr_ns', np.nan) if len(b_) else np.nan, 3)}"
            + ("  (这个实验训练时用了稠密目标)" if any((c.get("train_config") or {}).get("lambda_bd")
                                             for c in ckpts.values()) else "  (训练时没用稠密目标)"))
    sel_rows = [(s_, (ck_.get("best") or {})) for s_, ck_ in ckpts.items() if ck_.get("best")]
    if sel_rows:
        say("  (g) 每个seed选中的权重(16号第2批记录)：")
        for s_, b_ in sel_rows:
            hb_ = "  ".join(f"{h}@{v.get('epoch_progress')}" for h, v in
                            (b_.get("head_best") or {}).items())
            say(f"      seed{s_}: 按{b_.get('metric')}选中 val#{b_.get('val_index')} "
                f"@epoch{b_.get('epoch_progress')}，停止原因={b_.get('stop_reason')}"
                + (f"；各头最优 {hb_}" if hb_ else ""))
    say(f"\n  写出 {len(written)} 个文件到 {outdir}/ ：")
    for w in written:
        say(f"    {w}")
    say(f"\n总用时 {(time.time() - t_start) / 60:.1f} 分钟。把 {outdir}/summary.txt 整个贴回来即可。")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary) + "\n")
    with open(os.path.join(outdir, "summary_facts.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(data_facts=facts, condition_diag=diag, head_c_offset=dict(
            val_macro_f1=best[0], down=best[1], up=best[2]),
            baseline_offsets=dict(TF_prior=best_tf, TFxbound_prior=best_tfb),
            head_c_offset_perhead=(dict(val_macro_f1=best_ph[0], down=best_ph[1], up=best_ph[2])
                                   if best_ph else None),
            run_name=run_name, ctx_mode=run_ctx[0], seeds=seeds_ok, seeds_perhead=seeds_ph,
            export_format=EXPORT_FORMAT, prior_stack=stack_info,
            dense_target=dense_target if ybd_eval is not None else None,
            ensemble_ci={k: v for k, v in ci_ens.items() if k.endswith("_ci")}),
            fh, ensure_ascii=False, indent=2, default=str)
    return dict(metrics_long=ml, metrics_test_wide=wide, per_tf=per_tf, per_gene=per_gene, ds=ds)


if __name__ == "__main__":
    # 第2批：自动发现要导出的实验(见文件头第2批第1条)，逐个调用 run_export，Dataset 只建一次
    _cfg = dict(CONFIG)
    _runs, _root = _cfg.pop("runs"), _cfg.pop("ckpt_root")
    _res_root, _force = _cfg.pop("results_root"), _cfg.pop("force")
    if _runs == "auto":
        _runs = []
        if glob.glob(os.path.join(_root, "seed*_best.pt")):
            _runs.append(("run1", _root))
        for _d in sorted(os.listdir(_root)) if os.path.isdir(_root) else []:
            _p = os.path.join(_root, _d)
            if os.path.isdir(_p) and glob.glob(os.path.join(_p, "seed*_best.pt")):
                _runs.append((_d, _p))
    _todo = []
    for _name, _cdir in _runs:
        _out = os.path.join(_res_root, _name)
        _cks = glob.glob(os.path.join(_cdir, "seed*_best.pt"))
        _summ = os.path.join(_out, "summary.txt")
        if not _cks:
            print(f"[跳过] {_name}: {_cdir} 下没有 seed*_best.pt")
        elif (not _force and os.path.exists(_summ)
              and os.path.getmtime(_summ) > max(os.path.getmtime(f) for f in _cks)
              and str(json.load(open(os.path.join(_out, "summary_facts.json"), encoding="utf-8"))
                      .get("export_format", "")) >= EXPORT_FORMAT
              if os.path.exists(os.path.join(_out, "summary_facts.json")) else False):
            print(f"[跳过] {_name}: {_summ} 比它的 {len(_cks)} 个 checkpoint 都新、导出格式也是最新"
                  "(CONFIG['force']=True 可强制重导)")
        else:
            _todo.append((_name, _cdir, _out))
    print(f"要导出的实验: {[t[0] for t in _todo] or '无'}")
    _ds = None
    for _name, _cdir, _out in _todo:
        _res = run_export(run_name=_name, ckpt_dir=_cdir, outdir=_out, ds=_ds, **_cfg)
        _ds = _res["ds"]
