# scripts/tflayout/25_head_a_all_genes.py

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
    # 2026-09-30a 第10批：参照=新主线 v8；第9批的 (v3_ce_marker_dense, v4_abl_no_cis, v8_headA_all) 结果在 _head_a_all/ 里留档
    # 2026-10-01a 第11批：只比 v8 和补到 5 seed 的 no_knockout(第10批是 v8 + 四个 v10_*，结果在 _head_a_all_b10/)
    runs=("v8_headA_all", "v10_abl_no_knockout"),  # 缺 checkpoint 的自动跳过
    reference="v8_headA_all",
    variant="perhead",       # A 头用哪套权重："perhead"(报告口径)或 "selected"
    splits=("test", "val"),
    n_boot=1000,             # 按基因 bootstrap 次数
    device=None,             # None=有 GPU 用 GPU
    num_workers=4,
    eval_batch_size=64,      # 每行一个不同基因，cis 分支显存比平时大，跟 24 号一样压到 64
    repro_tol=1e-3,          # 网格基因跟 17 号导出的最大差容忍度(bf16 + batch 组成不同，24 号实测 1e-3~2.3e-3)；>10 倍直接停
    outdir="out/results/_head_a_all_b11",  # 2026-10-01a 第11批(第10批 _head_a_all_b10/、第9批 _head_a_all/ 不覆盖；26 号读第9批那份)
)


def run_head_a_all(paths, ckpt_root, results_root, runs, reference, variant, splits, n_boot, device, num_workers,
                   eval_batch_size, repro_tol, outdir):
    """唯一入口，各段见文件头。"""
    t_all = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def load(fn):  # 跟 18/22/24 号同一种按路径动态加载
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
            return float("nan")  # 2026-09-30a：近常数(集成后的浮点抖动)也算常数，见文件头第10批第2条
        return float(np.corrcoef(x, y)[0, 1])

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

    tl = load("16_train_loop.py")
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    say("=" * 78)
    say(f"Head A 全部基因评估(25 号，第11批 2026-10-01a)  {time.strftime('%Y-%m-%d %H:%M:%S')}  16号代码版本 {tl.CODE_VERSION}  "
        f"设备={device}  权重={variant}")
    say("=" * 78)

    # ---- 0. 伪标签表 + Dataset ----
    ha = pd.read_parquet(paths["head_a"])
    if "split" not in ha.columns:
        raise SystemExit(f"{paths['head_a']} 没有 split 列(先确认 08 号产出)")
    lbl = pd.read_parquet(paths["labels"], columns=["gene_id"])
    grid = set(lbl["gene_id"].tolist())
    d0 = sorted(set(pd.read_parquet(paths["layout"], columns=["tf"])["tf"]))[0]
    keep = ha[ha["split"].isin(list(splits))]
    genes = sorted(keep.index.tolist(), key=lambda g: (list(splits).index(keep.loc[g, "split"]), str(g)))
    pseudo = pd.DataFrame(dict(gene_id=genes, tf_depleted=d0, log2fc=np.nan, direction_3class="ns"))
    p_path = os.path.join(outdir, "_headA_pseudo_labels.parquet")
    pseudo.to_parquet(p_path, index=False)
    ds = tl.TFLayoutDataset(paths["layout"], p_path, paths["head_a"], paths["sgd"],
                            paths["promoter_tokens"], paths["bpe_tokenizer"])
    ga = ds.samples["gene_id"].to_numpy()
    sp = np.array([ha.loc[g, "split"] for g in ga], dtype=object)
    in_grid = np.array([g in grid for g in ga], bool)
    has_site = np.array([len(ds.layout_by_gene.get(g, {"tf_idx": []})["tf_idx"]) > 0 for g in ga], bool)
    y_true = np.array([ds.head_a.get(g, np.nan) for g in ga], np.float64)
    strata = {"全部": lambda m: m, "网格": lambda m: m & in_grid, "网格外": lambda m: m & ~in_grid,
              "网格外有位点": lambda m: m & ~in_grid & has_site, "网格外无位点": lambda m: m & ~in_grid & ~has_site}
    say(f"伪标签表：{len(ga)} 个 Head A 基因(split {list(splits)})，D 固定为 {d0}(Head A=ŷ_WT，跟 D 无关)")
    say("\n[1] 基因集合与 Head A 真值(z-score，08 号只用 train 基因拟合)")
    for s_ in splits:
        base = sp == s_
        parts = []
        for nm in ("网格", "网格外有位点", "网格外无位点"):
            m = strata[nm](base) & np.isfinite(y_true)
            parts.append(f"{nm} {int(m.sum())} 个(均值 {np.nanmean(y_true[m]):+.3f}±{np.nanstd(y_true[m]):.3f})"
                         if m.any() else f"{nm} 0 个")
        say(f"  {s_}: 共 {int(base.sum())} 个；" + "；".join(parts))

    # ---- 1. 逐实验逐 seed 推理 ----
    class_counts = torch.tensor([1.0, 1.0, 1.0])  # 只取 Head A 预测，loss 不看
    preds, seed_cfg = {}, {}
    for run in runs:
        cks = []
        for p in sorted(glob.glob(os.path.join(ckpt_root, run, "seed*_best.pt"))):
            mt = re.match(r"seed(\d+)_best\.pt$", os.path.basename(p))
            if mt:
                cks.append((int(mt.group(1)), p))
        if not cks:
            say(f"\n提示：{ckpt_root}/{run}/ 下没有 seed*_best.pt，跳过 {run}")
            continue
        E = None
        ep = os.path.join(results_root, run, "predictions_test.parquet")
        if os.path.exists(ep):
            E = pd.read_parquet(ep)
        preds[run] = {}
        for s, p in sorted(cks):
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:
                ck = torch.load(p, map_location="cpu")
            cfg = ck.get("train_config") or {}
            if not ck.get("model_kwargs"):
                say(f"  {run} seed{s}: checkpoint 没有 model_kwargs(老格式)，跳过")
                continue
            model = tl.SiameseHeadsModel(**ck["model_kwargs"])
            hs = ck.get("head_states") or {}
            use_a = variant == "perhead" and "A" in hs
            model.load_state_dict(hs["A"] if use_a else ck["model_state"], strict=True)
            model = model.to(device)
            model.eval()
            ds.set_ctx_mode(cfg.get("ctx_mode", "legacy"), float(cfg.get("ctx_clip", 3.0)), verbose=False)
            if hasattr(ds, "set_ablation"):
                ds.set_ablation(cfg.get("ablation", "none"), verbose=False)
            ekw = dict(device=device, num_workers=int(num_workers), amp=cfg.get("amp", "bf16"),
                       forward_mode=cfg.get("forward_mode", "grouped"), eval_batch_size=int(eval_batch_size),
                       layout_buckets=1, lambda_bd=0.0)
            t0 = time.time()
            ev = tl.evaluate(model, ds, list(range(len(ds))), class_counts, batch_size=128, **ekw)
            ya = to_np(ev["y_a_pred"])
            preds[run][s] = ya
            seed_cfg[(run, s)] = dict(head_a_all=bool(cfg.get("head_a_all")), ablation=cfg.get("ablation", "none"),
                                      code_version=ck.get("code_version"), weights="A头最优" if use_a else "选中")
            rep = ""
            col = f"y_a_seed{s}" + ("_ph" if use_a else "")
            if E is not None and col in E.columns:
                ref = E.drop_duplicates("gene_id").set_index("gene_id")[col]
                mm = (sp == "test") & in_grid & np.isin(ga, ref.index.to_numpy())
                if mm.any():
                    dmax = float(np.max(np.abs(ya[mm] - ref.reindex(ga[mm]).to_numpy(np.float64))))
                    rep = f"；test 网格基因跟 17 号导出 {col} 最大差 {dmax:.2e}"
                    if dmax > 10 * repro_tol:
                        raise SystemExit(f"{run} seed{s} 复现核对失败(最大差 {dmax:.3g})：模型重建或 Dataset 设置跟 17 号不一致")
            say(f"  {run} seed{s}: 推理 {time.time() - t0:.0f} 秒(ablation={seed_cfg[(run, s)]['ablation']}，"
                f"head_a_all={seed_cfg[(run, s)]['head_a_all']}，权重={seed_cfg[(run, s)]['weights']}){rep}")
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    if reference not in preds or not preds[reference]:
        raise SystemExit(f"参照 {reference} 没有任何可用的 seed")
    done_runs = [r for r in runs if r in preds and preds[r]]

    # ---- 2. 各实验全部 seed 集成 ----
    rng = np.random.default_rng(0)
    boot_idx = {}
    for s_ in splits:
        for nm, f in strata.items():
            ix = np.where(f(sp == s_) & np.isfinite(y_true))[0]
            boot_idx[(s_, nm)] = (ix, [rng.choice(ix, len(ix)) for _ in range(int(n_boot))] if len(ix) >= 5 else [])
    rows_m = []
    say(f"\n[2] 各实验全部 seed 集成的 Head A r [95% CI 按基因 bootstrap {n_boot} 次]；逐seed 均值±std")
    for run in done_runs:
        ens = np.mean(list(preds[run].values()), 0)
        say(f"  ── {run}({len(preds[run])} 个 seed {sorted(preds[run])})")
        for s_ in splits:
            cells = []
            for nm in strata:
                ix, bts = boot_idx[(s_, nm)]
                if len(ix) < 5:
                    continue
                r_ = pearson(ens[ix], y_true[ix])
                lo, hi = ci_of([pearson(ens[b], y_true[b]) for b in bts])
                per = [pearson(v[ix], y_true[ix]) for v in preds[run].values()]
                if not np.isfinite(r_):  # 2026-09-30a：常数预测，r 没有定义(见文件头第10批第2条)
                    cells.append(f"{nm}({len(ix)}) 常数预测(r 无定义)")
                else:
                    cells.append(f"{nm}({len(ix)}) {r_:.3f}[{lo:.3f},{hi:.3f}] 逐seed "
                                 + (f"{np.nanmean(per):.3f}±{np.nanstd(per):.3f}" if np.isfinite(per).any() else "无定义"))
                rows_m.append(dict(run=run, seeds=len(preds[run]), split=s_, stratum=nm, n_genes=len(ix), r=r_,
                                   ci_lo=lo, ci_hi=hi, r_seed_mean=float(np.nanmean(per)), r_seed_std=float(np.nanstd(per))))
            say(f"    {s_}: " + "  ".join(cells))
    pd.DataFrame(rows_m).to_csv(os.path.join(outdir, "metrics_all_seeds.csv"), index=False)

    # ---- 3. 配对差(共有 seed 上各自重算集成) ----
    rows_d = []
    say(f"\n[3] 配对差 = 实验 − 参照 {reference}(在两者共有 seed 上各自重算集成；按基因配对 bootstrap；逐seed 配对差括号里=变好的 seed 数)")
    for run in done_runs:
        if run == reference:
            continue
        common = sorted(set(preds[run]) & set(preds[reference]))
        if not common:
            say(f"  {run}: 跟参照没有共有 seed，跳过")
            continue
        e_r = np.mean([preds[run][s] for s in common], 0)
        e_0 = np.mean([preds[reference][s] for s in common], 0)
        say(f"  ── {run} − {reference}，共有 seed {common}")
        for s_ in splits:
            cells = []
            for nm in strata:
                ix, bts = boot_idx[(s_, nm)]
                if len(ix) < 5:
                    continue
                d_ = pearson(e_r[ix], y_true[ix]) - pearson(e_0[ix], y_true[ix])
                lo, hi = ci_of([pearson(e_r[b], y_true[b]) - pearson(e_0[b], y_true[b]) for b in bts])
                ps = [pearson(preds[run][s][ix], y_true[ix]) - pearson(preds[reference][s][ix], y_true[ix]) for s in common]
                star = " ↑" if lo > 0 else (" ↓" if hi < 0 else "")
                if not np.isfinite(d_):  # 2026-09-30a：有一边是常数预测
                    cells.append(f"{nm} 无定义(常数预测)")
                else:
                    cells.append(f"{nm} {d_:+.3f}[{lo:+.3f},{hi:+.3f}]{star} 逐seed {np.nanmean(ps):+.3f}"
                                 f"({sum(x > 0 for x in ps)}/{len(ps)})")
                rows_d.append(dict(run=run, reference=reference, seeds=str(common), split=s_, stratum=nm, n_genes=len(ix),
                                   r_run=pearson(e_r[ix], y_true[ix]), r_ref=pearson(e_0[ix], y_true[ix]), delta=d_,
                                   ci_lo=lo, ci_hi=hi, seed_delta_mean=float(np.mean(ps)),
                                   seeds_better=int(sum(x > 0 for x in ps))))
            say(f"    {s_}: " + "  ".join(cells))
    pd.DataFrame(rows_d).to_csv(os.path.join(outdir, "delta_vs_reference.csv"), index=False)

    # ---- 4. 组间效应拆解 ----
    say("\n[4] 组间效应：各组(网格/网格外有位点/网格外无位点)的真值与预测均值；\"组内中心化 r\"=各组分别减去自己的真值均值和预测均值后"
        "再合并算 r(扣掉组间均值差)。原始合并 r 明显高于组内中心化 r -> 合并数字有一部分来自\"模型分得清哪组整体表达高低\"")
    groups = {"网格": in_grid, "网格外有位点": ~in_grid & has_site, "网格外无位点": ~in_grid & ~has_site}
    for s_ in splits:
        base = (sp == s_) & np.isfinite(y_true)
        say(f"  {s_}: 真值均值 " + "  ".join(f"{k} {np.nanmean(y_true[base & m]):+.3f}" for k, m in groups.items()
                                        if (base & m).any()))
        for run in done_runs:
            ens = np.mean(list(preds[run].values()), 0)
            yc, pc = y_true.copy(), ens.copy()
            for m in groups.values():
                mm = base & m
                if mm.any():
                    yc[mm] -= np.nanmean(y_true[mm])
                    pc[mm] -= np.nanmean(ens[mm])
            say(f"    {run}: 预测均值 " + "  ".join(f"{k} {np.nanmean(ens[base & m]):+.3f}" for k, m in groups.items()
                                              if (base & m).any())
                + f"；合并 r {pearson(ens[base], y_true[base]):.3f} vs 组内中心化 r {pearson(pc[base], yc[base]):.3f}")

    # ---- 5. 逐基因表 + 图 ----
    per_gene = pd.DataFrame(dict(gene_id=ga, split=sp, in_grid=in_grid, has_sites=has_site, y_a_true=y_true))
    for run in done_runs:
        per_gene[f"{run}_ens"] = np.mean(list(preds[run].values()), 0)
        for s, v in preds[run].items():
            per_gene[f"{run}_seed{s}"] = v
    per_gene.to_csv(os.path.join(outdir, "per_gene_head_a.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dm = pd.DataFrame(rows_m)
        if dm.empty:
            raise ImportError("没有可画的指标")
        fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
        for ax, s_ in zip(axes[0], splits):
            sub = dm[dm["split"] == s_]
            nms = list(strata)
            w = 0.8 / max(len(done_runs), 1)
            for i, run in enumerate(done_runs):
                q = sub[sub["run"] == run].set_index("stratum").reindex(nms)
                x = np.arange(len(nms)) + i * w
                ax.bar(x, q["r"], w, yerr=[q["r"] - q["ci_lo"], q["ci_hi"] - q["r"]], label=run, capsize=2)
            ax.set_xticks(np.arange(len(nms)) + 0.4 - w / 2)
            ax.set_xticklabels(["all", "grid", "off-grid", "off-grid+sites", "off-grid no sites"], rotation=20, fontsize=8)
            ax.set_title(f"Head A r ({s_}, ensemble of all seeds)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig", "head_a_all_genes.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        say("(没有 matplotlib，跳过画图)")

    say("\n判读(第11批，预先写下，建议，不是硬规则；详见 18 号文件头第11批)：")
    say("  本脚本 [3] 的 test/网格 应跟 19 号(_compare_b11)[3] perhead 的 A_r 同号同量级(同一批基因，只差 bf16/batch 组成 ~1e-3)。")
    say("  no_knockout 不改 Head A 的输入(Head A=ŷ_WT)，是阴性对照：第10批 2 对 2 在 test 网格上 −.033(val −.002)。5 对 5 应回到 |Δ|≲0.01；")
    say("  仍然 ≥0.02 且 27 号合成区间不含0 -> 删位点通过共享主干影响了 Head A(讨论里写一句)；否则第10批那个 −.033 记为训练随机性。")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    say(f"\n写出 {outdir}/summary.txt、metrics_all_seeds.csv、delta_vs_reference.csv、per_gene_head_a.csv、"
        f"fig/head_a_all_genes.png  (用时 {(time.time() - t_all) / 60:.1f} 分钟)")


if __name__ == "__main__":
    run_head_a_all(**CONFIG)