# scripts/tflayout/29_deletion_specificity.py

import glob
import importlib.util
import os
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd

try:  # 包装类要放在模块层(DataLoader 用 spawn/forkserver 时要能 pickle)；沙盒没有 torch 时退成空基类
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    class _TorchDataset:  # noqa: D401
        pass

CONFIG = dict(
    paths=dict(layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
               head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
               promoter_tokens="out/promoter_token_ids.parquet",
               bpe_tokenizer="out/bpe_tokenizer.json"),
    ckpt_root="out/checkpoints",
    results_root="out/results",
    run_with="v8_headA_all",           # W：训练时删位点(论文主模型)
    variant="perhead",                 # B/C 头用哪套权重
    n_boot=500,                         # 按基因整群 bootstrap 次数
    min_rows_tf=20,                     # (q3) 该 TF 作为 D' 被删的行数下限
    min_sig_tf=8,                       # (q3) 算 role 需要的 train D∈L_g 显著行(down+up)下限
    wrong_seed=20261002,                # 选 D' 的哈希种子(固定，可复现)
    device=None,                        # None=有 GPU 用 GPU
    num_workers=4,
    eval_batch_size=512,
    smoke_first=True, smoke_genes=40,
    repro_tol=1e-3,
    n_check_rows=2000,                  # (s5) 抽样核对的行数
    outdir="out/results/_spec_control",
)


class WrongKnockoutView(_TorchDataset):
    """对 09 号 TFLayoutDataset 的只读包装：对 wrong_by_idx 里登记的样本(全局下标 -> D' 的 tf_idx)，
    把 layout_d 从"L_g 去掉 D"换成"L_g 去掉 D'"；其余样本、其余字段原样透传。base 必须处在 ablation=none。"""

    def __init__(self, base, wrong_by_idx):
        self.base = base
        self.wrong_by_idx = wrong_by_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        d2 = self.wrong_by_idx.get(int(idx), -1)
        if d2 >= 0:
            lg = item["layout_wt"]
            keep = lg["tf_idx"] != d2
            item["layout_d"] = {k: v[keep] for k, v in lg.items()}
        return item

    def __getattr__(self, name):  # 只在常规属性找不到时才进来：把 ds.samples / ds.tf2idx / ds.collate_grouped 等透传给 base
        if name in ("base", "wrong_by_idx") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__dict__["base"], name)


def run_deletion_specificity(paths, ckpt_root, results_root, run_with, variant, n_boot, min_rows_tf, min_sig_tf,
                             wrong_seed, device, num_workers, eval_batch_size, smoke_first, smoke_genes, repro_tol,
                             n_check_rows, outdir):
    """唯一入口，各段见文件头。"""
    t_all = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    lines, md, tex = [], [], []
    METS = ("B_r", "B_sign_acc", "B_r_within_tf", "C_auroc_down", "C_auprc_down", "C_auroc_up", "C_auprc_up")
    SHOW = ("C_auprc_down", "C_auprc_up", "C_auroc_down", "C_auroc_up", "B_r_within_tf", "B_r", "B_sign_acc")
    SHORT = dict(B_r="B_r", B_sign_acc="B_sign", B_r_within_tf="B_r|TF", C_auroc_down="C_AUCdn",
                 C_auprc_down="C_APdn", C_auroc_up="C_AUCup", C_auprc_up="C_APup")
    TEXM = dict(B_r=r"$r_B$", B_sign_acc="Sign acc.", B_r_within_tf=r"$r_B$ within TF",
                C_auroc_down=r"AUROC$_\downarrow$", C_auprc_down=r"AUPRC$_\downarrow$",
                C_auroc_up=r"AUROC$_\uparrow$", C_auprc_up=r"AUPRC$_\uparrow$")
    CELLS = ("W_del", "W_nodel", "W_wrong")
    CELL_DESC = {"W_del": "删对的 TF(主模型原样)", "W_nodel": "不删(28 号反事实)", "W_wrong": "删错的 TF(格式不变、内容错)"}
    # (名字, 变体, 参照)
    CONTRASTS = (("Δ_spec", "W_wrong", "W_del"), ("Δ_wn", "W_wrong", "W_nodel"), ("Δ_infer", "W_nodel", "W_del"))

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def load(fn):
        name = f"_tflayout_{fn[:-3]}"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, fn))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    def choose_wrong_tf(gene_id, d_idx, counts, seed):
        """为 (基因, 被耗竭 TF D) 选"被误删"的 TF D'(纯函数，便于单测)。
        counts：{tf_idx: 该基因启动子上的位点数}。规则：D 不在 L_g 里、或启动子上没有别的 TF -> 返回 -1(不做错配删除)；
        否则在"别的 TF"里选位点数跟 D 的位点数差最小的，平局时用 (基因, D, seed) 的 crc32 作随机种子选一个。"""
        n_d = int(counts.get(int(d_idx), 0))
        if n_d <= 0:
            return -1
        cand = [(int(t), int(n)) for t, n in counts.items() if int(t) != int(d_idx) and int(n) > 0]
        if not cand:
            return -1
        gap = min(abs(n - n_d) for _, n in cand)
        ties = sorted(t for t, n in cand if abs(n - n_d) == gap)
        rng = np.random.default_rng(zlib.crc32(f"{gene_id}|{int(d_idx)}|{int(seed)}".encode("utf-8")))
        return int(ties[int(rng.integers(len(ties)))])

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

    def auroc_auprc(score, pos):  # 跟 19/27/28 号逐行相同
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

    def tex_table(caption, label, header, rows, colspec, notes=""):
        out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
               r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%",
               rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
        for r_ in rows:
            out.append(r_ if (r_.strip() in ("\\midrule", "\\addlinespace") or r_.rstrip().endswith("\\\\"))
                       else (r_ + r" \\"))
        out += [r"\bottomrule", r"\end{tabular}}"]
        if notes:
            out.append(rf"\par\smallskip\parbox{{0.97\linewidth}}{{\scriptsize {notes}}}")
        out.append(r"\end{table}")
        tex.append("\n".join(out) + "\n")

    tl = load("16_train_loop.py")
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "CPU"
    say("=" * 100)
    say(f"删位点特异性对照(29 号，第13批 2026-10-02b)  {time.strftime('%Y-%m-%d %H:%M:%S')}  16号代码版本 {tl.CODE_VERSION}  "
        f"设备={device}({gpu_name})  权重={variant}")
    say("=" * 100)

    # ------------------------------------------------------------------ [0] Dataset、切分、D∈L_g、选 D'
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
    gene_counts = {}
    for g, lg in ds.layout_by_gene.items():
        ti = np.asarray(lg["tf_idx"], np.int64)
        if ti.size:
            u_, c_ = np.unique(ti, return_counts=True)
            gene_counts[g] = {int(t_): int(n_) for t_, n_ in zip(u_.tolist(), c_.tolist())}
    dep_idx = np.array([ds.tf2idx.get(t, -1) for t in tfs_all], np.int64)
    nds_all = np.array([gene_counts.get(g, {}).get(int(d), 0) for g, d in zip(genes_all, dep_idx)], np.int64)
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
    sig_te = np.isfinite(yb_te)
    idx2tf = {int(i): t for t, i in ds.tf2idx.items()}

    wrong_te = np.array([choose_wrong_tf(g, d, gene_counts.get(g, {}), wrong_seed) if inl else -1
                         for g, d, inl in zip(g_te, dep_idx[te], in_lg_te)], np.int64)
    pair_te = wrong_te >= 0                                     # 可配对：D∈L_g 且启动子上还有别的 TF
    n_wr = np.array([gene_counts.get(g, {}).get(int(w), 0) if w >= 0 else 0 for g, w in zip(g_te, wrong_te)], np.int64)
    say(f"\n[0] test：{len(te)} 行 = {len(uniq_g)} 基因 × {n_tf_te} 个被耗竭 TF；D∈L_g {int(in_lg_te.sum())} 行({in_lg_te.mean():.1%})，"
        f"其中可配对(启动子上还有别的 TF)的 {int(pair_te.sum())} 行({pair_te.sum() / max(in_lg_te.sum(), 1):.1%})；"
        f"可配对且显著 {int((pair_te & sig_te).sum())}(down {int((pair_te & (yc_te == 0)).sum())} / up {int((pair_te & (yc_te == 2)).sum())})")
    if pair_te.any():
        gap = np.abs(n_wr[pair_te] - nds_te[pair_te])
        say(f"    错配 TF D' 的位点数跟 D 的位点数：完全相同 {float((gap == 0).mean()):.1%}，差 ≤1 {float((gap <= 1).mean()):.1%}，"
            f"平均差 {float(gap.mean()):.2f}；被选为 D' 的不同 TF {len(set(wrong_te[pair_te].tolist()))} 个")
    wrong_by_idx = {int(i): int(w) for i, w in zip(te, wrong_te) if w >= 0}
    mS = pair_te                                                  # 主分析子集
    rowsS = np.flatnonzero(mS)
    boot_S = [br[mS[br]] for br in boot_rows]
    boot_sig = [br[mS[br] & (yc_te[br] != 1)] for br in boot_rows]
    wds = WrongKnockoutView(ds, wrong_by_idx)

    exports = {}
    p_ = os.path.join(results_root, run_with, "predictions_test.parquet")
    if os.path.exists(p_):
        E = pd.read_parquet(p_)
        pos = pd.Index((E["gene_id"].astype(str) + "|" + E["tf_depleted"].astype(str)).to_numpy()).get_indexer(key_te)
        if (pos < 0).any():
            say(f"    ⚠ {run_with} 的 17 号导出有 {int((pos < 0).sum())} 条 test 行对不上，(s1) 跳过")
        else:
            exports[run_with] = E.iloc[pos].reset_index(drop=True)
    else:
        say(f"    (没有 {p_}：(s1) 跳过)")
    checks = []

    # (s5) 抽样核对包装类
    ds.set_ablation("none", verbose=False)
    samp = rowsS[np.random.default_rng(1).permutation(len(rowsS))[:int(n_check_rows)]]
    bad5, n5 = 0, 0
    for r_ in samp:
        gi = int(te[r_])
        it = wds[gi]
        ti_wt = np.asarray(it["layout_wt"]["tf_idx"], np.int64)
        ti_d = np.asarray(it["layout_d"]["tf_idx"], np.int64)
        d_, w_ = int(dep_idx[gi]), int(wrong_te[r_])
        ok5 = (not (ti_d == w_).any()) and ((ti_d == d_).sum() == (ti_wt == d_).sum()) and \
              (len(ti_wt) - len(ti_d) == int((ti_wt == w_).sum()))
        n5 += 1
        bad5 += int(not ok5)
    checks.append(("s5", bad5 == 0, f"包装类抽样核对 {n5} 行：layout_d 没有 D'、保留 D、少掉的位点数=D' 的位点数；不一致 {bad5} 行"))
    say(f"    (s5) {checks[-1][2]} " + ("✓" if bad5 == 0 else "⚠(停：包装类没按设计工作，不要用下面的数)"))
    if bad5:
        raise SystemExit("(s5) 失败：WrongKnockoutView 没有只删 D'")

    # ------------------------------------------------------------------ [1] 推理(W × 共有 seed × 3 种推理)
    def seeds_of(run):
        out = {}
        for p__ in sorted(glob.glob(os.path.join(ckpt_root, run, "seed*_best.pt"))):
            mt = re.match(r"seed(\d+)_best\.pt$", os.path.basename(p__))
            if mt:
                out[int(mt.group(1))] = p__
        return out

    def load_ck(p__):
        try:
            return torch.load(p__, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(p__, map_location="cpu")

    def build(ck):
        cfg = ck.get("train_config") or {}
        if not ck.get("model_kwargs"):
            raise SystemExit("checkpoint 没有 model_kwargs(第2批之前的老格式)，29 号不支持")
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
                   huber_delta=cfg.get("huber_delta", 1.0), lambda_bd=0.0)
        return model, cfg, ekw

    def infer(model, ck, rows, ekw, dsx):
        hs = (ck.get("head_states") or {}) if variant == "perhead" else {}
        out = {}
        for h in ("B", "C"):
            st = hs.get(h) if hs else None
            model.load_state_dict(st if st is not None else ck["model_state"], strict=True)
            ev = tl.evaluate(model, dsx, list(rows), class_counts, batch_size=128, **ekw)
            if h == "B":
                out["yb"] = to_np(ev["y_b_pred"])
            else:
                pr = softmax_np(to_np(ev["logits_c"]))
                out["pdn"], out["pup"] = pr[:, 0], pr[:, 2]
        model.load_state_dict(ck["model_state"], strict=True)
        out["used_ph"] = bool(hs) and ("B" in hs) and ("C" in hs)
        return out

    def run_mode(model, ck, rows, ekw, mode):
        if mode == "del":
            ds.set_ablation("none", verbose=False)
            return infer(model, ck, rows, ekw, ds)
        if mode == "nodel":
            ds.set_ablation("no_knockout", verbose=False)
            return infer(model, ck, rows, ekw, ds)
        ds.set_ablation("none", verbose=False)           # wrong：底座 none，包装类把 layout_d 换成"去掉 D'"
        return infer(model, ck, rows, ekw, wds)

    sw = seeds_of(run_with)
    seeds = sorted(sw)
    say(f"\n[1] 推理：W={run_with} seed {seeds}；三种推理 del / nodel / wrong")
    if not seeds:
        raise SystemExit(f"{ckpt_root}/{run_with}/ 没有 seed*_best.pt")
    k = len(seeds)
    if smoke_first:
        ck = load_ck(sw[seeds[0]])
        model, _, ekw = build(ck)
        sm_rows = te[np.isin(g_te, uniq_g[:int(smoke_genes)])]
        t0 = time.time()
        for m_ in ("del", "nodel", "wrong"):
            o = run_mode(model, ck, sm_rows, ekw, m_)
            if not all(np.isfinite(o[x]).all() for x in ("yb", "pdn", "pup")):
                raise SystemExit(f"冒烟：推理模式 {m_} 的预测里有非有限值")
        dt = time.time() - t0
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        est = dt / max(len(sm_rows), 1) * len(te) * 3 * k / 60
        say(f"  冒烟通过：{len(sm_rows)} 行 × 3 种推理 × 2 个头 用 {dt:.1f} 秒；全量预计 {est:.1f} 分钟(不含载入，冒烟有预热开销，偏高)")
    preds = {c: {} for c in CELLS}
    for s in seeds:
        ck = load_ck(sw[s])
        model, cfg, ekw = build(ck)
        t0 = time.time()
        for m_ in ("del", "nodel", "wrong"):
            preds[f"W_{m_}"][s] = run_mode(model, ck, te, ekw, m_)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        o = preds["W_del"][s]
        E = exports.get(run_with)
        sfx = "_ph" if o["used_ph"] else ""
        rep = ""
        if E is not None and f"y_b_seed{s}{sfx}" in E.columns and f"p_down_seed{s}{sfx}" in E.columns:
            dmax = max(float(np.nanmax(np.abs(o["yb"] - E[f"y_b_seed{s}{sfx}"].to_numpy(np.float64)))),
                       float(np.nanmax(np.abs(o["pdn"] - E[f"p_down_seed{s}{sfx}"].to_numpy(np.float64)))),
                       float(np.nanmax(np.abs(o["pup"] - E[f"p_up_seed{s}{sfx}"].to_numpy(np.float64)))))
            checks.append(("s1", dmax <= 10 * repro_tol, f"{run_with} seed{s} W_del vs 17 号导出 最大差 {dmax:.2e}"))
            rep = f"；(s1) W_del 跟 17 号导出最大差 {dmax:.2e}"
            if dmax > 10 * repro_tol:
                raise SystemExit(f"(s1) 复现核对失败(seed{s} 最大差 {dmax:.3g})：模型重建或 Dataset 设置跟 17 号不一致")
        np_ = ~mS
        d2 = max(float(np.max(np.abs(preds["W_wrong"][s][x][np_] - preds["W_del"][s][x][np_])))
                 for x in ("yb", "pdn", "pup")) if np_.any() else 0.0
        checks.append(("s2", d2 <= repro_tol, f"{run_with} seed{s} 不可配对行 W_wrong vs W_del 最大差 {d2:.2e}"))
        say(f"  W seed{s}：三种推理共 {time.time() - t0:.0f} 秒(权重={'各头最优' if o['used_ph'] else '选中'}){rep}；"
            f"(s2) 不可配对行 W_wrong vs W_del 最大差 {d2:.2e}" + ("" if d2 <= repro_tol else " ⚠"))
    ens = {c: {x: np.mean([preds[c][s][x] for s in seeds], 0) for x in ("yb", "pdn", "pup")} for c in CELLS}

    # ------------------------------------------------------------------ [2] 指标与三个对比(只在可配对的 D∈L_g 行上)
    def mets(P, rows):
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
    point = {c: mets(ens[c], rowsS) for c in CELLS}
    boots = {c: [mets(ens[c], br) for br in boot_S] for c in CELLS}
    per_seed = {(c, s): mets(preds[c][s], rowsS) for c in CELLS for s in seeds}
    say(f"\n[2] 三格比较(只看可配对的 D∈L_g 行：{len(rowsS)} 行，显著 {int((mS & sig_te).sum())}；{k}-seed 集成；"
        f"bootstrap {n_boot} 次按基因整群，用时 {time.time() - t2:.0f} 秒)")
    say("  Δ = 变体 − 参照；†=bootstrap CI 不含0(只含 test 抽样，三格用同一批权重，没有训练随机性)；逐 seed = 变体比参照好的 seed 个数")
    say("  " + " " * 44 + "".join(f"{SHORT[m]:>10}" for m in SHOW))
    for c in CELLS:
        say(f"  {c:8s}{CELL_DESC[c]:<34}" + "".join(f"{point[c][m]:>10.3f}" for m in SHOW))
    cell_rows, con_rows, crow = [], [], {}
    for c in CELLS:
        for m in METS:
            cell_rows.append(dict(cell=c, desc=CELL_DESC[c], metric=m, ensemble=point[c][m],
                                  **{f"seed{s}": per_seed[(c, s)][m] for s in seeds}))
    pd.DataFrame(cell_rows).to_csv(os.path.join(outdir, "cells_spec.csv"), index=False)
    for cname, var, ref in CONTRASTS:
        cells = []
        for m in SHOW:
            d0 = point[var][m] - point[ref][m]
            lo, hi = ci_of([a[m] - b[m] for a, b in zip(boots[var], boots[ref])])
            ps = [per_seed[(var, s)][m] - per_seed[(ref, s)][m] for s in seeds]
            better = int(sum(x > 0 for x in ps))
            sig_ = bool(np.isfinite(lo) and (lo > 0 or hi < 0))
            crow[(cname, m)] = dict(delta=d0, ci_lo=lo, ci_hi=hi, test_sig=sig_, seeds_better=better, k=k,
                                    seed_mean=float(np.mean(ps)), seed_sd=float(np.std(ps, ddof=1)) if k > 1 else 0.0)
            con_rows.append(dict(contrast=cname, variant=var, reference=ref, metric=m, **crow[(cname, m)]))
            cells.append(f"{SHORT[m]} {f3(d0, True)}[{f3(lo, True)},{f3(hi, True)}]{'†' if sig_ else ''}({better}/{k})")
        say(f"  {cname:8s}{var} − {ref}：" + "  ".join(cells))
    pd.DataFrame(con_rows).to_csv(os.path.join(outdir, "contrasts_spec.csv"), index=False)

    # ------------------------------------------------------------------ [3] 逐 TF：错配删除的方向是不是跟被删 TF 自己的角色一致
    say("\n[3] 逐 TF 方向(只看可配对的 D∈L_g 行；e_C=该推理让 P(dn)−P(up) 增加多少，相对不删；>0=推向 down；"
        "role_T=(n_down−n_up)/(n_down+n_up)，T 自己作为被耗竭 TF、在 train 的 D∈L_g 显著行上)")

    def eff_c(tag):
        Pc, Pn = ens[tag], ens["W_nodel"]
        return (Pc["pdn"] - Pc["pup"]) - (Pn["pdn"] - Pn["pup"])

    eC_del, eC_wrong = eff_c("W_del"), eff_c("W_wrong")
    m_ = in_lg_all[tr] & (yc_all[tr] != 1)
    q_ = pd.DataFrame(dict(tf=tfs_all[tr][m_], dn=(yc_all[tr][m_] == 0).astype(int), up=(yc_all[tr][m_] == 2).astype(int))).groupby("tf").sum()
    q_["n_sig"] = q_["dn"] + q_["up"]
    q_["role"] = (q_["dn"] - q_["up"]) / q_["n_sig"].clip(lower=1)
    role_tr = q_
    dfw = pd.DataFrame(dict(tf_removed=[idx2tf.get(int(w), "?") for w in wrong_te[rowsS]], eC=eC_wrong[rowsS]))
    pw = dfw.groupby("tf_removed").agg(n_rows=("eC", "size"), eC_wrong=("eC", "mean"))
    dft = pd.DataFrame(dict(tf_removed=t_te[rowsS], eC=eC_del[rowsS]))
    pt_ = dft.groupby("tf_removed").agg(n_rows_true=("eC", "size"), eC_true=("eC", "mean"))
    per_tf = pw.join(pt_, how="outer").join(role_tr[["role", "n_sig"]].rename(columns={"role": "role_train", "n_sig": "n_sig_train"}))
    per_tf.reset_index().rename(columns={"index": "tf"}).to_csv(os.path.join(outdir, "per_tf_wrong.csv"), index=False)
    trng = np.random.default_rng(1)
    rho = {}
    for tag, col, nrow in (("错配删除(被删的是 D')", "eC_wrong", "n_rows"), ("真删(被删的是 D)", "eC_true", "n_rows_true")):
        q = per_tf[(per_tf[nrow].fillna(0) >= min_rows_tf) & (per_tf["n_sig_train"].fillna(0) >= min_sig_tf)]
        n_q = len(q)
        bix = [trng.integers(0, n_q, n_q) for _ in range(int(n_boot))] if n_q >= 5 else []
        xv, yv = q[col].to_numpy(np.float64), q["role_train"].to_numpy(np.float64)
        r0 = spearman(xv, yv)
        lo, hi = ci_of([spearman(xv[ix], yv[ix]) for ix in bix])
        rho[col] = (r0, lo, hi, n_q)
        say(f"  {tag}：Spearman(ε_C, role_被删TF) {f3(r0, True)}[{f3(lo, True)},{f3(hi, True)}]({n_q} 个 TF：作为被删 TF 的行数≥{min_rows_tf}、"
            f"train 显著≥{min_sig_tf})")
    # 行级：e_C 的 TF 内合并 AUROC(down vs up，标签是被耗竭 TF D 的)
    sig_S = rowsS[yc_te[rowsS] != 1]
    pos_dn = yc_te[sig_S] == 0
    score_full = (ens["W_del"]["pdn"] - ens["W_del"]["pup"])
    sc = {"e_C 真删": eC_del, "e_C 错配删": eC_wrong, "完整输出 P(dn)−P(up)": score_full}
    wt0, wtb = {}, {}
    for nm, v in sc.items():
        wt0[nm] = pooled_wtf(v[sig_S], pos_dn, tcode_te[sig_S])
        wtb[nm] = [pooled_wtf(v[br], yc_te[br] == 0, tcode_te[br]) for br in boot_sig]
    dpair = [a - b for a, b in zip(wtb["e_C 真删"], wtb["e_C 错配删"])]
    dlo, dhi = ci_of(dpair)
    say(f"  行级(可配对且显著 {len(sig_S)} 行，down {int(pos_dn.sum())} / up {int((~pos_dn).sum())})，区分 down/up 的 TF 内合并 AUROC [按基因整群 bootstrap]：")
    for nm in sc:
        lo, hi = ci_of(wtb[nm])
        say(f"    {nm:<22}{f3(wt0[nm])}[{f3(lo)},{f3(hi)}]")
    d_w = wt0["e_C 真删"] - wt0["e_C 错配删"]
    say(f"    真删 − 错配删 = {f3(d_w, True)}[{f3(dlo, True)},{f3(dhi, True)}]")
    pd.DataFrame(dict(gene_id=g_te[rowsS], tf_depleted=t_te[rowsS], tf_removed=[idx2tf.get(int(w), "?") for w in wrong_te[rowsS]],
                      n_D_sites=nds_te[rowsS], n_removed_wrong=n_wr[rowsS], y_c=yc_te[rowsS], y_b=yb_te[rowsS],
                      eC_true=eC_del[rowsS], eC_wrong=eC_wrong[rowsS])).to_csv(
        os.path.join(outdir, "rows_wrong_DinLg.csv.gz"), index=False, compression="gzip")

    # ------------------------------------------------------------------ [4] 规则核对
    say("\n[4] 规则核对(预先写在文件头；\"明显\"=|Δ|≥0.02 且 CI 不含0)")

    def clear(cn, m, sign, thr=0.02):
        r_ = crow[(cn, m)]
        return bool(np.isfinite(r_["delta"]) and r_["delta"] * sign >= thr and r_["test_sig"])

    def fmt(cn, m):
        r_ = crow[(cn, m)]
        return f"{SHORT[m]} {f3(r_['delta'], True)}[{f3(r_['ci_lo'], True)},{f3(r_['ci_hi'], True)}]({r_['seeds_better']}/{k} 变好)"

    rules = []
    rules.append(("(q1) Δ_spec 的 C_APdn 和 C_APup 都明显 <0(删错比删对差 -> 删位点效应是 D 特异的)",
                  clear("Δ_spec", "C_auprc_down", -1) and clear("Δ_spec", "C_auprc_up", -1),
                  fmt("Δ_spec", "C_auprc_down") + "  " + fmt("Δ_spec", "C_auprc_up")))
    rules.append(("(q2) Δ_wn 的 C_APdn、C_APup 都不明显 >0(删错相对不删没有通用收益)",
                  not (clear("Δ_wn", "C_auprc_down", 1) or clear("Δ_wn", "C_auprc_up", 1)),
                  fmt("Δ_wn", "C_auprc_down") + "  " + fmt("Δ_wn", "C_auprc_up")))
    r3 = rho.get("eC_wrong", (np.nan,) * 4)
    rules.append(("(q3) 错配删除的 ε_C 跟被删 TF 自己的 role 的 Spearman >0 且 TF bootstrap CI 不含0",
                  bool(np.isfinite(r3[1]) and r3[0] > 0 and r3[1] > 0),
                  f"错配 {f3(r3[0], True)}[{f3(r3[1], True)},{f3(r3[2], True)}]({r3[3]} 个 TF)；真删 "
                  f"{f3(rho.get('eC_true', (np.nan,))[0], True)}"))
    rules.append(("(q4) e_C 的 TF 内合并 AUROC：真删 − 错配删 ≥0.05 且配对 bootstrap CI 不含0",
                  bool(np.isfinite(dlo) and d_w >= 0.05 and dlo > 0),
                  f"真删 {f3(wt0['e_C 真删'])}  错配 {f3(wt0['e_C 错配删'])}  差 {f3(d_w, True)}[{f3(dlo, True)},{f3(dhi, True)}]"))
    for nm, ok_, txt in rules:
        say(f"  {'成立' if ok_ else '不成立'}  {nm}：{txt}")
    chk_rows = []
    for tag in ("s1", "s2", "s5"):
        q = [x for x in checks if x[0] == tag]
        if q:
            bad = [x for x in q if not x[1]]
            txt = (q[0][2] if not bad else "；".join(x[2] for x in bad)) + (f"(共 {len(q)} 项)" if len(q) > 1 else "")
            chk_rows.append(dict(rule=f"自检 ({tag})", holds=not bad, numbers=txt))
            say(f"  自检 ({tag}) {'✓' if not bad else '⚠'} {txt}")
    pd.DataFrame([dict(rule=nm, holds=ok_, numbers=txt) for nm, ok_, txt in rules] + chk_rows).to_csv(
        os.path.join(outdir, "rule_checks_spec.csv"), index=False)

    # Table 7(LaTeX) 与 paper_numbers_spec.md
    cols7 = ("C_auprc_down", "C_auprc_up", "C_auroc_down", "C_auroc_up", "B_r_within_tf")
    lab7 = {"Δ_spec": r"$\Delta$ wrong-TF deletion $-$ correct deletion", "Δ_wn": r"$\Delta$ wrong-TF deletion $-$ no deletion",
            "Δ_infer": r"$\Delta$ no deletion $-$ correct deletion"}
    trows = [r"Correct deletion (reference) & " + " & ".join(f3(point["W_del"][x]) for x in cols7)]
    for cname, _, _ in CONTRASTS:
        cells = []
        for x in cols7:
            r_ = crow[(cname, x)]
            cells.append(f"${r_['delta']:+.3f}" + (r"^{\dagger}$" if r_["test_sig"] else "$"))
        trows.append(lab7[cname] + " & " + " & ".join(cells))
    tex_table(rf"Specificity control for explicit site deletion at inference ({k}-seed ensemble, test set, {len(rowsS)} pairs where the depleted TF has "
              r"sites in the promoter and at least one other TF is present). Wrong-TF deletion removes all sites of another TF present in the promoter "
              r"(chosen to match the depleted TF's site count) while keeping the depleted TF's own sites; the input format is identical to training.",
              "tab:spec", ["Comparison"] + [TEXM[x] for x in cols7], trows, "l" + "c" * len(cols7),
              notes=r"$\dagger$: 95\% gene-cluster bootstrap CI excludes 0. All three rows use the same trained weights, so no training variability is involved.")
    md.append(f"\n## Table 7 删位点特异性对照(29 号，{time.strftime('%Y-%m-%d %H:%M')}；{k}-seed 集成，{variant}，可配对 D∈L_g {len(rowsS)} 行)— {outdir}/contrasts_spec.csv\n")
    for cname, var, ref in CONTRASTS:
        md.append(f"- {cname}({var} − {ref})：" + "；".join(fmt(cname, x) for x in SHOW))
    md.append(f"- 逐 TF Spearman(ε_C, role_被删TF)：错配 {f3(r3[0], True)}[{f3(r3[1], True)},{f3(r3[2], True)}]({r3[3]} 个 TF)；"
              f"真删 {f3(rho.get('eC_true', (np.nan,))[0], True)}")
    md.append(f"- e_C TF 内合并 AUROC：真删 {f3(wt0['e_C 真删'])}；错配 {f3(wt0['e_C 错配删'])}；完整输出 {f3(wt0['完整输出 P(dn)−P(up)'])}；"
              f"真删−错配 {f3(d_w, True)}[{f3(dlo, True)},{f3(dhi, True)}]")
    for nm, ok_, txt in rules:
        md.append(f"- {'成立' if ok_ else '不成立'}：{nm} —— {txt}")
    with open(os.path.join(outdir, "paper_numbers_spec.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(outdir, "tables_spec.tex"), "w", encoding="utf-8") as fh:
        fh.write("% 29_deletion_specificity.py 自动生成；需要 \\usepackage{booktabs,graphicx,amssymb}\n\n" + "\n".join(tex))
    say(f"\n写出 {outdir}/：summary.txt、cells_spec.csv、contrasts_spec.csv、per_tf_wrong.csv、rows_wrong_DinLg.csv.gz、rule_checks_spec.csv、"
        f"tables_spec.tex、paper_numbers_spec.md  (用时 {(time.time() - t_all) / 60:.1f} 分钟)")
    say("下一步：./run_all.sh --tables --skip-arch-selftest(27 号把 tables_spec.tex / paper_numbers_spec.md / [29号] 规则并进 _paper/)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(point=point, crow=crow, rules=rules)


if __name__ == "__main__":
    run_deletion_specificity(**CONFIG)