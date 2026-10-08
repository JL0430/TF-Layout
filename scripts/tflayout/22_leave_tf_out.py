# scripts/tflayout/22_leave_tf_out.py

import gc
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

LTO_VERSION = "2026-09-27a"

CONFIG = dict(
    paths=dict(layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
               head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
               promoter_tokens="out/promoter_token_ids.parquet",
               bpe_tokenizer="out/bpe_tokenizer.json"),
    experiment="v3_ce_marker_dense",              # 18 号 CONFIG["experiments"] 里的预设名(必须 ctx_mode=marker)
    main_results_dir="out/results/v3_ce_marker_dense",  # 同一配置在标准划分上的 17 号导出(对照)
    dense_target="out/head_b_dense_target.parquet",  # 预设里 dense_target="@dense" 时用它
    n_folds=3,
    seed=42,          # 每折训练用的 seed，也是对照里取哪个 seed 的标准模型
    fold_seed=0,      # 分折时并列样本量的随机打散(跟 v1 相同 -> 同样的分折)
    min_sig=20,       # 合格 TF：val+test 染色体上至少这么多个显著样本
    n_boot=500,
    outdir="out/results/_lto2",          # v2 结果目录(v1 在 out/results/_lto/，不动)
    ckpt_dir="out/checkpoints/_lto2",    # 每折权重(选中+各头最优，约 150MB/折)
    strategies=("raw", "zero", "mean", "knn_bind", "knn_emb"),  # 没见过的 TF 怎么表示，见文件头 v2 第2条
    knn_k=10,
    seen_control=True,   # 同一折在见过的 TF 上评估一次(健康检查，见文件头 v2 第4条)
    check_rows=2000,     # 不变性自检用多少条见过的 TF 样本
    v1_dir="out/results/_lto",           # 有 v1 结果时打印 raw 对 v1 的对照
    smoke_first=True, smoke_steps=200, smoke_eval_rows=4000,  # 冒烟：训练 200 步 + 在 4000 条被留出样本上走一遍全部策略
    dry_run=False,    # True=只打印分折就退出
    force=False,      # True=忽略已有权重/预测，全部重训
)


def run_lto(paths, experiment, main_results_dir, dense_target, n_folds, seed, fold_seed, min_sig,
            n_boot, outdir, ckpt_dir="out/checkpoints/_lto2",
            strategies=("raw", "zero", "mean", "knn_bind", "knn_emb"), knn_k=10, seen_control=True,
            check_rows=2000, v1_dir="out/results/_lto", smoke_first=True, smoke_steps=200,
            smoke_eval_rows=4000, dry_run=False, force=False):
    """唯一入口，协议见文件头。"""
    t_all = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    lines = []
    strategies = tuple(strategies)
    known = ("raw", "zero", "mean", "knn_bind", "knn_emb")
    bad_s = [s_ for s_ in strategies if s_ not in known]
    if bad_s or "raw" not in strategies:
        raise SystemExit(f"strategies 只能从 {known} 里选、且必须含 raw，收到 {strategies}")

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def load(fn):  # 跟 18 号同一种按路径动态加载(文件名以数字开头，不能 import)
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
        n1 = int(pos.sum())
        n0 = len(pos) - n1
        if n1 < 3 or n0 < 3:
            return float("nan"), float("nan")
        sc = np.asarray(score, np.float64)
        _, inv_, cnt_ = np.unique(sc, return_inverse=True, return_counts=True)
        avg_rank = np.cumsum(cnt_) - (cnt_ - 1) / 2.0
        r = avg_rank[inv_.reshape(-1)]
        auc = float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
        p = pos[np.argsort(-sc, kind="mergesort")]
        prec = np.cumsum(p) / np.arange(1, len(p) + 1)
        return auc, float(prec[p].sum() / n1)

    def softmax_np(lg):
        lg = np.asarray(lg, np.float64)
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(1, keepdims=True)

    tl = load("16_train_loop.py")
    m18 = load("18_run_experiments.py")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say("=" * 78)
    say(f"leave-TF-out v2({LTO_VERSION})  {time.strftime('%Y-%m-%d %H:%M:%S')}  实验配置={experiment}  16号代码版本 "
        f"{tl.CODE_VERSION}  设备={device}  策略={list(strategies)}")
    say("=" * 78)

    # ---- 1. 解析预设(跟 18 号 resolve 同一规则) ----
    ex, base = m18.CONFIG["experiments"], dict(m18.CONFIG["base"])

    def resolve(name, depth=0):
        if name not in ex:
            raise SystemExit(f"18 号 CONFIG['experiments'] 里没有 {name}")
        if depth > 5:
            raise SystemExit(f"{name} 的 _base 嵌套太深")
        e = dict(ex[name])
        parent = e.pop("_base", None)
        return dict(resolve(parent, depth + 1), **e) if parent else e

    c = dict(base, **resolve(experiment))
    if c.get("dense_target") == "@dense":
        c["dense_target"] = dense_target
    if c.get("ctx_mode") != "marker":
        raise SystemExit(f"leave-TF-out 必须用 ctx_mode=marker，{experiment} 是 {c.get('ctx_mode')}")
    if c.get("lambda_bd") and not (c.get("dense_target") and os.path.exists(c["dense_target"])):
        raise SystemExit(f"{experiment} 用稠密目标(λ_Bd={c['lambda_bd']})，但找不到 {c.get('dense_target')}")
    if c.get("num_workers") is None:
        n_cpu = os.cpu_count() or 4
        c["num_workers"] = 8 if n_cpu > 8 else max(1, n_cpu - 1)
    sig_params = inspect.signature(tl.run_one_seed).parameters
    kw = {k: v for k, v in c.items()
          if k in sig_params and k not in ("ds", "split_idx", "seed", "class_counts")}
    kw["device"] = device
    say(f"训练配置(相对 base 的改动): "
        f"{ {k: v for k, v in c.items() if base.get(k, '<无>') != v} }")

    # ---- 2. Dataset、切分、合格 TF、分折(跟 v1 同一算法) ----
    ds = tl.TFLayoutDataset(paths["layout"], paths["labels"], paths["head_a"], paths["sgd"],
                            paths["promoter_tokens"], paths["bpe_tokenizer"])
    split_idx = tl.build_split_indices(ds, paths["head_a"])
    kw["n_tf"], kw["pad_token_id"] = ds.n_tf, ds.pad_id
    digest_src = {k: kw[k] for k in sorted(kw) if k not in ("device", "num_workers", "eval_batch_size")}
    cfg_digest = hashlib.sha1(json.dumps(digest_src, sort_keys=True, default=str).encode()).hexdigest()[:12]
    smp = ds.samples
    gene_arr, tf_arr = smp["gene_id"].to_numpy(), smp["tf_depleted"].to_numpy()
    yb_all = smp["log2fc"].to_numpy(np.float64)
    yc_all = smp["direction_3class"].map(ds.class2idx).to_numpy().astype(np.int64)
    split_of = np.full(len(smp), "", dtype=object)
    for sp in ("train", "val", "test"):
        split_of[np.asarray(split_idx[sp], dtype=np.int64)] = sp
    vt = np.asarray(list(split_idx["val"]) + list(split_idx["test"]), dtype=np.int64)
    n_sig = pd.Series(np.isfinite(yb_all[vt])).groupby(tf_arr[vt]).sum()
    elig = n_sig[n_sig >= min_sig]
    say(f"\n{len(n_sig)} 个被耗竭 TF 里，val+test 染色体上显著样本 ≥{min_sig} 的合格 TF 有 {len(elig)} 个"
        f"(共 {int(elig.sum())} 个显著样本)")
    if len(elig) < n_folds:
        raise SystemExit("合格 TF 少于折数，调小 n_folds 或 min_sig")
    rng = np.random.default_rng(fold_seed)
    names = [str(t) for t in elig.index]
    names = [names[i] for i in rng.permutation(len(names))]
    names.sort(key=lambda t: -int(elig[t]))  # 稳定排序：并列保持随机顺序
    folds = [[] for _ in range(n_folds)]
    for i, t in enumerate(names):
        r_ = i % (2 * n_folds)
        folds[r_ if r_ < n_folds else 2 * n_folds - 1 - r_].append(t)
    for f, held in enumerate(folds):
        say(f"  折{f}: {len(held)} 个 TF，显著样本 {int(sum(elig[t] for t in held))}  {sorted(held)}")
    v1_folds = os.path.join(v1_dir, "folds.json") if v1_dir else ""
    if v1_folds and os.path.exists(v1_folds):
        with open(v1_folds, encoding="utf-8") as fh:
            same = json.load(fh).get("folds") == [sorted(h) for h in folds]
        say(f"  跟 v1({v1_folds})的分折{'完全相同' if same else '不同(数据或 CONFIG 变过？raw 跟 v1 不可直接对照)'}")
    with open(os.path.join(outdir, "folds.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(experiment=experiment, seed=seed, fold_seed=fold_seed, min_sig=min_sig,
                       folds=[sorted(h) for h in folds], lto_version=LTO_VERSION), fh, ensure_ascii=False, indent=1)
    if dry_run:
        say("\ndry_run=True：只看分折，不训练")
        return dict(folds=folds)

    exact = smp["direction_3class"].value_counts()
    class_counts = torch.tensor([float(exact.get(k, 0.0)) for k in ("down", "ns", "up")])
    lg_sets = {g: set(np.asarray(v["tf_idx"]).tolist()) for g, v in ds.layout_by_gene.items()}

    def in_lg_of(idx):
        return np.array([ds.tf2idx.get(tf_arr[i], -1) in lg_sets.get(gene_arr[i], ()) for i in idx], bool)

    # 结合谱(knn_bind 用)：TF×基因 的位点数，log1p 后按行归一化——ChEC-seq 输入数据，178 个 TF 都有
    gene_list = sorted(ds.layout_by_gene, key=str)
    bind = np.zeros((ds.n_tf, len(gene_list)), np.float64)
    for j, gid in enumerate(gene_list):
        ti = np.asarray(ds.layout_by_gene[gid]["tf_idx"], np.int64)
        if ti.size:
            np.add.at(bind[:, j], ti, 1.0)
    bind = np.log1p(bind)
    bind_unit = bind / np.maximum(np.linalg.norm(bind, axis=1, keepdims=True), 1e-12)
    tf_names = list(ds.tf_list)

    def prep_ds():  # 复用权重、不经过 run_one_seed 时，Dataset 的条件编码/消融/稠密目标要手动设成训练时的样子
        ds.set_ctx_mode(c.get("ctx_mode", "marker"), c.get("ctx_clip", 3.0), verbose=False)
        if hasattr(ds, "set_ablation"):
            ds.set_ablation(c.get("ablation", "none"), verbose=False)
        if hasattr(ds, "set_dense_target"):
            ds.set_dense_target(c.get("dense_target"), verbose=False)

    def knn_weights(unit, k_tf, trained):
        tr = np.asarray(trained, np.int64)
        sims = unit[tr] @ unit[k_tf]
        order = np.argsort(-sims, kind="stable")[:int(knn_k)]
        nb, s_ = tr[order], np.clip(sims[order], 0.0, None)
        w = s_ / s_.sum() if s_.sum() > 0 else np.full(len(nb), 1.0 / len(nb))
        return nb, w, sims[order]

    def strategy_maps(held_tf, trained_tf, e_tf_unit):
        """每个策略 -> (列映射, 行映射)：列映射 {k: None(置0) 或 (邻居下标, 权重)}；行映射 {k: (邻居下标, 权重)}。"""
        tr = np.asarray(trained_tf, np.int64)
        mean_w = (tr, np.full(len(tr), 1.0 / len(tr)))
        maps, nbr_rows = {}, []
        for s_ in strategies:
            if s_ == "raw":
                continue
            col, row = {}, {}
            for k in held_tf:
                if s_ in ("zero", "mean"):
                    nw = mean_w
                else:
                    unit = bind_unit if s_ == "knn_bind" else e_tf_unit
                    nb, w, sims = knn_weights(unit, k, tr)
                    nw = (nb, w)
                    nbr_rows.append(dict(strategy=s_, tf=tf_names[k], **{
                        f"nb{j + 1}": f"{tf_names[int(nb[j])]}({w[j]:.2f},cos={sims[j]:.2f})"
                        for j in range(min(3, len(nb)))}))
                row[k] = nw
                col[k] = None if s_ == "zero" else nw
            maps[s_] = (col, row)
        return maps, nbr_rows

    def surgery(sd, col_map, row_map):
        """权重手术(见文件头 v2 第2条)：换掉被留出 TF 的条件 MLP 第一层那一列，差值挪进偏置 -> WT 前向逐位不变；
        tf_embed_corr / tf_bias_c 的对应行一并替换。输入输出都是 state_dict(不改原字典里的张量)。"""
        out = dict(sd)
        kW = [k for k in sd if k.endswith("fusion.condition.net.0.weight")]
        kB = [k for k in sd if k.endswith("fusion.condition.net.0.bias")]
        if len(kW) != 1 or len(kB) != 1:
            raise RuntimeError(f"state_dict 里找不到唯一的条件 MLP 第一层(weight {kW}, bias {kB})")
        kW, kB = kW[0], kB[0]
        W0 = sd[kW].detach().to(torch.float64)
        W, b = W0.clone(), sd[kB].detach().to(torch.float64).clone()
        for k, nw in col_map.items():
            if nw is None:
                new = torch.zeros_like(W0[:, k])
            else:
                ii = torch.as_tensor(np.asarray(nw[0], np.int64), dtype=torch.long, device=W0.device)
                ww = torch.as_tensor(np.asarray(nw[1], np.float64), dtype=torch.float64, device=W0.device)
                new = (W0[:, ii] * ww).sum(1)
            b += W0[:, k] - new
            W[:, k] = new
        out[kW], out[kB] = W.to(sd[kW].dtype), b.to(sd[kB].dtype)
        for suf in ("tf_embed_corr.weight", "tf_bias_c.weight"):
            ks = [k for k in sd if k.endswith(suf)]
            if not ks:
                continue
            E0 = sd[ks[0]].detach().to(torch.float64)
            E = E0.clone()
            for k, nw in row_map.items():
                ii = torch.as_tensor(np.asarray(nw[0], np.int64), dtype=torch.long, device=E0.device)
                ww = torch.as_tensor(np.asarray(nw[1], np.float64), dtype=torch.float64, device=E0.device)
                E[k] = (E0[ii] * ww.unsqueeze(1)).sum(0)
            out[ks[0]] = E.to(sd[ks[0]].dtype)
        return out

    def make_loader(rows, kw_f):
        try:
            return tl._make_loader(ds, list(rows), "eval", kw_f.get("forward_mode", "grouped"),
                                   int(kw_f.get("batch_size", 192)), 1, int(kw_f.get("eval_batch_size", 512)), 0,
                                   int(kw_f.get("num_workers", 0)), device, True, int(kw_f.get("layout_buckets", 1)))
        except Exception as e:  # noqa: BLE001 —— 只影响速度
            say(f"  提示：复用 DataLoader 失败({type(e).__name__}: {e})，每次评估各建一次")
            return None

    def eval_strategies(model, head_states, rows, maps, kw_f, with_ph=True):
        """在 rows 上把每个策略都评估一遍，返回 {策略: {"sel": (y_b, prob(n,3), y_a), "ph": (y_b, prob) 或 None}}。
        跑完 model 恢复成选中的原始权重。"""
        eval_kw = tl._eval_kwargs_from(kw_f)
        ld = make_loader(rows, kw_f)
        if ld is not None:
            eval_kw = dict(eval_kw, loader=ld)
        sel_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        res = {}
        try:
            for s_ in strategies:
                if s_ == "raw":
                    sd_s, hs_s = sel_sd, head_states
                else:
                    col, row = maps[s_]
                    sd_s = surgery(sel_sd, col, row)
                    hs_s = {h: surgery(v, col, row) for h, v in head_states.items()} if head_states else None
                model.load_state_dict(sd_s)
                ev = tl.evaluate(model, ds, list(rows), class_counts, **eval_kw)
                model.load_state_dict(sel_sd)
                sel = (np.asarray(ev["y_b_pred"], np.float64), softmax_np(ev["logits_c"]),
                       np.asarray(ev["y_a_pred"], np.float64))
                ph = None
                if with_ph and hs_s:
                    ev_ph = tl.evaluate_per_head(model, hs_s, ds, list(rows), class_counts, **eval_kw)
                    ph = (np.asarray(ev_ph["y_b_pred"], np.float64), softmax_np(ev_ph["logits_c"]))
                    model.load_state_dict(sel_sd)
                res[s_] = dict(sel=sel, ph=ph)
        finally:
            model.load_state_dict(sel_sd)
            del ld
            gc.collect()
        return res

    def fold_split(held):
        train_keep, _ = tl.leave_tf_out_split(ds, held, base_idx=split_idx["train"])
        val_keep, val_held = tl.leave_tf_out_split(ds, held, base_idx=split_idx["val"])
        test_keep, test_held = tl.leave_tf_out_split(ds, held, base_idx=split_idx["test"])
        return dict(train=train_keep, val=val_keep, test=list(split_idx["test"])), val_held, test_held, test_keep

    def ckpt_path(f):
        return os.path.join(ckpt_dir, f"fold{f}_seed{seed}.pt")

    def ckpt_ok(f, held):
        p = ckpt_path(f)
        if force or not os.path.exists(p):
            return None
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:  # 很老的 torch 没有 weights_only 参数
            ck = torch.load(p, map_location="cpu")
        ok = (ck.get("held") == sorted(held) and ck.get("seed") == seed and ck.get("experiment") == experiment
              and ck.get("code_version") == tl.CODE_VERSION and ck.get("cfg_digest") == cfg_digest)
        if not ok:
            say(f"  折{f}: {p} 的配置跟本次不一致，重训(旧文件会被覆盖)")
        return ck if ok else None

    def train_fold(f, held, kw_f, smoke=False):
        mod_split, val_held, test_held, test_keep = fold_split(held)
        held_idx = list(val_held) + list(test_held)
        assert not (set(held) & set(tf_arr[mod_split["train"]])) and \
            not (set(held) & set(tf_arr[mod_split["val"]])), "被留出的 TF 不应出现在训练/选模型的 val 里"
        held_tf = [ds.tf2idx[t] for t in held]
        trained_tf = sorted({ds.tf2idx[t] for t in set(tf_arr[mod_split["train"]]) if t in ds.tf2idx})
        say(f"\n===== 折{f}{'(冒烟)' if smoke else ''}: 训练 {len(mod_split['train'])} 条(已去掉 {len(held)} 个 TF，"
            f"训练过的 TF {len(trained_tf)} 个)、选模型 val {len(mod_split['val'])} 条；评估集 {len(held_idx)} 条"
            f"(val {len(val_held)} + test {len(test_held)}) =====")
        t_f = time.time()
        ck = None if smoke else ckpt_ok(f, held)
        if ck is not None:
            model = tl.SiameseHeadsModel(**ck["model_kwargs"]).to(device)
            model.load_state_dict(ck["model_state"])
            head_states, best_meta, source = ck.get("head_states"), dict(ck.get("best") or {}), "复用权重"
            prep_ds()
            say(f"  折{f}: 复用 {ckpt_path(f)}(选中@epoch{best_meta.get('epoch_progress')})，只重新评估")
        else:
            model, history, best = tl.run_one_seed(ds, mod_split, seed, class_counts=class_counts, **kw_f)
            head_states = best.get("head_states")
            best_meta = {k: v for k, v in best.items() if k != "head_states"}
            source = "训练"
            if not smoke:
                os.makedirs(ckpt_dir, exist_ok=True)
                tmp = ckpt_path(f) + ".tmp"
                torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                head_states=head_states, model_kwargs=best["model_kwargs"], best=best_meta,
                                history=history, held=sorted(held), held_idx=held_tf, trained_idx=trained_tf,
                                tf_list=tf_names, seed=seed, experiment=experiment, code_version=tl.CODE_VERSION,
                                cfg_digest=cfg_digest, lto_version=LTO_VERSION), tmp)
                os.replace(tmp, ckpt_path(f))
                say(f"  折{f}: 权重已存 {ckpt_path(f)}")
            del history
        model.eval()
        ekey = [k for k in model.state_dict() if k.endswith("layout.token_embed.e_tf.weight")]
        if ekey:
            e_tf = model.state_dict()[ekey[0]].detach().to(torch.float64).cpu().numpy()
            e_tf_unit = e_tf / np.maximum(np.linalg.norm(e_tf, axis=1, keepdims=True), 1e-12)
        else:
            say("  提示：state_dict 里没有 layout.token_embed.e_tf.weight，knn_emb 退回结合谱相似度")
            e_tf_unit = bind_unit
        maps, nbr_rows = strategy_maps(held_tf, trained_tf, e_tf_unit)
        eval_rows = held_idx
        if smoke and len(held_idx) > smoke_eval_rows:
            eval_rows = sorted(np.random.default_rng(0).choice(held_idx, int(smoke_eval_rows), replace=False).tolist())
        res = eval_strategies(model, head_states, eval_rows, maps, kw_f)
        # ---- 不变性自检：见过的 TF 样本上手术前后应逐位(浮点舍入内)相同；被留出样本上 Head A 应相同 ----
        seen_rows = list(mod_split["val"]) + list(test_keep)
        chk = sorted(np.random.default_rng(1).choice(seen_rows, min(int(check_rows), len(seen_rows)),
                                                     replace=False).tolist())
        res_chk = eval_strategies(model, None, chk, maps, kw_f, with_ph=False)
        inv = {}
        for s_ in strategies:
            if s_ == "raw":
                continue
            inv[s_] = dict(
                seen_yb=float(np.max(np.abs(res_chk[s_]["sel"][0] - res_chk["raw"]["sel"][0]))),
                seen_p=float(np.max(np.abs(res_chk[s_]["sel"][1] - res_chk["raw"]["sel"][1]))),
                held_ya=float(np.max(np.abs(res[s_]["sel"][2] - res["raw"]["sel"][2]))))
        worst = max((max(v.values()) for v in inv.values()), default=0.0)
        say(f"  不变性自检(手术前后最大差，应≈0)：" + "；".join(
            f"{s_}: 见过TF y_b {v['seen_yb']:.1e} / 概率 {v['seen_p']:.1e}，被留出样本 Head A {v['held_ya']:.1e}"
            for s_, v in inv.items()) + ("  ✓" if worst <= 1e-3 else "  ⚠ 超过1e-3，手术没有保持 WT 不变，结果先别用"))
        if smoke and worst > 0.05:
            raise SystemExit("冒烟：权重手术破坏了 WT/见过的 TF 的预测(>0.05)，把上面的日志贴回来")
        df = pd.DataFrame(dict(
            fold=f, gene_id=gene_arr[eval_rows], tf_depleted=tf_arr[eval_rows], split=split_of[eval_rows],
            y_b_true=yb_all[eval_rows], y_c_true=yc_all[eval_rows], in_lg=in_lg_of(eval_rows)))
        for s_ in strategies:
            for tag in ("sel", "ph"):
                v = res[s_][tag]
                if v is None:
                    continue
                df[f"yb__{s_}__{tag}"], df[f"pdn__{s_}__{tag}"], df[f"pup__{s_}__{tag}"] = v[0], v[1][:, 0], v[1][:, 2]
        df_seen = None
        if seen_control and not smoke:
            t_s = time.time()
            eval_kw = tl._eval_kwargs_from(kw_f)  # 只评估 raw：其它策略在见过的 TF 上按构造跟 raw 相同(上面自检已核对)
            ld = make_loader(seen_rows, kw_f)
            if ld is not None:
                eval_kw = dict(eval_kw, loader=ld)
            ev = tl.evaluate(model, ds, list(seen_rows), class_counts, **eval_kw)
            del ld
            gc.collect()
            yb_s, pr_s = np.asarray(ev["y_b_pred"], np.float64), softmax_np(ev["logits_c"])
            df_seen = pd.DataFrame(dict(fold=f, gene_id=gene_arr[seen_rows], tf_depleted=tf_arr[seen_rows],
                                        split=split_of[seen_rows], y_b_true=yb_all[seen_rows],
                                        y_c_true=yc_all[seen_rows], in_lg=in_lg_of(seen_rows),
                                        yb=yb_s, pdn=pr_s[:, 0], pup=pr_s[:, 2]))
            say(f"  已见 TF 对照：{len(seen_rows)} 条，评估 {time.time() - t_s:.0f} 秒")
        meta = dict(fold=f, held=sorted(held), seed=seed, experiment=experiment, code_version=tl.CODE_VERSION,
                    cfg_digest=cfg_digest, lto_version=LTO_VERSION, strategies=list(strategies), source=source,
                    best_epoch_progress=best_meta.get("epoch_progress"), stop_reason=best_meta.get("stop_reason"),
                    minutes=(time.time() - t_f) / 60, n_held_rows=len(eval_rows), invariance=inv,
                    n_trained_tf=len(trained_tf))
        del model, head_states, res, res_chk
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return df, df_seen, meta, nbr_rows

    # ---- 3. 冒烟 + 逐折(可断点续跑：预测文件齐全且配置一致就跳过；只有权重时只重新评估) ----
    def pred_path(f):
        return os.path.join(outdir, f"fold{f}_pred.csv.gz")

    def seen_path(f):
        return os.path.join(outdir, f"fold{f}_seen.csv.gz")

    def fold_reusable(f):
        mp = os.path.join(outdir, f"fold{f}_meta.json")
        if force or not (os.path.exists(pred_path(f)) and os.path.exists(mp)):
            return False
        if seen_control and not os.path.exists(seen_path(f)):
            return False
        with open(mp, encoding="utf-8") as fh:
            m_ = json.load(fh)
        return (m_.get("held") == sorted(folds[f]) and m_.get("seed") == seed
                and m_.get("experiment") == experiment and m_.get("code_version") == tl.CODE_VERSION
                and m_.get("cfg_digest") == cfg_digest and m_.get("strategies") == list(strategies))

    todo = [f for f in range(n_folds) if not fold_reusable(f)]
    need_train = [f for f in todo if ckpt_ok(f, folds[f]) is None]
    if smoke_first and need_train:
        say(f"\n===== 冒烟：折{need_train[0]}、{smoke_steps} 步(不存结果)，走一遍全部策略 + 不变性自检 =====")
        t_s = time.time()
        df_s, _, meta_s, _ = train_fold(need_train[0], folds[need_train[0]],
                                        dict(kw, n_epochs=1, max_steps=int(smoke_steps)), smoke=True)
        cols = [cn for cn in df_s.columns if cn.startswith(("yb__", "pdn__", "pup__"))]
        ok = bool(len(df_s)) and all(bool(np.isfinite(df_s[cn]).all()) for cn in cols)
        say(f"冒烟{'通过' if ok else '失败(预测里有 NaN)'}：{time.time() - t_s:.0f} 秒，评估样本 {len(df_s)} 条，"
            f"{len(cols) // 3} 套预测")
        if not ok:
            raise SystemExit("冒烟没通过，把上面的日志贴回来")
    metas, nbr_all = [], []
    for f in range(n_folds):
        mp = os.path.join(outdir, f"fold{f}_meta.json")
        if f not in todo:
            say(f"\n折{f}: 已有同配置结果，跳过({pred_path(f)})")
            with open(mp, encoding="utf-8") as fh:
                metas.append(json.load(fh))
            continue
        df_f, df_seen, meta, nbr = train_fold(f, folds[f], kw)
        df_f.to_csv(pred_path(f), index=False, compression="gzip")
        if df_seen is not None:
            df_seen.to_csv(seen_path(f), index=False, compression="gzip")
        meta["knn_neighbors"] = nbr
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, default=str)
        metas.append(meta)
        say(f"折{f} 完成({meta['source']})：{meta['minutes']:.0f} 分钟，已存 {pred_path(f)}")
    for m_ in metas:
        nbr_all.extend(dict(fold=m_["fold"], **r_) for r_ in m_.get("knn_neighbors", []))
    if nbr_all:
        pd.DataFrame(nbr_all).to_csv(os.path.join(outdir, "knn_neighbors.csv"), index=False)

    # ---- 4. 汇总 ----
    P = pd.concat([pd.read_csv(pred_path(f)) for f in range(n_folds)], ignore_index=True)
    say(f"\n汇总：{len(P)} 条被留出样本，{P['tf_depleted'].nunique()} 个 TF，{P['gene_id'].nunique()} 个基因，"
        f"显著样本 {int(P['y_b_true'].notna().sum())}；D∈L_g 的样本 {int(P['in_lg'].sum())} 条")
    comp = {}
    for sp in ("val", "test"):
        pth = os.path.join(main_results_dir, f"predictions_{sp}.parquet")
        if os.path.exists(pth):
            comp[sp] = pd.read_parquet(pth)
    variants = {}
    for s_ in strategies:
        variants[f"LTO[{s_}](selected)"] = (f"yb__{s_}__sel", f"pdn__{s_}__sel", f"pup__{s_}__sel")
    for s_ in strategies:
        if f"yb__{s_}__ph" in P.columns:
            variants[f"LTO[{s_}](perhead)"] = (f"yb__{s_}__ph", f"pdn__{s_}__ph", f"pup__{s_}__ph")
    id_key, D_std = None, None
    if len(comp) == 2:
        D_std = pd.concat(comp.values(), ignore_index=True)
        cols_seed = [f"y_b_seed{seed}", f"p_down_seed{seed}", f"p_up_seed{seed}"]
        seeds_all = sorted(int(cn[len("y_b_seed"):]) for cn in D_std.columns
                           if cn.startswith("y_b_seed") and cn[len("y_b_seed"):].isdigit())
        if all(cn in D_std.columns for cn in cols_seed) and seeds_all:
            Dm = D_std[["gene_id", "tf_depleted"]].copy()
            Dm["y_b_id"], Dm["p_down_id"], Dm["p_up_id"] = (D_std[cols_seed[0]], D_std[cols_seed[1]],
                                                            D_std[cols_seed[2]])
            Dm["y_b_ens"] = np.mean([D_std[f"y_b_seed{s2}"].to_numpy(np.float64) for s2 in seeds_all], 0)
            Dm["p_down_ens"] = np.mean([D_std[f"p_down_seed{s2}"].to_numpy(np.float64) for s2 in seeds_all], 0)
            Dm["p_up_ens"] = np.mean([D_std[f"p_up_seed{s2}"].to_numpy(np.float64) for s2 in seeds_all], 0)
            D_std = Dm.drop_duplicates(["gene_id", "tf_depleted"])
            n0 = len(P)
            P = P.merge(D_std, on=["gene_id", "tf_depleted"], how="left")
            miss = int(P["y_b_id"].isna().sum())
            if miss:
                say(f"⚠ {miss}/{n0} 条被留出样本在 {main_results_dir} 的导出里找不到，对照只用能对上的行")
            id_key = f"标准模型 seed{seed}(见过这些 TF)"
            variants[id_key] = ("y_b_id", "p_down_id", "p_up_id")
            variants[f"标准模型 {len(seeds_all)}-seed 集成(上界参照)"] = ("y_b_ens", "p_down_ens", "p_up_ens")
        else:
            D_std = None
            say(f"⚠ {main_results_dir} 的导出里没有 seed{seed} 的列，跳过对照")
    else:
        say(f"⚠ 找不到 {main_results_dir}/predictions_val|test.parquet，跳过\"见过 vs 没见过\"对照(先 --export)")
    valid = P["y_b_id"].notna().to_numpy() if id_key else np.ones(len(P), bool)

    tf_code = pd.factorize(P["tf_depleted"])[0]
    n_tfc = int(tf_code.max()) + 1
    yb_t, yc_t = P["y_b_true"].to_numpy(np.float64), P["y_c_true"].to_numpy(np.int64)
    strata = {"全部": np.ones(len(P), bool), "D∈L_g": P["in_lg"].to_numpy(bool),
              "D∉L_g": ~P["in_lg"].to_numpy(bool)}
    V = {k: tuple(P[c_].to_numpy(np.float64) for c_ in cols) for k, cols in variants.items()}
    mkeys = ("B_r", "B_sign", "B_r_within_tf", "C_AUCdn", "C_APdn", "C_AUCup", "C_APup")
    short = dict(B_r="B_r", B_sign="B_sign", B_r_within_tf="B_r|TF", C_AUCdn="C_AUCdn", C_APdn="C_APdn",
                 C_AUCup="C_AUCup", C_APup="C_APup")

    def mets(rows, key, yb_=None, yc_=None, tc_all=None, VV=None):
        yb_ = yb_t if yb_ is None else yb_
        yc_ = yc_t if yc_ is None else yc_
        tc_all = tf_code if tc_all is None else tc_all
        yp, pdn, pup = (V if VV is None else VV)[key]
        ok = np.isfinite(yb_[rows]) & np.isfinite(yp[rows])
        p_, t_, tc_ = yp[rows][ok], yb_[rows][ok], tc_all[rows][ok]
        nt = int(tc_all.max()) + 1 if len(tc_all) else 1
        cnt = np.maximum(np.bincount(tc_, minlength=nt), 1)
        mp_ = np.bincount(tc_, weights=p_, minlength=nt) / cnt
        mt_ = np.bincount(tc_, weights=t_, minlength=nt) / cnt
        out = dict(n=int(len(rows)), n_sig=int(ok.sum()), B_r=pearson(p_, t_),
                   B_sign=float(np.mean(np.sign(p_) == np.sign(t_))) if len(p_) else float("nan"),
                   B_r_within_tf=pearson(p_ - mp_[tc_], t_ - mt_[tc_]))
        out["C_AUCdn"], out["C_APdn"] = auroc_auprc(pdn[rows], yc_[rows] == 0)
        out["C_AUCup"], out["C_APup"] = auroc_auprc(pup[rows], yc_[rows] == 2)
        return out

    genes = P["gene_id"].to_numpy()
    uniq, inv = np.unique(genes, return_inverse=True)
    rows_of_gene = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    brng = np.random.default_rng(0)
    boot = [np.concatenate([rows_of_gene[g] for g in brng.integers(0, len(uniq), len(uniq))])
            for _ in range(int(n_boot))]
    sel_keys = [f"LTO[{s_}](selected)" for s_ in strategies]
    ci_keys = sel_keys + ([id_key] if id_key else [])
    rows_out = []

    def ci_of(arr):
        arr = np.asarray(arr, np.float64)
        arr = arr[np.isfinite(arr)]
        return (float(np.quantile(arr, .025)), float(np.quantile(arr, .975))) if len(arr) else (np.nan, np.nan)

    say(f"\n[1] 被留出 TF 上的指标(点估计 [95% CI，按基因整群 bootstrap {n_boot} 次；只给\"选中权重\"的各策略和同 seed 标准模型算 CI]；"
        "对照只用能对上的行)")
    say("    策略：raw=原样(=v1)；zero=翻转 D 变空操作、只剩删位点；mean=训练过的 TF 的均值；knn_bind/knn_emb=按结合谱/"
        "TF 嵌入最像的训练过的 TF 加权(见文件头 v2 第2条)")
    for sname, smask in strata.items():
        base_rows = np.where(smask & valid)[0]
        if len(base_rows) == 0:
            continue
        say(f"  ── 分层 {sname}：{len(base_rows)} 条样本，显著 {int(np.isfinite(yb_t[base_rows]).sum())} 条 ──")
        pt = {k: mets(base_rows, k) for k in variants}
        bt = {k: [] for k in ci_keys}
        keep_mask = smask & valid
        for br in boot:
            rr = br[keep_mask[br]]
            for k in ci_keys:
                bt[k].append(mets(rr, k))
        for k in variants:
            parts = []
            for mk in mkeys:
                v = pt[k][mk]
                if k in bt:
                    lo, hi = ci_of([b[mk] for b in bt[k]])
                    parts.append(f"{short[mk]}={v:.3f}[{lo:.3f},{hi:.3f}]")
                else:
                    lo, hi = np.nan, np.nan
                    parts.append(f"{short[mk]}={v:.3f}")
                rows_out.append(dict(stratum=sname, variant=k, metric=mk, value=v, ci_lo=lo, ci_hi=hi,
                                     n=pt[k]["n"], n_sig=pt[k]["n_sig"]))
            say(f"    {k}: " + "  ".join(parts))
        pairs = ([(k, id_key, "−标准模型(同seed)") for k in sel_keys] if id_key else []) + \
                [(k, "LTO[raw](selected)", "−raw") for k in sel_keys if k != "LTO[raw](selected)"]
        for a_, b_, lab in pairs:
            parts = []
            for mk in mkeys:
                d0 = pt[a_][mk] - pt[b_][mk]
                lo, hi = ci_of([x[mk] - y[mk] for x, y in zip(bt[a_], bt[b_])])
                flag = "↑" if lo > 0 else ("↓" if hi < 0 else "")
                parts.append(f"{short[mk]}={d0:+.3f}[{lo:+.3f},{hi:+.3f}]{flag}")
                rows_out.append(dict(stratum=sname, variant=f"{a_}{lab}", metric=mk, value=d0, ci_lo=lo,
                                     ci_hi=hi, n=pt[a_]["n"], n_sig=pt[a_]["n_sig"]))
            say(f"    Δ {a_}{lab}(配对): " + "  ".join(parts))
    pd.DataFrame(rows_out).to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    # ---- 4b. raw 对 v1(同 seed、同分折重训；只核对点估计) ----
    v1m = os.path.join(v1_dir, "metrics.csv") if v1_dir else ""
    if v1m and os.path.exists(v1m):
        m1 = pd.read_csv(v1m)
        m1 = m1[(m1["stratum"] == "全部") & (m1["variant"] == "LTO(selected)")].set_index("metric")["value"]
        now = pd.DataFrame(rows_out)
        now = now[(now["stratum"] == "全部") & (now["variant"] == "LTO[raw](selected)")].set_index("metric")["value"]
        say("\n[1b] raw 对 v1(同 seed 同分折重训；GPU 非确定性下差 ≤~0.03 算正常)：" + "  ".join(
            f"{short[mk]} {m1.get(mk, np.nan):.3f}->{now.get(mk, np.nan):.3f}" for mk in mkeys))

    # ---- 5. 已见 TF 对照 ----
    if seen_control and all(os.path.exists(seen_path(f)) for f in range(n_folds)):
        S = pd.concat([pd.read_csv(seen_path(f)) for f in range(n_folds)], ignore_index=True)
        if D_std is not None:
            S = S.merge(D_std, on=["gene_id", "tf_depleted"], how="left")  # 左连接保持 S 的行顺序
        VS = {"LTO[raw] 见过的 TF": tuple(S[c_].to_numpy(np.float64) for c_ in ("yb", "pdn", "pup"))}
        ok_rows = np.arange(len(S))
        if D_std is not None:
            VS["标准模型 同seed 同一批行"] = tuple(S[c_].to_numpy(np.float64) for c_ in ("y_b_id", "p_down_id", "p_up_id"))
            ok_rows = np.where(np.isfinite(S["y_b_id"].to_numpy(np.float64)))[0]
        yb_s, yc_s = S["y_b_true"].to_numpy(np.float64), S["y_c_true"].to_numpy(np.int64)
        tc_s = pd.factorize(S["tf_depleted"])[0]
        say(f"\n[2] 已见 TF 对照(每折模型在它训练过的 TF 的 val+test 样本上；{len(S)} 条，3 折合并，点估计；"
            "LTO 模型健康时应跟标准模型接近)")
        for k in VS:
            m_ = mets(ok_rows, k, yb_s, yc_s, tc_s, VS)
            say(f"    {k}: " + "  ".join(f"{short[mk]}={m_[mk]:.3f}" for mk in mkeys))

    # ---- 6. 逐 TF ----
    tf_rows = []
    for tf, g in P.groupby("tf_depleted"):
        idx = g.index.to_numpy()
        rec = dict(tf_depleted=tf, fold=int(g["fold"].iloc[0]), n_rows=len(g),
                   n_sig=int(g["y_b_true"].notna().sum()), n_down=int((g["y_c_true"] == 0).sum()),
                   n_up=int((g["y_c_true"] == 2).sum()), frac_in_lg=float(g["in_lg"].mean()))
        for k in sel_keys + ([id_key] if id_key else []):
            yp, pdn, pup = V[k]
            ok = np.isfinite(yb_t[idx]) & np.isfinite(yp[idx])
            tag = "id" if k == id_key else k[4:k.index("]")]
            rec[f"r_b_{tag}"] = pearson(yp[idx][ok], yb_t[idx][ok]) if ok.sum() >= 10 else np.nan
            rec[f"aucdn_{tag}"] = auroc_auprc(pdn[idx], yc_t[idx] == 0)[0]
            rec[f"aucup_{tag}"] = auroc_auprc(pup[idx], yc_t[idx] == 2)[0]
        tf_rows.append(rec)
    pt_df = pd.DataFrame(tf_rows).sort_values("n_sig", ascending=False)
    pt_df.to_csv(os.path.join(outdir, "per_tf.csv"), index=False)
    ok_tf = pt_df[pt_df["n_sig"] >= 20]
    say(f"\n[3] 逐 TF(显著样本 ≥20 的 {len(ok_tf)} 个；完整表见 per_tf.csv；格式：中位 [四分位]，<0(r)或<0.5(AUROC) 的 TF 数)")
    for tag in [s_ for s_ in strategies] + (["id"] if id_key else []):
        parts = []
        for col, lab in ((f"r_b_{tag}", "TF内r_b"), (f"aucdn_{tag}", "AUROC dn"), (f"aucup_{tag}", "AUROC up")):
            if col in ok_tf.columns and ok_tf[col].notna().any():
                thr = 0.5 if "auc" in col else 0.0
                parts.append(f"{lab} {ok_tf[col].median():.3f} [{ok_tf[col].quantile(.25):.3f}~"
                             f"{ok_tf[col].quantile(.75):.3f}] <{thr:g}:{int((ok_tf[col] < thr).sum())}")
        say(f"  {'标准模型同seed' if tag == 'id' else tag:>14s}: " + "  ".join(parts))
    if nbr_all:
        say(f"\n[4] knn 邻居(每个被留出 TF 前3个，完整见 knn_neighbors.csv)，例：")
        for r_ in nbr_all[:6]:
            say(f"    {r_['strategy']} {r_['tf']}: " + "  ".join(str(r_.get(f'nb{j}', '')) for j in (1, 2, 3)))
    say("\n[5] 各折: " + "；".join(
        f"折{m_['fold']} {m_.get('source', '')} {m_['minutes']:.0f}分钟(选中@epoch{m_.get('best_epoch_progress')}，"
        f"{m_.get('stop_reason')})" for m_ in metas))
    say("\n判读(预先写下，建议，不是硬规则)：(a) [1b] raw≈v1、[2] 见过的 TF 上 LTO≈标准模型，否则先查训练；(b) zero 的 D∈L_g 层\n"
        "AUROC 下界 >0.5 -> 只靠删位点就能泛化；(c) mean 的 AUROC 下界 >0.5 -> 反向来自未训练的 marker 列(推测成立)，再看 23 号\n"
        "基因先验；(d) knn_* 对 raw 以外再看它对 mean 的差(23 号 [2] 段给配对 CI)，>0 才是\"借邻居 TF 外推\"的证据；\n"
        "(e) 全部策略 AUROC ≤0.5 或 B 不超过基因先验 -> 不宣称泛化。只有 1 个 seed/折。")
    say(f"\n写出 {outdir}/summary.txt、folds.json、fold*_pred.csv.gz、fold*_seen.csv.gz、fold*_meta.json、per_tf.csv、"
        f"metrics.csv、knn_neighbors.csv；权重在 {ckpt_dir}/  (总用时 {(time.time() - t_all) / 60:.1f} 分钟)")
    say("下一步：python scripts/tflayout/23_lto_diagnose.py(run_all.sh --lto 会自动接着跑)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return dict(folds=folds, metrics=pd.DataFrame(rows_out), per_tf=pt_df)


if __name__ == "__main__":
    run_lto(**CONFIG)
