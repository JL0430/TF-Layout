# scripts/tflayout/26_paper_baselines.py

import itertools
import os
import time

import numpy as np
import pandas as pd

CONFIG = dict(
    paths=dict(layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
               head_a="out/head_a_baseline_logtpm.parquet",
               promoter_tokens="out/promoter_token_ids.parquet",
               promoter_seq="out/promoter_seq.parquet"),          # 2026-09-30a：6-mer 基线用(11 号产出)
    results_root="out/results",
    # 2026-10-05 第12批：G1 闸门(v11 对 B7b-bag 的对等集成比较)。基线部分 seed=0，预期与冻结结果一致(没核对)，只换模型侧：
    #   论文冻结结果 = model_run "v8_headA_all" + outdir "out/results/_baselines"(27 号读它；要重算 v8 就把这两处改回去)
    model_run="v11_wt_bc",        # 模型侧用哪个实验的导出；没有就退回 fallback_run
    fallback_run="v3_ce_marker_dense",
    # 25 号产出(1108 基因的模型预测)；按顺序找第一个存在且有 <model_run>_ens 列的；都没有就只比 820 网格基因
    head_a_all_csv=("out/results/_head_a_all/per_gene_head_a.csv", "out/results/_head_a_all_b10/per_gene_head_a.csv"),
    citra_pred=None,              # 可选：CITRA 逐基因预测 csv(第一列 gene_id、第二列预测值)。None=跳过
    variant="ens_ph",             # 模型用哪套权重的集成：\"ens_ph\"(各头最优，报告口径)或 \"ens\"
    token_top=4000,               # promoter token 词袋保留最常见的这么多个 token
    kmer_k=6,                     # 2026-09-30a：k-mer 基线的 k(正反链合并后 k=6 是 2080 维)
    # 2026-09-30a：扩到 1e5(第9批上界 1000，token/k-mer 岭回归可能顶在上界)；在 val 上按 r 选，顶边界会打印警告
    # 2026-10-01a 第11批：两头再扩(第10批 A2k 顶在 1e5；Head B 顶在 0.1，见文件头第11批第1、2条)
    ridge_alphas=(1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6),
    ridge_alphas_small_block=(0.1, 10.0, 1000.0),  # 分块岭回归里小块(8 个 layout 特征)的 alpha 候选
    logit_l2=1.0,                 # Head C 逻辑回归的 L2(sklearn 的 C=1/logit_l2；numpy 退路同一个量级)
    logit_iters=1500,             # 没有 sklearn 时全批量梯度下降步数(第9批 400)
    logit_lr=0.5,
    c_class_weight_options=("none", "balanced"),  # 2026-09-30a：Head C 基线的类别权重在 val 上按 AUPRC 均值选
    offset_grid=(-8.0, 5.0, 0.25),  # 2026-09-30a：第9批 (−3,3) 太窄(B4/B5 顶边界)；步长跟 17/19 号一样
    n_boot=300,                   # 按基因整群 bootstrap 次数
    n_boot_tf=200,                # 2026-09-30a：TF 内 AUROC 的 bootstrap 次数(每次要按 TF 分组排名，慢一些)
    min_pos_per_tf=5,             # TF 内 AUROC 只算正例 ≥ 这个数的 TF(跟 17 号 (h) 一样)
    strata_models=("B2_TF×结合先验", "B4_pair+TF先验", "B5_B4+实测WT表达", "B6_gbdt(同B5特征)",
                   "B7a_gbdt+flat(无WT)", "B7b_gbdt+flat(+WT)", "B7a_bag(无WT)", "B7b_bag(+WT)"),  # [2c] 跟模型比的基线(第12批 v4 加 B7，第14批 v5 加 bag)
    stack_baseline=("B6_gbdt(同B5特征)", "B5_B4+实测WT表达"),  # [3] 堆叠用的基线：按顺序取第一个存在的(第一个堆叠行=27 号 Table 2 的堆叠行，保持 B6)
    stack_extra=("B7a_bag(无WT)", "B7b_bag(+WT)"),  # 2026-10-02c 第14批：额外再对这些基线各堆叠一次(没有就跳过)
    # 2026-10-02a 第12批 v4：B7 平铺 layout GBDT(见文件头)
    flat_layout=True,              # False=整体关掉 B7
    flat_with_wt=True,             # 是否同时拟合带实测 WT 表达的 B7b
    # 2026-10-02c 第14批 v5：Head B / Head C 的轮数候选分开(第13批 B7b 的 Head B 选在上界 500，B7a 的 Head C 选在下界 100)
    # 各自训练到候选里最大的轮数、在这些轮数里按 val 选；旧键 stages 仍可用(两头都用它)
    # 2026-10-02d 第15批 v6：Head B 候选扩到 4000(第14批 B7a 的 Head B 又顶在 1200)；逐轮只在 val 上预测，选定后全量预测一次
    flat_gbdt=dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                   stages_b=(100, 200, 300, 500, 800, 1200, 1600, 2000, 2500, 3000, 4000),
                   stages_c=(25, 50, 75, 100, 150, 200, 300)),
    flat_class_weight_options=("none", "balanced"),
    # 2026-10-02c 第14批 v5：B7a/B7b 的 bagging 对等集成(每个成员随机 80% 训练基因)；None 或 n=0 关闭
    # 2026-10-02d 第15批 v6：再多训练 n_extra_null 个成员(种子接着往后排)，只用来量"5 个成员 vs 另外 5 个"的重训噪声(Table 6 的 ‡/§)；
    #   报告用的 B7_bag 仍是前 n 个成员(跟 v5 同种子)；null_splits=从 C(2n−1,n−1) 种切分里最多抽几种，null_pairs=单成员两两差最多抽几对
    flat_bag=dict(n=5, frac=0.8, n_extra_null=5, null_splits=60, null_pairs=20),
    fair_ensemble=True,            # [2d] 公平集成对照(模型单 seed 对 B7 单模型、模型集成对 bag)；False 关闭
    fair_noise=True,               # 2026-10-02d 第15批 v6：[2d] 加训练随机性的合成区间 ‡ / 保守区间 §(见文件头第15批第1条)；False 回到 v5 行为
    n_boot_fair=200,               # [2d] 的 bootstrap 次数(取前 n 次抽样，跟 [2]/[2b] 配对；不超过 n_boot 和 n_boot_tf)
    seed=0,
    use_sklearn=True,             # 有 sklearn 就多算 GBDT、逻辑回归用 lbfgs；没有自动退回 numpy 实现
    outdir="out/results/_baselines_v11",  # 2026-10-05：新目录，不覆盖 27 号读的冻结 _baselines/
)


def run_paper_baselines(paths, results_root, model_run, fallback_run, head_a_all_csv, citra_pred,
                        variant, token_top, kmer_k, ridge_alphas, ridge_alphas_small_block, logit_l2, logit_iters,
                        logit_lr, c_class_weight_options, offset_grid, n_boot, n_boot_tf, min_pos_per_tf,
                        strata_models, stack_baseline, seed, use_sklearn, outdir,
                        flat_layout=True, flat_with_wt=True, flat_gbdt=None, flat_class_weight_options=("none", "balanced"),
                        stack_extra=(), flat_bag=None, fair_ensemble=True, n_boot_fair=200, fair_noise=True):
    """唯一入口，各段见文件头。"""
    t_all = time.time()
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)
    lines = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def pearson(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or x.std() <= 1e-12 * max(1.0, abs(x.mean())) or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def auroc(score, pos):
        score, pos = np.asarray(score, np.float64), np.asarray(pos, bool)
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 == 0 or n0 == 0:
            return float("nan")
        r = pd.Series(score).rank().to_numpy()
        return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def avg_precision(score, pos):
        score, pos = np.asarray(score, np.float64), np.asarray(pos, bool)
        if pos.sum() == 0:
            return float("nan")
        o = np.argsort(-score, kind="mergesort")
        p = pos[o]
        prec = np.cumsum(p) / np.arange(1, len(p) + 1)
        return float(np.sum(prec * p) / p.sum())

    def macro_f1(pred, true):
        f1s = []
        for c in range(3):
            tp = float(np.sum((pred == c) & (true == c)))
            fp = float(np.sum((pred == c) & (true != c)))
            fn = float(np.sum((pred != c) & (true == c)))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f1s))

    def ci_of(arr):
        arr = np.asarray(arr, np.float64)
        arr = arr[np.isfinite(arr)]
        return (float(np.quantile(arr, .025)), float(np.quantile(arr, .975))) if len(arr) else (np.nan, np.nan)

    def star_of(lo, hi):
        return "↑" if lo > 0 else ("↓" if hi < 0 else " ")

    def standardize(X, fit_mask):
        X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = X[fit_mask].mean(0), X[fit_mask].std(0) + 1e-9
        return (X - mu) / sd

    def ridge_fit_predict(X, y, fit_mask, sel_mask, alphas, n_small=0, alphas_small=None):
        """闭式 ridge(截距不正则；特征按 fit_mask 标准化)，alpha 在 sel_mask 上按 Pearson r 选。
        2026-09-30a：n_small>0 时是分块岭回归——前 n_small 列一个 alpha(从 alphas_small 选)、其余列另一个 alpha(从 alphas 选)。
        返回 (全体预测, 选中的 alpha 元组, 是否顶在网格边界)。"""
        Xd = np.column_stack([np.ones(len(X)), standardize(X, fit_mask)])
        y = np.asarray(y, np.float64)
        A, b = Xd[fit_mask], y[fit_mask]
        Gm, rhs = A.T @ A, A.T @ b
        p = Xd.shape[1] - 1
        grid = ([(a,) for a in alphas] if n_small <= 0
                else list(itertools.product(alphas_small or alphas, alphas)))
        best, best_r, best_a = None, -np.inf, grid[len(grid) // 2]
        for al in grid:
            pen = (np.full(p, float(al[0])) if n_small <= 0
                   else np.r_[np.full(n_small, float(al[0])), np.full(p - n_small, float(al[1]))])
            try:
                w = np.linalg.solve(Gm + np.diag(np.r_[0.0, pen]), rhs)
            except np.linalg.LinAlgError:
                continue
            pr = Xd @ w
            r_ = pearson(pr[sel_mask], y[sel_mask]) if np.any(sel_mask) else np.nan
            if np.isfinite(r_) and r_ > best_r:
                best, best_r, best_a = pr, r_, tuple(float(a) for a in al)
        if best is None:
            pen = np.full(p, float(best_a[-1]))
            best = Xd @ np.linalg.solve(Gm + np.diag(np.r_[0.0, pen]), rhs)
        edge = best_a[-1] in (float(min(alphas)), float(max(alphas)))
        return best, best_a, edge

    def logit_numpy(Xs, y_cls, fit_mask, balanced):
        """没有 sklearn 时的退路：多项逻辑回归(3类，L2，带动量的全批量梯度下降)。Xs 已标准化。"""
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        A = Xd[fit_mask]
        yy = np.asarray(y_cls, np.int64)[fit_mask]
        n, d = A.shape
        cnt = np.array([max(int((yy == c).sum()), 1) for c in range(3)], np.float64)
        sw = (n / (3.0 * cnt))[yy] if balanced else np.ones(n)
        sw = sw / sw.mean()
        Y = np.zeros((n, 3))
        Y[np.arange(n), yy] = 1.0
        W, V = np.zeros((d, 3)), np.zeros((d, 3))
        for _ in range(int(logit_iters)):
            z = A @ W
            z -= z.max(1, keepdims=True)
            P = np.exp(z)
            P /= P.sum(1, keepdims=True)
            Gd = A.T @ ((P - Y) * sw[:, None]) / n + float(logit_l2) * W / n
            Gd[0] -= float(logit_l2) * W[0] / n
            V = 0.9 * V - float(logit_lr) * Gd
            W = W + V
        z = Xd @ W
        z -= z.max(1, keepdims=True)
        P = np.exp(z)
        return P / P.sum(1, keepdims=True)

    def logit_fit_predict(X, y_cls, fit_mask, balanced):
        """多项逻辑回归，返回全体的概率矩阵(列顺序 down/ns/up)。2026-09-30a：有 sklearn 用 lbfgs 收敛到底。"""
        Xs = standardize(X, fit_mask)
        if sk_lr is not None:
            m_ = sk_lr(C=1.0 / float(logit_l2), max_iter=2000, class_weight="balanced" if balanced else None)
            m_.fit(Xs[fit_mask], np.asarray(y_cls, np.int64)[fit_mask])
            P = np.zeros((len(Xs), 3))
            P[:, list(m_.classes_)] = m_.predict_proba(Xs)
            return P
        return logit_numpy(Xs, y_cls, fit_mask, balanced)

    def tune_offset(p_val, y_val, grid):
        """在 val 上搜 down/up 的 log 概率偏置，使 macro-F1 最大(跟 17/19 号同一种做法)。返回 (偏置, 是否顶边界)。"""
        lo, hi, st = grid
        cand = np.arange(lo, hi + 1e-9, st)
        lp = np.log(np.clip(p_val, 1e-12, 1.0))
        best, bf = (0.0, 0.0), -np.inf
        for bd in cand:
            for bu in cand:
                f = macro_f1((lp + np.array([bd, 0.0, bu])).argmax(1), y_val)
                if f > bf:
                    bf, best = f, (float(bd), float(bu))
        edge = any(abs(v - lo) < 1e-9 or abs(v - cand[-1]) < 1e-9 for v in best)
        return best, edge

    def apply_offset(p_mat, off):
        return (np.log(np.clip(p_mat, 1e-12, 1.0)) + np.array([off[0], 0.0, off[1]])).argmax(1)

    def log_odds(P):
        P = np.clip(np.asarray(P, np.float64), 1e-9, 1.0)
        return np.column_stack([np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / P[:, 1])])

    def pad(s_, w):
        """按显示宽度补空格(CJK 算2列)，让贴回来的表格不错位。"""
        s_ = str(s_)
        wid = sum(2 if ord(ch) > 0x2E80 else 1 for ch in s_)
        return s_ + " " * max(w - wid, 0)

    def ascii_of(s_):
        """图里只能用 ASCII(服务器没中文字体会出方块，跟 17 号同一条约定)。"""
        s_ = str(s_).replace("模型 ", "model ").replace("模型", "model")
        return "".join(ch if ord(ch) < 128 else "_" for ch in s_)[:22]

    def three_col(df, c_dn, c_up):
        m = np.column_stack([pd.to_numeric(df[c_dn], errors="coerce").to_numpy(np.float64),
                             np.zeros(len(df)),
                             pd.to_numeric(df[c_up], errors="coerce").to_numpy(np.float64)])
        m[:, 1] = np.clip(1.0 - m[:, 0] - m[:, 2], 1e-9, 1.0)
        return m

    def kmer_counts(seqs, k):
        """正反链合并的 k-mer 计数(含 N 的窗口跳过)。返回 (n_seq × n_canonical) 矩阵。"""
        lut = np.full(256, -1, np.int64)
        for ch, v in zip("ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3)):
            lut[ord(ch)] = v
        pw = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
        allc = np.arange(4 ** k, dtype=np.int64)
        digits = (allc[:, None] // pw[None, :]) % 4
        rc_all = (3 - digits[:, ::-1]) @ pw
        canon = np.minimum(allc, rc_all)
        uniq = np.unique(canon)
        col_of = np.full(4 ** k, -1, np.int64)
        col_of[uniq] = np.arange(len(uniq))
        out = np.zeros((len(seqs), len(uniq)), np.float64)
        for i, s_ in enumerate(seqs):
            if not isinstance(s_, str) or len(s_) < k:
                continue
            a = lut[np.frombuffer(s_.encode("ascii", "replace"), np.uint8)]
            win = np.lib.stride_tricks.sliding_window_view(a, k)
            ok = (win >= 0).all(1)
            if not ok.any():
                continue
            code = win[ok] @ pw
            out[i] = np.bincount(col_of[canon[code]], minlength=len(uniq))
        return out

    say("=" * 100)
    say(f"论文基线对比(26 号，第15批 v6 2026-10-02d)  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * 100)

    sk_gb, sk_lr = None, None
    if use_sklearn:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression
            sk_gb = (HistGradientBoostingRegressor, HistGradientBoostingClassifier)
            sk_lr = LogisticRegression
        except ImportError:
            say("    (没有 sklearn：跳过 GBDT 基线，逻辑回归退回 numpy 梯度下降；pip install scikit-learn 之后重跑即可)")

    # ----------------------------------------------------------------- 0. 重建网格
    lay = pd.read_parquet(paths["layout"])
    lbl = pd.read_parquet(paths["labels"])
    ha = pd.read_parquet(paths["head_a"])
    if "split" not in ha.columns:
        raise SystemExit(f"{paths['head_a']} 没有 split 列(先确认 08 号产出)")
    y_col = [c for c in ha.columns if c != "split"][0]
    up2tf = {t.upper(): t for t in sorted(set(lay["tf"]))}
    dep = lbl["tf_depleted"].astype(str).str.strip().str.upper().map(lambda u: up2tf.get(u))
    n_bad = int(dep.isna().sum())
    lbl = lbl.assign(tf_depleted=dep).dropna(subset=["tf_depleted"]).reset_index(drop=True)
    lbl["split"] = lbl["gene_id"].map(ha["split"].to_dict())
    n_nosplit = int(lbl["split"].isna().sum())
    if n_nosplit:
        say(f"警告：{n_nosplit} 行的基因在 Head A 标签里没有 split，已丢弃(正常应为0)")
        lbl = lbl.dropna(subset=["split"]).reset_index(drop=True)
    cls2i = {"down": 0, "ns": 1, "up": 2}
    lbl["y_c"] = lbl["direction_3class"].map(cls2i).fillna(1).astype(int)
    lbl["y_b"] = pd.to_numeric(lbl["log2fc"], errors="coerce")
    nby = lbl["split"].value_counts().to_dict()
    say(f"[0] 网格重建：{len(lbl)} 行(丢弃 tf_depleted 对不上的 {n_bad} 行，跟 09 号同一规则)；"
        f"train/val/test = {nby.get('train', 0)}/{nby.get('val', 0)}/{nby.get('test', 0)}"
        f"  (应为 399528/84692/101680；对不上先别看下面的数)")

    lay = lay.assign(_ma=(pd.to_numeric(lay["res_id"], errors="coerce").fillna(0) > 0.5).astype(float),
                     _ad=np.abs(pd.to_numeric(lay["site_pos"], errors="coerce").fillna(0.0)))
    g_agg = lay.groupby("gene_id").agg(n_sites=("tf", "size"), n_unique_tf=("tf", "nunique"),
                                       sum_a=("a", "sum"), mean_a=("a", "mean"), sum_m=("m", "sum"),
                                       mean_m=("m", "mean"), frac_motif=("_ma", "mean"),
                                       min_dist_any=("_ad", "min"))
    gd_agg = lay.groupby(["gene_id", "tf"]).agg(d_n_sites=("tf", "size"), d_max_a=("a", "max"),
                                                d_sum_a=("a", "sum"), d_min_dist=("_ad", "min"),
                                                d_frac_motif=("_ma", "mean"))
    say(f"    layout：{len(lay)} 个位点 / {len(g_agg)} 个基因；(基因,TF) 对 {len(gd_agg)} 个")

    G = lbl.join(g_agg, on="gene_id").join(gd_agg, on=["gene_id", "tf_depleted"])
    G["bound"] = np.isfinite(pd.to_numeric(G["d_n_sites"], errors="coerce")).astype(float)
    for c in ("n_sites", "n_unique_tf", "sum_a", "mean_a", "sum_m", "mean_m", "frac_motif",
              "d_n_sites", "d_max_a", "d_sum_a", "d_frac_motif"):
        G[c] = pd.to_numeric(G[c], errors="coerce").fillna(0.0)
    for c in ("min_dist_any", "d_min_dist"):
        G[c] = pd.to_numeric(G[c], errors="coerce").fillna(1500.0)
    ha_y = pd.to_numeric(ha[y_col], errors="coerce")
    G["y_a_meas"] = G["gene_id"].map(ha_y.to_dict()).astype(float)
    tr = (G["split"] == "train").to_numpy()
    va = (G["split"] == "val").to_numpy()
    te = (G["split"] == "test").to_numpy()
    G["y_a_meas"] = G["y_a_meas"].fillna(float(G.loc[tr, "y_a_meas"].mean()))

    glob_p = np.array([float((G.loc[tr, "y_c"] == c).mean()) for c in range(3)])
    cnt_tf = (G[tr].groupby(["tf_depleted", "y_c"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1, 2], fill_value=0))
    p_tf = {t: (row.to_numpy() + 1.0) / (row.sum() + 3.0) for t, row in cnt_tf.iterrows()}
    cnt_tfb = (G[tr].groupby(["tf_depleted", "bound", "y_c"]).size().unstack(fill_value=0)
               .reindex(columns=[0, 1, 2], fill_value=0))
    p_tfb = {k: (row.to_numpy() + 10.0 * p_tf.get(k[0], glob_p)) / (row.sum() + 10.0)
             for k, row in cnt_tfb.iterrows()}
    P_tf = np.stack([p_tf.get(t, glob_p) for t in G["tf_depleted"]])
    P_tfb = np.stack([p_tfb.get((t, b), p_tf.get(t, glob_p)) for t, b in zip(G["tf_depleted"], G["bound"])])
    tf_mean_b = G[tr & np.isfinite(G["y_b"])].groupby("tf_depleted")["y_b"].mean().to_dict()
    glob_b = float(G.loc[tr & np.isfinite(G["y_b"]), "y_b"].mean())
    col_tfmean = G["tf_depleted"].map(tf_mean_b).fillna(glob_b).to_numpy(np.float64)

    # ----------------------------------------------------------------- 1. Head A
    say("\n[1] Head A(基因级 z(log TPM))：岭回归 alpha 在 val 基因上按 r 选(分块岭回归两块各选一个)；模型=各 seed 集成")
    ok = np.isfinite(ha_y.to_numpy(np.float64))
    genesA = np.array(ha.index.tolist(), dtype=object)[ok]
    yA = ha_y.to_numpy(np.float64)[ok]
    spA = ha["split"].to_numpy()[ok]
    in_gridA = np.isin(genesA, np.array(sorted(set(lbl["gene_id"])), dtype=object))
    has_siteA = np.isin(genesA, g_agg.index.to_numpy())
    grpA = np.where(in_gridA, 0, np.where(has_siteA, 1, 2))  # 0 网格 / 1 网格外有位点 / 2 网格外无位点(同 25 号 [4])
    trA, vaA, teA = spA == "train", spA == "val", spA == "test"
    fA = ["n_sites", "n_unique_tf", "sum_a", "mean_a", "sum_m", "mean_m", "frac_motif", "min_dist_any"]
    ga_r = g_agg.reindex(genesA)
    XA_lay = np.column_stack([pd.to_numeric(ga_r[c], errors="coerce")
                              .fillna(1500.0 if c == "min_dist_any" else 0.0).to_numpy(np.float64) for c in fA])
    say(f"    Head A 基因 {len(genesA)} 个(train/val/test = {int(trA.sum())}/{int(vaA.sum())}/{int(teA.sum())})；"
        f"test 里网格 {int((teA & in_gridA).sum())}、网格外有位点 {int((teA & (grpA == 1)).sum())}、网格外无位点 "
        f"{int((teA & (grpA == 2)).sum())}(应为 4524/940/1108、820/136/152，跟 25 号 [1] 一致)")

    XA_tok = None
    if os.path.exists(paths["promoter_tokens"]):
        tk = pd.read_parquet(paths["promoter_tokens"])
        g2t = {g: np.asarray(v, np.int64) for g, v in zip(tk["gene_id"], tk["token_ids"])}
        have = [g for g in genesA if g in g2t]
        if have:
            vc = pd.Series(np.concatenate([g2t[g] for g in have])).value_counts()
            keep = vc.index.to_numpy()[:int(token_top)]
            pos = {int(t): i for i, t in enumerate(keep)}
            XA_tok = np.zeros((len(genesA), len(keep)), np.float32)
            for i, g in enumerate(genesA):
                ids = g2t.get(g)
                if ids is None:
                    continue
                idx = [pos[int(t)] for t in ids if int(t) in pos]
                if idx:
                    np.add.at(XA_tok[i], np.asarray(idx, np.int64), 1.0)
            XA_tok = np.log1p(XA_tok).astype(np.float64)
            say(f"    promoter token 词袋：{len(have)}/{len(genesA)} 个基因有序列，保留最常见的 {len(keep)} 个 token"
                f"(全词表 {len(vc)} 个)")
    if XA_tok is None:
        say(f"    没有 {paths['promoter_tokens']}，跳过序列词袋基线(A2/A3/A4)")

    XA_kmer = None  # 2026-09-30a：6-mer 计数
    sp_path = paths.get("promoter_seq")
    if sp_path and os.path.exists(sp_path):
        try:
            sq = pd.read_parquet(sp_path)
            gcol = next((c for c in ("gene_id", "gene", "orf", "systematic_name", "name") if c in sq.columns), None)
            scol = "seq" if "seq" in sq.columns else next(
                (c for c in sq.columns if sq[c].dtype == object and isinstance(sq[c].iloc[0], str)
                 and len(sq[c].iloc[0]) >= 100), None)
            gids = (sq[gcol] if gcol is not None else pd.Series(sq.index, index=sq.index)).astype(str).to_numpy()
            if scol is None:
                raise ValueError(f"找不到序列列(列名 {list(sq.columns)[:10]})")
            g2s = dict(zip(gids, sq[scol].tolist()))
            n_have = sum(1 for g in genesA if str(g) in g2s)
            if n_have < 0.5 * len(genesA):
                raise ValueError(f"只有 {n_have}/{len(genesA)} 个 Head A 基因在序列文件里(基因列={gcol!r}，列名 "
                                 f"{list(sq.columns)[:10]})")
            XA_kmer = np.log1p(kmer_counts([g2s.get(str(g)) for g in genesA], int(kmer_k)))
            say(f"    promoter {kmer_k}-mer(正反链合并)：{n_have}/{len(genesA)} 个基因有序列，{XA_kmer.shape[1]} 维"
                f"(基因列={gcol!r}，序列列={scol!r})")
        except Exception as e:  # noqa: BLE001 —— 解析不出来就跳过这几行基线，不影响其余
            say(f"    (k-mer 基线跳过：{type(e).__name__}: {e})")
            XA_kmer = None
    else:
        say(f"    没有 {sp_path}，跳过 k-mer 基线(A2k/A3k/A4k)")

    predA, noteA, warnA = {}, {}, []

    def add_ridge(name, X, n_small=0):
        pr, al, edge = ridge_fit_predict(X, yA, trA, vaA, ridge_alphas, n_small=n_small,
                                         alphas_small=ridge_alphas_small_block)
        predA[name] = pr
        noteA[name] = (f"alpha={al[0]:g}" if len(al) == 1 else f"alpha(layout块)={al[0]:g}, alpha(序列块)={al[1]:g}") + \
            f", 特征{X.shape[1]}"
        if edge and X.shape[1] > 50:  # 特征少的模型按 r 选 alpha 时对 alpha 不敏感，顶边界不要紧
            warnA.append(f"{name} 的 alpha={al[-1]:g} 顶在网格边界上(高维序列模型，说明可能还没调到最好)")

    add_ridge("A1_layout_ridge", XA_lay)
    if XA_tok is not None:
        add_ridge("A2_token_ridge", XA_tok)
        add_ridge("A3_layout+token_分块ridge", np.column_stack([XA_lay, XA_tok]), n_small=XA_lay.shape[1])
    if XA_kmer is not None:
        add_ridge(f"A2k_{kmer_k}mer_ridge", XA_kmer)
        add_ridge(f"A3k_layout+{kmer_k}mer_分块ridge", np.column_stack([XA_lay, XA_kmer]), n_small=XA_lay.shape[1])
    if sk_gb is not None:
        for nm, X in (("A4_gbdt(layout+token)", np.column_stack([XA_lay, XA_tok]) if XA_tok is not None else None),
                      (f"A4k_gbdt(layout+{kmer_k}mer)", np.column_stack([XA_lay, XA_kmer]) if XA_kmer is not None
                       else None),
                      ("A4L_gbdt(layout)", XA_lay if XA_tok is None and XA_kmer is None else None)):
            if X is None:
                continue
            m_ = sk_gb[0](max_iter=300, learning_rate=0.06, random_state=int(seed))
            m_.fit(X[trA], yA[trA])
            predA[nm] = m_.predict(X)
            noteA[nm] = f"HistGradientBoosting, 特征{X.shape[1]}"
    for k, v in noteA.items():
        say(f"    拟合 {pad(k, 30)}{v}")
    for w_ in warnA:
        say(f"    ⚠ {w_}")
    if citra_pred:
        if os.path.exists(citra_pred):
            cp = pd.read_csv(citra_pred)
            cmap = dict(zip(cp[cp.columns[0]].astype(str),
                            pd.to_numeric(cp[cp.columns[1]], errors="coerce")))
            predA["CITRA"] = np.array([cmap.get(str(g), np.nan) for g in genesA], np.float64)
            noteA["CITRA"] = f"{citra_pred}，覆盖 {int(np.isfinite(predA['CITRA']).sum())}/{len(genesA)} 个基因"
            say(f"    CITRA：{noteA['CITRA']}")
        else:
            say(f"    (CONFIG['citra_pred']={citra_pred} 不存在，跳过 CITRA 行)")

    run = model_run if os.path.exists(os.path.join(results_root, model_run, "predictions_test.parquet")) \
        else fallback_run
    if run != model_run:
        say(f"    提示：找不到 {model_run} 的导出，模型侧退回 {fallback_run}")
    pt = pd.read_parquet(os.path.join(results_root, run, "predictions_test.parquet"))
    pv = pd.read_parquet(os.path.join(results_root, run, "predictions_val.parquet"))
    vsuf = "_ph" if variant == "ens_ph" else ""
    mdlA = f"模型 {run}"
    for csv_ in ([head_a_all_csv] if isinstance(head_a_all_csv, str) else list(head_a_all_csv)):
        if mdlA in predA or not os.path.exists(csv_):
            continue
        pg = pd.read_csv(csv_)
        col = f"{run}_ens"
        if col in pg.columns:
            mp = dict(zip(pg["gene_id"].astype(str), pd.to_numeric(pg[col], errors="coerce")))
            predA[mdlA] = np.array([mp.get(str(g), np.nan) for g in genesA], np.float64)
            say(f"    模型 Head A 预测读自 25 号 {csv_} 的 {col} 列(全部 Head A 基因)")
    if mdlA not in predA:
        mp = pt.drop_duplicates("gene_id").set_index("gene_id")[f"ens{vsuf}_y_a"].to_dict()
        predA[mdlA] = np.array([mp.get(g, np.nan) for g in genesA], np.float64)
        say(f"    模型 Head A 预测读自 {run}/predictions_test.parquet 的 ens{vsuf}_y_a(只有网格基因)")

    def centered_r(v, ix):
        """组内中心化 r(跟 25 号 [4] 同一定义)：各组分别扣掉自己的真值均值和预测均值再合并。"""
        yy, vv, gg = yA[ix].copy(), np.asarray(v, np.float64)[ix].copy(), grpA[ix]
        okv = np.isfinite(vv) & np.isfinite(yy)
        for c in (0, 1, 2):
            m = (gg == c) & okv
            if m.any():
                yy[m] -= yy[m].mean()
                vv[m] -= vv[m].mean()
        return pearson(vv[okv], yy[okv])

    rng = np.random.default_rng(int(seed))
    setsA = {"test 全部": teA, "test 网格": teA & in_gridA, "test 网格外": teA & ~in_gridA,
             "test 网格外有位点": teA & (grpA == 1), "test 网格外无位点": teA & (grpA == 2),  # 2026-10-01a 第11批第3条
             "test 全部(组内中心化)": teA}
    bootA = {}
    for nm, m in setsA.items():
        ix = np.where(m)[0]
        bootA[nm] = (ix, bootA["test 全部"][1] if nm == "test 全部(组内中心化)" else
                     [rng.choice(ix, len(ix)) for _ in range(int(n_boot))])
    rowsA = []
    say("    " + pad("模型/基线", 34) + "".join(pad(k.replace("test ", "").replace("(组内中心化)", "组内中心化"), 24)
                                          for k in setsA))  # 2026-10-01a：列跟 setsA 走
    for k, v in predA.items():
        cells = []
        for nm, (ix, bts) in bootA.items():
            fr = centered_r if nm.endswith("(组内中心化)") else (lambda vv, ii: pearson(vv[ii], yA[ii]))
            r_ = fr(v, ix)
            lo, hi = ci_of([fr(v, b) for b in bts])
            dd = dlo = dhi = np.nan
            if k != mdlA and mdlA in predA:
                mv = predA[mdlA]
                dd = fr(mv, ix) - r_
                dlo, dhi = ci_of([fr(mv, b) - fr(v, b) for b in bts])
            cells.append(f"{r_:.3f}[{lo:.3f},{hi:.3f}]" if np.isfinite(r_) else "常数预测(r无定义)")
            rowsA.append(dict(model=k, gene_set=nm, n_genes=len(ix), r=r_, ci_lo=lo, ci_hi=hi,
                              delta_model_minus_base=dd, delta_ci_lo=dlo, delta_ci_hi=dhi,
                              note=noteA.get(k, "")))
        say("    " + pad(k, 34) + "".join(pad(c, 24) for c in cells))
    say("    Δ = 模型 − 基线(按基因配对 bootstrap；↑/↓=CI 不含0)：")
    dA = pd.DataFrame(rowsA)
    for k in predA:
        if k == mdlA:
            continue
        cs = []
        for nm in setsA:
            q = dA[(dA["model"] == k) & (dA["gene_set"] == nm)].iloc[0]
            cs.append(f"{nm.replace('test ', '')} {q['delta_model_minus_base']:+.3f}"
                      f"[{q['delta_ci_lo']:+.3f},{q['delta_ci_hi']:+.3f}]"
                      f"{star_of(q['delta_ci_lo'], q['delta_ci_hi']).strip()}"
                      if np.isfinite(q["delta_model_minus_base"]) else
                      f"{nm.replace('test ', '')} 无定义(基线常数预测)")  # 2026-10-01a
        say("      对 " + pad(k, 30) + "  ".join(cs))
    dA.to_csv(os.path.join(outdir, "table1_head_a.csv"), index=False)

    # ----------------------------------------------------------------- 2. Head B/C
    say("\n[2] Head B/C(test 全部行)：Head B=ridge(只在显著行上拟合)、Head C=多项逻辑回归(先验以对数几率进模型；类别权重在 val "
        "上按 AUPRC 均值选)；每条基线和模型的 down/up 偏置都在 val 上搜、原样套到 test")
    F_gene = ["n_sites", "n_unique_tf", "sum_a", "mean_a", "sum_m", "mean_m", "frac_motif",
              "min_dist_any", "y_a_meas"]
    F_pair = ["bound", "d_n_sites", "d_max_a", "d_sum_a", "d_min_dist", "d_frac_motif",
              "n_sites", "n_unique_tf", "sum_a", "mean_a", "frac_motif", "min_dist_any"]
    lo_tf, lo_tfb = log_odds(P_tf), log_odds(P_tfb)  # 2026-09-30a：先验改成对数几率(第9批是线性概率)
    prior = [lo_tf[:, 0], lo_tf[:, 1], lo_tfb[:, 0], lo_tfb[:, 1], col_tfmean]

    def mat(cols, extra=()):
        return np.column_stack([pd.to_numeric(G[c], errors="coerce").to_numpy(np.float64)
                                for c in cols] + list(extra))

    XB = {"B3_gene_only": mat(F_gene),
          "B4_pair+TF先验": mat(F_pair, prior),
          "B5_B4+实测WT表达": mat(F_pair + ["y_a_meas"], prior)}
    yb = G["y_b"].to_numpy(np.float64)
    yc = G["y_c"].to_numpy(np.int64)
    sig = np.isfinite(yb)
    fitB, fitC = tr & sig, tr
    ycv_all = yc[va]

    def val_ap(P):
        return float(np.nanmean([avg_precision(P[va][:, 0], ycv_all == 0), avg_precision(P[va][:, 2], ycv_all == 2)]))

    predB, predC, noteB = {}, {}, {}
    for k, X in XB.items():
        pb, a_, edge = ridge_fit_predict(X, np.nan_to_num(yb), fitB, va & sig, ridge_alphas)
        predB[k] = pb
        cands = {}
        for cw in c_class_weight_options:
            cands[cw] = logit_fit_predict(X, yc, fitC, balanced=(cw == "balanced"))
        cw_best = max(cands, key=lambda c_: val_ap(cands[c_]))
        predC[k] = cands[cw_best]
        edge_txt = ("" if not edge else "(下边界≈不正则，特征少，无影响)" if (a_[0] == float(min(ridge_alphas))
                    and X.shape[1] <= 50) else "(顶边界)")  # 2026-10-01a 第11批第2条
        noteB[k] = (f"ridge alpha={a_[0]:g}{edge_txt}, 逻辑回归类别权重={cw_best}"
                    f"(val AUPRC均值 " + "/".join(f"{c_}:{val_ap(P_):.3f}" for c_, P_ in cands.items()) +
                    f"), 特征{X.shape[1]}, {'sklearn lbfgs' if sk_lr is not None else 'numpy GD'}")
        say(f"    拟合 {pad(k, 18)}{noteB[k]}(显著训练行 {int(fitB.sum())}、全部训练行 {int(fitC.sum())})")
    if sk_gb is not None:
        X = XB["B5_B4+实测WT表达"]
        r_ = sk_gb[0](max_iter=300, learning_rate=0.06, random_state=int(seed))
        r_.fit(X[fitB], yb[fitB])
        predB["B6_gbdt(同B5特征)"] = r_.predict(X)
        cands = {}
        for cw in c_class_weight_options:
            c_ = sk_gb[1](max_iter=300, learning_rate=0.06, random_state=int(seed))
            if cw == "balanced":
                cnt = np.array([max(int((yc[fitC] == c).sum()), 1) for c in range(3)], np.float64)
                c_.fit(X[fitC], yc[fitC], sample_weight=(len(yc[fitC]) / (3.0 * cnt))[yc[fitC]])
            else:
                c_.fit(X[fitC], yc[fitC])
            pp = np.zeros((len(G), 3))
            pp[:, list(c_.classes_)] = c_.predict_proba(X)
            cands[cw] = pp
        cw_best = max(cands, key=lambda c_: val_ap(cands[c_]))
        predC["B6_gbdt(同B5特征)"] = cands[cw_best]
        noteB["B6_gbdt(同B5特征)"] = (f"HistGradientBoosting, 类别权重={cw_best}(val AUPRC均值 " +
                                      "/".join(f"{c_}:{val_ap(P_):.3f}" for c_, P_ in cands.items()) +
                                      f"), 特征{X.shape[1]}")
        say(f"    拟合 {pad('B6_gbdt', 18)}{noteB['B6_gbdt(同B5特征)']}")

    # ----------------------------------------------------------------- 2026-10-02a 第12批 v4：B7 平铺 layout GBDT
    # 2026-10-02c 第14批 v5：Head B/C 轮数网格分开并修边界；新增 bagging 对等集成 B7a_bag / B7b_bag(见文件头第14批)
    B7A, B7B = "B7a_gbdt+flat(无WT)", "B7b_gbdt+flat(+WT)"
    B7A_BAG, B7B_BAG = "B7a_bag(无WT)", "B7b_bag(+WT)"
    bag_cfg = dict(flat_bag or {})
    bag_n, bag_frac = int(bag_cfg.get("n", 0) or 0), float(bag_cfg.get("frac", 0.8))
    bag_extra = int(bag_cfg.get("n_extra_null", 0) or 0) if bag_n > 0 else 0   # 2026-10-02d 第15批 v6
    bag_mem = {}       # 2026-10-02d：tag -> [(Head B 预测 float32, Head C 概率 float32), ...]，前 bag_n 个=报告用的 bag，其余只量噪声
    tune_rows = []     # 2026-10-02d：table_b7_tuning.csv(Methods 用)
    if flat_layout and sk_gb is None:
        say("    (没有 sklearn：跳过 B7 平铺 layout GBDT；pip install scikit-learn 之后重跑)")
    if flat_layout and sk_gb is not None:
        gbp = dict(flat_gbdt or dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                                    stages_b=(100, 200, 300, 500, 800, 1200, 1600, 2000, 2500, 3000, 4000),
                                    stages_c=(25, 50, 75, 100, 150, 200, 300)))
        st_old = gbp.pop("stages", (100, 200, 300, 400, 500))   # 旧键：两头共用
        stages_b = tuple(sorted(int(s_) for s_ in gbp.pop("stages_b", st_old)))
        stages_c = tuple(sorted(int(s_) for s_ in gbp.pop("stages_c", st_old)))
        set_b, set_c, max_b, max_c = set(stages_b), set(stages_c), stages_b[-1], stages_c[-1]
        tf_names = sorted(set(lay["tf"]))
        tf_ix = {t_: i_ for i_, t_ in enumerate(tf_names)}
        gene_ids = pd.unique(G["gene_id"])
        gix = {g_: i_ for i_, g_ in enumerate(gene_ids)}
        cnt_gt = np.zeros((len(gene_ids), len(tf_names)), np.float32)
        mind_gt = np.full((len(gene_ids), len(tf_names)), 1500.0, np.float32)
        sub_ = gd_agg.reset_index()
        sub_ = sub_[sub_["gene_id"].isin(gene_ids) & sub_["tf"].isin(tf_names)]
        gi_ = sub_["gene_id"].map(gix).to_numpy(np.int64)
        ti_ = sub_["tf"].map(tf_ix).to_numpy(np.int64)
        cnt_gt[gi_, ti_] = sub_["d_n_sites"].to_numpy(np.float32)
        mind_gt[gi_, ti_] = sub_["d_min_dist"].to_numpy(np.float32)
        g_row_s, d_row_s = G["gene_id"].map(gix), G["tf_depleted"].map(tf_ix)
        if g_row_s.isna().any() or d_row_s.isna().any():
            raise SystemExit("B7：有网格行的基因/被耗竭 TF 对不上 layout 的 TF 词表(先查 [0] 段的丢弃名单)")
        g_row, d_row = g_row_s.to_numpy(np.int64), d_row_s.to_numpy(np.int64)
        say(f"\n    B7 平铺 layout：{len(gene_ids)} 个基因 × {len(tf_names)} 个 TF，每个 TF 两个特征(位点数、距 ATG 最近距离)；"
            f"有位点的(基因,TF)格 {int((cnt_gt > 0).sum())} 个；HistGB 轮数候选 Head B {stages_b}、Head C {stages_c}"
            f"(第15批 v6：逐轮只在 val 上预测，选定轮数后再全量预测一次)")
        gid_arr = G["gene_id"].to_numpy()
        tr_genes = pd.unique(gid_arr[tr])
        va_ix = np.flatnonzero(va)

        def mk_est(kind, ncol, n_iter):
            kw = dict(learning_rate=float(gbp.get("learning_rate", 0.06)), max_leaf_nodes=int(gbp.get("max_leaf_nodes", 31)),
                      min_samples_leaf=int(gbp.get("min_samples_leaf", 40)), max_iter=int(n_iter), early_stopping=False,
                      random_state=int(seed))
            cls_ = sk_gb[0] if kind == "reg" else sk_gb[1]
            try:
                return cls_(categorical_features=[ncol - 1], **kw), True   # 最后一列=被耗竭 TF 的类别码
            except TypeError:   # 很老的 sklearn 没有 categorical_features：当数值用
                return cls_(**kw), False

        def bal_w(mask_):
            cnt_c = np.array([max(int((yc[mask_] == c).sum()), 1) for c in range(3)], np.float64)
            return (len(yc[mask_]) / (3.0 * cnt_c))[yc[mask_]]

        def at_stage(gen_, it_stop):
            """2026-10-02d：从 staged_predict(_proba) 生成器里取第 it_stop 轮的预测(取到就停，不往后算)。"""
            last = None
            for it_, pr_ in enumerate(gen_, start=1):
                last = pr_
                if it_ == it_stop:
                    break
            return np.array(last, copy=True)

        def clf_full(clf_, X_, it_):
            P_ = np.zeros((X_.shape[0], 3))
            P_[:, list(clf_.classes_)] = at_stage(clf_.staged_predict_proba(X_), it_)
            return P_

        for tag, with_wt in ((B7A, False), (B7B, True)):
            if with_wt and not flat_with_wt:
                continue
            t0 = time.time()
            base = XB["B5_B4+实测WT表达"] if with_wt else XB["B4_pair+TF先验"]
            X = np.hstack([base.astype(np.float32), cnt_gt[g_row], mind_gt[g_row], d_row.astype(np.float32)[:, None]])
            say(f"    开始拟合 {tag}：{X.shape[1]} 维特征；1 个 Head B 回归 + {len(flat_class_weight_options)} 个 Head C 三分类"
                f"(没有进度条，估几分钟到十几分钟一个)")
            reg, cat_ok = mk_est("reg", X.shape[1], max_b)
            reg.fit(X[fitB], yb[fitB])
            vs_ix = np.flatnonzero(va & sig)
            curve_b = {}
            for it_, pr_ in enumerate(reg.staged_predict(X[vs_ix]), start=1):   # 2026-10-02d：只在 val 显著行上逐轮
                if it_ in set_b:
                    curve_b[it_] = pearson(pr_, yb[vs_ix])
            okb = {k_: v_ for k_, v_ in curve_b.items() if np.isfinite(v_)}
            best_itb = max(okb, key=lambda k_: okb[k_]) if okb else max_b
            best_r = float(okb.get(best_itb, np.nan))
            predB[tag] = at_stage(reg.staged_predict(X), best_itb)
            cands7 = {}
            Xv_ = X[va_ix]
            ycv7 = yc[va_ix]
            for cw in flat_class_weight_options:
                clf, _ = mk_est("clf", X.shape[1], max_c)
                if cw == "balanced":
                    clf.fit(X[fitC], yc[fitC], sample_weight=bal_w(fitC))
                else:
                    clf.fit(X[fitC], yc[fitC])
                curve_c = {}
                for it_, pp_ in enumerate(clf.staged_predict_proba(Xv_), start=1):   # 2026-10-02d：只在 val 行上逐轮
                    if it_ in set_c:
                        Pv7 = np.zeros((len(va_ix), 3))
                        Pv7[:, list(clf.classes_)] = pp_
                        curve_c[it_] = float(np.nanmean([avg_precision(Pv7[:, 0], ycv7 == 0), avg_precision(Pv7[:, 2], ycv7 == 2)]))
                okc = {k_: v_ for k_, v_ in curve_c.items() if np.isfinite(v_)}
                bit_ = max(okc, key=lambda k_: okc[k_]) if okc else max_c
                bs_ = float(okc.get(bit_, np.nan))
                cands7[cw] = (clf_full(clf, X, bit_), bs_, bit_, curve_c)
                del clf
            del Xv_
            cw_best = max(cands7, key=lambda c_: (cands7[c_][1] if np.isfinite(cands7[c_][1]) else -1.0))
            predC[tag] = cands7[cw_best][0]
            itc_best = int(cands7[cw_best][2])
            edge_b = "(⚠ 顶在轮数上界，可能还没调到最好)" if best_itb == stages_b[-1] else ("(⚠ 顶在轮数下界)" if best_itb == stages_b[0] else "")
            edge_c = "(⚠ 顶在轮数上界)" if itc_best == stages_c[-1] else ("(⚠ 顶在轮数下界)" if itc_best == stages_c[0] else "")
            noteB[tag] = (f"HistGB 平铺 layout({len(tf_names)} TF×[位点数,最近距离]+D 类别码{'(类别特征)' if cat_ok else '(当数值)'}"
                          f"{'+实测WT表达' if with_wt else ''})，Head B 轮数={best_itb}{edge_b}(val r {best_r:.3f})，"
                          f"Head C 类别权重={cw_best} 轮数={itc_best}{edge_c}(val AUPRC均值 "
                          + "/".join(f"{c_}:{cands7[c_][1]:.3f}" for c_ in cands7) + f")，特征{X.shape[1]}")
            say(f"    拟合 {pad(tag, 22)}{noteB[tag]}(用时 {(time.time() - t0) / 60:.1f} 分钟)")
            say(f"      Head B val r 曲线(轮数:r)：" + "  ".join(f"{k_}:{v_:.4f}" for k_, v_ in sorted(curve_b.items())))
            say(f"      Head C val AUPRC均值 曲线({cw_best})：" + "  ".join(f"{k_}:{v_:.4f}" for k_, v_ in sorted(cands7[cw_best][3].items())))
            if best_itb == stages_b[-1] or itc_best in (stages_c[0], stages_c[-1]):
                say(f"    ⚠ {tag} 的轮数选在候选网格边界上：把这行贴回来，必要时在 CONFIG['flat_gbdt'] 里把 stages_b/stages_c 再往外扩")
            tune_rows.append(dict(model=tag, head="B", selected=int(best_itb), val_score=best_r, edge=bool(best_itb in (stages_b[0], stages_b[-1])),
                                  grid=str(stages_b), class_weight="", curve=";".join(f"{k_}:{v_:.4f}" for k_, v_ in sorted(curve_b.items())),
                                  learning_rate=float(gbp.get("learning_rate", 0.06)), max_leaf_nodes=int(gbp.get("max_leaf_nodes", 31)),
                                  min_samples_leaf=int(gbp.get("min_samples_leaf", 40)), n_features=int(X.shape[1])))
            for cw, (_, bs_, bit_, crv_) in cands7.items():
                tune_rows.append(dict(model=tag, head=f"C({cw}{'，选中' if cw == cw_best else ''})", selected=int(bit_), val_score=bs_,
                                      edge=bool(bit_ in (stages_c[0], stages_c[-1])), grid=str(stages_c), class_weight=cw,
                                      curve=";".join(f"{k_}:{v_:.4f}" for k_, v_ in sorted(crv_.items())),
                                      learning_rate=float(gbp.get("learning_rate", 0.06)), max_leaf_nodes=int(gbp.get("max_leaf_nodes", 31)),
                                      min_samples_leaf=int(gbp.get("min_samples_leaf", 40)), n_features=int(X.shape[1])))
            del cands7

            if bag_n > 0:   # ---- 2026-10-02c 第14批 v5：bagging 对等集成(超参沿用上面单模型选定的)
                t1 = time.time()   # 2026-10-02d 第15批 v6：共训练 bag_n + bag_extra 个成员；前 bag_n 个(种子同 v5)=报告用的 bag，其余只量噪声
                bag_tag = B7B_BAG if with_wt else B7A_BAG
                n_pick = max(int(round(bag_frac * len(tr_genes))), 1)
                mem_ = []
                pb_sum, pc_sum = np.zeros(len(G)), np.zeros((len(G), 3))   # 报告用的 bag 按 v5 的 float64 累加(成员另存 float32 只给零分布用)
                for m_ in range(bag_n + bag_extra):
                    chosen = np.random.default_rng(int(seed) + 1000 + m_).choice(tr_genes, size=n_pick, replace=False)
                    rm = tr & np.isin(gid_arr, chosen)
                    fB_m, fC_m = rm & sig, rm
                    reg_m, _ = mk_est("reg", X.shape[1], best_itb)
                    reg_m.fit(X[fB_m], yb[fB_m])
                    pb_m64 = reg_m.predict(X)
                    clf_m, _ = mk_est("clf", X.shape[1], itc_best)
                    if cw_best == "balanced":
                        clf_m.fit(X[fC_m], yc[fC_m], sample_weight=bal_w(fC_m))
                    else:
                        clf_m.fit(X[fC_m], yc[fC_m])
                    P_m = np.zeros((len(G), 3))
                    P_m[:, list(clf_m.classes_)] = clf_m.predict_proba(X)
                    if m_ < bag_n:
                        pb_sum += pb_m64
                        pc_sum += P_m
                    mem_.append((pb_m64.astype(np.float32), P_m.astype(np.float32)))
                    del reg_m, clf_m
                bag_mem[tag] = mem_
                predB[bag_tag] = pb_sum / bag_n
                predC[bag_tag] = pc_sum / bag_n
                noteB[bag_tag] = (f"{tag} 的 bagging 集成：{bag_n} 个成员，每个用随机 {bag_frac:.0%} 的训练基因({n_pick}/{len(tr_genes)})，"
                                  f"超参沿用单模型(Head B 轮数 {best_itb}，Head C 类别权重 {cw_best} 轮数 {itc_best})，Head B 取预测均值、Head C 取概率均值"
                                  + (f"；另训练 {bag_extra} 个成员只用来量重训噪声([2d] 的 ‡/§)" if bag_extra else ""))
                say(f"    拟合 {pad(bag_tag, 22)}{noteB[bag_tag]}(用时 {(time.time() - t1) / 60:.1f} 分钟)")
            del X, reg

    pos_of = dict(zip(zip(G["gene_id"], G["tf_depleted"]), range(len(G))))
    try:
        ord_t = np.array([pos_of[k] for k in zip(pt["gene_id"], pt["tf_depleted"])], np.int64)
        ord_v = np.array([pos_of[k] for k in zip(pv["gene_id"], pv["tf_depleted"])], np.int64)
    except KeyError as e:
        raise SystemExit(f"17 号导出里有本脚本重建不出来的 (基因,TF) 键 {e}：先查 09/17 号的过滤规则")
    if not (np.array_equal(np.sort(ord_t), np.sort(np.where(te)[0]))
            and np.array_equal(np.sort(ord_v), np.sort(np.where(va)[0]))):
        raise SystemExit("17 号导出的 test/val 行跟本脚本重建的对不上(行集合不同)，先查 09/17 号的过滤规则")
    if len(ord_t) == 0 or len(ord_v) == 0:
        raise SystemExit("17 号导出的 test 或 val 行是空的，先确认 out/results/<实验>/predictions_*.parquet")
    d_chk = float(np.nanmax(np.abs(np.nan_to_num(pd.to_numeric(pt["y_b_true"], errors="coerce")
                                                 .to_numpy(np.float64)) - np.nan_to_num(yb[ord_t]))))
    say(f"    跟 17 号导出逐行核对 y_b_true：最大差 {d_chk:.2e}(应为 0)")
    if d_chk > 1e-8:
        raise SystemExit("标签对不上，停(不要用下面的数)")

    mdlB = f"模型 {run}({variant})"
    models = {mdlB: (pd.to_numeric(pt[f"ens{vsuf}_y_b"], errors="coerce").to_numpy(np.float64),
                     three_col(pt, f"ens{vsuf}_p_down", f"ens{vsuf}_p_up"),
                     three_col(pv, f"ens{vsuf}_p_down", f"ens{vsuf}_p_up")),
              "B1_TF均值": (pd.to_numeric(pt["base_TF_mean_y_b"], errors="coerce").to_numpy(np.float64),
                            three_col(pt, "base_TF_p_down", "base_TF_p_up"),
                            three_col(pv, "base_TF_p_down", "base_TF_p_up")),
              "B2_TF×结合先验": (pd.to_numeric(pt["base_TFxbound_y_b"], errors="coerce").to_numpy(np.float64),
                                 three_col(pt, "base_TFxbound_p_down", "base_TFxbound_p_up"),
                                 three_col(pv, "base_TFxbound_p_down", "base_TFxbound_p_up"))}
    val_b = {mdlB: (pd.to_numeric(pv[f"ens{vsuf}_y_b"], errors="coerce").to_numpy(np.float64)
                    if f"ens{vsuf}_y_b" in pv.columns else None)}
    for k in predB:
        models[k] = (predB[k][ord_t], predC[k][ord_t], predC[k][ord_v])
        val_b[k] = predB[k][ord_v]
    oracle_key = "ORACLE_同基因其它TF(参照)"
    if "ORACLE_gene_otherTF_y_b" in pt.columns:
        models[oracle_key] = (
            pd.to_numeric(pt["ORACLE_gene_otherTF_y_b"], errors="coerce").to_numpy(np.float64),
            models["B1_TF均值"][1], models["B1_TF均值"][2])  # ORACLE 只有 Head B，C 列占位、不解读

    ybt, yct = yb[ord_t], yc[ord_t]
    ycv = yc[ord_v]
    tfs_t = G["tf_depleted"].to_numpy()[ord_t]
    genes_t = G["gene_id"].to_numpy()[ord_t]
    tcode_t, tuniq_t = pd.factorize(pd.Series(tfs_t))
    bound_t = G["bound"].to_numpy()[ord_t] > 0.5
    offs, pred_c, edge_off = {}, {}, []
    for k, (_, cp, cv) in models.items():
        offs[k], e_ = tune_offset(cv, ycv, offset_grid)
        pred_c[k] = apply_offset(cp, offs[k])
        if e_ and k != oracle_key:
            edge_off.append(k)
    say("    val 上搜到的偏置(down,up)：" + "；".join(f"{k.split('(')[0]} {offs[k]}" for k in models))
    if edge_off:
        say(f"    ⚠ 偏置顶在搜索范围 {offset_grid[:2]} 的边界上：{edge_off}(它们的 macro-F1 可能被低估)")

    def metrics_of(ix, bp, cpb, cpr):
        yb_, yc_ = ybt[ix], yct[ix]
        s_ = np.isfinite(yb_)
        out = dict(B_r=pearson(bp[ix][s_], yb_[s_]),
                   B_sign=float(np.mean(np.sign(bp[ix][s_]) == np.sign(yb_[s_]))) if s_.any() else np.nan,
                   C_AUCdn=auroc(cpb[ix][:, 0], yc_ == 0), C_AUCup=auroc(cpb[ix][:, 2], yc_ == 2),
                   C_APdn=avg_precision(cpb[ix][:, 0], yc_ == 0),
                   C_APup=avg_precision(cpb[ix][:, 2], yc_ == 2),
                   C_mF1=macro_f1(cpr[ix], yc_))
        if s_.any():
            d = pd.DataFrame(dict(t=tfs_t[ix][s_], p=bp[ix][s_], y=yb_[s_]))
            out["B_r|TF"] = pearson(d["p"] - d.groupby("t")["p"].transform("mean"),
                                    d["y"] - d.groupby("t")["y"].transform("mean"))
        else:
            out["B_r|TF"] = np.nan
        return out

    codes, uniq = pd.factorize(pd.Series(genes_t))
    by_gene = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng2 = np.random.default_rng(int(seed))
    boots = [np.concatenate([by_gene[i] for i in rng2.choice(len(uniq), len(uniq))]) for _ in range(int(n_boot))]
    all_ix = np.arange(len(ord_t))

    def blank_c(d):
        for m_ in ("C_AUCdn", "C_AUCup", "C_APdn", "C_APup", "C_mF1"):
            d[m_] = float("nan")
        return d

    mets = {k: metrics_of(all_ix, v[0], v[1], pred_c[k]) for k, v in models.items()}
    mets_b = {k: [metrics_of(b, v[0], v[1], pred_c[k]) for b in boots] for k, v in models.items()}
    if oracle_key in mets:  # ORACLE 只有 Head B 的定义，Head C 列不打印
        mets[oracle_key] = blank_c(mets[oracle_key])
        mets_b[oracle_key] = [blank_c(x) for x in mets_b[oracle_key]]
    keys = ["B_r", "B_sign", "B_r|TF", "C_AUCdn", "C_AUCup", "C_APdn", "C_APup", "C_mF1"]
    say(f"    test 行 {len(ord_t)}，显著 {int(np.isfinite(ybt).sum())}，"
        f"down/up = {int((yct == 0).sum())}/{int((yct == 2).sum())}；bootstrap {n_boot} 次(按基因整群)")
    say("    " + pad("模型/基线", 34) + "".join(f"{k:>10}" for k in keys))
    rowsB, rowsD = [], []
    for k in models:
        say("    " + pad(k, 34) + "".join(f"{mets[k][m]:>10.3f}" for m in keys))
        for m in keys:
            lo, hi = ci_of([x[m] for x in mets_b[k]])
            rowsB.append(dict(model=k, metric=m, value=mets[k][m], ci_lo=lo, ci_hi=hi,
                              offset=str(offs[k]), note=noteB.get(k, "")))
    say("    Δ = 模型 − 基线 [95% CI 按基因整群配对 bootstrap]，↑/↓=CI 不含0：")
    for k in models:
        if k == mdlB:
            continue
        cs = []
        for m in keys:
            d_ = mets[mdlB][m] - mets[k][m]
            lo, hi = ci_of([a[m] - b[m] for a, b in zip(mets_b[mdlB], mets_b[k])])
            cs.append(f"{d_:+.3f}" + star_of(lo, hi))
            rowsD.append(dict(model=mdlB, baseline=k, metric=m, delta=d_, ci_lo=lo, ci_hi=hi))
        say("      对 " + pad(k, 30) + "".join(f"{c:>10}" for c in cs))
    say("    (ORACLE 行用了同基因其它 TF 的 test 标签，只是分解分析的参照，不是可用的方法；它的 Head C 列是占位，别读)")
    if "B4_pair+TF先验" in mets and "B2_TF×结合先验" in mets:
        chk = [(m, mets["B4_pair+TF先验"][m] - mets["B2_TF×结合先验"][m]) for m in ("C_AUCdn", "C_AUCup")]
        say("    自检：B4 含 B2 的全部信息(TF×结合先验的对数几率)，排序不该比 B2 差：" +
            "、".join(f"{m} B4−B2 {d_:+.3f}" for m, d_ in chk) +
            ("  ✓" if all(d_ > -0.01 for _, d_ in chk) else
             "  ⚠ B4 比 B2 差超过 0.01(第9批是 −0.05/−0.06) -> 逻辑回归拟合还有问题，Head C 的 B3~B5 数字先别进论文，把这段贴回来"))
    pd.DataFrame(rowsB).to_csv(os.path.join(outdir, "table2_head_bc.csv"), index=False)
    pd.DataFrame(rowsD).to_csv(os.path.join(outdir, "delta_model_vs_baseline.csv"), index=False)

    # ----------------------------------------------------------------- 2b. TF 内 AUROC
    say(f"\n[2b] Head C 的 TF 内 AUROC(只比同一个被耗竭 TF 内部的基因排序；≥{min_pos_per_tf} 个正例的 TF)：中位数(跟 17 号 (h) "
        f"同定义)与合并值(=Σ_t U_t / Σ_t n1·n0，同一 TF 内随机一对正/负例排对的概率)；[95% CI 按基因整群 bootstrap "
        f"{n_boot_tf} 次]；B1(TF 先验)在 TF 内是常数，应恰为 0.500")
    nT = len(tuniq_t)

    def within_tf(ix, score, cls):
        t_ = tcode_t[ix]
        p_ = (yct[ix] == cls).astype(np.float64)
        rk = pd.Series(np.asarray(score, np.float64)[ix]).groupby(t_).rank(method="average").to_numpy()
        n_ = np.bincount(t_, minlength=nT).astype(np.float64)
        n1 = np.bincount(t_, weights=p_, minlength=nT)
        r1 = np.bincount(t_, weights=rk * p_, minlength=nT)
        n0 = n_ - n1
        okt = (n1 >= min_pos_per_tf) & (n0 >= 1)
        if not okt.any():
            return np.nan, np.nan, 0
        u = r1[okt] - n1[okt] * (n1[okt] + 1) / 2
        return float(np.median(u / (n1[okt] * n0[okt]))), float(u.sum() / (n1[okt] * n0[okt]).sum()), int(okt.sum())

    tf_models = [k for k in models if k != oracle_key]
    bt_tf = boots[:int(n_boot_tf)]
    wt, wt_b = {}, {}
    for k in tf_models:
        cp = models[k][1]
        for cls, nm in ((0, "dn"), (2, "up")):
            wt[(k, nm)] = within_tf(all_ix, cp[:, cls], cls)
            wt_b[(k, nm)] = [within_tf(b, cp[:, cls], cls) for b in bt_tf]
    rowsT = []
    say("    " + pad("模型/基线", 34) + "  中位dn     合并dn[95% CI]        中位up     合并up[95% CI]        (TF 数 dn/up)")
    for k in tf_models:
        cells = []
        for nm in ("dn", "up"):
            med, poo, nt_ = wt[(k, nm)]
            lo, hi = ci_of([x[1] for x in wt_b[(k, nm)]])
            mlo, mhi = ci_of([x[0] for x in wt_b[(k, nm)]])
            cells.append(f"{med:>7.3f}   {poo:.3f}[{lo:.3f},{hi:.3f}]")
            rowsT.append(dict(model=k, direction=nm, median_auc=med, median_ci_lo=mlo, median_ci_hi=mhi,
                              pooled_auc=poo, pooled_ci_lo=lo, pooled_ci_hi=hi, n_tf=nt_))
        say("    " + pad(k, 34) + "   ".join(cells) + f"   ({wt[(k, 'dn')][2]}/{wt[(k, 'up')][2]})")
    say("    Δ = 模型 − 基线(合并 TF 内 AUROC；中位数的差在括号里)，95% CI 配对：")
    best_base = {}
    for k in tf_models:
        if k == mdlB:
            continue
        cs = []
        for nm in ("dn", "up"):
            d_ = wt[(mdlB, nm)][1] - wt[(k, nm)][1]
            lo, hi = ci_of([a[1] - b[1] for a, b in zip(wt_b[(mdlB, nm)], wt_b[(k, nm)])])
            dm = wt[(mdlB, nm)][0] - wt[(k, nm)][0]
            cs.append(f"{nm} {d_:+.3f}[{lo:+.3f},{hi:+.3f}]{star_of(lo, hi).strip()}(中位 {dm:+.3f})")
            rowsT.append(dict(model=f"Δ {mdlB} − {k}", direction=nm, pooled_auc=d_, pooled_ci_lo=lo, pooled_ci_hi=hi,
                              median_auc=dm))
            if k.startswith(("B3", "B4", "B5", "B6", "B7")):
                if nm not in best_base or wt[(k, nm)][1] > wt[(best_base[nm], nm)][1]:
                    best_base[nm] = k
        say("      对 " + pad(k, 30) + "   ".join(cs))
    if best_base:
        say("    B3~B7 里合并 TF 内 AUROC 最强的：" + "；".join(f"{nm}={k}({wt[(k, nm)][1]:.3f})" for nm, k in best_base.items())
            + " -> 判读规则 (e) 看模型对它的那一行")
    pd.DataFrame(rowsT).to_csv(os.path.join(outdir, "table3_within_tf_auc.csv"), index=False)

    # ----------------------------------------------------------------- 2c. 按 D∈L_g 分层
    s_models = [k for k in strata_models if k in models]
    say(f"\n[2c] 按 D∈L_g 分层(被耗竭 TF 在该基因启动子上有位点)：模型与 {s_models} 的关键指标，Δ=模型−基线[95% CI 配对整群 bootstrap "
        f"{n_boot} 次]，偏置沿用 [2] 在 val 上搜到的")
    keys_s = ["B_r", "B_r|TF", "C_AUCdn", "C_APdn", "C_AUCup", "C_APup"]
    rowsS = []
    for s_nm, s_mask in (("D∈L_g", bound_t), ("D∉L_g", ~bound_t)):
        ix = all_ix[s_mask]
        bts = [b[s_mask[b]] for b in boots]
        say(f"  {s_nm}：{len(ix)} 行，显著 {int(np.isfinite(ybt[ix]).sum())}，down/up {int((yct[ix] == 0).sum())}/"
            f"{int((yct[ix] == 2).sum())}")
        say("    " + pad("", 30) + "".join(f"{k:>18}" for k in keys_s))
        mm = {k: metrics_of(ix, models[k][0], models[k][1], pred_c[k]) for k in [mdlB] + s_models}
        mb = {k: [metrics_of(b, models[k][0], models[k][1], pred_c[k]) for b in bts] for k in [mdlB] + s_models}
        say("    " + pad(mdlB[:28], 30) + "".join(f"{mm[mdlB][m]:>18.3f}" for m in keys_s))
        for k in s_models:
            cs = []
            for m in keys_s:
                d_ = mm[mdlB][m] - mm[k][m]
                lo, hi = ci_of([a[m] - b[m] for a, b in zip(mb[mdlB], mb[k])])
                cs.append(f"{mm[k][m]:.3f} Δ{d_:+.3f}{star_of(lo, hi)}")
                rowsS.append(dict(stratum=s_nm, baseline=k, metric=m, model_value=mm[mdlB][m], base_value=mm[k][m],
                                  delta=d_, ci_lo=lo, ci_hi=hi))
            say("    " + pad(k[:28], 30) + "".join(f"{c:>18}" for c in cs))
    pd.DataFrame(rowsS).to_csv(os.path.join(outdir, "table4_strata.csv"), index=False)

    # ----------------------------------------------------------------- 2d. 公平集成对照(2026-10-02c 第14批 v5)
    rowsF, valsF, null_rowsF = [], [], []
    nbf = int(min(int(n_boot_fair), len(boots), len(bt_tf)))
    if fair_ensemble:
        suf_s = "_ph" if variant == "ens_ph" else ""

        def seeds_in(df_):
            got = []
            for c_ in df_.columns:
                if not str(c_).startswith("y_b_seed"):
                    continue
                tail = str(c_)[len("y_b_seed"):]
                if suf_s:
                    if not tail.endswith(suf_s):
                        continue
                    tail = tail[:-len(suf_s)]
                elif tail.endswith("_ph"):
                    continue
                if tail.isdigit():
                    got.append(int(tail))
            return sorted(got)

        def has_cols(df_, s_):
            return all(f"{p_}{s_}{suf_s}" in df_.columns for p_ in ("y_b_seed", "p_down_seed", "p_up_seed"))

        seeds_m = [s_ for s_ in seeds_in(pt) if s_ in set(seeds_in(pv)) and has_cols(pt, s_) and has_cols(pv, s_)]
        mk_f = keys + ["wTF_dn", "wTF_up"]
        say(f"\n[2d] 公平集成对照：模型的逐 seed 单模型(seed {seeds_m}，各自在 val 上调偏置)对 B7 单模型；模型 {len(seeds_m)}-seed 集成对 B7 的 bagging 集成。"
            f"CI=按基因整群 bootstrap 前 {nbf} 次(跟 [2]/[2b] 同一批抽样，配对)，只含 test 基因抽样；标准差是 seed 之间的")
        if len(seeds_m) < 2:
            say("    (17 号导出里找不到 ≥2 个 seed 的 y_b_seed*/p_down_seed*/p_up_seed* 列：跳过 [2d])")
        else:
            ps_pt, ps_bt = [], []
            seed_pred = {}   # 2026-10-02d 第15批 v6：留着逐 seed 预测，下面造模型的"2 对 2"零分布
            for s_ in seeds_m:
                bp_ = pd.to_numeric(pt[f"y_b_seed{s_}{suf_s}"], errors="coerce").to_numpy(np.float64)
                cp_ = three_col(pt, f"p_down_seed{s_}{suf_s}", f"p_up_seed{s_}{suf_s}")
                seed_pred[s_] = (bp_, cp_)
                cv_ = three_col(pv, f"p_down_seed{s_}{suf_s}", f"p_up_seed{s_}{suf_s}")
                off_s, _ = tune_offset(cv_, ycv, offset_grid)
                cr_ = apply_offset(cp_, off_s)
                m0 = metrics_of(all_ix, bp_, cp_, cr_)
                m0["wTF_dn"], m0["wTF_up"] = within_tf(all_ix, cp_[:, 0], 0)[1], within_tf(all_ix, cp_[:, 2], 2)[1]
                mbs = []
                for b_ in boots[:nbf]:
                    mm_ = metrics_of(b_, bp_, cp_, cr_)
                    mm_["wTF_dn"], mm_["wTF_up"] = within_tf(b_, cp_[:, 0], 0)[1], within_tf(b_, cp_[:, 2], 2)[1]
                    mbs.append(mm_)
                ps_pt.append(m0)
                ps_bt.append(mbs)
                say(f"    seed {s_}：偏置 {off_s}  " + "  ".join(f"{m_} {m0[m_]:.3f}" for m_ in mk_f))

            def pack_model(k):
                p0 = dict(mets[k])
                p0["wTF_dn"], p0["wTF_up"] = wt[(k, "dn")][1], wt[(k, "up")][1]
                b0 = {m_: np.array([x[m_] for x in mets_b[k][:nbf]], np.float64) for m_ in keys}
                b0["wTF_dn"] = np.array([x[1] for x in wt_b[(k, "dn")][:nbf]], np.float64)
                b0["wTF_up"] = np.array([x[1] for x in wt_b[(k, "up")][:nbf]], np.float64)
                return p0, b0

            F, F_sd = {}, {}
            L_ENS, L_SGL = f"模型 {len(seeds_m)}-seed 集成", f"模型 单seed均值(n={len(seeds_m)})"
            F[L_ENS] = pack_model(mdlB)
            F[L_SGL] = ({m_: float(np.nanmean([d_[m_] for d_ in ps_pt])) for m_ in mk_f},
                        {m_: np.nanmean(np.array([[mb[i][m_] for i in range(nbf)] for mb in ps_bt], np.float64), axis=0)
                         for m_ in mk_f})
            F_sd[L_SGL] = {m_: float(np.nanstd([d_[m_] for d_ in ps_pt], ddof=1)) for m_ in mk_f}
            for k in (B7A, B7A_BAG, B7B, B7B_BAG, "B6_gbdt(同B5特征)"):
                if k in models and (k, "dn") in wt:
                    F[k] = pack_model(k)
            say("    " + pad("", 34) + "".join(f"{m_:>9}" for m_ in mk_f))
            for lab, (p0, _) in F.items():
                sd_ = F_sd.get(lab)
                say("    " + pad(lab[:32], 34) + "".join(f"{p0[m_]:>9.3f}" for m_ in mk_f))
                if sd_:
                    say("    " + pad("    (seed 间标准差)", 34) + "".join(f"{sd_[m_]:>9.3f}" for m_ in mk_f))
                for m_ in mk_f:
                    valsF.append(dict(label=lab, metric=m_, value=p0[m_], seed_sd=(sd_ or {}).get(m_, np.nan)))

            def cmp_rows(label, a, b):
                if a not in F or b not in F:
                    return None
                (pa, ba), (pb_, bb) = F[a], F[b]
                res = {}
                for m_ in mk_f:
                    d_ = pa[m_] - pb_[m_]
                    lo, hi = ci_of(ba[m_] - bb[m_])
                    res[m_] = (d_, lo, hi)
                    rowsF.append(dict(comparison=label, a=a, b=b, metric=m_, delta=d_, ci_lo=lo, ci_hi=hi,
                                      a_value=pa[m_], b_value=pb_[m_]))
                say(f"    {pad(label[:46], 48)}" + "".join(f"{res[m_][0]:>+8.3f}{star_of(res[m_][1], res[m_][2]).strip() or ' '}" for m_ in mk_f))
                return res

            say("    Δ(CI 不含0 标 ↑/↓)：" + pad("", 24) + "".join(f"{m_:>9}" for m_ in mk_f))
            RES = {}
            for lab, a, b in (("(i) 集成−B7a_bag(对等集成)", L_ENS, B7A_BAG), ("(j) 单seed均值−B7a(单对单)", L_SGL, B7A),
                              ("    集成−B7b_bag", L_ENS, B7B_BAG), ("    单seed均值−B7b", L_SGL, B7B),
                              ("    集成−B6(参照)", L_ENS, "B6_gbdt(同B5特征)"),
                              ("(k) 集成增益 B7a_bag−B7a", B7A_BAG, B7A), ("    集成增益 B7b_bag−B7b", B7B_BAG, B7B),
                              ("    模型集成增益 集成−单seed均值", L_ENS, L_SGL)):
                RES[lab.strip()] = cmp_rows(lab.strip(), a, b)
            if B7A_BAG not in F:
                say("    (没有 B7a_bag 行：CONFIG['flat_bag'] 关了或没有 sklearn；(i)(k) 跳过，只有 (j))")

            def clear_pos(res, ms=("B_r|TF", "C_APdn", "wTF_dn")):
                return [m_ for m_ in ms if np.isfinite(res[m_][0]) and res[m_][0] >= 0.02 and res[m_][1] > 0]

            def clear_neg(res, ms=("B_r|TF", "C_APdn", "wTF_dn")):
                return [m_ for m_ in ms if np.isfinite(res[m_][0]) and res[m_][0] <= -0.02 and res[m_][2] < 0]

            say("    判读(规则见文件头第14批；\"明显\"=Δ≥0.02 且 CI 下界>0)：")
            for tag_, lab_ in (("(i)", "(i) 集成−B7a_bag(对等集成)"), ("(j)", "(j) 单seed均值−B7a(单对单)")):
                res_ = RES.get(lab_)
                if res_ is None:
                    continue
                ok_ = clear_pos(res_)
                bad_ = clear_neg(res_)
                verdict = ("成立：可写\"结构化 layout 编码器有可测增益\"" if len(ok_) >= 2 else
                           "不成立：不写\"优于平铺特征\"，贡献点=layout 信息 + 孪生训练 + 评估协议")
                say(f"      {tag_} 三项(B_r|TF、C_APdn、wTF_dn)里明显更好的 {len(ok_)} 项 {ok_}；明显更差的 {bad_}  ->  {verdict}")
            rk_ = RES.get("(k) 集成增益 B7a_bag−B7a")
            if rk_ is not None:
                say(f"      (k) B7a_bag−B7a 在 C_APdn {rk_['C_APdn'][0]:+.3f}[{rk_['C_APdn'][1]:+.3f},{rk_['C_APdn'][2]:+.3f}] -> "
                    + ("集成对 GBDT 也有明显增益，(i) 的对等性成立" if rk_["C_APdn"][0] >= 0.02 and rk_["C_APdn"][1] > 0
                       else "集成对 GBDT 增益不明显(<0.02 或 CI 含0)：(i) 里模型的集成优势可能只是集成本身，写论文时要同时报单 seed 成绩"))

            # ---- 2026-10-02d 第15批 v6：训练随机性的合成区间 ‡ / 保守区间 §(定义见文件头第15批第1条)
            if fair_noise:
                t_n = time.time()
                mk_n = [m_ for m_ in mk_f if m_ != "C_mF1"]   # C_mF1 要逐个集成在 val 上调偏置，零分布里不算(Table 6 也不用它)
                K = len(seeds_m)

                def ens_metrics(bp_e, cp_e):
                    mm_ = metrics_of(all_ix, bp_e, cp_e, np.asarray(cp_e).argmax(1))
                    mm_["wTF_dn"], mm_["wTF_up"] = within_tf(all_ix, cp_e[:, 0], 0)[1], within_tf(all_ix, cp_e[:, 2], 2)[1]
                    return {m_: mm_[m_] for m_ in mk_n}

                def rms_of(diffs):
                    a_ = np.asarray(diffs, np.float64)
                    a_ = a_[np.isfinite(a_)]
                    return (float(np.sqrt(np.mean(a_ ** 2))) if len(a_) else np.nan), int(len(a_))

                rm1, rm2, cache_m = {}, {}, {}

                def mens(sub):
                    if sub not in cache_m:
                        cache_m[sub] = ens_metrics(np.mean([seed_pred[s_][0] for s_ in sub], 0),
                                                   np.mean([seed_pred[s_][1] for s_ in sub], 0))
                    return cache_m[sub]

                for kk in (1, 2):
                    subs = list(itertools.combinations(seeds_m, kk))
                    pairs = [(a, b) for a in subs for b in subs if a < b and not set(a) & set(b)]
                    for m_ in mk_n:
                        r_, n_ = rms_of([mens(a)[m_] - mens(b)[m_] for a, b in pairs]) if pairs else (np.nan, 0)
                        (rm1 if kk == 1 else rm2)[m_] = r_
                        null_rowsF.append(dict(source="模型", k=kk, metric=m_, rms=r_, n_pairs=n_, members=str(seeds_m)))
                rg = {}
                n_spl = int(dict(flat_bag or {}).get("null_splits", 60) or 60)
                n_prs = int(dict(flat_bag or {}).get("null_pairs", 20) or 20)
                for tag_ in (B7A, B7B):
                    mem_ = bag_mem.get(tag_)
                    if not mem_ or bag_n < 1 or len(mem_) < 2 * bag_n:
                        continue
                    nm_ = len(mem_)
                    bpt = [np.asarray(x_[0], np.float64)[ord_t] for x_ in mem_]
                    cpt = [np.asarray(x_[1], np.float64)[ord_t] for x_ in mem_]
                    cache_g = {}

                    def gens(sub):
                        if sub not in cache_g:
                            cache_g[sub] = ens_metrics(np.mean([bpt[i_] for i_ in sub], 0), np.mean([cpt[i_] for i_ in sub], 0))
                        return cache_g[sub]

                    rng_n = np.random.default_rng(int(seed) + 7)
                    seen, spl = set(), []
                    for _ in range(20 * n_spl):
                        if len(spl) >= n_spl:
                            break
                        pm = rng_n.permutation(nm_)
                        a_, b_ = tuple(sorted(int(i_) for i_ in pm[:bag_n])), tuple(sorted(int(i_) for i_ in pm[bag_n:2 * bag_n]))
                        key_ = frozenset((a_, b_))
                        if key_ not in seen:
                            seen.add(key_)
                            spl.append((a_, b_))
                    all_pr = list(itertools.combinations(range(nm_), 2))
                    pick = rng_n.choice(len(all_pr), size=min(n_prs, len(all_pr)), replace=False)
                    prs = [((all_pr[i_][0],), (all_pr[i_][1],)) for i_ in sorted(pick)]
                    rN, r1 = {}, {}
                    for m_ in mk_n:
                        rN[m_], nN_ = rms_of([gens(a_)[m_] - gens(b_)[m_] for a_, b_ in spl])
                        r1[m_], n1_ = rms_of([gens(a_)[m_] - gens(b_)[m_] for a_, b_ in prs])
                        null_rowsF.append(dict(source=f"{tag_} bag", k=bag_n, metric=m_, rms=rN[m_], n_pairs=nN_, members=f"{nm_} 个成员"))
                        null_rowsF.append(dict(source=f"{tag_} 单成员", k=1, metric=m_, rms=r1[m_], n_pairs=n1_, members=f"{nm_} 个成员"))
                    rg[tag_] = (rN, r1)
                fb_note = [] if (B7A in rg or B7A_BAG not in F) else ["B7a_bag 没有额外成员，σ 按跟模型集成同量级处理"]
                if B7B_BAG in F and B7B not in rg:
                    fb_note.append("B7b_bag 没有额外成员，σ 按跟模型集成同量级处理")

                def var_of(lab, m_, mode):
                    """该行\"重训一次\"的方差(mode='tot' 对应 ‡，'cons' 对应 §)。"""
                    r2_ = rm2.get(m_, np.nan)
                    if lab == L_ENS:
                        return r2_ ** 2 / K if mode == "tot" else r2_ ** 2 / 2
                    if lab == L_SGL:
                        return rm1.get(m_, np.nan) ** 2 / (2 * K)
                    if lab in (B7A_BAG, B7B_BAG):
                        tg_ = B7A if lab == B7A_BAG else B7B
                        if tg_ in rg:
                            return rg[tg_][0].get(m_, np.nan) ** 2 / 2
                        return r2_ ** 2 / K if mode == "tot" else r2_ ** 2 / 2   # 退路：跟模型集成同量级
                    if lab in (B7A, B7B):
                        if mode == "tot":
                            return 0.0
                        return (rg[lab][1].get(m_, np.nan) ** 2 / 2) if lab in rg else rm1.get(m_, np.nan) ** 2 / 2
                    if lab == "B6_gbdt(同B5特征)":
                        return 0.0
                    return np.nan

                for r_ in rowsF:
                    a_, b_, m_ = r_["a"], r_["b"], r_["metric"]
                    if {a_, b_} == {L_ENS, L_SGL}:   # 两边共用同一批 seed，不独立，不给
                        sd_t = sd_c = np.nan
                    else:
                        sd_t = float(np.sqrt(var_of(a_, m_, "tot") + var_of(b_, m_, "tot")))
                        sd_c = float(np.sqrt(var_of(a_, m_, "cons") + var_of(b_, m_, "cons")))
                    se_ = (r_["ci_hi"] - r_["ci_lo"]) / 3.92 if np.isfinite(r_["ci_lo"]) and np.isfinite(r_["ci_hi"]) else np.nan
                    for pre, sd_ in (("tot", sd_t), ("cons", sd_c)):
                        if np.isfinite(sd_) and np.isfinite(se_):
                            w_ = 1.96 * float(np.sqrt(se_ ** 2 + sd_ ** 2))
                            lo_, hi_ = r_["delta"] - w_, r_["delta"] + w_
                        else:
                            lo_ = hi_ = np.nan
                        r_[f"{pre}_lo"], r_[f"{pre}_hi"] = lo_, hi_
                        r_[f"{pre}_sig"] = bool(np.isfinite(lo_) and (lo_ > 0 or hi_ < 0))
                    r_["noise_sd"], r_["noise_sd_cons"] = sd_t, sd_c
                show_n = ["B_r|TF", "C_APdn", "wTF_dn", "C_APup", "C_AUCdn"]
                say(f"    含训练随机性的区间(‡=合成区间不含0；§=保守区间不含0；文件头第15批第1条；零分布用时 {time.time() - t_n:.0f} 秒)：")
                say("      零分布 rms(两个独立重训之间的差)：" + "  ".join(
                    f"{m_}: 模型k1 {rm1.get(m_, np.nan):.4f}/k2 {rm2.get(m_, np.nan):.4f}"
                    + "".join(f"、{tg_.split('_')[0]} bag{bag_n} {rg[tg_][0][m_]:.4f}/单成员 {rg[tg_][1][m_]:.4f}" for tg_ in rg)
                    for m_ in show_n))
                if fb_note:
                    say("      注意：" + "；".join(fb_note))
                for lab_ in [x_ for x_ in dict.fromkeys(r_["comparison"] for r_ in rowsF)]:
                    q_ = {r_["metric"]: r_ for r_ in rowsF if r_["comparison"] == lab_}
                    if not any(np.isfinite(q_[m_].get("tot_lo", np.nan)) for m_ in show_n if m_ in q_):
                        continue
                    say(f"      {pad(lab_[:30], 32)}" + "  ".join(
                        f"{m_} {q_[m_]['delta']:+.3f}[{q_[m_]['tot_lo']:+.3f},{q_[m_]['tot_hi']:+.3f}]"
                        f"{'‡' if q_[m_]['tot_sig'] else ''}{'§' if q_[m_]['cons_sig'] else ''}"
                        for m_ in show_n if m_ in q_ and np.isfinite(q_[m_].get("tot_lo", np.nan))))

                def clear_tot(lab_):
                    q_ = {r_["metric"]: r_ for r_ in rowsF if r_["comparison"] == lab_}
                    ms_ = ("B_r|TF", "C_APdn", "wTF_dn")
                    good = [m_ for m_ in ms_ if m_ in q_ and q_[m_]["delta"] >= 0.02 and np.isfinite(q_[m_]["tot_lo"]) and q_[m_]["tot_lo"] > 0]
                    good_c = [m_ for m_ in good if np.isfinite(q_[m_]["cons_lo"]) and q_[m_]["cons_lo"] > 0]
                    return good, good_c

                for tag_, lab_ in (("(i‡)", "(i) 集成−B7a_bag(对等集成)"), ("(j‡)", "(j) 单seed均值−B7a(单对单)")):
                    if not any(r_["comparison"] == lab_ for r_ in rowsF):
                        continue
                    g_, gc_ = clear_tot(lab_)
                    var_ = ("Variant A(两项以上，摘要写\"对等集成下仍有增益\"，带上这几项的数字)" if len(g_) >= 2 else
                            "Variant A-lite(只有一项：摘要只点这一项，其余写打平)" if len(g_) == 1 else
                            "Variant B(没有一项：不写任何\"优于平铺特征\")")
                    say(f"      {tag_} 三项(B_r|TF、C_APdn、wTF_dn)里明显‡的 {len(g_)} 项 {g_}(其中 § 也成立 {gc_})  ->  "
                        + (var_ if tag_ == "(i‡)" else ("单模型也有增益" if len(g_) >= 2 else "单模型打平/不稳：正文写\"单模型打平\"，所有\"优于\"都带\"集成后\"")))
    if rowsF:   # 没有数据(关了 [2d] 或导出里没有逐 seed 列)就不写空文件，27 号读不到会提示并跳过 Table 6
        pd.DataFrame(rowsF).to_csv(os.path.join(outdir, "table6_fair_ensemble.csv"), index=False)
        pd.DataFrame(valsF).to_csv(os.path.join(outdir, "table6_values.csv"), index=False)
    if null_rowsF:   # 2026-10-02d 第15批 v6
        pd.DataFrame(null_rowsF).to_csv(os.path.join(outdir, "table6_null.csv"), index=False)
    if tune_rows:    # 2026-10-02d 第15批 v6：B7 的轮数/类别权重选择(Methods 用)
        pd.DataFrame(tune_rows).to_csv(os.path.join(outdir, "table_b7_tuning.csv"), index=False)

    # ----------------------------------------------------------------- 3. 互补性：val 上堆叠
    sb_list = []
    sb0 = next((k for k in ((stack_baseline,) if isinstance(stack_baseline, str) else stack_baseline) if k in models), None)
    if sb0 is not None:
        sb_list.append(sb0)
    sb_list += [k for k in ((stack_extra,) if isinstance(stack_extra, str) else tuple(stack_extra or ())) if k in models and k not in sb_list]
    rowsK = []
    if not sb_list:
        say("\n[3] 堆叠：找不到 CONFIG['stack_baseline'] 里的基线，跳过")
    ybv = yb[ord_v]
    sgv = np.isfinite(ybv)
    for sb in sb_list:
        stk = f"STACK(模型+{sb.split('(')[0]})"
        say(f"\n[3] 互补性：在 val 上把 模型 和 {sb} 的预测堆叠，套到 test(Head B=OLS 只用 val 显著行；Head C=不加权多项逻辑回归，"
            "输入=两者的 log(p_dn/p_ns)、log(p_up/p_ns))；Δ 的 CI 按基因整群配对 bootstrap")
        stack_b = None
        if val_b.get(mdlB) is not None and val_b.get(sb) is not None:
            Xv = np.column_stack([np.ones(len(ybv)), val_b[mdlB], val_b[sb]])
            w = np.linalg.lstsq(Xv[sgv], ybv[sgv], rcond=None)[0]
            stack_b = np.column_stack([np.ones(len(ord_t)), models[mdlB][0], models[sb][0]]) @ w
            say(f"    Head B 堆叠系数(截距, 模型, 基线) = [{w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f}](val 显著行 {int(sgv.sum())})")
        else:
            say("    (val 导出里没有模型的 Head B 预测列，Head B 堆叠跳过)")
        Zv = np.column_stack([log_odds(models[mdlB][2]), log_odds(models[sb][2])])
        Zt = np.column_stack([log_odds(models[mdlB][1]), log_odds(models[sb][1])])
        Zall = np.vstack([Zv, Zt])
        fit_m = np.r_[np.ones(len(Zv), bool), np.zeros(len(Zt), bool)]
        Pall = logit_fit_predict(Zall, np.r_[ycv, yct], fit_m, balanced=False)
        Pv_s, Pt_s = Pall[:len(Zv)], Pall[len(Zv):]
        off_s, _ = tune_offset(Pv_s, ycv, offset_grid)
        say(f"    Head C 堆叠：val 行 {len(Zv)} 上拟合；偏置 {off_s}(同一份 val 上调，macro-F1 略乐观，只看 AUROC/AUPRC)")
        models_k = {mdlB: (models[mdlB][0], models[mdlB][1], pred_c[mdlB]),
                    sb: (models[sb][0], models[sb][1], pred_c[sb]),
                    stk: (stack_b if stack_b is not None else np.full(len(ord_t), np.nan), Pt_s,
                          apply_offset(Pt_s, off_s))}
        mk = {k: metrics_of(all_ix, *v) for k, v in models_k.items()}
        mkb = {k: [metrics_of(b, *v) for b in boots] for k, v in models_k.items()}
        say("    " + pad("", 34) + "".join(f"{k:>10}" for k in keys))
        for k in models_k:
            say("    " + pad(k, 34) + "".join(f"{mk[k][m]:>10.3f}" for m in keys))
        for ref in (sb, mdlB):
            cs = []
            for m in keys:
                d_ = mk[stk][m] - mk[ref][m]
                lo, hi = ci_of([a[m] - b[m] for a, b in zip(mkb[stk], mkb[ref])])
                cs.append(f"{d_:+.3f}" + star_of(lo, hi))
                rowsK.append(dict(stack=stk, reference=ref, metric=m, stack_value=mk[stk][m], ref_value=mk[ref][m],
                                  delta=d_, ci_lo=lo, ci_hi=hi))
            say("      堆叠 − " + pad(ref, 28) + "".join(f"{c:>10}" for c in cs))
        say("    (堆叠 − 基线 显著>0 = 模型带了基线特征之外的信息；堆叠 − 模型 显著>0 = 基线特征里有模型缺的信息)")
    pd.DataFrame(rowsK).to_csv(os.path.join(outdir, "table5_stack.csv"), index=False)

    # ----------------------------------------------------------------- 4. 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
        for j, gs in enumerate(("test 全部", "test 全部(组内中心化)")):
            sub = dA[dA["gene_set"] == gs].reset_index(drop=True)
            ax[0].barh(np.arange(len(sub)) + 0.4 * j, sub["r"].to_numpy(), 0.4,
                       xerr=[np.clip(sub["r"] - sub["ci_lo"], 0, None), np.clip(sub["ci_hi"] - sub["r"], 0, None)],
                       capsize=2, label=["all test genes", "within-group centered"][j])
            ax[0].set_yticks(np.arange(len(sub)) + 0.2)
            ax[0].set_yticklabels([ascii_of(s) for s in sub["model"]], fontsize=7)
        ax[0].set_xlabel("Head A Pearson r (test)")
        ax[0].legend(fontsize=7)
        dt = pd.DataFrame([r_ for r_ in rowsT if not str(r_["model"]).startswith("Δ")])
        names = [k for k in tf_models]
        for j, nm in enumerate(("dn", "up")):
            q = dt[dt["direction"] == nm].set_index("model").reindex(names)
            ax[1].bar(np.arange(len(names)) + 0.4 * j, q["pooled_auc"].to_numpy(), 0.4,
                      yerr=[np.clip(q["pooled_auc"] - q["pooled_ci_lo"], 0, None),
                            np.clip(q["pooled_ci_hi"] - q["pooled_auc"], 0, None)], capsize=2, label=nm)
        ax[1].axhline(0.5, color="k", lw=.6, ls="--")
        ax[1].set_xticks(np.arange(len(names)) + 0.2)
        ax[1].set_xticklabels([ascii_of(s)[:14] for s in names], rotation=25, fontsize=7)
        ax[1].set_title("Head C within-TF AUROC (pooled)")
        ax[1].legend(fontsize=7)
        db = pd.DataFrame(rowsB)
        show = ["B_r", "B_r|TF", "C_APdn"]
        w = 0.8 / len(show)
        names = list(models)
        for i, m in enumerate(show):
            q = db[db["metric"] == m].set_index("model").reindex(names)
            ax[2].bar(np.arange(len(names)) + i * w, q["value"].to_numpy(), w, label=m)
        ax[2].set_xticks(np.arange(len(names)) + 0.4 - w / 2)
        ax[2].set_xticklabels([ascii_of(s)[:14] for s in names], rotation=25, fontsize=7)
        ax[2].legend(fontsize=7)
        ax[2].set_title("Head B/C vs baselines (test)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig", "baselines.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001 —— 画图失败不该让整个脚本白跑
        say(f"    (画图跳过: {e})")

    say("\n判读(第9批 (a)~(d)、第10批 (e)~(g)，预先写下，建议，不是硬规则；详见本脚本文件头)：")
    say("  (a) 模型对 B4 在 B_r|TF、C_APdn 上 |Δ|≥0.02 且 CI 不含0 -> 论文可以写\"深模型相对浅层基线有增量\"；")
    say("  (b) 只赢 B1/B2 而赢不了 B4 -> 如实报告，卖点改成机制/可解释性；")
    say("  (c) B5/B6 明显强于模型 -> \"实测 WT 表达\"这条通道比结构更重要，讨论里要写(跟 Kang 2022 G3 一致)；")
    say("  (d) Head A：模型对 A3/A3k(分块岭回归)、A4/A4k(GBDT)里最强的那条 Δ<0.02 -> 不能宣称 layout 分支对表达预测有贡献；")
    say("  (e) [2b] 模型合并 TF 内 AUROC 比 B3~B7 最强的高 ≥0.03 且 CI 不含0 -> 卖点\"TF 内排序\"可写(对照换成最强基线)；"
        "否则改写卖点、删掉只拿先验比的那句；")
    say("  (f) [3] 堆叠−基线 明显>0 -> 模型带来基线特征之外的信息；堆叠−模型 明显>0 -> 基线特征里有模型缺的信息(讨论/局限)；")
    say("  (g) [2c] 描述性：深模型对 B4 的增量主要在 D∈L_g 层 -> 增量来自\"被耗竭 TF 在不在/在哪\"。")
    say('  (h) 第12批 v4：B7a=同信息(无实测WT表达)的平铺 layout GBDT。模型对 B7a 在 B_r|TF、C_APdn、合并 TF 内 AUROC 上 ≥0.02 且 CI 不含0 -> 可写"结构化 layout 编码器比平铺特征更好"；')
    say('      差在 ±0.02 内 -> 如实写"layout 信息是主因，编码方式没有可测的额外收益"(贡献点从结构改成信息来源 + 孪生训练 + 评估协议)；')
    say('      B7a 明显更强 -> 必须写进讨论并重新评估主张。B7b(带实测WT表达)只是上限型对照。')
    say('  (i)~(l) 第14批 v5：公平集成对照，见 [2d] 与文件头第14批。(i) 模型集成−B7a_bag 在 B_r|TF/C_APdn/TF 内 AUROC(dn) 三项里 ≥2 项明显 -> 可写结构增益；否则不写；')
    say('      (j) 单 seed 均值−B7a 同样判；(k) B7a_bag−B7a 的 C_APdn ≥0.02 才说明集成对 GBDT 也有用；(l) [3] 堆叠(模型+B7b_bag)−B7b_bag 明显>0 -> 可写模型与最强基线互补。')
    say('  (i‡)(j‡) 第15批 v6：同 (i)(j) 三项，但"明显"改用含训练随机性的合成区间 ‡(Δ≥0.02 且下界>0)；(i‡) ≥2 项 -> 摘要 Variant A，恰 1 项 -> Variant A-lite(只点那一项)，0 项 -> Variant B。')
    say(f"\n写出 {outdir}/summary.txt、table1_head_a.csv、table2_head_bc.csv、delta_model_vs_baseline.csv、"
        f"table3_within_tf_auc.csv、table4_strata.csv、table5_stack.csv、table6_fair_ensemble.csv、table6_values.csv、table6_null.csv、"
        f"table_b7_tuning.csv、fig/baselines.png  (用时 {(time.time() - t_all) / 60:.1f} 分钟)")
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as fh:  # 2026-10-01a：用时也写进去
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_paper_baselines(**CONFIG)