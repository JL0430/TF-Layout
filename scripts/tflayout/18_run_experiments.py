# scripts/tflayout/18_run_experiments.py

import datetime
import importlib.util
import json
import os
import sys
import time
import traceback


# ------------------------------------------------------------------------------
PLANS = dict(
    # 主线补满5个seed(3个新seed≈3小时)
    b4a_main=[("v3_ce_marker_dense", (42, 123, 456, 789, 2024))],
    # 输入消融，各2个seed(≈2小时/个，共≈6小时)：回答 layout/cis/位点删除各自贡献多少
    b4a_ablate=[("v4_abl_no_knockout", (42, 123)), ("v4_abl_no_layout", (42, 123)),
                ("v4_abl_no_cis", (42, 123))],
    # 可选(≈4小时)：训练配方——EMA 与 Head C 基因自身倾向，各2个seed
    b4a_model=[("v4_ema", (42, 123)), ("v4_hcgene", (42, 123))],
    # 可选(≈4小时)：稠密目标权重的剂量-反应(0.3/3；第3批只测了1.0)
    b4a_extra=[("v3_ce_marker_dense_lbd3", (42, 123)), ("v3_ce_marker_dense_lbd03", (42, 123))],
    # 可选(≈3小时)：relative 主线补满5个seed，做"上界/对照"(v2_ce 有 42/123，新增 456/789/2024)
    b4a_relative5=[("v2_ce", (42, 123, 456, 789, 2024))],
    # EMA/hcgene 有用之后：组合配置也补到5个seed(把实验名换成胜出的那个)
    b4b_final=[("v4_hcgene_ema", (42, 123, 456, 789, 2024))],
    # ---- 2026-09-26b 第5批(见文件头) ----
    # 位置/间距消融，各2个seed(≈4小时)：本项目核心问题"位置和间距有没有被模型用到"
    b5a_position=[("v5_abl_no_position", (42, 123)), ("v5_abl_shuffle_position", (42, 123))],
    # no_cis 补到5个seed(42/123 复用，新训3个；≈2~3小时，未实测)：决定主线要不要去掉 cis 分支
    b5a_nocis5=[("v4_abl_no_cis", (42, 123, 456, 789, 2024))],
    # 可选(≈2小时)：λ_Bd=0.3，看主线相对 v2_ce 的 Head A 代价能不能收回
    b5a_lbd03=[("v3_ce_marker_dense_lbd03", (42, 123))],
    # ---- 2026-09-27a 第6批(见文件头) ----
    # 绝对位置 vs 相对间距：整体平移(Δd 不变)，2个seed(≈2小时)
    b6a_shift=[("v6_abl_shift_position", (42, 123))],
    # 三个位置消融都补到3个seed(已有的自动跳过；新训 no_position/shuffle 的 456、shift 的 456，≈3小时)
    b6a_pos3=[("v5_abl_no_position", (42, 123, 456)), ("v5_abl_shuffle_position", (42, 123, 456)),
              ("v6_abl_shift_position", (42, 123, 456))],
    # ---- 2026-09-28a 第7批(见文件头) ----
    # Head C 加基因自身倾向项 ψ_Cg(z_wt)，2个seed(≈2小时)；判读规则见文件头第7批
    b7a_hcgene=[("v4_hcgene", (42, 123))],
    # ---- 2026-09-28b 第8批(见文件头) ----
    # Head A 用全部 Head A 基因训练/选模型，2个seed(≈2.5~3小时)；判读规则见文件头第8批
    b8a_headA_all=[("v8_headA_all", (42, 123))],
    # 规则(a)成立之后再跑：补到5个seed(已有的自动跳过，新训 456/789/2024，≈4小时)
    b8b_headA_all5=[("v8_headA_all", (42, 123, 456, 789, 2024))],
    # ---- 2026-09-29a 第9批(见文件头)：就用上面的 b8b_headA_all5(已有 42/123 自动跳过，只新训 456/789/2024) ----
    # ---- 2026-09-30a 第10批(见文件头)：在新主线 v8 上重做三个输入消融，各2个seed(≈6~8小时) ----
    b10a_ablate_v8=[("v10_abl_no_cis", (42, 123)), ("v10_abl_no_layout", (42, 123)),
                    ("v10_abl_no_knockout", (42, 123))],
    # 可选(≈2.5小时)：Head A 权重降到 0.3(Head A 在训练基因上记忆，见文件头第10批(2))
    b10b_lambda_a=[("v10_lambda_a03", (42, 123))],
    # ---- 2026-10-01a 第11批(见文件头)：核心消融补到 5 个 seed(已有的 42/123 自动跳过，新训 3 个≈3.5 小时) ----
    b11a_knockout5=[("v10_abl_no_knockout", (42, 123, 456, 789, 2024))],
    # 定义了、默认不跑(≈3.2 小时)：no_cis 补到 5 个 seed(见文件头第11批(2))
    b11b_nocis5=[("v10_abl_no_cis", (42, 123, 456, 789, 2024))],
    # ---- 2026-10-05 第12批(见文件头)：实测 WT 表达接入 Head B/C ----
    # pilot：2 个 seed(≈2.6 小时，未实测)；先过冒烟和 G0
    b12a_wt_pilot=[("v11_wt_bc", (42, 123))],
    # G0 通过后补到 5 个 seed(42/123 同代码同配置自动复用，只新训 456/789/2024，≈4 小时，未实测)
    b12b_wt_main5=[("v11_wt_bc", (42, 123, 456, 789, 2024))],
    # G1 通过且决定换稿时才跑：论文消融三件套(no_knockout 5 个 seed、no_layout/no_cis 各 2 个，≈12 小时，未实测)
    b12c_wt_ablate=[("v11_abl_no_knockout", (42, 123, 456, 789, 2024)), ("v11_abl_no_layout", (42, 123)),
                    ("v11_abl_no_cis", (42, 123))],
)
# 默认(第11批)：只补 no_knockout(≈3.5 小时)。第10批的 b10a_ablate_v8 + b10b_lambda_a 已跑完(2026-10-01 03:27)。
# 2026-10-05 第12批：默认只跑 pilot；G0 通过后改成 ("b12b_wt_main5",)(已跑的 seed 自动跳过)
ACTIVE_PLAN = ("b12b_wt_main5",)
_PLAN_ITEMS = [x for _n in ACTIVE_PLAN for x in PLANS[_n]]  # 2026-09-27a：CONFIG["plan"] 由它合并而来

# ------------------------------------------------------------------------------
# 配置：全部写死在这里，改完直接运行
# ------------------------------------------------------------------------------
CONFIG = dict(
    paths=dict(layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
               head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
               promoter_tokens="out/promoter_token_ids.parquet",
               bpe_tokenizer="out/bpe_tokenizer.json"),
    ckpt_root="out/checkpoints",
    results_root="out/results",
    dense_target="out/head_b_dense_target.parquet",  # 2026-09-25b：21号产出，v3_*_dense 预设用它
    # 所有实验共用的部分 = run1 的规格和提速配置(status 第7节第一段)
    base=dict(vocab_size=4000, d_model=256, n_heads=8, cis_layers=6, lay_layers=4,
              batch_size=192, lr=1e-4, n_boot=1000, num_workers=None,  # None=按CPU核数估
              amp="bf16", forward_mode="grouped", tfs_per_gene=4, eval_batch_size=512,
              layout_buckets=1, lambda_b=1.0, lambda_c=1.0, lambda_sign=0.1,
              n_epochs=30, patience=5, weight_decay=1e-5),
    experiments=dict(
        # ---- 第2批全部改动(status 8.2.1~2.6)，对应 16 号文件头第12条 ----
        v2_full=dict(
            ctx_mode="relative",          # 09号：WT全1、D位=0、其它TF=2^log2FC、去泄漏
            head_c_mode="delta_tf",       # 15号：Head C 加逐TF偏置 b_C[D]
            dropout=0.2,                  # 0.1 -> 0.2
            loss_b="huber", huber_delta=1.0,  # Head B 重尾
            grad_clip=1.0,
            optimizer="adamw", weight_decay=0.05,
            lr_schedule="warmup_cosine", warmup_steps=500, min_lr_ratio=0.05,
            n_epochs=15,                  # 余弦按15个epoch排；run1最优epoch都≤14
            val_every=0.25, patience=12,  # 每1/4个epoch验证一次，连续12次(=3个epoch)没改善才停
            select_metric="composite", save_per_head_best=True),
        # ---- 逐项消融(v2_full 有提升后再跑) ----
        v2_ctx_legacy=dict(_base="v2_full", ctx_mode="legacy"),
        v2_ctx_marker=dict(_base="v2_full", ctx_mode="marker"),
        v2_headc_delta=dict(_base="v2_full", head_c_mode="delta"),
        v2_headc_gene=dict(_base="v2_full", head_c_mode="delta_tf_gene"),
        v2_mse_noclip=dict(_base="v2_full", loss_b="mse", grad_clip=0.0),
        v2_no_reg=dict(_base="v2_full", optimizer="adam", weight_decay=1e-5, dropout=0.1,
                       lr_schedule="constant", warmup_steps=0),
        v2_select_valloss=dict(_base="v2_full", select_metric="val_loss"),
        # ---- 2026-09-25 第2.5轮(见文件头) ----
        v2_ce=dict(_base="v2_full", gamma=0.0),                  # Head C：focal -> 交叉熵
        v2_cb=dict(_base="v2_ce", beta=0.99999),                 # + 真正起作用的类别权重
        # ---- 2026-09-25b 第3批(见文件头)。dense_target="@dense" 运行时替换成 CONFIG["dense_target"] ----
        v3_ce_marker=dict(_base="v2_ce", ctx_mode="marker"),
        v3_ce_dense=dict(_base="v2_ce", dense_target="@dense", lambda_bd=1.0, dense_on="ns"),
        v3_ce_marker_dense=dict(_base="v3_ce_marker", dense_target="@dense", lambda_bd=1.0,
                                dense_on="ns"),
        v3_ce_marker_dense_lbd03=dict(_base="v3_ce_marker_dense", lambda_bd=0.3),
        v3_ce_marker_dense_lbd3=dict(_base="v3_ce_marker_dense", lambda_bd=3.0),
        # ---- 2026-09-26a 第4批(见文件头)。都以当前主线 v3_ce_marker_dense 为底 ----
        v4_abl_no_knockout=dict(_base="v3_ce_marker_dense", ablation="no_knockout"),  # 不删 D 的位点
        v4_abl_no_layout=dict(_base="v3_ce_marker_dense", ablation="no_layout"),      # L_g 置空
        v4_abl_no_cis=dict(_base="v3_ce_marker_dense", ablation="no_cis"),            # cis 换 [UNK]
        v4_hcgene=dict(_base="v3_ce_marker_dense", head_c_mode="delta_tf_gene"),  # Head C + 基因自身倾向
        v4_ema=dict(_base="v3_ce_marker_dense", ema_decay=0.999),                  # 权重 EMA(16号 26a)
        v4_hcgene_ema=dict(_base="v4_hcgene", ema_decay=0.999),                    # 两者都有用时的组合
        # ---- 2026-09-26b 第5批(见文件头)。位置消融，只改 09 号 L_g 的 pos 字段 ----
        v5_abl_no_position=dict(_base="v3_ce_marker_dense", ablation="no_position"),            # 坐标全置0
        v5_abl_shuffle_position=dict(_base="v3_ce_marker_dense", ablation="shuffle_position"),  # 基因内置换坐标
        # ---- 2026-09-27a 第6批(见文件头)。整体平移坐标：Δd 不变、绝对位置被抹掉 ----
        v6_abl_shift_position=dict(_base="v3_ce_marker_dense", ablation="shift_position"),
        # ---- 2026-09-28b 第8批(见文件头)。网格外基因的 Head A 伪样本并进训练/验证(16 号文件头第17条) ----
        v8_headA_all=dict(_base="v3_ce_marker_dense", head_a_all=True),
        # ---- 2026-09-30a 第10批(见文件头)。以新主线 v8 为底：三个输入消融 + Head A 降权 ----
        v10_abl_no_cis=dict(_base="v8_headA_all", ablation="no_cis"),            # cis 换 [UNK]
        v10_abl_no_layout=dict(_base="v8_headA_all", ablation="no_layout"),      # L_g 置空
        v10_abl_no_knockout=dict(_base="v8_headA_all", ablation="no_knockout"),  # 不删 D 的位点
        v10_lambda_a03=dict(_base="v8_headA_all", lambda_a=0.3),                 # Head A 权重 1.0 -> 0.3
        # ---- 2026-10-05 第12批(见文件头)。实测 WT 表达只进 Head B/C(15 号第4批 wt_input；16 号第18条)。
        #      新实验必须用新名字：同名实验改 wt_input 会被 16 号判"配置不同"，把旧 checkpoint 改名 .bak 再重训 ----
        v11_wt_bc=dict(_base="v8_headA_all", wt_input="head_bc"),
        v11_abl_no_knockout=dict(_base="v11_wt_bc", ablation="no_knockout"),  # 不删 D 的位点
        v11_abl_no_layout=dict(_base="v11_wt_bc", ablation="no_layout"),      # L_g 置空
        v11_abl_no_cis=dict(_base="v11_wt_bc", ablation="no_cis"),            # cis 换 [UNK]
    ),
    # 执行计划：见上面的 PLANS / ACTIVE_PLAN(改 ACTIVE_PLAN 的名字就切计划)。已跑完的 seed 会自动跳过。
    # 2026-09-27a：同一个实验在几个计划里重复出现时合并成一条(seed 取并集、保持首次出现的顺序)，避免重复导出
    plan=[(_n, tuple(dict.fromkeys(_s for _n2, _ss in _PLAN_ITEMS if _n2 == _n for _s in _ss)))
          for _n in dict.fromkeys(_n for _n, _ in _PLAN_ITEMS)],
    plan_name="+".join(ACTIVE_PLAN),
    smoke_first=True,     # 正式跑之前先冒烟(见文件头执行顺序1)
    smoke_steps=300,
    auto_export=True,     # 每个实验训完调 17 号导出
    auto_compare=False,   # 2026-10-05 改 False：19 号 CONFIG 仍指着 v8 + v10_abl_no_knockout 和论文引用的 _compare_b11，自动重跑会重写冻结结果；要对比用 ./run_all.sh --compare
    stop_on_error=True,   # 某个实验出错：True=停下；False=打印报错、继续下一个
)


def run_experiments(paths, ckpt_root, results_root, base, experiments, plan, smoke_first=True,
                    smoke_steps=300, auto_export=True, auto_compare=True, stop_on_error=True,
                    dense_target="out/head_b_dense_target.parquet", plan_name=None):
    """唯一入口，步骤见文件头"执行顺序"。"""
    t_all = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    mods = {}
    for tag, fn in (("16", "16_train_loop.py"), ("17", "17_export_results.py"),
                    ("19", "19_compare_runs.py")):
        name = f"_tflayout_{fn[:-3]}"
        if name in sys.modules:
            mods[tag] = sys.modules[name]
            continue
        path = os.path.join(here, fn)
        if not os.path.exists(path):
            if tag == "16":
                raise SystemExit(f"找不到 {path}")
            print(f"提示：找不到 {fn}，跳过对应步骤")
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m  # 跟 15/16 号 _load_module 同样的注册顺序(torch.compile 需要)
        spec.loader.exec_module(m)
        mods[tag] = m
    tl = mods["16"]
    import torch
    print("=" * 78)
    print(f"对照实验(第10批 2026-09-30a，计划 {plan_name})  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  16号代码版本 "
          f"{tl.CODE_VERSION}  PyTorch {torch.__version__}  "
          f"GPU={'有: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")
    print("=" * 78)

    # ---- 解析预设(_base 继承，最多嵌套5层) ----
    def resolve(name, depth=0):
        if name not in experiments:
            raise SystemExit(f"CONFIG['experiments'] 里没有实验 {name}")
        if depth > 5:
            raise SystemExit(f"实验 {name} 的 _base 嵌套太深(循环引用？)")
        e = dict(experiments[name])
        parent = e.pop("_base", None)
        return dict(resolve(parent, depth + 1), **e) if parent else e

    cfg_base = dict(base)
    if cfg_base.get("num_workers") is None:
        n_cpu = os.cpu_count() or 4
        cfg_base["num_workers"] = 8 if n_cpu > 8 else max(1, n_cpu - 1)
    plan_cfg = []
    for name, seeds in plan:
        c = dict(cfg_base, **resolve(name))
        if c.get("dense_target") == "@dense":  # 2026-09-25b：占位符换成真实路径
            c["dense_target"] = dense_target
        if c.get("lambda_bd") and not (c.get("dense_target") and os.path.exists(c["dense_target"])):
            raise SystemExit(f"实验 {name} 要用稠密目标(λ_Bd={c['lambda_bd']})，但找不到 "
                             f"{c.get('dense_target')}——先跑 python scripts/tflayout/21_build_dense_target.py"
                             "(或 ./run_all.sh --dense-target)")
        plan_cfg.append((name, tuple(int(x) for x in seeds), c))
        diff = {k: v for k, v in c.items() if cfg_base.get(k, "<无>") != v}
        print(f"计划: {name}  seeds={list(seeds)}  -> {os.path.join(ckpt_root, name)}/")
        print(f"      相对 base 的改动: {diff}")
    if not plan_cfg:
        raise SystemExit("CONFIG['plan'] 是空的")

    # ---- Dataset 只建一次 ----
    t0 = time.time()
    ds = tl.TFLayoutDataset(paths["layout"], paths["labels"], paths["head_a"], paths["sgd"],
                            paths["promoter_tokens"], paths["bpe_tokenizer"])
    split_idx = tl.build_split_indices(ds, paths["head_a"])
    print(f"Dataset 建好({time.time() - t0:.0f}秒)：{len(ds)} 条样本、{ds.n_tf} 个TF；"
          "三种 ctx 编码的统计：")
    for mode in ("relative", "marker", "legacy"):
        ds.set_ctx_mode(mode)

    record = dict(started=f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                  code_version=tl.CODE_VERSION, plan=[], smoke=None)

    # ---- 1. 冒烟 ----
    def _missing(name, seeds):  # 这个实验还没有 checkpoint 的 seed
        return [s_ for s_ in seeds
                if not os.path.exists(os.path.join(ckpt_root, name, f"seed{s_}_best.pt"))]

    def _novelty(c):  # 新代码路径越多越值得先冒烟：Head A 全基因 > EMA > 输入消融 > 基因自身倾向 > 稠密目标
        return (4 * bool(c.get("head_a_all")) + 3 * (float(c.get("ema_decay") or 0.0) > 0.0)
                + 2 * (c.get("ablation", "none") != "none")
                + (c.get("head_c_mode") == "delta_tf_gene") + (float(c.get("lambda_bd") or 0.0) > 0.0)
                + 5 * (c.get("wt_input", "none") != "none"))  # 2026-10-05：实测 WT 表达是新代码路径，最该先冒烟

    smoke_cand = [x for x in plan_cfg if _missing(x[0], x[1])]
    if smoke_first and not smoke_cand:
        print("\n所有计划里的 seed 都已有 checkpoint，跳过冒烟(只会做导出/对比)")
    def _sig(c):  # 新代码路径的签名：(输入消融, 是否 EMA, 是否 Head C 基因项, 是否 Head A 全基因[2026-09-28b])
        return (c.get("ablation", "none"), float(c.get("ema_decay") or 0.0) > 0.0,
                c.get("head_c_mode") == "delta_tf_gene", bool(c.get("head_a_all")),
                c.get("wt_input", "none") != "none")

    # 2026-09-26a：还有 seed 要训练的实验里，每种\"新代码路径签名\"各冒烟一次(三个消融各走一遍，no_layout 是整批
    # layout 全空、d_rows 恒空的路径，真实训练里从没整批走过，必须先冒烟)；一种新路径都没有时退回新颖度最高的那个
    smoke_list, _seen = [], set()
    for x in smoke_cand:
        sg = _sig(x[2])
        if sg != ("none", False, False, False, False) and sg not in _seen:
            _seen.add(sg)
            smoke_list.append(x)
    if not smoke_list and smoke_cand:
        smoke_list = [max(smoke_cand, key=lambda x: _novelty(x[2]))]
    record["smoke"] = []
    for name0, seeds0, c0 in (smoke_list if smoke_first else []):
        cs = dict(c0, n_epochs=1, max_steps=int(smoke_steps), n_boot=50)
        print(f"\n===== 冒烟：{name0} 的配置、seed{seeds0[0]}、{smoke_steps} 步(不存 checkpoint) =====")
        t_s = time.time()
        res = tl.run_full_training(**paths, **cs, seeds=(seeds0[0],), save_dir=None, ds=ds,
                                   split_idx=split_idx, resume=False)
        r0 = res["main_results"][0]
        h = r0["history"]
        ok = bool(h) and all(v == v for v in (h[-1]["train_loss"], h[-1]["val_loss"],
                                              r0["test_eval"]["loss"]))
        tr_sec = sum(x["train_sec"] for x in h)
        per_step = tr_sec / max(h[-1]["step"], 1) if h else float("nan")
        n_steps_epoch = h[-1]["step"] / max(h[-1]["epoch_progress"], 1e-9) if h else float("nan")
        est_ep = per_step * n_steps_epoch / 60
        print(f"冒烟{'通过' if ok else '失败(loss 出现 NaN)'}({name0})：用时 {time.time() - t_s:.0f} 秒；"
              f"训练每步约 {per_step * 1000:.0f} ms，每 epoch {n_steps_epoch:.0f} 步 ≈ "
              f"{est_ep:.1f} 分钟(不含验证)；验证一次 {h[-1]['val_sec']:.0f} 秒")
        record["smoke"].append(dict(name=name0, ok=ok, sec_per_step=per_step,
                                    steps_per_epoch=n_steps_epoch, est_min_per_epoch=est_ep))
        if not ok:
            raise SystemExit("冒烟没通过，先把上面的日志贴回来")

    # ---- 2~3. 逐个实验训练 + 导出 ----
    for name, seeds, c in plan_cfg:
        save_dir = os.path.join(ckpt_root, name)
        outdir = os.path.join(results_root, name)
        print(f"\n===== 实验 {name}  seeds={list(seeds)}  checkpoint -> {save_dir}/ =====")
        t_e = time.time()
        entry = dict(name=name, seeds=list(seeds), config=c, save_dir=save_dir, outdir=outdir)
        try:
            res = tl.run_full_training(**paths, **c, seeds=seeds, save_dir=save_dir, ds=ds,
                                       split_idx=split_idx, resume=True)
            entry["train_min"] = (time.time() - t_e) / 60
            entry["per_seed"] = []
            for r in res["main_results"]:
                te, ph = r["test_eval"], r.get("test_eval_per_head") or {}
                entry["per_seed"].append(dict(
                    seed=r["seed"], best=r.get("best"),
                    **{f"test_{k}": te.get(k) for k in ("r_a_gene", "r_b", "macro_f1_c",
                                                         "auprc_down", "auprc_up", "r_bd",
                                                         "r_bd_ns")},
                    **{f"test_ph_{k}": ph.get(k) for k in ("r_a_gene", "r_b", "macro_f1_c",
                                                            "auprc_down", "auprc_up")}))
        except Exception as e:  # noqa: BLE001 —— 按 stop_on_error 决定停不停
            entry["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            record["plan"].append(entry)
            if stop_on_error:
                raise
            continue
        if auto_export and "17" in mods:
            try:
                ex = mods["17"]
                ekw = {k: ex.CONFIG[k] for k in ("layout", "labels", "head_a", "sgd",
                                                 "promoter_tokens", "bpe_tokenizer", "device",
                                                 "num_workers", "eval_batch_size", "n_boot",
                                                 "min_sig_per_tf", "offset_grid",
                                                 "write_csv_gz", "make_figures")}
                ekw["dense_target"] = ex.CONFIG.get("dense_target", dense_target)
                ekw.update({k: paths[k] for k in paths})
                ex.run_export(run_name=name, ckpt_dir=save_dir, outdir=outdir, ds=ds, **ekw)
                entry["exported"] = True
            except Exception as e:  # noqa: BLE001 —— 导出失败不影响已存好的 checkpoint
                entry["export_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                print(f"⚠ {name} 导出失败(checkpoint 已存好，可以之后单独跑 17 号)")
        record["plan"].append(entry)

    # ---- 4. 对比 ----
    if auto_compare and "19" in mods:
        try:
            cmp_cfg = dict(mods["19"].CONFIG, results_root=results_root)
            mods["19"].run_compare(**cmp_cfg)
        except Exception as e:  # noqa: BLE001
            record["compare_error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    # ---- 5. 摘要 ----
    print(f"\n===== 全部结束，总用时 {(time.time() - t_all) / 60:.1f} 分钟 =====")
    for e in record["plan"]:
        if "error" in e:
            print(f"  {e['name']}: ✗ {e['error']}")
            continue
        print(f"  {e['name']}: 训练 {e.get('train_min', float('nan')):.0f} 分钟"
              f"{'，已导出到 ' + e['outdir'] if e.get('exported') else ''}")
        for r in e.get("per_seed", []):
            def f(v):
                return "nan" if v is None or v != v else f"{v:.4f}"
            print(f"    seed{r['seed']}: 选中权重 r_a_gene={f(r['test_r_a_gene'])} "
                  f"r_b={f(r['test_r_b'])} F1={f(r['test_macro_f1_c'])} AUPRC dn/up="
                  f"{f(r['test_auprc_down'])}/{f(r['test_auprc_up'])}"
                  + (f" r_bd={f(r.get('test_r_bd'))}/不显著{f(r.get('test_r_bd_ns'))}"
                     if r.get("test_r_bd") is not None and r.get("test_r_bd") == r.get("test_r_bd") else "")
                  + (f" | 各头最优 r_a_gene={f(r['test_ph_r_a_gene'])} r_b={f(r['test_ph_r_b'])} "
                     f"F1={f(r['test_ph_macro_f1_c'])}" if r.get("test_ph_r_b") is not None
                     else ""))
    cmp_out = (mods["19"].CONFIG.get("outdir") if "19" in mods else None) or os.path.join(results_root, "_compare")
    print("  (macro-F1 这里是 argmax 口径；调过偏置的同 seed 公平对比看 19 号的 "
          f"{os.path.join(cmp_out, 'summary.txt')})")
    os.makedirs(os.path.join(results_root, "_experiments"), exist_ok=True)
    rec_path = os.path.join(results_root, "_experiments",
                            f"{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  本次配置+结果摘要 -> {rec_path}")
    return record


if __name__ == "__main__":
    run_experiments(**CONFIG)