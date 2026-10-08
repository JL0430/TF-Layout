# scripts/tflayout/09_torch_dataset.py

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_SHUFFLE_SEED = 20260926  # 2026-09-26b：shuffle_position 的固定随机种子(跟训练 seed 无关，见文件头第5批)
_SHIFT_SEED = 20260927    # 2026-09-27a：shift_position 的固定随机种子(跟训练 seed 无关，见文件头第6批)
_SHIFT_BP = 1000          # 2026-09-27a：shift_position 的整体偏移范围 ±_SHIFT_BP bp(见文件头第6批)
_JITTER_SEED = 20260928   # 2026-09-28a：位置探针 jitter 的固定随机种子(只用于推理，见文件头第7批)


def _length_buckets(sorted_lens, max_chunks: int = 1, per_token: int = 12,
                    chunk_penalty: int = 50000):
    """第三轮提速：给 layout 分支分段。sorted_lens 是升序的每行真实位点数(Python int
    列表)，把它切成至多 max_chunks 段连续区间，最小化
        Σ 段内行数 × (段内最长+1) × (段内最长+1+per_token) + chunk_penalty × 段数
    (+1 是 layout 分支固定拼的空token；(m+1)² 那部分对应注意力和距离偏置，per_token 那部分
    对应逐 token 的线性层，12 是按"距离偏置是 fp32 五维张量、访存为主"粗估的相对权重；
    chunk_penalty 近似多切一段带来的额外 kernel 启动开销。三个常数都是启发式，信心6/10)。
    返回 [(start, end, n_pad), ...]，n_pad=max(段内最长,1)。O(max_chunks·n²) 的小 DP，
    n 是一个 batch 里的基因数或 D 侧行数(几十到一百多)，在 worker 里算，开销可忽略。"""
    n = len(sorted_lens)
    if n == 0:
        return []
    if max_chunks <= 1 or n == 1:  # 不分段(输入不要求有序，所以取 max 而不是最后一个)
        return [(0, n, max(max(int(x) for x in sorted_lens), 1))]

    def cost(s, e):
        m = max(int(sorted_lens[e - 1]), 1) + 1
        return (e - s) * m * (m + per_token) + chunk_penalty

    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(max_chunks + 1)]
    arg = [[0] * (n + 1) for _ in range(max_chunks + 1)]
    best[0][0] = 0
    for c in range(1, max_chunks + 1):
        for e in range(1, n + 1):
            for s0 in range(c - 1, e):
                if best[c - 1][s0] == inf:
                    continue
                v = best[c - 1][s0] + cost(s0, e)
                if v < best[c][e]:
                    best[c][e], arg[c][e] = v, s0
    c_best = min(range(1, max_chunks + 1), key=lambda c: best[c][n])  # 同代价取段数少的
    bounds, e = [], n
    for c in range(c_best, 0, -1):
        s0 = arg[c][e]
        bounds.append((s0, e))
        e = s0
    bounds.reverse()
    return [(s0, e, max(int(sorted_lens[e - 1]), 1)) for s0, e in bounds]


class TFLayoutDataset(Dataset):
    def __init__(self, layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
                head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
                promoter_tokens="out/promoter_token_ids.parquet",
                bpe_tokenizer="out/bpe_tokenizer.json", ctx_mode: str = "legacy",
                ctx_clip: float = 3.0, dense_target: str = None):
        lay = pd.read_parquet(layout)
        lbl = pd.read_parquet(labels)

        # TF 词表只信 lay["tf"]（来自 bigWig 文件名，00→04 反复确认过是干净的178个）。
        # lbl["tf_depleted"] 来自 Excel Table-S3c 的列名，是另一条独立路径解析出来的，
        # 实测发现两边有些名字对不上(大小写/空格之类)——如果直接取并集当词表，会凭空
        # 多出几个"幽灵TF"，而且更严重的是：__getitem__ 里靠 tf2idx 找 D 自己的 token
        # 来删(敲除)，名字对不上就找不到真正的 token，等于"敲了个寂寞"，L_g\D 和 L_g
        # 完全一样，那个 TF 的孪生训练样本就白搭了。这里做大小写/首尾空格归一化去对齐，
        # 归一化后还是对不上的样本直接丢弃并打印出来，不能悄悄留着当"新TF"用。
        lay_tfs = sorted(set(lay["tf"]))
        lay_tfs_upper = {t.upper(): t for t in lay_tfs}
        dep_norm = lbl["tf_depleted"].astype(str).str.strip().str.upper().map(
            lambda u: lay_tfs_upper.get(u))
        n_bad = int(dep_norm.isna().sum())
        if n_bad:
            bad_names = sorted(set(lbl.loc[dep_norm.isna(), "tf_depleted"]))
            print(f"警告：{n_bad} 条样本的 tf_depleted 跟 layout 里的 TF 名对不上"
                  f"(大小写/空格标准化后仍找不到)，已丢弃这些样本，涉及名字: {bad_names}")
        lbl = lbl.assign(tf_depleted=dep_norm).dropna(subset=["tf_depleted"])

        self.tf_list = lay_tfs
        self.tf2idx = {t: i for i, t in enumerate(self.tf_list)}
        self.n_tf = len(self.tf_list)
        self.class2idx = {"down": 0, "ns": 1, "up": 2}
        self._fields = ["pos", "strand", "a", "m", "res"]

        # 每个基因的 L_g，预先按 gene_id 分组存成 dict，训练时 O(1) 取，避免每个
        # __getitem__ 都重新过滤一遍整张长表
        self.layout_by_gene = {}
        for gid, grp in lay.groupby("gene_id", sort=False):
            grp = grp.sort_values("site_pos")
            self.layout_by_gene[gid] = dict(
                tf_idx=np.array([self.tf2idx[t] for t in grp["tf"]], dtype=np.int64),
                pos=grp["site_pos"].to_numpy(np.float32),
                strand=grp["motif_strand"].to_numpy(np.float32),
                a=grp["a"].to_numpy(np.float32),
                m=grp["m"].to_numpy(np.float32),
                res=grp["res_id"].to_numpy(np.float32),
            )
        self._empty_lg = dict(tf_idx=np.zeros(0, np.int64),
                              **{f: np.zeros(0, np.float32) for f in self._fields})

        # ---- cis 分支：gene_id -> BPE token id 数组，13号脚本的产出。WT/D 共用同一份
        # （见文件头说明：DNA序列不随TF耗竭改变，改变的信息已经在layout_wt/layout_d
        # 的差异里）。pad_id/unk_id 现读 bpe_tokenizer.json，不硬编码。
        self.pad_id, self.unk_id = 0, 1
        if os.path.exists(bpe_tokenizer):
            try:
                with open(bpe_tokenizer) as fh:
                    vocab = json.load(fh)["model"]["vocab"]
                self.pad_id, self.unk_id = vocab["[PAD]"], vocab["[UNK]"]
                print(f"cis 分支：从 {bpe_tokenizer} 读到 [PAD] id={self.pad_id}, "
                      f"[UNK] id={self.unk_id}（应该跟13号脚本当时打印的一致，"
                      "麻烦核对一下）")
            except Exception as e:
                print(f"警告：解析 {bpe_tokenizer} 失败({e})，退回未经确认的默认假设 "
                      "pad_id=0/unk_id=1——如果跟实际训练出的 tokenizer 不一致，"
                      "cis 分支的 padding/mask 会全错")
        else:
            print(f"警告：找不到 {bpe_tokenizer}，cis 分支的 pad_id/unk_id 用未经确认"
                  "的默认假设 0/1（先跑13_train_bpe.py）")

        self.gene2tokens = {}
        if os.path.exists(promoter_tokens):
            tok_df = pd.read_parquet(promoter_tokens)
            # 【2026-09-23】np.asarray 在 token_ids 已经是 ndarray 的情况下不会拷贝，
            # 这里的来源是 parquet 列(pyarrow 后端)，拿到的底层 buffer 常常是只读的，
            # 会导致 collate_fn 里 torch.from_numpy(...) 触发
            # "given NumPy array is not writable" 的 UserWarning(真实训练日志里
            # 出现过，09_torch_dataset.py:276)。换成 np.array(强制拷贝一份)就是
            # 可写的了，信心9/10——只是让底层 buffer 可写，不改变任何 token 值。
            self.gene2tokens = {gid: np.array(ids, dtype=np.int64)
                               for gid, ids in zip(tok_df["gene_id"], tok_df["token_ids"])}
            print(f"cis 分支：读到 {len(self.gene2tokens)} 个基因的 promoter token 序列")
        else:
            print(f"警告：找不到 {promoter_tokens}，cis 分支全部退化成占位 [UNK] token"
                  "（先跑11_extract_promoter_seq.py + 13_train_bpe.py）")
        # 找不到对应序列的 gene_id 用长度1的 [UNK] 占位，不用 [PAD]——原因见文件头
        # "接入cis分支"说明：整行全 PAD 会让 CisTransformer 的 softmax 出 NaN。
        self._empty_cis = np.array([self.unk_id], dtype=np.int64)

        n_gene_miss = sum(1 for gid in lbl["gene_id"].unique()
                          if gid not in self.gene2tokens)
        if self.gene2tokens and n_gene_miss:
            print(f"警告：Head B/C 标签里有 {n_gene_miss} 个 gene_id 在 promoter token "
                  "里找不到对应序列，这些样本的 cis 分支会用 [UNK] 占位（正常应该是0，"
                  "如果这个数字很大，说明 gene_id 全集对不上，需要回头查）")

        # Head A：baseline log(TPM)。08脚本第1节还没确认tpm/文件结构之前，这里读不到
        # 文件就全部留 NaN 占位，不阻塞 Dataset 本身能跑通。
        self.gene_split = {}  # 第2批：ctx_mode=relative 去泄漏要用(TF 基因在哪个 split)
        if head_a and os.path.exists(head_a):
            _ha = pd.read_parquet(head_a)
            self.head_a = _ha.iloc[:, 0]
            if "split" in _ha.columns:
                self.gene_split = {str(k).upper(): v for k, v in _ha["split"].items()}
        else:
            self.head_a = pd.Series(dtype=float)
            print("警告：head_a_baseline_logtpm.parquet 不存在，Head A 标签全部是 NaN "
                  "占位，正式训练前需要先跑通08脚本第1节、确认tpm/文件结构")

        # TF名 -> 它自己的系统名(基因ID)，用于把"TF自身在条件D下的表达变化"接到ctx向量上；
        # 映射逻辑跟00/01/02/03保持一致
        sgd_df = pd.read_csv(sgd, sep="\t", header=None, dtype=str, quoting=3)
        keep_mask = sgd_df[3].fillna("").str.match(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$") | \
            (sgd_df[1] == "ORF")
        sg = sgd_df[keep_mask & sgd_df[3].notna()]
        name2sys = {}
        for sysn, ali in zip(sg[3], sg[5]):
            for a_ in str(ali).split("|") if isinstance(ali, str) else []:
                if a_.strip():
                    name2sys.setdefault(a_.strip().upper(), sysn.upper())
        for sysn, std in zip(sg[3], sg[4]):
            if isinstance(std, str) and std.strip():
                name2sys[std.strip().upper()] = sysn.upper()
        for sysn in sg[3]:
            name2sys[sysn.upper()] = sysn.upper()
        self._tf2gene = {t: name2sys.get(t.upper()) for t in self.tf_list
                         if name2sys.get(t.upper())}

        self._lbl_lookup = lbl.set_index(["gene_id", "tf_depleted"])
        self.samples = lbl[["gene_id", "tf_depleted", "log2fc",
                            "direction_3class"]].reset_index(drop=True)

        # 【2026-09-23 性能修复，信心9/10——纯缓存优化，不改变任何数值】_build_ctx(dep_tf)
        # 的输出只取决于 dep_tf，按"数据集里实际出现过的 dep_tf 种类"缓存一次，__getitem__
        # 直接查表(原来每条样本重算，是"GPU 0%利用率"的真实瓶颈之一)。
        # 【2026-09-24 第2批】缓存的构建挪进 set_ctx_mode(见文件头第2批说明)，legacy 模式下
        # 算的是同一个东西(_build_ctx + D位置0)，逐位不变。
        self._fc_sig = {(g, d): float(v) for g, d, v in
                        zip(lbl["gene_id"], lbl["tf_depleted"], lbl["log2fc"]) if pd.notna(v)}
        self._gene2tfidx = {}
        for t, gsys in self._tf2gene.items():
            self._gene2tfidx.setdefault(str(gsys).upper(), []).append(self.tf2idx[t])
        self.ctx_mode = None
        self._ablation = "none"  # 2026-09-26a，见文件头；set_ablation 可切换
        self._layout_alt = None  # 2026-09-26b：no_position/shuffle_position 时的替代 L_g 字典
        self._layout_alt_cache = {}
        self._layout_probe = None  # 2026-09-28a：推理用的位置探针(见文件头第7批)；None=不用
        self._probe = ("none", 0.0)
        self.set_ctx_mode(ctx_mode, ctx_clip)

        # __getitem__ 访问加速(信心9/10，纯访问方式改写，跟 self.samples 逐行
        # 数值完全一致)：self.samples.iloc[idx] 每次都要现场装一个混合dtype的
        # pandas Series，585900条样本×最多30个epoch，这部分装箱开销单独看不大，
        # 但跟上面 ctx 那个大头比起来这个是"顺手一起处理掉"，不是本次瓶颈的主因。
        self._gene_id_arr = self.samples["gene_id"].to_numpy()
        self._tf_depleted_arr = self.samples["tf_depleted"].to_numpy()
        self._log2fc_arr = self.samples["log2fc"].to_numpy(dtype=np.float64)
        self._y_c_idx_arr = self.samples["direction_3class"].map(self.class2idx).to_numpy()
        # 【2026-09-25b 第3批】稠密 log2FC(Head B 辅助目标)，默认全 NaN(=不用)，见文件头
        self._y_bd_arr = np.full(len(self.samples), np.nan, dtype=np.float64)
        self._dense_key = (None, None)
        # 【2026-09-28b 第8批】真实样本行数；set_head_a_all(True) 时伪样本追加在它后面(见文件头第8批)
        self._n_real = len(self.samples)
        self._head_a_all = False
        self._pseudo_cache = None
        self.pseudo_idx = {}
        if dense_target:
            self.set_dense_target(dense_target)

    def __len__(self):
        return len(self.samples)

    def set_ctx_mode(self, mode: str = "legacy", clip: float = 3.0, verbose: bool = True):
        """切换条件编码(见文件头【2026-09-24 第2批】)。只重建两样缓存：WT 向量
        self._ctx_wt_vec 和按 D 缓存的 self._ctx_d_cache，__getitem__ 直接查表。
        同一模式重复调用直接返回。num_workers>0 时要在建 DataLoader 之前调(worker 拿的是
        建 loader 那一刻的数据集副本)——16 号 run_one_seed 就是这么做的。"""
        if mode not in ("legacy", "relative", "marker"):
            raise ValueError(f"ctx_mode 只能是 legacy/relative/marker，收到 {mode}")
        if mode == self.ctx_mode and float(clip) == getattr(self, "_ctx_clip", None):
            return
        n = self.n_tf
        deps = list(self.samples["tf_depleted"].unique())
        cache, stats = {}, {}
        if mode == "legacy":
            wt = np.zeros(n, dtype=np.float32)  # WT条件暂不区分"活性"高低，留0(原写法)
            for dep_tf in deps:
                ctx = self._build_ctx(dep_tf)
                if dep_tf in self.tf2idx:
                    ctx[self.tf2idx[dep_tf]] = 0.0  # 跟原 __getitem__ 里的显式置零逻辑一致
                cache[dep_tf] = ctx
        else:
            wt = np.ones(n, dtype=np.float32)
            usable, masked, unmapped = [], [], []
            for t in self.tf_list:
                g = self._tf2gene.get(t)
                if g is None:
                    unmapped.append(t)
                elif self.gene_split.get(str(g).upper()) == "train":
                    usable.append(t)
                else:
                    masked.append(t)  # val/test 染色体上的(或 split 未知的，保守起见一起屏蔽)
            n_changed = []
            for dep_tf in deps:
                ctx = np.ones(n, dtype=np.float32)
                if mode == "relative":
                    for t in usable:
                        if t == dep_tf:
                            continue
                        v = self._fc_sig.get((self._tf2gene[t], dep_tf))
                        if v is not None and np.isfinite(v):
                            ctx[self.tf2idx[t]] = float(2.0 ** np.clip(v, -clip, clip))
                    n_changed.append(int((ctx != 1.0).sum()))
                if dep_tf in self.tf2idx:
                    ctx[self.tf2idx[dep_tf]] = 0.0  # 真正的"D 显式置0"：WT 那一位是1
                cache[dep_tf] = ctx
            stats = dict(usable=len(usable), masked=len(masked), unmapped=len(unmapped),
                         n_changed=n_changed)
        self._ctx_wt_vec = wt
        self._ctx_d_cache = cache
        self._ctx_self_mask = (mode == "relative")
        self.ctx_mode, self._ctx_clip = mode, float(clip)
        if verbose:
            n_same = sum(1 for v in cache.values() if np.array_equal(v, wt))
            msg = (f"ctx_mode={mode}：{len(cache)} 个被耗竭 TF，ctx_d 跟 ctx_wt 完全相同的 "
                   f"{n_same} 个")
            if mode != "legacy":
                msg += (f"；TF 维度里 {stats['usable']} 个用训练染色体上的自身基因、"
                        f"{stats['masked']} 个(自身基因在 val/test 或 split 未知)恒为1、"
                        f"{stats['unmapped']} 个找不到自身基因恒为1")
            if mode == "relative" and stats["n_changed"]:
                nc = np.asarray(stats["n_changed"])
                msg += (f"；每个 D 条件下有显著变化(≠1)的 TF 维度数 中位 {int(np.median(nc))}、"
                        f"为0的 D {int((nc == 0).sum())} 个(这些 D 只剩 D 位=0 这一个信号)")
            if mode != "legacy" and not self.gene_split:
                msg += ("  ⚠ 没读到 head_a 文件的 split 列，全部 TF 维度都被当成\"split 未知\"\n"
                        "屏蔽成1(relative 退化成 marker)——先确认 08 号产出")
            print(msg)

    def set_ablation(self, mode: str = "none", verbose: bool = True):
        """【2026-09-26a 第4批】切换输入消融(见文件头)：none / no_knockout / no_layout / no_cis；
        【2026-09-26b 第5批】新增 no_position / shuffle_position(只动位点坐标，见文件头第5批)。
        【2026-09-27a 第6批】新增 shift_position(每个基因整体平移，位点两两 Δd 不变，见文件头第6批)。
        只改 __getitem__ 取到的输入；同一模式重复调用直接返回。num_workers>0 时要在建 DataLoader
        之前调(16 号 run_one_seed 就是这么做的)。"""
        modes = ("none", "no_knockout", "no_layout", "no_cis", "no_position", "shuffle_position",
                 "shift_position")
        if mode not in modes:
            raise ValueError(f"ablation 只能是 {'/'.join(modes)}，收到 {mode}")
        if mode == self._ablation:
            return
        if self._layout_probe is not None:  # 2026-09-28a：探针依赖消融底座，换模式就清掉
            if verbose:
                print(f"位置探针 {self._probe} 随 ablation 切换被清除")
            self._layout_probe, self._probe = None, ("none", 0.0)
        stats = None
        if mode in ("no_position", "shuffle_position", "shift_position"):
            if mode not in self._layout_alt_cache:
                alt, n_site, n_moved, offs = {}, 0, 0, []
                for i, gid in enumerate(sorted(self.layout_by_gene, key=str)):
                    lg = self.layout_by_gene[gid]
                    n = len(lg["tf_idx"])
                    if mode == "no_position":
                        new = {k: (np.zeros_like(v) if k == "pos" else v) for k, v in lg.items()}
                        n_moved += int((lg["pos"] != 0).sum())
                    elif mode == "shift_position":  # 2026-09-27a：整体平移同一个整数偏移，顺序和 Δd 都不变
                        rng = np.random.default_rng([_SHIFT_SEED, i])
                        off = int(rng.integers(-_SHIFT_BP, _SHIFT_BP + 1)) if n else 0
                        new = {k: ((v + np.float32(off)).astype(v.dtype) if k == "pos" else v)
                               for k, v in lg.items()}
                        n_moved += int(n) if off != 0 else 0
                        if n:
                            offs.append(off)
                    else:
                        rng = np.random.default_rng([_SHUFFLE_SEED, i])
                        pos_new = lg["pos"][rng.permutation(n)] if n > 1 else lg["pos"].copy()
                        order = np.argsort(pos_new, kind="stable")
                        new = {k: (pos_new[order] if k == "pos" else v[order]) for k, v in lg.items()}
                        n_moved += int((pos_new != lg["pos"]).sum())
                    n_site += n
                    alt[gid] = {k: np.ascontiguousarray(v) for k, v in new.items()}
                st_ = dict(n_gene=len(alt), n_site=n_site, n_moved=n_moved)
                if offs:  # 2026-09-27a：shift_position 的偏移统计(只打印用)
                    st_.update(off_abs_mean=float(np.mean(np.abs(offs))), off_sd=float(np.std(offs)))
                self._layout_alt_cache[mode] = (alt, st_)
            self._layout_alt, stats = self._layout_alt_cache[mode]
        else:
            self._layout_alt = None
        self._ablation = mode
        if verbose:
            msg = f"输入消融 ablation={mode}"
            if stats:
                msg += (f"：{stats['n_gene']} 个基因、{stats['n_site']} 个位点，坐标被改动的位点 {stats['n_moved']} 个"
                        f"({stats['n_moved'] / max(stats['n_site'], 1):.1%}；shuffle 时单位点基因/同坐标位点不变)"
                        + (f"；shift 偏移 |δ| 均值 {stats['off_abs_mean']:.0f}bp、标准差 {stats['off_sd']:.0f}bp"
                           if "off_abs_mean" in stats else ""))
            print(msg)

    def set_position_probe(self, kind: str = "none", value: float = 0.0, verbose: bool = True) -> dict:
        """【2026-09-28a 第7批】推理用的位置探针(见文件头第7批)：kind=none/offset/jitter；offset 的 value=Δ(bp，整数)，
        jitter 的 value=σ(bp，>0)。只能在 ablation=none 或 shift_position 下用。返回统计字典(位点数、实际位移的均值/标准差)。"""
        kinds = ("none", "offset", "jitter")
        if kind not in kinds:
            raise ValueError(f"position probe 只能是 {'/'.join(kinds)}，收到 {kind}")
        value = float(value)
        if kind == "none" or (kind == "offset" and value == 0.0):
            self._layout_probe, self._probe = None, ("none", 0.0)
            if verbose:
                print("位置探针：关闭(L_g 跟当前 ablation 下的原样一致)")
            return dict(kind="none", n_site=0, disp_mean=0.0, disp_sd=0.0)
        if self._ablation not in ("none", "shift_position"):
            raise ValueError(f"位置探针只能在 ablation=none/shift_position 下用，当前是 {self._ablation}")
        if kind == "offset" and value != round(value):
            raise ValueError(f"offset 必须是整数 bp(坐标是整数，保持 float32 精确)，收到 {value}")
        if kind == "jitter" and not value > 0:
            raise ValueError(f"jitter 的 σ 必须 >0，收到 {value}")
        base = self.layout_by_gene if self._ablation == "none" else self._layout_alt
        new_lay, disp, n_site = {}, [], 0
        for i, gid in enumerate(sorted(self.layout_by_gene, key=str)):
            lg = base.get(gid, self._empty_lg)
            n = len(lg["tf_idx"])
            n_site += n
            if n == 0:
                new_lay[gid] = lg
                continue
            if kind == "offset":
                d_ = np.full(n, value, dtype=np.float64)
                new = {k: ((v + np.float32(value)).astype(v.dtype) if k == "pos" else v) for k, v in lg.items()}
            else:
                rng = np.random.default_rng([_JITTER_SEED, i, int(round(value * 10))])
                d_ = np.rint(rng.normal(0.0, value, n))
                pos_new = (lg["pos"].astype(np.float64) + d_).astype(np.float32)
                order = np.argsort(pos_new, kind="stable")
                new = {k: (pos_new[order] if k == "pos" else v[order]) for k, v in lg.items()}
            disp.append(d_)
            new_lay[gid] = {k: np.ascontiguousarray(v) for k, v in new.items()}
        self._layout_probe, self._probe = new_lay, (kind, value)
        dd = np.concatenate(disp) if disp else np.zeros(0)
        st = dict(kind=kind, value=value, n_site=int(n_site),
                  disp_mean=float(dd.mean()) if dd.size else 0.0, disp_sd=float(dd.std()) if dd.size else 0.0)
        if verbose:
            print(f"位置探针 {kind}={value:g}(底座 ablation={self._ablation})：{len(new_lay)} 个基因、{n_site} 个位点，"
                  f"位移均值 {st['disp_mean']:+.1f}bp、标准差 {st['disp_sd']:.1f}bp")
        return st

    def set_dense_target(self, path: str = None, column: str = "log2fc_dense",
                         verbose: bool = True) -> dict:
        """【2026-09-25b 第3批】加载/清空 Head B 的稠密 log2FC 目标(见文件头)。path=None -> 全 NaN。
        返回 dict(n=样本数, n_finite=有稠密值的样本数, frac=覆盖率, by_split={split: 覆盖率})。"""
        key = (os.path.abspath(path) if path else None, column if path else None)
        if key == self._dense_key:
            return getattr(self, "_dense_stats", dict(n=len(self.samples), n_finite=0, frac=0.0))
        arr = np.full(len(self.samples), np.nan, dtype=np.float64)
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到稠密目标文件 {path}(先跑 21_build_dense_target.py)")
            d = pd.read_parquet(path) if not str(path).endswith((".csv", ".csv.gz")) \
                else pd.read_csv(path)
            for c in ("gene_id", "tf_depleted", column):
                if c not in d.columns:
                    raise ValueError(f"{path} 缺列 {c}(现有列: {list(d.columns)})")
            lay_up = {t.upper(): t for t in self.tf_list}
            src = pd.DataFrame({
                "_g": d["gene_id"].astype(str).str.strip().str.upper(),
                "_t": d["tf_depleted"].astype(str).str.strip().str.upper().map(lay_up),
                "_v": pd.to_numeric(d[column], errors="coerce").astype(np.float64)})
            src = src.dropna(subset=["_t"]).drop_duplicates(subset=["_g", "_t"], keep="first")
            src["_t"] = src["_t"].astype(str).str.upper()
            keys = pd.DataFrame({
                "_g": self.samples["gene_id"].astype(str).str.strip().str.upper(),
                "_t": self.samples["tf_depleted"].astype(str).str.strip().str.upper()})
            merged = keys.merge(src, on=["_g", "_t"], how="left", sort=False)
            if len(merged) != len(keys):
                raise RuntimeError("set_dense_target: 合并后行数变了(按键去重没生效？)")
            arr = np.array(merged["_v"].to_numpy(dtype=np.float64), dtype=np.float64)  # 拷贝：to_numpy 可能是只读视图
            arr[getattr(self, "_n_real", len(arr)):] = np.nan  # 2026-09-28b：伪样本行永远没有稠密目标
        self._y_bd_arr = arr
        self._dense_key = key
        _nr = getattr(self, "_n_real", len(arr))  # 2026-09-28b：统计只看真实样本行(打印口径跟以前一样)
        fin = np.isfinite(arr[:_nr])
        by_split = {}
        if self.gene_split and path:
            sp = self.samples["gene_id"].iloc[:_nr].astype(str).str.upper().map(self.gene_split).to_numpy()
            for k in ("train", "val", "test"):
                m = sp == k
                by_split[k] = float(fin[m].mean()) if m.any() else float("nan")
        ns_mask = ~np.isfinite(self._log2fc_arr[:_nr])
        stats = dict(n=int(_nr), n_finite=int(fin.sum()), frac=float(fin.mean()) if _nr else 0.0,
                     frac_ns=float(fin[ns_mask].mean()) if ns_mask.any() else float("nan"),
                     by_split=by_split, path=path, column=column if path else None)
        self._dense_stats = stats
        if verbose:
            if path:
                print(f"稠密目标：{path}[{column}] -> {stats['n_finite']}/{stats['n']} 条样本有值"
                      f"({stats['frac']:.1%}；不显著样本里 {stats['frac_ns']:.1%})；按 split "
                      f"{ {k: round(v, 4) for k, v in by_split.items()} }")
            else:
                print("稠密目标：未加载(y_bd 全是 NaN)")
        return stats

    def set_head_a_all(self, on: bool = False, verbose: bool = True) -> dict:
        """【2026-09-28b 第8批】打开/关闭\"网格外基因的 Head A 伪样本\"(见文件头第8批)。打开：在 self.samples 末尾追加伪样本行，
        self.pseudo_idx={split: 行下标}；关闭：截回 self._n_real 行。同一状态重复调用直接返回。num_workers>0 时要在建
        DataLoader 之前调(16 号 run_one_seed 就是这么做的)。返回统计字典。"""
        on = bool(on)
        if on == self._head_a_all:
            return dict(getattr(self, "_head_a_all_stats", dict(on=False)))
        n0 = self._n_real
        if not on:
            self.samples = self.samples.iloc[:n0]
            self._gene_id_arr = self._gene_id_arr[:n0]
            self._tf_depleted_arr = self._tf_depleted_arr[:n0]
            self._log2fc_arr = self._log2fc_arr[:n0]
            self._y_c_idx_arr = self._y_c_idx_arr[:n0]
            self._y_bd_arr = self._y_bd_arr[:n0]
            self.pseudo_idx = {}
            self._head_a_all = False
            self._head_a_all_stats = dict(on=False)
            if verbose:
                print(f"Head A 全基因伪样本：关闭(样本数回到 {n0})")
            return dict(self._head_a_all_stats)
        if self._pseudo_cache is None:
            grid = set(self.samples["gene_id"].iloc[:n0].tolist())
            deps = sorted(set(self.samples["tf_depleted"].iloc[:n0].tolist()), key=str)
            ha = self.head_a
            genes, splits = [], []
            n_nan, n_nosplit = 0, 0
            for g in sorted(ha.index.tolist(), key=str):
                if g in grid:
                    continue
                v = ha.get(g)
                if v is None or not np.isfinite(float(v)):
                    n_nan += 1
                    continue
                sp = self.gene_split.get(str(g).upper())
                if sp not in ("train", "val", "test"):
                    n_nosplit += 1
                    continue
                genes.append(g)
                splits.append(sp)
            rows_g, rows_d, rows_sp, n_nosite = [], [], [], 0
            n_per = len(deps)
            for g, sp in zip(genes, splits):
                lg = self.layout_by_gene.get(g, self._empty_lg)
                in_lg = set(int(t) for t in lg["tf_idx"].tolist())
                if not in_lg:
                    n_nosite += 1
                cand = [d for d in deps if self.tf2idx.get(d, -1) not in in_lg] or deps
                rows_g += [g] * n_per
                rows_d += [cand[i % len(cand)] for i in range(n_per)]
                rows_sp += [sp] * n_per
            new = pd.DataFrame(dict(gene_id=pd.Series(rows_g, dtype=self.samples["gene_id"].dtype),
                                    tf_depleted=pd.Series(rows_d, dtype=self.samples["tf_depleted"].dtype),
                                    log2fc=np.full(len(rows_g), np.nan, dtype=np.float64),
                                    direction_3class="ignore"))
            sp_arr = np.asarray(rows_sp, dtype=object)
            pidx = {k: (n0 + np.flatnonzero(sp_arr == k)).astype(np.int64) for k in ("train", "val", "test")}
            st = dict(on=True, n_real=int(n0), n_pseudo=int(len(new)), rows_per_gene=int(n_per),
                      n_genes={k: int(sum(1 for x in splits if x == k)) for k in ("train", "val", "test")},
                      n_genes_no_site=int(n_nosite), n_skipped_nan=int(n_nan), n_skipped_nosplit=int(n_nosplit))
            self._pseudo_cache = (new, pidx, st)
        new, pidx, st = self._pseudo_cache
        self.samples = pd.concat([self.samples.iloc[:n0], new], ignore_index=True)
        self._gene_id_arr = np.concatenate([self._gene_id_arr[:n0], new["gene_id"].to_numpy()])
        self._tf_depleted_arr = np.concatenate([self._tf_depleted_arr[:n0], new["tf_depleted"].to_numpy()])
        self._log2fc_arr = np.concatenate([self._log2fc_arr[:n0], np.full(len(new), np.nan)])
        self._y_c_idx_arr = np.concatenate([np.asarray(self._y_c_idx_arr[:n0]).astype(np.int64),
                                            np.full(len(new), -1, dtype=np.int64)])
        self._y_bd_arr = np.concatenate([self._y_bd_arr[:n0], np.full(len(new), np.nan)])
        self.pseudo_idx = {k: v.copy() for k, v in pidx.items()}
        self._head_a_all = True
        self._head_a_all_stats = dict(st)
        if verbose:
            print(f"Head A 全基因伪样本：打开——网格外基因 train/val/test = {st['n_genes']['train']}/{st['n_genes']['val']}/"
                  f"{st['n_genes']['test']}(其中启动子上没有任何位点的 {st['n_genes_no_site']} 个)，每基因 {st['rows_per_gene']} 行，"
                  f"共追加 {st['n_pseudo']} 行(y_b/y_bd=NaN、y_c=-1)；跳过 Head A 标签非有限的 {st['n_skipped_nan']} 个、"
                  f"split 未知的 {st['n_skipped_nosplit']} 个；样本数 {n0} -> {len(self.samples)}")
        return dict(st)

    def _build_ctx(self, depleted_tf):
        """其余TF的"活性代理"：它自己的基因在条件D下的log2FC，查不到填0。
        见文件头说明——这是唯一按默认实现、没有标准答案的部分。"""
        ctx = np.zeros(self.n_tf, dtype=np.float32)
        for t in self.tf_list:
            if t == depleted_tf:
                continue
            gene = self._tf2gene.get(t)
            if gene is None:
                continue
            key = (gene, depleted_tf)
            if key in self._lbl_lookup.index:
                v = self._lbl_lookup.loc[key, "log2fc"]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                ctx[self.tf2idx[t]] = 0.0 if pd.isna(v) else float(v)
        return ctx

    def __getitem__(self, idx):
        gid = self._gene_id_arr[idx]
        dep_tf = self._tf_depleted_arr[idx]

        abl = self._ablation  # 2026-09-26a：none 时下面每一行都跟原来一样
        if abl == "no_layout":
            lg = self._empty_lg
        elif self._layout_probe is not None:  # 2026-09-28a：推理用位置探针(已按当前 ablation 的底座算好)
            lg = self._layout_probe.get(gid, self._empty_lg)
        elif abl in ("no_position", "shuffle_position", "shift_position"):  # 2026-09-26b/27a：只换了 pos 字段的替代 L_g
            lg = self._layout_alt.get(gid, self._empty_lg)
        else:
            lg = self.layout_by_gene.get(gid, self._empty_lg)
        if abl in ("no_knockout", "no_layout"):
            lg_minus_d = lg  # 不删 D 的位点(no_layout 时本来就是空的)；下游只读，共用同一份数组
        else:
            keep = lg["tf_idx"] != self.tf2idx.get(dep_tf, -1)
            lg_minus_d = {k: v[keep] for k, v in lg.items()}

        cis_ids = self._empty_cis if abl == "no_cis" else self.gene2tokens.get(gid, self._empty_cis)

        # ctx_d 现在是查表(见 __init__ 里的 _ctx_d_cache 说明)，不再每条样本重算。
        # .copy() 是防御性的：缓存命中的是同一份 numpy buffer，下游目前只读用
        # (torch.from_numpy 之后马上会在 collate_fn 里被 torch.stack 拷进新张量)，
        # 理论上不需要 copy，但178个float32的复制成本可以忽略不计，换来"以后
        # 谁不小心原地改了这个数组也不会串到其它样本"的安全性，划算。
        ctx_d = self._ctx_d_cache[dep_tf].copy()
        ctx_wt = self._ctx_wt_vec.copy()  # legacy=全0(原写法)；relative/marker=全1
        if self._ctx_self_mask:
            # 第2批去泄漏②：g 自己是某个 TF 的基因时，那一维置1(中性)，D 位保持0
            for k in self._gene2tfidx.get(str(gid).upper(), ()):
                if ctx_d[k] != 0.0:
                    ctx_d[k] = 1.0

        y_b = self._log2fc_arr[idx]
        return dict(
            gene_id=gid, tf_depleted=dep_tf,
            layout_wt=lg, layout_d=lg_minus_d,
            cis_ids=cis_ids,
            ctx_wt=torch.from_numpy(ctx_wt), ctx_d=torch.from_numpy(ctx_d),
            y_a=torch.tensor(float(self.head_a.get(gid, np.nan)), dtype=torch.float32),
            y_b=torch.tensor(float(y_b) if pd.notna(y_b) else float("nan"),
                             dtype=torch.float32),
            y_c=torch.tensor(int(self._y_c_idx_arr[idx]), dtype=torch.long),
            y_bd=torch.tensor(float(self._y_bd_arr[idx]), dtype=torch.float32),  # 第3批，NaN=无
        )

    def collate_fn(self, batch):
        """变长 L_g 补 0 到同一 batch 内最长长度，附带 bool mask 标出哪些位置是真实
        token；空 token(某基因一个位点都没有)按图1b建议要在模型里加一个可学习占位
        向量，这里只负责补0和给mask，占位向量的embedding放到模型代码里实现。
        cis_ids 同样按 batch 内最长长度补 pad_id，不需要额外 mask 字段——
        12_cis_transformer.py 内部会自己用 token_ids==pad_token_id 算 mask，
        只要这里补的 pad_id 跟传给 CisTransformer 构造函数的 pad_token_id 一致就行。
        从 @staticmethod 改成实例方法是因为 pad_cis 需要 self.pad_id；调用方式不变，
        DataLoader(..., collate_fn=ds.collate_fn) 原来就是按绑定方法用的。"""
        fields = ["pos", "strand", "a", "m", "res"]

        def pad(key):
            lens = [len(b[key]["tf_idx"]) for b in batch]
            maxlen = max(max(lens), 1)
            out = {f: torch.zeros(len(batch), maxlen, dtype=torch.float32) for f in fields}
            out["tf_idx"] = torch.zeros(len(batch), maxlen, dtype=torch.long)
            mask = torch.zeros(len(batch), maxlen, dtype=torch.bool)
            for i, b in enumerate(batch):
                n = len(b[key]["tf_idx"])
                if n == 0:
                    continue
                out["tf_idx"][i, :n] = torch.from_numpy(b[key]["tf_idx"])
                for f in fields:
                    out[f][i, :n] = torch.from_numpy(b[key][f])
                mask[i, :n] = True
            out["mask"] = mask
            return out

        def pad_cis():
            lens = [len(b["cis_ids"]) for b in batch]
            maxlen = max(max(lens), 1)
            out = torch.full((len(batch), maxlen), self.pad_id, dtype=torch.long)
            for i, b in enumerate(batch):
                n = len(b["cis_ids"])
                out[i, :n] = torch.from_numpy(b["cis_ids"])
            return out

        return dict(
            gene_id=[b["gene_id"] for b in batch],
            tf_depleted=[b["tf_depleted"] for b in batch],
            layout_wt=pad("layout_wt"),
            layout_d=pad("layout_d"),
            cis_ids=pad_cis(),
            ctx_wt=torch.stack([b["ctx_wt"] for b in batch]),
            ctx_d=torch.stack([b["ctx_d"] for b in batch]),
            y_a=torch.stack([b["y_a"] for b in batch]),
            y_b=torch.stack([b["y_b"] for b in batch]),
            y_c=torch.stack([b["y_c"] for b in batch]),
            # 第3批：有 y_bd 的样本才带这个键(旧式假数据集的样本没有它，行为不变)
            **({"y_bd": torch.stack([b["y_bd"] for b in batch])} if "y_bd" in batch[0] else {}),
        )

    @staticmethod
    def group_collated(full, tf2idx, pad_id, layout_buckets: int = 1):
        """把 collate_fn 的输出(每条样本一行)重组成按基因去重的形状，供 15 号脚本
        SiameseHeadsModel.forward_grouped 使用(见文件头【2026-09-23 第二轮提速】)。
        写成 staticmethod 是为了让 16 号脚本的假数据集、15 号脚本的自检也能直接复用
        同一份分组逻辑，不各自抄一遍。返回字典(G=batch里不同基因数，B=样本数，
        B_d=需要单独算 D 侧 layout 的样本数)：
          cis_ids (G,Lc) / layout_wt 各字段 (G,N) / ctx_wt (G,n_tf)：每个基因一份
          sample_gene (B,) long：第 i 条样本属于第几个基因
          layout_d 各字段 (B_d,N_d) / d_rows (B_d,) long：只含 L_g\\D != L_g 的样本
          ctx_d (B,n_tf) / dep_tf_idx (B,) long / y_a,y_b,y_c(,y_bd) (B,)：逐样本
          gene_id / tf_depleted：逐样本字符串列表，跟 collate_fn 一致
          layout_wt_splits / layout_d_splits：[(起,止,padding长度),...]，第三轮提速新增，
            见文件头；layout_buckets=1 时各只有一段(等于不分段，行为跟之前一样)
        """
        genes = full["gene_id"]
        slot_of, first_rows, sample_gene = {}, [], []
        for i, g in enumerate(genes):
            s = slot_of.get(g)
            if s is None:
                s = len(first_rows)
                slot_of[g] = s
                first_rows.append(i)
            sample_gene.append(s)
        first_rows = torch.tensor(first_rows, dtype=torch.long)
        sample_gene = torch.tensor(sample_gene, dtype=torch.long)

        lw, ld = full["layout_wt"], full["layout_d"]
        need_d = lw["mask"].sum(1) != ld["mask"].sum(1)
        d_rows = torch.nonzero(need_d, as_tuple=False).flatten()

        # ---- 第三轮提速：按 L_g 长度排序基因槽位、按 L_g\D 长度排序 d_rows，再分段 ----
        len_wt = lw["mask"].index_select(0, first_rows).sum(1)
        len_d = ld["mask"].index_select(0, d_rows).sum(1)
        if layout_buckets > 1:
            order = torch.sort(len_wt, stable=True).indices
            new_slot = torch.empty_like(order)
            new_slot[order] = torch.arange(len(order))
            first_rows, len_wt = first_rows[order], len_wt[order]
            sample_gene = new_slot[sample_gene]
            order_d = torch.sort(len_d, stable=True).indices
            d_rows, len_d = d_rows[order_d], len_d[order_d]
        wt_splits = _length_buckets(len_wt.tolist(), layout_buckets)
        d_splits = _length_buckets(len_d.tolist(), layout_buckets)

        def take(lay, rows):
            sub = {k: v.index_select(0, rows) for k, v in lay.items()}
            n = int(sub["mask"].sum(1).max()) if rows.numel() else 0
            n = max(n, 1)  # 跟 collate_fn 一样至少留1列，全空也有合法形状
            return {k: v[:, :n].contiguous() for k, v in sub.items()}

        layout_wt_g = take(lw, first_rows)
        layout_d_sub = take(ld, d_rows)

        cis_full = full["cis_ids"]
        cis_g = cis_full.index_select(0, first_rows)
        n_cis = max(int((cis_g != pad_id).sum(1).max()), 1)
        cis_g = cis_g[:, :n_cis].contiguous()
        ctx_wt_g = full["ctx_wt"].index_select(0, first_rows)

        # ---- 前提校验：同一基因的 WT 侧输入必须逐元素相同，否则分组前向会算错 ----
        if not torch.equal(full["ctx_wt"], ctx_wt_g.index_select(0, sample_gene)):
            raise ValueError("group_collated: 同一基因的 ctx_wt 在 batch 内不一致——分组"
                             "前向假设 WT 条件只取决于基因；如果改过 ctx_wt 的定义，请用 "
                             "16 号脚本 --forward legacy")
        if (not torch.equal(cis_full[:, :n_cis], cis_g.index_select(0, sample_gene))
                or bool((cis_full[:, n_cis:] != pad_id).any())):
            raise ValueError("group_collated: 同一基因的 cis_ids 在 batch 内不一致")
        n_wt = layout_wt_g["mask"].shape[1]
        for k, v in lw.items():
            if (not torch.equal(v[:, :n_wt], layout_wt_g[k].index_select(0, sample_gene))
                    or bool(v[:, n_wt:].any())):
                raise ValueError(f"group_collated: 同一基因的 layout_wt[{k}] 在 batch 内不一致")

        dep_tf_idx = torch.tensor([tf2idx[t] for t in full["tf_depleted"]], dtype=torch.long)
        return dict(
            gene_id=full["gene_id"], tf_depleted=full["tf_depleted"],
            cis_ids=cis_g, layout_wt=layout_wt_g, ctx_wt=ctx_wt_g,
            sample_gene=sample_gene, layout_d=layout_d_sub, d_rows=d_rows,
            ctx_d=full["ctx_d"], dep_tf_idx=dep_tf_idx,
            y_a=full["y_a"], y_b=full["y_b"], y_c=full["y_c"],
            layout_wt_splits=wt_splits, layout_d_splits=d_splits,
            **({"y_bd": full["y_bd"]} if "y_bd" in full else {}),  # 第3批，逐样本
        )

    def collate_grouped(self, batch, layout_buckets: int = 1):
        """DataLoader 用的 collate_fn(分组版)：先走原来的 collate_fn，再重组。
        调用方式跟 collate_fn 一样：DataLoader(..., collate_fn=ds.collate_grouped)；
        要分段就用 functools.partial(ds.collate_grouped, layout_buckets=3)。"""
        return TFLayoutDataset.group_collated(self.collate_fn(batch), self.tf2idx,
                                              self.pad_id, layout_buckets)


def run_check_dataset(layout="out/tf_layout.parquet", labels="out/head_bc_labels.parquet",
                      head_a="out/head_a_baseline_logtpm.parquet", sgd="data/SGD_features.tab",
                      promoter_tokens="out/promoter_token_ids.parquet",
                      bpe_tokenizer="out/bpe_tokenizer.json", batch_size=8,
                      dense_target="out/head_b_dense_target.parquet"):
    ds = TFLayoutDataset(layout, labels, head_a, sgd, promoter_tokens, bpe_tokenizer)
    # 第3批：21 号产出存在就顺手核对一次覆盖率(只打印，最后清空，不影响下面的检查)
    if dense_target and os.path.exists(dense_target):
        st = ds.set_dense_target(dense_target)
        v = ds._y_bd_arr
        sig = np.isfinite(ds._log2fc_arr) & np.isfinite(v)
        if sig.sum() >= 3:
            print(f"  稠密值 vs 显著 log2FC(显著且有稠密值的 {int(sig.sum())} 条)：Pearson r="
                  f"{np.corrcoef(v[sig], ds._log2fc_arr[sig])[0, 1]:.3f}(21 号校准后应≈20 号的0.96以上)")
        item = ds[int(np.flatnonzero(np.isfinite(v))[0])] if st["n_finite"] else ds[0]
        print(f"  __getitem__ 的 y_bd 字段: {float(item['y_bd']):.4f}")
        ds.set_dense_target(None)
    else:
        print(f"(没找到 {dense_target}，跳过稠密目标核对；需要时先跑 21_build_dense_target.py)")
    print(f"样本数(gene×depleted_tf): {len(ds)}；TF 词表大小: {ds.n_tf}")
    # 第2批：三种条件编码各打印一次统计(只重建两份小缓存，很快)，最后切回 legacy
    print(f"head_a 文件 split 列：读到 {len(ds.gene_split)} 个基因的 split"
          f"({dict(pd.Series(list(ds.gene_split.values())).value_counts()) if ds.gene_split else '无'})")
    for _mode in ("relative", "marker", "legacy"):
        ds.set_ctx_mode(_mode)
    _g_tf = [g for g in ds.samples["gene_id"].unique() if str(g).upper() in ds._gene2tfidx]
    print(f"是某个 TF 自身基因、在 relative 模式下会做\"自身标签屏蔽\"的基因 {len(_g_tf)} 个")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)
    batch = next(iter(dl))
    print("一个 batch 的形状：")
    print(f"  layout_wt.tf_idx {tuple(batch['layout_wt']['tf_idx'].shape)}  "
          f"layout_d.tf_idx {tuple(batch['layout_d']['tf_idx'].shape)}")
    print(f"  cis_ids {tuple(batch['cis_ids'].shape)}  "
          f"（非 pad_id={ds.pad_id} 的位置占比: "
          f"{(batch['cis_ids'] != ds.pad_id).float().mean():.2f}）")
    print(f"  ctx_wt {tuple(batch['ctx_wt'].shape)}  ctx_d {tuple(batch['ctx_d'].shape)}")
    print(f"  y_a {tuple(batch['y_a'].shape)}  y_b {tuple(batch['y_b'].shape)}  "
          f"y_c {tuple(batch['y_c'].shape)}")
    print(f"  y_b 里 NaN(非显著，算 masked MSE 时要跳过) 的比例: "
          f"{batch['y_b'].isnan().float().mean():.2f}")
    print("  y_c 类别分布(这个batch)：",
          {k: int((batch["y_c"] == v).sum()) for k, v in ds.class2idx.items()})

    # 分组版 collate(第二轮提速新增)：挑4个基因的全部样本组成一个batch，看去重效果
    rng = np.random.default_rng(0)
    genes4 = rng.choice(ds.samples["gene_id"].unique(), size=4, replace=False)
    idx4 = np.flatnonzero(np.isin(ds._gene_id_arr, genes4))
    g = ds.collate_grouped([ds[int(i)] for i in idx4])
    print(f"分组版 collate：{len(idx4)} 条样本 -> {g['cis_ids'].shape[0]} 个基因"
          f"(cis/WT layout 只各算 {g['cis_ids'].shape[0]} 份)；其中 L_g\\D != L_g、"
          f"需要单独算 D 侧 layout 的样本 {g['d_rows'].numel()} 条")
    print(f"  cis_ids {tuple(g['cis_ids'].shape)}  layout_wt.tf_idx "
          f"{tuple(g['layout_wt']['tf_idx'].shape)}  layout_d.tf_idx "
          f"{tuple(g['layout_d']['tf_idx'].shape)}  sample_gene {tuple(g['sample_gene'].shape)}")
    # 第三轮提速：同一批样本按 L_g 长度分段(至多3段)的切分结果
    g3 = ds.collate_grouped([ds[int(i)] for i in idx4], layout_buckets=3)
    print(f"  分段(layout_buckets=3)：WT 侧 {g3['layout_wt_splits']}  D 侧 "
          f"{g3['layout_d_splits']}  (每段 = (起,止,padding长度))")

    # 第5批(2026-09-26b)：两种位置消融的不变量核对——全部基因逐个比，只读，最后切回 none
    # 第6批(2026-09-27a)：加 shift_position——位点顺序/属性不变、每个基因偏移恒定且 |δ|≤_SHIFT_BP、两两 Δd 逐位不变
    for _m in ("no_position", "shuffle_position", "shift_position"):
        ds.set_ablation(_m)
        bad = []
        for gid, a_ in ds.layout_by_gene.items():
            b_ = ds._layout_alt[gid]
            site_a = sorted(zip(a_["tf_idx"].tolist(), a_["strand"].tolist(), a_["a"].tolist(),
                                a_["m"].tolist(), a_["res"].tolist()))
            site_b = sorted(zip(b_["tf_idx"].tolist(), b_["strand"].tolist(), b_["a"].tolist(),
                                b_["m"].tolist(), b_["res"].tolist()))
            if _m == "no_position":
                pos_ok = bool((b_["pos"] == 0).all())
            elif _m == "shuffle_position":
                pos_ok = np.array_equal(np.sort(a_["pos"]), b_["pos"])
            else:  # shift_position：同一顺序下逐位点差值恒定、在范围内，且相邻间距逐位相同
                d_ = b_["pos"].astype(np.float64) - a_["pos"].astype(np.float64)
                pos_ok = bool(d_.size == 0 or (np.all(d_ == d_[0]) and abs(d_[0]) <= _SHIFT_BP
                                               and np.array_equal(np.diff(a_["pos"]), np.diff(b_["pos"]))
                                               and np.array_equal(a_["tf_idx"], b_["tf_idx"])))
            if site_a != site_b or not pos_ok:
                bad.append(gid)
        # 敲除一致性：挑一条 D∈L_g 的样本，layout_d 应等于"替代 L_g 删掉 D 的 token"
        _hit = None
        for _i in range(len(ds)):
            _g, _d = ds._gene_id_arr[_i], ds._tf_depleted_arr[_i]
            if _g in ds.layout_by_gene and ds.tf2idx.get(_d, -1) in set(ds.layout_by_gene[_g]["tf_idx"].tolist()):
                _hit = _i
                break
        ko_ok = None
        if _hit is not None:
            it = ds[_hit]
            keep = it["layout_wt"]["tf_idx"] != ds.tf2idx[it["tf_depleted"]]
            ko_ok = all(np.array_equal(it["layout_wt"][k][keep], it["layout_d"][k]) for k in it["layout_wt"])
        print(f"  位置消融 {_m}：{len(ds.layout_by_gene)} 个基因里位点属性多重集/坐标不变量不满足的 {len(bad)} 个"
              f"(应为0{'，例: ' + str(bad[:5]) if bad else ''})；敲除一致性 {ko_ok}(应为 True)")
    # 第7批(2026-09-28a)：推理用位置探针的不变量——offset：同序、逐位点差值恒为 Δ；jitter：位点属性多重集不变、位移标准差≈σ；
    # 在 shift_position 底座上也核对一次 offset(24 号拿 shift 模型当对照要用)；最后关掉探针
    for _abl, _kind, _val in (("none", "offset", 100), ("none", "jitter", 50), ("shift_position", "offset", -100)):
        ds.set_ablation(_abl, verbose=False)
        base_ = ds.layout_by_gene if _abl == "none" else ds._layout_alt
        st_ = ds.set_position_probe(_kind, _val, verbose=False)
        bad = []
        for gid, a_ in base_.items():
            b_ = ds._layout_probe[gid]
            if _kind == "offset":
                ok_ = (np.array_equal(a_["tf_idx"], b_["tf_idx"])
                       and np.array_equal(b_["pos"].astype(np.float64) - a_["pos"].astype(np.float64),
                                          np.full(len(a_["pos"]), float(_val))))
            else:
                site_a = sorted(zip(a_["tf_idx"].tolist(), a_["strand"].tolist(), a_["a"].tolist(),
                                    a_["m"].tolist(), a_["res"].tolist()))
                site_b = sorted(zip(b_["tf_idx"].tolist(), b_["strand"].tolist(), b_["a"].tolist(),
                                    b_["m"].tolist(), b_["res"].tolist()))
                ok_ = site_a == site_b and bool(np.all(np.diff(b_["pos"]) >= 0))
            if not ok_:
                bad.append(gid)
        print(f"  位置探针 {_kind}={_val}(底座 {_abl})：不变量不满足的基因 {len(bad)} 个(应为0"
              f"{'，例: ' + str(bad[:5]) if bad else ''})；位移均值 {st_['disp_mean']:+.1f}bp、标准差 {st_['disp_sd']:.1f}bp"
              + (f"(应为 {_val:+d}/0.0)" if _kind == "offset" else f"(应≈0/≈{_val})"))
        ds.set_position_probe("none", verbose=False)
    ds.set_ablation("none")
    # 第8批(2026-09-28b)：Head A 全基因伪样本的不变量——真实行逐位不变、伪基因不在网格里、每个伪基因行数相同、
    # 标签是 NaN/NaN/-1、D 尽量不在 L_g 里、关掉后完全复原
    snap = (ds._gene_id_arr.copy(), ds._tf_depleted_arr.copy(), ds._log2fc_arr.copy(),
            np.asarray(ds._y_c_idx_arr).copy(), ds._y_bd_arr.copy(), len(ds))
    st8 = ds.set_head_a_all(True)
    n0 = ds._n_real
    real_same = (np.array_equal(ds._gene_id_arr[:n0], snap[0]) and np.array_equal(ds._tf_depleted_arr[:n0], snap[1])
                 and np.array_equal(ds._log2fc_arr[:n0], snap[2], equal_nan=True)
                 and np.array_equal(np.asarray(ds._y_c_idx_arr[:n0]).astype(np.int64), snap[3].astype(np.int64)))
    pg = ds._gene_id_arr[n0:]
    grid = set(snap[0].tolist())
    cnt = pd.Series(pg).value_counts()
    tfsets = {g: set(v["tf_idx"].tolist()) for g, v in ds.layout_by_gene.items()}
    d_in = sum(1 for g, d in zip(pg, ds._tf_depleted_arr[n0:]) if ds.tf2idx.get(d, -1) in tfsets.get(g, ()))
    lab_ok = (bool(np.isnan(ds._log2fc_arr[n0:]).all()) and bool(np.isnan(ds._y_bd_arr[n0:]).all())
              and bool((np.asarray(ds._y_c_idx_arr[n0:]) == -1).all()))
    idx_ok = sum(len(v) for v in ds.pseudo_idx.values()) == len(pg) and all(
        len(v) == 0 or (v.min() >= n0 and v.max() < len(ds)) for v in ds.pseudo_idx.values())
    it = ds[int(n0)] if len(pg) else None
    print(f"  Head A 全基因伪样本：真实 {n0} 行逐位不变 {real_same}(应为 True)；伪基因落在网格里的 "
          f"{sum(1 for g in cnt.index if g in grid)} 个(应为0)；每个伪基因行数 {sorted(set(cnt.tolist()))}(应只有一个值=被耗竭TF数)；"
          f"标签 NaN/NaN/-1 {lab_ok}(应为 True)；pseudo_idx 覆盖且越界检查 {idx_ok}；D∈L_g 的伪样本 {d_in} 条"
          f"(只有启动子上几乎所有被耗竭TF都有位点时才可能>0)")
    if it is not None:
        print(f"    伪样本 __getitem__：y_a={float(it['y_a']):.3f}(应为有限值)  y_b={float(it['y_b'])}  y_c={int(it['y_c'])}"
              f"  y_bd={float(it['y_bd'])}")
    ds.set_dense_target(dense_target if dense_target and os.path.exists(dense_target) else None, verbose=False)
    pz = bool(np.isnan(ds._y_bd_arr[n0:]).all())
    ds.set_dense_target(None, verbose=False)
    ds.set_head_a_all(False)
    back = (len(ds) == snap[5] and np.array_equal(ds._gene_id_arr, snap[0])
            and np.array_equal(np.asarray(ds._y_c_idx_arr).astype(np.int64), snap[3].astype(np.int64)))
    print(f"    打开伪样本时加载稠密目标，伪样本行仍全是 NaN: {pz}(应为 True)；关闭后完全复原: {back}(应为 True)；"
          f"网格外基因 train/val/test = {st8['n_genes']}(status 第8节预计 1302/257/288)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="out/tf_layout.parquet")
    ap.add_argument("--labels", default="out/head_bc_labels.parquet")
    ap.add_argument("--head-a", default="out/head_a_baseline_logtpm.parquet")
    ap.add_argument("--sgd", default="data/SGD_features.tab")
    ap.add_argument("--promoter-tokens", default="out/promoter_token_ids.parquet")
    ap.add_argument("--bpe-tokenizer", default="out/bpe_tokenizer.json")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    run_check_dataset(a.layout, a.labels, a.head_a, a.sgd, a.promoter_tokens,
                      a.bpe_tokenizer, a.batch_size)
