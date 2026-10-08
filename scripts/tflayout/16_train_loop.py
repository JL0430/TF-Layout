# scripts/tflayout/16_train_loop.py

import argparse
import contextlib
import functools
import gc
import importlib.util
import logging
import math
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


def _load_module(name: str, filename: str):
    """按文件路径动态加载模块，原因和实现跟15_siamese_heads.py里的同名函数
    一致(10/12/14/09/15号文件名都以数字开头，不是合法Python标识符)。
    【2026-09-23e修复，见文件头(d)条】exec_module前必须把module注册进
    sys.modules[name]，否则torch.compile(TorchDynamo)追踪到15号脚本(以及它
    间接加载的10/12/14号脚本)里定义的函数、要解析全局变量时会报
    ModuleNotFoundError——这正是这一版要修的真实bug，详细推理见15号脚本文件头。
    顺手加"已加载过就直接返回缓存"的判断，避免同一个name被exec_module两次。"""
    if name in sys.modules:
        return sys.modules[name]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ds_mod = _load_module("_tflayout_09_torch_dataset", "09_torch_dataset.py")
_heads_mod = _load_module("_tflayout_15_siamese_heads", "15_siamese_heads.py")
TFLayoutDataset = _ds_mod.TFLayoutDataset
SiameseHeadsModel = _heads_mod.SiameseHeadsModel
compute_total_loss = _heads_mod.compute_total_loss
DEFAULT_CLASS_COUNTS_APPROX = _heads_mod.DEFAULT_CLASS_COUNTS_APPROX
_pearson_r = _heads_mod._pearson_r


def _default_class_counts() -> torch.Tensor:
    return torch.tensor([DEFAULT_CLASS_COUNTS_APPROX["down"],
                         DEFAULT_CLASS_COUNTS_APPROX["ns"],
                         DEFAULT_CLASS_COUNTS_APPROX["up"]])


def _dep_tf_idx(batch: dict, tf2idx: dict, device) -> torch.Tensor:
    return torch.tensor([tf2idx[t] for t in batch["tf_depleted"]], dtype=torch.long,
                        device=device)


CODE_VERSION = "2026-10-05a"  # 写进checkpoint，run_all.sh据此判断旧checkpoint能不能跳过
# 【2026-09-25a】训练数学定义跟这些旧版本相同(本版只加了默认值=原行为的参数和诊断)，它们存的
# checkpoint 在配置一致时照样可以断点续跑复用(18号把 v2_full 从2个seed补到5个时不会重训42/123)
# 【2026-09-25b】同理：25b 只加了默认值=原行为的稠密辅助目标(文件头第14条)，25a 的 checkpoint 也兼容
# 【2026-09-26a】同理：26a 只加了默认关的 EMA(文件头第15条)，25b 的 checkpoint(第3批三个实验的6个seed)
# 在配置一致时照样复用
# 【2026-09-28b】同理：28b 只加了默认关的 head_a_all(文件头第17条)，26a 的全部 checkpoint 在配置一致时照样复用
# 【2026-10-05a】同理：05a 只加了默认 none 的 wt_input(文件头第18条)，28b 的全部 checkpoint(v8/v10 等)在配置一致时照样复用
_COMPATIBLE_CODE_VERSIONS = ("2026-09-24a", "2026-09-25a", "2026-09-25b", "2026-09-26a", "2026-09-28b")
# 旧 checkpoint 的 train_config 里没有的新参数，按这里的默认值参与"配置是否一致"的比较
_NEW_KEY_DEFAULTS = dict(min_delta=1e-6, lambda_bd=0.0, dense_on="ns", dense_target=None,
                         dense_delta=None, dense_target_digest=None, ema_decay=0.0,
                         ablation="none", head_a_all=False, wt_input="none")
# 【2026-09-23g】上一版docstring(第11条e)写"升到2026-09-23f"但这行没跟着改，版本号
# 卡在"e"没动——是个真bug(不影响checkpoint复用判断，因为compile相关参数都在
# _CONFIG_KEYS_NOT_AFFECTING_RESULT里，但版本号本身应该准确)，这次一并修上，见文件头
# 第11条(f)。
# 【2026-09-24a 第2批】训练侧改进(见文件头第12条)，checkpoint 新增字段；默认参数下训练行为
# 跟 g 版相同，但 train_config 多了新键，g 版的 checkpoint 不会被判为可复用(本来也不该复用：
# 第2批的实验都存在 out/checkpoints/<实验名>/ 下，不会碰 run1 那5个文件)。


def _to_device(obj, device):
    """张量/嵌套字典整体搬到device(pin_memory配合non_blocking异步拷贝)，其余原样返回。"""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _autocast(device, amp: str):
    """amp='bf16' 且在支持 bf16 的 CUDA 上 -> bf16 autocast；其余情况(CPU、amp='off'、
    GPU 不支持 bf16) -> 空上下文，也就是纯 fp32，跟原来一样。"""
    if (amp == "bf16" and str(device).startswith("cuda") and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _forward_batch(model, batch, tf2idx, device, forward_mode: str = "legacy"):
    """forward_mode='legacy'：原来的逐样本前向(batch 来自 ds.collate_fn)；
    'grouped'：分组前向(batch 来自 ds.collate_grouped，见文件头第9条(a))。"""
    # 2026-10-05a(文件头第18条)：模型 wt_input!="none" 时，把实测 WT 表达(= batch["y_a"])只传给 Head B/C；
    # 默认 none 时 wt_expr=None，跟原来逐位相同。getattr 兼容 torch.compile 包装(属性查找转发到原模型)
    wt_expr = None
    if getattr(model, "wt_input", "none") != "none":
        wt_expr = batch["y_a"].to(device, non_blocking=True)
    if forward_mode == "grouped":
        g = _to_device({k: batch[k] for k in ("cis_ids", "layout_wt", "layout_d",
                                              "sample_gene", "d_rows", "ctx_wt",
                                              "ctx_d", "dep_tf_idx")}, device)
        return model.forward_grouped(g["cis_ids"], g["layout_wt"], g["layout_d"],
                                     g["sample_gene"], g["d_rows"], g["ctx_wt"],
                                     g["ctx_d"], g["dep_tf_idx"],
                                     batch.get("layout_wt_splits"),
                                     batch.get("layout_d_splits"), wt_expr=wt_expr)
    dep_idx = _dep_tf_idx(batch, tf2idx, device)
    return model(_to_device(batch["layout_wt"], device),
                _to_device(batch["layout_d"], device),
                batch["cis_ids"].to(device, non_blocking=True),
                batch["ctx_wt"].to(device, non_blocking=True),
                batch["ctx_d"].to(device, non_blocking=True), dep_idx, wt_expr=wt_expr)


class GeneGroupedBatchSampler:
    """训练用 batch_sampler(DataLoader 的协议接口：__iter__ 产出下标列表、__len__)。
    每个 epoch：每个基因的样本先各自打乱、切成 K 条一块，所有块再整体打乱，每
    batch_size//K 块拼成一个 batch。每条样本每个 epoch 恰好出现一次；K=1 时等价于
    原来的 shuffle=True(每条样本独立打乱)。每次 __iter__ 用 (seed, 第几次迭代) 做随机
    种子，同一 seed 可复现、不同 epoch 顺序不同。见文件头第9条(c)，信心6/10。"""

    def __init__(self, gene_of_sample, indices, batch_size: int, tfs_per_gene: int = 1,
                 seed: int = 0):
        idx = np.asarray(indices, dtype=np.int64)
        groups = {}
        for i, g in zip(idx.tolist(), np.asarray(gene_of_sample)[idx].tolist()):
            groups.setdefault(g, []).append(i)
        self.groups = [np.asarray(v, dtype=np.int64) for v in groups.values()]
        self.k = max(1, int(tfs_per_gene))
        self.chunks_per_batch = max(1, int(batch_size) // self.k)
        self.seed = int(seed)
        self._n_iter = 0
        n_chunks = sum(-(-len(g) // self.k) for g in self.groups)
        self._len = -(-n_chunks // self.chunks_per_batch)

    def __len__(self):
        return self._len

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self._n_iter])
        self._n_iter += 1
        chunks = []
        for g in self.groups:
            p = rng.permutation(g)
            chunks.extend(p[s:s + self.k] for s in range(0, len(p), self.k))
        order = rng.permutation(len(chunks))
        cpb = self.chunks_per_batch
        for b in range(0, len(order), cpb):
            yield np.concatenate([chunks[j] for j in order[b:b + cpb]]).tolist()


def _make_loader(ds, idx, kind: str, forward_mode: str, batch_size: int,
                 tfs_per_gene: int, eval_batch_size: int, seed: int, num_workers: int,
                 device, persistent: bool, layout_buckets: int = 1):
    """统一造 DataLoader，返回 (loader, batches)。
    kind='train'：legacy 跟原来一模一样(Subset+shuffle=True)；grouped 用
      GeneGroupedBatchSampler。batches=None。
    kind='eval'：batches 是预先定好的下标列表的列表(evaluate 要靠它把输出放回 idx
      顺序)——legacy 按 idx 顺序每 batch_size 条一块(跟原来 Subset+shuffle=False
      完全一样)；grouped 按基因整组打包(同一基因的全部样本进同一个 batch，凑满
      eval_batch_size 条左右换下一个 batch，基因顺序=在 idx 里首次出现的顺序)。
    layout_buckets(第三轮提速)：分组 collate 时 layout 分支按长度分几段，见 09 号文件头。"""
    pin = str(device).startswith("cuda")
    kw = dict(num_workers=num_workers, pin_memory=pin,
              persistent_workers=(persistent and num_workers > 0))
    collate = (functools.partial(ds.collate_grouped, layout_buckets=int(layout_buckets))
               if forward_mode == "grouped" else ds.collate_fn)
    if kind == "train":
        if forward_mode == "legacy":
            return DataLoader(Subset(ds, list(idx)), batch_size=batch_size, shuffle=True,
                              collate_fn=collate, **kw), None
        sampler = GeneGroupedBatchSampler(ds.samples["gene_id"].to_numpy(), idx,
                                          batch_size, tfs_per_gene, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, **kw), None
    idx = list(idx)
    if forward_mode == "grouped":
        genes = ds.samples["gene_id"].to_numpy()[np.asarray(idx, dtype=np.int64)]
        groups = {}
        for i, g in zip(idx, genes.tolist()):
            groups.setdefault(g, []).append(i)
        batches, cur = [], []
        for grp in groups.values():
            if cur and len(cur) + len(grp) > eval_batch_size:
                batches.append(cur)
                cur = []
            cur = cur + grp
        if cur:
            batches.append(cur)
    else:
        batches = [idx[s:s + batch_size] for s in range(0, len(idx), batch_size)]
    return DataLoader(ds, batch_sampler=batches, collate_fn=collate, **kw), batches


def _make_optimizer(model, lr: float, weight_decay: float, device, kind: str = "adam"):
    """Adam；有 CUDA 时用 fused 实现(同一个算法，一次 kernel 更新全部参数)，老版本
    PyTorch 不支持 fused 参数时退回普通实现。
    【2026-09-24 第2批】kind="adamw"：解耦权重衰减(Loshchilov & Hutter 2019)，按常见做法只对
    ≥2 维的参数(线性层权重、embedding 表)做衰减，偏置/LayerNorm/门控 c0/池化 query 这类
    1 维或标量参数不衰减。默认 kind="adam" 跟原来完全一样(Adam 的 weight_decay 是 L2 耦合在
    梯度里的，run1 用的 1e-5 基本不起作用)。"""
    fused = str(device).startswith("cuda")
    if kind == "adamw":
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if p.requires_grad:
                (decay if p.ndim >= 2 else no_decay).append(p)
        groups = [dict(params=decay, weight_decay=weight_decay),
                  dict(params=no_decay, weight_decay=0.0)]
        try:
            return torch.optim.AdamW(groups, lr=lr, fused=fused)
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(groups, lr=lr)
    if kind != "adam":
        raise ValueError(f"optimizer 只能是 adam/adamw，收到 {kind}")
    try:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _make_scheduler(opt, schedule: str, warmup_steps: int, total_steps: int,
                    min_lr_ratio: float = 0.05):
    """【2026-09-24 第2批】学习率调度，每个优化步调一次。schedule="constant" 且
    warmup_steps=0 时返回 None(=原来的恒定学习率，逐位不变)；"constant"+warmup>0 = 线性热身
    后恒定；"warmup_cosine" = 线性热身 warmup_steps 步，之后按余弦从 1 降到 min_lr_ratio(到
    total_steps 为止，total_steps = n_epochs×每epoch步数，早停时自然停在中途)。"""
    if schedule not in ("constant", "warmup_cosine"):
        raise ValueError(f"lr_schedule 只能是 constant/warmup_cosine，收到 {schedule}")
    warmup_steps = max(0, int(warmup_steps))
    if schedule == "constant" and warmup_steps == 0:
        return None
    span = max(1, int(total_steps) - warmup_steps)

    def lam(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        prog = min(1.0, (step - warmup_steps) / span)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lam)


class _CompileWarmupTimeout(RuntimeError):
    """【2026-09-23f新增，见文件头第11条(e)】bench ⑧节热身阶段累计用时超过
    --compile-warmup-timeout 时主动抛出，让调用方按"这组编译对照跑不动"处理、
    跳过继续，而不需要用户手动Ctrl+Z。只在 run_benchmark 里使用，不影响训练
    路径(训练路径靠 cache_size_limit 兜底，见 _maybe_compile_model)。"""


_SYMPY_INTERP_LOGGER_SILENCED = False


def _silence_noisy_dynamo_logging():
    """【2026-09-23f新增，纯日志降噪，不影响任何计算，信心9/10】用户这次贴回来的
    bench_--compile.txt 里，`torch/utils/_sympy/interp.py` 这个 logger 打印的
    `pow_by_natural(...)` WARNING 连续刷了几十行，把真正需要盯着看的报错/结果
    行淹没掉。这里只把这一个 logger 调到 ERROR 级别，不碰 torch._dynamo/
    torch._inductor 其它 logger(那些可能还有有用的编译诊断信息)。只做一次，
    避免重复设置；如果这台机器上 PyTorch 版本不同、logger 名字对不上，这里
    只是不生效，不影响功能，也不会报错(logging.getLogger对不存在的名字也是
    合法调用，只是没有已有handler去消费它)。"""
    global _SYMPY_INTERP_LOGGER_SILENCED
    if _SYMPY_INTERP_LOGGER_SILENCED:
        return
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    _SYMPY_INTERP_LOGGER_SILENCED = True


def _resolve_compile_dynamic(mode: str):
    """【2026-09-23f新增，见文件头第11条(e)③】把命令行/训练函数里的字符串
    compile_dynamic 转成 torch.compile 的 dynamic 实参：'auto'(默认) -> None
    (Dynamo自动判断哪些维度要动态化)，'true' -> True(上一轮的写法，从第一次
    调用起把全部维度都当符号量，这次bench_--compile.txt里观察到反复重新编译
    的就是这个模式)。real/smoke模式默认用'auto'；如果下一次`--mode bench
    --compile`里 dynamic=True 那组反而更快、也没有触发重编译问题，可以用
    `--compile-dynamic true`切回去，见run_benchmark ⑧节末尾的建议行。不认识
    的字符串按'auto'处理并打印一行提示，不报错。"""
    if mode == "true":
        return True
    if mode != "auto":
        print(f"提示：compile_dynamic 传了未识别的值 {mode!r}，按 'auto' 处理"
              "(可选值只有 auto/true)")
    return None


def _maybe_compile_model(model, compile_model: bool, tag: str = "",
                         cache_size_limit: int = 16, dynamic=None,
                         scope: str = "full"):
    """【2026-09-23d 第四轮新增，见文件头第11条(b)；2026-09-23f 本次(Claude改)
    在原基础上加安全网，见文件头第11条(e)；2026-09-23g 新增 scope 参数，见文件头
    第11条(f)】compile_model=True 时用 torch.compile 包模型的一部分。直接对
    已绑定方法赋值，而不是 `model = torch.compile(model)`：forward_grouped
    是绕开 nn.Module.__call__ 的自定义入口方法，`torch.compile(model)` 包出来的
    壳子只拦截 `model(...)` 这种调用方式，`model.forward_grouped(...)` 会绕过它
    命中未编译的原始方法，包了等于没包。torch.compile 按 Python 字节码 trace，
    不区分内部是写 `self.forward(...)` 还是 `self(...)`，所以10/12/14号脚本
    内部的调用写法不用跟着改。

    scope【2026-09-23g新增，信心6/10，见文件头第11条(f)】：
      'full'(默认，行为跟上一轮完全一致)：编译 model.forward 和
          model.forward_grouped 这两个入口方法整个，包括里面
          `if d_rows.numel()>0` 这条数据依赖分支和 fusion/头的全部子调用。
      'submodules'：只编译 model.cis.forward 和 model.layout.forward 这两个
          子模块本身，forward/forward_grouped 整个编排逻辑(含那条数据依赖分支、
          fusion 的 4 个子方法、三个头)留在 eager，完全不进 Dynamo trace。选
          这两个子模块是直接抄 bench ③ 自己量出来的占比：cis 32%+WT layout
          31%+D layout 31%=94%，融合+头只占 6%，压缩范围到这两个最贵的模块
          理论上能拿到大部分编译收益，同时把"每个batch形状都变+有数据依赖
          分支"这个 compile 最不友好的编排逻辑整段排除在外——这正是
          bench_--compile.txt 观察到的反复重编译最可能的触发点(根因分析见
          文件头第11条(e))。`model.layout.forward = torch.compile(...)`这一行
          能同时接住 forward()里 `self.layout(**layout_wt)`(走 __call__)和
          forward_chunked()里 `self.forward(**sub)`(直接调用、绕开 __call__)
          两种调用方式，跟本函数给 forward/forward_grouped 单独赋值是同一个
          技巧，不是新机制。没有在真实机器上验证过 submodules 是否真的比 full
          更不容易触发重编译，这是本轮唯一还没实测的假设。

    dynamic【2026-09-23f，默认从 True 改成 None，信心6/10，见文件头第11条
    (e)③】：None=Dynamo自动判断哪些维度真的会变才标记成符号量(官方推荐的默认
    行为)；True=从第一次调用起就把全部输入维度都当符号量(上一轮的写法)。这个
    模型 batch 间基因数/序列长度天天在变，理论上迟早会被 Dynamo 自动标记成
    动态，但"要经过几次重新编译才收敛"、"auto 模式下 `if d_rows.numel()>0`
    这种数据依赖分支表现如何"都没有实测依据，所以改成参数、由调用方决定，
    run_benchmark ⑧节这次两种都测。

    cache_size_limit【2026-09-23f新增，信心8/10，机制是 PyTorch 官方文档行为，
    数值本身没调过】：编译前设置 torch._dynamo.config.cache_size_limit 和
    accumulated_cache_size_limit。这是这一轮要修的核心问题——用户这次
    `bash run_all.sh --bench --compile` 贴回来的日志显示，dynamic=True 编译
    后第一次真正调用就陷入长达两分多钟的重复重新编译(guard miss→recompile，
    sympy 对 `pow_by_natural([VR[3,int_oo],VR[-1,-1]])` 这类无穷上界的符号
    倒数运算给不出确定值域，导致每次重新编译都很慢)，用户最后手动Ctrl+Z中止，
    详细根因分析见文件头第11条(e)。设了 cache_size_limit 之后，某个调用点
    重新编译次数一旦超过这个数，Dynamo 会自动放弃继续编译、退回eager跑那次
    调用，不会再无限期重编译——训练(run_one_seed/run_full_training)和 bench
    ⑧节都会用到这层保护。这不能保证"编译速度好"，只保证"不会因为反复重编译而
    悬在那里不动"，跟下面 bench ⑧节新增的 --compile-warmup-timeout 是两层
    互补的安全网(一层限制重编译次数、一层限制热身阶段总墙钟时间)，不是重复。

    失败(比如 PyTorch 版本太老、触发了 dynamo 不支持的写法)不应该让整个训练
    崩掉，try/except 后退回未编译模型、只打印一行提示。信心7/10——上面这套
    "为什么包这两个方法、为什么必须能兼容 dynamic 参数"的推理站得住，但这个
    具体模型第一次尝试 compile 就在真实机器上遇到了预期外的重编译问题(见上)，
    没有这台机器的GPU/profiling权限验证过这次修复真的有效——用前务必先跑
    `--mode bench --compile` 看新增的 ⑧ 节对比结果(现在会测 auto/True 两种
    dynamic 模式)。
    【2026-09-23e补充说明，本轮仍然成立】下面这层try/except只包得住
    `torch.compile(...)`这一行本身——它是懒编译，包装时不会触发真正的JIT编译，
    所以这里几乎不会抛异常；真正的编译在第一次调用被包装后的函数时才发生，那次
    调用不在这个函数里，这层try/except接不住(第一版的教训：真实环境下第一次
    调用直接报了 ModuleNotFoundError，从这里的try/except底下直接钻出去，见15
    号脚本文件头)。调用方(run_one_seed的训练循环本身、run_benchmark ⑧节)已经/
    需要在"第一次真正调用"那一步补上对应的容错，这个函数本身不用改。"""
    if not compile_model:
        return model
    _silence_noisy_dynamo_logging()
    try:
        torch._dynamo.config.cache_size_limit = cache_size_limit
        torch._dynamo.config.accumulated_cache_size_limit = cache_size_limit * 4
    except Exception as e:
        print(f"{tag}设置 torch._dynamo.config.cache_size_limit 失败"
              f"({type(e).__name__}: {e})，继续尝试 compile，但没有重编译次数上限"
              "这层保护了(不同 PyTorch 版本这个 config 项名字/存在与否可能不同)")
    if scope == "submodules":
        try:
            model.cis.forward = torch.compile(model.cis.forward, dynamic=dynamic)
            model.layout.forward = torch.compile(model.layout.forward, dynamic=dynamic)
            print(f"{tag}torch.compile 已启用(scope=submodules，dynamic={dynamic!r}，"
                  f"cache_size_limit={cache_size_limit}，只编译 cis.forward/"
                  "layout.forward 这两个子模块——按 bench ③ 的占比二者合计约占改之前"
                  "前向用时的94%；forward/forward_grouped 本身(含数据依赖分支、fusion/"
                  "头)留在 eager，不进 Dynamo trace；第一次真正调用时会有一次性JIT"
                  "编译耗时，不代表后续每步都这么慢)")
        except Exception as e:
            print(f"{tag}torch.compile 启用失败({type(e).__name__}: {e})，退回未编译模型"
                  "(不影响训练正确性，只是拿不到编译带来的提速；如果多次都在这台机器上"
                  "失败，建议去掉 --compile 或换回 --compile-scope full)")
        return model
    try:
        model.forward = torch.compile(model.forward, dynamic=dynamic)
        model.forward_grouped = torch.compile(model.forward_grouped, dynamic=dynamic)
        print(f"{tag}torch.compile 已启用(scope=full，dynamic={dynamic!r}，cache_size_limit="
              f"{cache_size_limit}，forward/forward_grouped 各编译一次；第一次真正"
              "调用时会有一次性JIT编译耗时，不代表后续每步都这么慢)")
    except Exception as e:
        print(f"{tag}torch.compile 启用失败({type(e).__name__}: {e})，退回未编译模型"
              "(不影响训练正确性，只是拿不到编译带来的提速；如果多次都在这台机器上失败，"
              "建议去掉 --compile)")
    return model


def _train_step(model, opt, batch, ds, device, forward_mode, amp, class_counts, loss_kw,
                grad_clip: float = 0.0, scheduler=None):
    """一步训练：autocast 下前向 -> fp32 算 loss -> 反传 -> (可选)梯度裁剪 -> 更新 ->
    (可选)学习率调度。返回 (parts, 样本数)，parts 里是 0 维 GPU 张量，不在这里同步
    (见文件头第9条)。【第2批】grad_clip>0 时按全局 L2 范数裁剪(clip_grad_norm_ 内部用
    clamp 算系数、不做 Python 判断，不触发 GPU 同步)，裁剪前的范数放进 parts["grad_norm"]；
    grad_clip=0、scheduler=None 时跟原来逐位相同(bench 调用方式不变)。"""
    with _autocast(device, amp):
        outputs = _forward_batch(model, batch, ds.tf2idx, device, forward_mode)
    y_bd = batch.get("y_bd") if loss_kw.get("lambda_bd") else None  # 2026-09-25b：只在 λ_bd>0 时用
    if loss_kw.get("lambda_bd") and y_bd is None:
        raise ValueError("lambda_bd>0 但 batch 里没有 y_bd(数据集没有稠密目标？)")
    loss, parts = compute_total_loss(
        outputs, batch["y_a"].to(device, non_blocking=True),
        batch["y_b"].to(device, non_blocking=True),
        batch["y_c"].to(device, non_blocking=True), class_counts,
        return_tensors=True, y_bd=y_bd.to(device, non_blocking=True) if y_bd is not None else None,
        **loss_kw)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip and grad_clip > 0:
        parts["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            float(grad_clip)).detach()
    opt.step()
    if scheduler is not None:
        scheduler.step()
    return parts, len(batch["y_a"])


def _report_need_d(ds) -> float:
    """统计"D∈L_g"(被耗竭的TF在这个基因的 L_g 里本来就有位点、需要单独算 D 侧
    layout)的样本占比——分组前向省掉的就是其余样本的 D 侧 layout/交叉注意力，这个
    比例越小省得越多。直接从数据算，不是估计。"""
    tfsets = {g: set(v["tf_idx"].tolist()) for g, v in ds.layout_by_gene.items()}
    genes = ds.samples["gene_id"].to_numpy()
    dep = ds.samples["tf_depleted"].map(ds.tf2idx).to_numpy()
    need = np.fromiter((d in tfsets.get(g, ()) for g, d in zip(genes, dep)),
                       dtype=bool, count=len(genes))
    frac = float(need.mean()) if len(need) else float("nan")
    print(f"D∈L_g(需要单独算 D 侧 layout)的样本: {int(need.sum())}/{len(need)} = "
          f"{frac:.2%}；其余样本的 D 侧 layout/交叉注意力直接复用 WT 结果")
    return frac


def _report_chunk_gene_diversity(ds, idx, chunk: int, name: str) -> float:
    """第三轮新增的诊断(只打印，不改任何计算)：val/test 的 loss 是"按 idx 顺序每 chunk
    条切一块、各算 compute_total_loss 再平均"(沿用最初的定义)，而 Head A 的 (1-r) 项是
    块内 Pearson r。如果样本在文件里是按基因连续排的，一块里只有一两个基因，块内 r 基本
    退化(标签几乎是常数)，val_loss 里 Head A 那一项就几乎只剩 MSE。这里直接数一下每块
    平均有几个不同基因，数字太小时提醒——是否要改早停判据的定义等看到这个数字再决定。"""
    genes = ds.samples["gene_id"].to_numpy()[np.asarray(idx, dtype=np.int64)]
    per = [len(set(genes[s:s + chunk].tolist())) for s in range(0, len(genes), chunk)]
    mean_g = float(np.mean(per)) if per else float("nan")
    msg = (f"{name} 按 idx 顺序每 {chunk} 条一块：平均每块 {mean_g:.1f} 个不同基因"
           f"(最少 {min(per) if per else 0})")
    if per and mean_g < 8:
        msg += ("  ⚠ 块内基因太少，Head A 的块内 Pearson r 基本退化，这个 loss 里 Head A "
                "那项几乎只剩 MSE(只是提醒，没有改早停判据)")
    print(msg)
    return mean_g


# --------------------------------------------------------------------------
# 1. 染色体holdout切分 + leave-TF-out切分
# --------------------------------------------------------------------------
def build_split_indices(ds, head_a_path: str = "out/head_a_baseline_logtpm.parquet") -> dict:
    """按ds.samples里的gene_id查head_a_baseline_logtpm.parquet的split列(08号
    脚本产出，染色体holdout: val=chrXIII/XIV, test=chrXV/XVI)，返回
    {"train":[...],"val":[...],"test":[...]}三个样本下标列表(下标是ds.samples
    里的行号，可以直接喂给torch.utils.data.Subset)。09号脚本的TFLayoutDataset
    只保留了log_tpm_baseline列，没保留split列，这里重新单独读一次。gene_id在
    这个文件里找不到的样本(理论上不应该发生，08脚本对全部6572个基因都产出了
    split)默认归入train并打印警告——不是我能在这台机器上确认会不会真的发生的
    事。"""
    split_df = pd.read_parquet(head_a_path)
    gene2split = split_df["split"].to_dict()
    # 2026-09-28b：只看真实样本行(09 号 set_head_a_all 的伪样本追加在 _n_real 之后，由 run_one_seed 自己并进去)
    genes = ds.samples["gene_id"].to_numpy()[:getattr(ds, "_n_real", len(ds))]
    splits = np.array([gene2split.get(g, "train") for g in genes])
    n_missing = int(sum(1 for g in genes if g not in gene2split))
    if n_missing:
        print(f"警告：{n_missing} 条样本的gene_id在{head_a_path}里找不到对应的"
              "split，已默认归入train(理论上不应该发生，08脚本对全部6572个基因"
              "都产出了split，如果这个数字不是0，需要回头查gene_id全集是否对得上)")
    idx = {s: np.where(splits == s)[0].tolist() for s in ("train", "val", "test")}
    print(f"按染色体holdout切分样本：train={len(idx['train'])}  "
          f"val={len(idx['val'])}  test={len(idx['test'])}")
    return idx


def leave_tf_out_split(ds, held_out_tfs, base_idx=None):
    """base_idx: 可选，限制在这个样本子集内再按TF切；不传则在全部样本上切。
    返回(keep_idx, heldout_idx)：keep_idx是tf_depleted不在held_out_tfs集合里的
    样本下标，heldout_idx是反过来。"留出的TF"这里定义成"被耗竭的那个TF"
    (tf_depleted)——status文件"TF维度也要做leave-TF-out验证"没有更细的操作化
    说明，理解成"模型训练时从没见过这个TF被敲除后的任何(gene,TF)样本，测试时
    专门看这些样本预测得怎么样"是最直接的读法，信心7/10。"""
    tf_depleted = ds.samples["tf_depleted"].to_numpy()
    held_out_set = set(held_out_tfs)
    is_heldout = np.array([t in held_out_set for t in tf_depleted])
    pool_mask = np.zeros(len(ds), dtype=bool)
    pool = np.arange(len(ds)) if base_idx is None else np.asarray(base_idx)
    pool_mask[pool] = True
    keep_idx = np.where(pool_mask & ~is_heldout)[0].tolist()
    heldout_idx = np.where(pool_mask & is_heldout)[0].tolist()
    return keep_idx, heldout_idx


# --------------------------------------------------------------------------
# 2. 单个seed的train/val循环(优化器 + 早停；第2批：子epoch验证/选模型判据/各头最优)
# --------------------------------------------------------------------------
def run_one_seed(ds, split_idx: dict, seed: int, n_tf: int, vocab_size: int,
                 pad_token_id: int = 0, d_model: int = 256, n_heads: int = 8,
                 cis_layers: int = 6, lay_layers: int = 4, batch_size: int = 64,
                 n_epochs: int = 30, lr: float = 1e-4, weight_decay: float = 1e-5,
                 patience: int = 5, lambda_b: float = 1.0, lambda_c: float = 1.0,
                 lambda_sign: float = 0.1, gamma: float = 2.0, beta: float = 0.999,
                 class_counts: torch.Tensor = None, device: str = "cpu",
                 log_every: int = 50, num_workers: int = 0, amp: str = "off",
                 forward_mode: str = "grouped", tfs_per_gene: int = 1,
                 eval_batch_size: int = 512, layout_buckets: int = 1,
                 compile_model: bool = False, compile_cache_limit: int = 16,
                 compile_dynamic: str = "auto", compile_scope: str = "full",
                 ctx_mode: str = "legacy", ctx_clip: float = 3.0, head_c_mode: str = "delta",
                 dropout: float = 0.1, lambda_a: float = 1.0, loss_b: str = "mse",
                 huber_delta: float = 1.0, optimizer: str = "adam", grad_clip: float = 0.0,
                 lr_schedule: str = "constant", warmup_steps: int = 0,
                 min_lr_ratio: float = 0.05, val_every: float = 1.0,
                 select_metric: str = "val_loss", save_per_head_best: bool = False,
                 max_steps: int = 0, min_delta: float = 1e-6, dense_target: str = None,
                 lambda_bd: float = 0.0, dense_on: str = "ns", dense_delta: float = None,
                 ema_decay: float = 0.0, ablation: str = "none", head_a_all: bool = False,
                 wt_input: str = "none"):
    """单个seed的完整train/val循环：优化器+早停(连续patience次验证没有改善就停，最后恢复
    到最优那次验证的权重)。返回 (训练好的model, 每次验证一条的历史, best信息字典)。

    num_workers【2026-09-23新增，信心8/10】：DataLoader后台进程数，默认0(单进程，不改变任何
    数值结果，只影响速度)；num_workers>0时persistent_workers=True，train/val两个loader在整个
    run_one_seed生命周期内只创建一次；pin_memory只在device是cuda时开。
    【第二轮提速】forward_mode('grouped'/'legacy')、amp('off'/'bf16')、tfs_per_gene(K，信心
    6/10)、eval_batch_size(只影响速度)，见文件头第9条。【第三轮】layout_buckets：只影响速度。
    【第四轮~2026-09-23g】compile_model/compile_cache_limit/compile_dynamic/compile_scope：
    torch.compile 相关，默认关，只影响速度，见文件头第11条(b)(e)(f)。

    【2026-09-24 第2批新增参数，见文件头第12条；每一个的默认值都等于原来的行为】
      ctx_mode/ctx_clip   条件编码 legacy/relative/marker(09号 set_ctx_mode；数据集没有这个
                          方法时(自检假数据)跳过并提示)
      head_c_mode         Head C 是否知道被耗竭的 TF：delta/delta_tf/delta_tf_gene(15号)
      dropout             各分支 dropout(原来写死0.1)
      lambda_a/loss_b/huber_delta   Head A 权重、Head B 用 mse 还是 huber(15号)
      optimizer           adam/adamw；weight_decay 沿用原参数
      grad_clip           >0 时全局梯度范数裁剪
      lr_schedule/warmup_steps/min_lr_ratio   constant/warmup_cosine
      val_every           每多少个 epoch 验证一次(1.0=每个epoch一次=原来；0.25=每1/4个epoch)。
                          patience 按"验证次数"计，val_every=1 时就是 epoch 数，跟原来一样
      select_metric       选模型/早停判据："val_loss"(原来：val 总loss) 或 "composite"
                          (= val r_a_gene + val r_b + val (AUPRC_down+AUPRC_up)/2，三个头各占约
                          一份，越大越好)。依据：17号导出的 val loss 分项里 A/B/C 约占
                          48%/46%/5%，Head C 在原判据里几乎没有发言权，而 Head C 的 macro-F1
                          正是最弱的一环；三项等权是我拍的，信心5/10
      save_per_head_best  另外按 A:r_a_gene、B:r_b、C:AUPRC均值 各自记住最优的那次验证的完整
                          权重(best["head_states"]，CPU 张量)，16/17号可以"A 头用 A 的最优权重、
                          B 用 B 的……"组合预测。开着时早停改成"选模型判据和三个头的指标都连续
                          patience 次没改善才停"。依据：run1 里 Head A 在头几个 epoch 就开始背
                          答案(训练 l_a 0.33→0.02)，而 B/C 的最优点更晚，单一判据只能折中
      max_steps           >0 时总共只训练这么多步(冒烟测试用，18号 smoke_first)
      min_delta【2026-09-25a】 选模型判据/各头指标要比之前最好值高出 min_delta 才算改善(默认1e-6
                          =原行为)；见文件头第13条(b)
      dense_target/lambda_bd/dense_on/dense_delta【2026-09-25b】 Head B 稠密 log2FC 辅助目标(文件头
                          第14条、15号第3批)：dense_target=21号产出路径(None=不加载)；lambda_bd=0 时
                          只监控(val 上报 l_bd/r_bd/r_bd_ns)，>0 时进训练 loss
      ablation【2026-09-26a】 输入消融 none/no_knockout/no_layout/no_cis(09 号 set_ablation，文件头
                          第16条)；每次调用都显式设置一次(默认 none)，所以 18 号让多个实验共用同一个
                          Dataset 时不会把上一个实验的消融带过来
      ema_decay【2026-09-26a】 >0 时对权重做滑动平均(文件头第15条)：验证/选模型/存权重用 EMA 权重，
                          训练本身不变；0=关(逐位等于 25b)
      head_a_all【2026-09-28b】 True 时网格外基因的 Head A 伪样本并进训练和验证(文件头第17条)；每次调用都显式设置，
                          返回前关掉；False=逐位等于 26a
      wt_input【2026-10-05a】 "head_bc" 时实测 WT 表达(batch["y_a"])只进 Head B/C(文件头第18条、15 号第4批)；
                          "none"=逐位等于 28b，且 model_kwargs 里不写这个键
    这一层函数的默认值故意取"跟原来行为最接近"的组合，更激进的默认值放在 18 号实验脚本的
    预设里，self_test 和外部调用者不会被意外改变行为。"""
    if wt_input not in ("none", "head_bc"):
        raise ValueError(f"wt_input 只能是 none/head_bc，收到 {wt_input}")
    torch.manual_seed(seed)
    if class_counts is None:
        class_counts = _default_class_counts()
    class_counts = class_counts.to(device)
    if hasattr(ds, "set_ctx_mode"):
        ds.set_ctx_mode(ctx_mode, ctx_clip, verbose=False)
    elif ctx_mode != "legacy":
        print(f"[seed{seed}] 提示：这个数据集对象没有 set_ctx_mode(自检假数据)，ctx_mode="
              f"{ctx_mode} 不生效，按它自己的条件编码训练")
    if hasattr(ds, "set_ablation"):  # 2026-09-26a：必须在建 DataLoader 之前；每次都显式设(含 none)
        ds.set_ablation(ablation, verbose=False)
    elif ablation != "none":
        raise ValueError(f"这个数据集对象没有 set_ablation，不能做 ablation={ablation}")
    # 2026-09-28b 第8批(文件头第17条)：Head A 全基因伪样本。每次都显式设(含 False)，18 号共用一个 Dataset 时不会串
    head_a_all = bool(head_a_all)
    ha_stats = None
    train_idx, val_idx = list(split_idx["train"]), list(split_idx["val"])
    if hasattr(ds, "set_head_a_all"):
        ha_stats = ds.set_head_a_all(head_a_all, verbose=False)
        if head_a_all:
            _pv = np.asarray(ds.pseudo_idx.get("val", []), dtype=np.int64)
            _nper = int(ha_stats.get("rows_per_gene") or 0)
            if _nper and len(_pv) % _nper == 0:  # 基因轮转：val loss 分块里基因多样(只影响日志里的 val_loss)
                _pv = _pv.reshape(-1, _nper).T.ravel()
            train_idx = train_idx + np.asarray(ds.pseudo_idx.get("train", []), dtype=np.int64).tolist()
            val_idx = val_idx + _pv.tolist()
    elif head_a_all:
        raise ValueError("这个数据集对象没有 set_head_a_all，不能做 head_a_all=True(先替换 09 号)")
    if select_metric not in ("val_loss", "composite"):
        raise ValueError(f"select_metric 只能是 val_loss/composite，收到 {select_metric}")
    # 2026-09-25b：稠密辅助目标(文件头第14条)。跟 set_ctx_mode 一样必须在建 DataLoader 之前
    dense_stats = None
    if dense_on not in ("ns", "all"):
        raise ValueError(f"dense_on 只能是 ns/all，收到 {dense_on}")
    if hasattr(ds, "set_dense_target"):
        dense_stats = ds.set_dense_target(dense_target, verbose=False)
    elif dense_target:
        print(f"[seed{seed}] 提示：这个数据集对象没有 set_dense_target，dense_target 不生效")
    if lambda_bd:
        _ybd = getattr(ds, "_y_bd_arr", None)
        _tr = np.asarray(split_idx["train"], dtype=np.int64)
        if _ybd is None or not np.isfinite(np.asarray(_ybd)[_tr]).any():
            raise ValueError(f"lambda_bd={lambda_bd}>0，但训练集里没有任何稠密目标(dense_target="
                             f"{dense_target!r})——先跑 21_build_dense_target.py 并传入它的产出路径")
    loss_kw = dict(lambda_b=lambda_b, lambda_c=lambda_c, lambda_sign=lambda_sign,
                   gamma=gamma, beta=beta, lambda_a=lambda_a, loss_b=loss_b,
                   huber_delta=huber_delta, lambda_bd=lambda_bd, dense_on=dense_on,
                   dense_delta=dense_delta)
    if head_a_all:  # 2026-09-28b：伪样本 y_c=-1，Head C 要跳过(15 号 c_ignore)；False 时 loss_kw 跟 26a 完全一样
        loss_kw["c_ignore"] = True

    train_loader, _ = _make_loader(ds, train_idx, "train", forward_mode,
                                   batch_size, tfs_per_gene, eval_batch_size, seed,
                                   num_workers, device, persistent=True,
                                   layout_buckets=layout_buckets)
    val_loader = _make_loader(ds, val_idx, "eval", forward_mode, batch_size, 1,
                              eval_batch_size, seed, num_workers, device, persistent=True,
                              layout_buckets=layout_buckets)

    model_kwargs = dict(n_tf=n_tf, vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                        cis_layers=cis_layers, lay_layers=lay_layers, dropout=dropout,
                        pad_token_id=pad_token_id, head_c_mode=head_c_mode)
    if wt_input != "none":  # 2026-10-05a：只在非默认时写，旧 checkpoint / none 的 model_kwargs 逐字节不变
        model_kwargs["wt_input"] = wt_input
    model = SiameseHeadsModel(**model_kwargs).to(device)
    model = _maybe_compile_model(model, compile_model, tag=f"[seed{seed}] ",
                                 cache_size_limit=compile_cache_limit,
                                 dynamic=_resolve_compile_dynamic(compile_dynamic),
                                 scope=compile_scope)
    opt = _make_optimizer(model, lr, weight_decay, device, kind=optimizer)
    n_steps_epoch = len(train_loader)
    total_steps = n_epochs * n_steps_epoch
    if max_steps and max_steps > 0:
        total_steps = min(total_steps, int(max_steps))
    sched = _make_scheduler(opt, lr_schedule, warmup_steps, n_epochs * n_steps_epoch,
                            min_lr_ratio)
    ema_decay = float(ema_decay)
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError(f"ema_decay 必须在 [0,1) 内(0=不用 EMA)，收到 {ema_decay}")
    ema_params = list(model.parameters()) if ema_decay > 0.0 else []  # 跟 model 共用同一批张量
    ema_shadow = [p.detach().clone() for p in ema_params]
    ema_cur = [p.detach() for p in ema_params]  # 视图，跟参数共用存储：优化器原地更新后自动是最新值
    ema_foreach = hasattr(torch, "_foreach_mul_") and hasattr(torch, "_foreach_add_")
    n_val = max(1, int(round(1.0 / val_every))) if 0 < val_every < 1 else 1
    val_points = sorted({max(0, int(round((i + 1) * n_steps_epoch / n_val)) - 1)
                         for i in range(n_val)})
    amp_on = not isinstance(_autocast(device, amp), contextlib.nullcontext)
    if amp == "bf16" and not amp_on:
        print(f"[seed{seed}] 提示：请求了 --amp bf16，但当前设备({device})不支持 bf16，"
              "退回 fp32")
    print(f"[seed{seed}] 配置: forward={forward_mode}  amp={'bf16' if amp_on else 'fp32'}  "
          f"batch_size={batch_size}  tfs_per_gene={tfs_per_gene if forward_mode == 'grouped' else 1}"
          f"  layout_buckets={layout_buckets if forward_mode == 'grouped' else 1}"
          f"  compile={compile_model}" +
          (f"(scope={compile_scope}  dynamic={compile_dynamic})" if compile_model else "") +
          f"  每epoch {n_steps_epoch} 步")
    print(f"[seed{seed}] 第2批配置: ctx={ctx_mode}  head_c={head_c_mode}  wt_input={wt_input}  dropout={dropout}  "
          f"loss_b={loss_b}" + (f"(δ={huber_delta})" if loss_b == "huber" else "") +
          f"  λ_A/B/C/s={lambda_a}/{lambda_b}/{lambda_c}/{lambda_sign}  {optimizer}"
          f"(lr={lr}, wd={weight_decay})  grad_clip={grad_clip}  lr_schedule={lr_schedule}"
          f"(warmup={warmup_steps}, min_ratio={min_lr_ratio})  每epoch验证{n_val}次"
          f"(第{[p + 1 for p in val_points]}步)  patience={patience}次验证  "
          f"选模型={select_metric}(min_delta={min_delta})  各头最优权重={'存' if save_per_head_best else '不存'}"
          + (f"  max_steps={max_steps}" if max_steps else ""))
    if ema_params:
        print(f"[seed{seed}] 第4批 EMA: ema_decay={ema_decay}(视野约 {1.0 / (1.0 - ema_decay):.0f} 步≈"
              f"{1.0 / (1.0 - ema_decay) / max(n_steps_epoch, 1):.2f} 个epoch)；验证/选模型/存 checkpoint "
              "用 EMA 权重，训练用原权重")
    if head_a_all and ha_stats:
        _ng = ha_stats["n_genes"]
        print(f"[seed{seed}] 第8批 Head A 全基因: 网格外基因 train/val/test={_ng['train']}/{_ng['val']}/{_ng['test']}"
              f"(无位点 {ha_stats['n_genes_no_site']} 个)，每基因 {ha_stats['rows_per_gene']} 条伪样本；训练 "
              f"{len(split_idx['train'])}+{len(train_idx) - len(split_idx['train'])} 条、验证 {len(split_idx['val'])}+"
              f"{len(val_idx) - len(split_idx['val'])} 条；val 的 r_a_gene/A 头选模型按全部 val 基因，B/C 指标只算真实行，"
              "val_loss 含伪样本分块(跟其它实验不可比，不参与选模型)")
    if dense_stats and dense_stats.get("n_finite"):
        print(f"[seed{seed}] 第3批稠密目标: {dense_target}  有值 {dense_stats['frac']:.1%}(不显著样本里 "
              f"{dense_stats.get('frac_ns', float('nan')):.1%})  λ_Bd={lambda_bd}"
              + (f"  dense_on={dense_on}  δ_d={huber_delta if dense_delta is None else dense_delta}"
                 if lambda_bd else "(只监控，不进 loss)"))

    sel_best, best_state, best_rec, bad = -float("inf"), None, None, 0
    head_keys = {"A": "val_r_a_gene", "B": "val_r_b", "C": "val_auprc_mean"}
    head_best = {h: dict(value=-float("inf"), state=None, rec=None) for h in head_keys} \
        if save_per_head_best else {}
    part_keys = ("total", "l_a", "l_b", "l_c", "l_sign", "l_bd")  # l_bd：2026-09-25b
    history, global_step, stop, stop_reason = [], 0, False, ""

    def _clone_state():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    acc = {k: torch.zeros((), device=device) for k in part_keys}
    clip_cnt, acc_n, t_seg = torch.zeros((), device=device), 0, time.time()
    gn_sum = torch.zeros((), device=device)
    for epoch in range(n_epochs):
        model.train()
        t_ep = time.time()
        t_log, samples_since_log = t_ep, 0
        ep_total, ep_steps, ep_samples, ep_val_sec = torch.zeros((), device=device), 0, 0, 0.0
        last_val_loss = float("nan")
        for step, batch in enumerate(train_loader):
            parts, nb = _train_step(model, opt, batch, ds, device, forward_mode, amp,
                                    class_counts, loss_kw, grad_clip, sched)
            global_step += 1
            if ema_params:  # 2026-09-26a：EMA 更新(不同步 GPU，见文件头第15条(a))
                _d = min(ema_decay, (1.0 + global_step) / (10.0 + global_step))
                with torch.no_grad():
                    if ema_foreach:
                        torch._foreach_mul_(ema_shadow, _d)
                        torch._foreach_add_(ema_shadow, ema_cur, alpha=1.0 - _d)
                    else:
                        for _e, _c in zip(ema_shadow, ema_cur):
                            _e.mul_(_d).add_(_c, alpha=1.0 - _d)
            ep_steps += 1
            ep_samples += nb
            samples_since_log += nb
            ep_total += parts["total"]
            for k in part_keys:
                acc[k] += parts[k]
            acc_n += 1
            if "grad_norm" in parts:
                clip_cnt += (parts["grad_norm"] > grad_clip).float()
                gn_sum += parts["grad_norm"]
            if log_every and step % log_every == 0:
                now = time.time()  # 下面 float() 会同步GPU，只在打日志时发生
                msg = {k: round(float(v), 4) for k, v in parts.items()}
                speed = ""
                if step > 0:
                    sps = samples_since_log / max(now - t_log, 1e-9)
                    eta = (n_steps_epoch - step - 1) * (now - t_ep - ep_val_sec) / (step + 1)
                    speed = f"  {sps:.0f}样本/s  本epoch预计还剩{eta / 60:.1f}分钟"
                lr_txt = f"  lr={opt.param_groups[0]['lr']:.2e}" if sched is not None else ""
                print(f"  [seed{seed}] epoch{epoch} step{step}/{n_steps_epoch} "
                      f"loss={msg}{lr_txt}{speed}")
                t_log, samples_since_log = now, 0
            hit_max = bool(max_steps) and global_step >= max_steps
            if step not in val_points and not hit_max:
                continue
            # ---------------- 验证(第2批：可以在 epoch 中间) ----------------
            t_train_seg = time.time() - t_seg
            if ema_params:  # 2026-09-26a：验证及后面的 _clone_state 都用 EMA 权重，回到训练权重在下面
                ema_backup = [p.detach().clone() for p in ema_params]
                with torch.no_grad():
                    for _p, _e in zip(ema_params, ema_shadow):
                        _p.copy_(_e)
            t_v = time.time()
            val_res = evaluate(model, ds, val_idx, class_counts,
                               batch_size=batch_size, device=device,
                               num_workers=num_workers, amp=amp, forward_mode=forward_mode,
                               eval_batch_size=eval_batch_size, loader=val_loader, **loss_kw)
            t_val = time.time() - t_v
            ep_val_sec += t_val
            last_val_loss = val_res["loss"]
            _ap = [v for v in (val_res["auprc_down"], val_res["auprc_up"]) if np.isfinite(v)]
            auprc_mean = float(np.mean(_ap)) if _ap else float("nan")  # 某类在 val 里缺席时只用另一类
            composite = float(np.nan_to_num(val_res["r_a_gene"]) + np.nan_to_num(val_res["r_b"])
                              + np.nan_to_num(auprc_mean))
            rec = dict(epoch=epoch, epoch_progress=round(global_step / n_steps_epoch, 4),
                       step=global_step, val_index=len(history),
                       train_loss=float(acc["total"] / max(acc_n, 1)),
                       **{f"train_{k}": float(acc[k] / max(acc_n, 1)) for k in part_keys[1:]},
                       val_loss=val_res["loss"],
                       **{f"val_{k}": float(val_res["loss_parts"][k]) for k in part_keys[1:]},
                       val_r_a_gene=val_res["r_a_gene"], val_r_b=val_res["r_b"],
                       val_rmse_b=val_res["rmse_b"], val_acc_c=val_res["acc_c"],
                       val_macro_f1_c=val_res["macro_f1_c"],
                       val_auroc_down=val_res["auroc_down"], val_auroc_up=val_res["auroc_up"],
                       val_auprc_down=val_res["auprc_down"], val_auprc_up=val_res["auprc_up"],
                       val_auprc_mean=float(auprc_mean), val_composite=composite,
                       val_r_bd=val_res.get("r_bd", float("nan")),
                       val_r_bd_ns=val_res.get("r_bd_ns", float("nan")),
                       lr=float(opt.param_groups[0]["lr"]),
                       clip_frac=float(clip_cnt / max(acc_n, 1)) if grad_clip > 0
                       else float("nan"),
                       grad_norm_mean=float(gn_sum / max(acc_n, 1)) if grad_clip > 0
                       else float("nan"),
                       train_sec=t_train_seg, val_sec=t_val)
            history.append(rec)
            score = -val_res["loss"] if select_metric == "val_loss" else composite
            improved = score > sel_best + min_delta
            if improved:
                sel_best, best_rec = score, rec
                best_state = _clone_state()
            head_improved = []
            for h, key in head_keys.items() if save_per_head_best else ():
                v = rec[key]
                if np.isfinite(v) and v > head_best[h]["value"] + min_delta:
                    head_best[h].update(value=float(v), rec=rec, state=_clone_state())
                    head_improved.append(h)
            bad = 0 if (improved or head_improved) else bad + 1
            if n_val > 1 or max_steps:
                print(f"  [seed{seed}] val#{rec['val_index']} @epoch{rec['epoch_progress']:.2f}: "
                      f"train_loss={rec['train_loss']:.4f}  val_loss={rec['val_loss']:.4f}"
                      f"(A/B/C/s={rec['val_l_a']:.3f}/{rec['val_l_b']:.3f}/{rec['val_l_c']:.3f}/"
                      f"{rec['val_l_sign']:.3f})  r_a_gene={rec['val_r_a_gene']:.4f}  "
                      f"r_b={rec['val_r_b']:.4f}  AUPRC dn/up={rec['val_auprc_down']:.3f}/"
                      f"{rec['val_auprc_up']:.3f}  composite={composite:.4f}"
                      + (f"  稠密 l_bd={rec['val_l_bd']:.3f} r_bd={rec['val_r_bd']:.3f}"
                         f"(不显著 {rec['val_r_bd_ns']:.3f})" if np.isfinite(rec["val_r_bd"]) else "")
                      + f"{'  ★' if improved else ''}"
                      f"{('  头' + ''.join(head_improved) + '↑') if head_improved else ''}"
                      f"  (验证{t_val:.0f}秒)")
            if ema_params:  # 换回训练权重(此时 best_state/各头最优权重已经克隆完)
                with torch.no_grad():
                    for _p, _b in zip(ema_params, ema_backup):
                        _p.copy_(_b)
                del ema_backup
            acc = {k: torch.zeros((), device=device) for k in part_keys}
            clip_cnt, acc_n, t_seg = torch.zeros((), device=device), 0, time.time()
            gn_sum = torch.zeros((), device=device)
            model.train()
            if bad >= patience:
                stop, stop_reason = True, "patience"
                break
            if hit_max:
                stop, stop_reason = True, "max_steps"
                break
        train_mean = float(ep_total / ep_steps) if ep_steps else float("nan")
        t_train = time.time() - t_ep - ep_val_sec
        print(f"[seed{seed}] epoch{epoch}: train_loss={train_mean:.4f}  "
              f"val_loss={last_val_loss:.4f}  (训练{t_train / 60:.1f}分钟、"
              f"{ep_samples / max(t_train, 1e-9):.0f}样本/s；验证{ep_val_sec:.0f}秒)")
        if stop:
            if stop_reason == "patience":
                crit = "val loss没下降" if select_metric == "val_loss" else "composite没提高"
                if save_per_head_best:
                    crit += "、三个头的指标也都没提高"
                print(f"[seed{seed}] 连续{patience}次验证{crit}，早停于epoch{epoch}"
                      f"(最优epoch={best_rec['epoch'] if best_rec else -1}, "
                      f"val_loss={best_rec['val_loss'] if best_rec else float('nan'):.4f})")
            else:
                print(f"[seed{seed}] 达到 max_steps={max_steps}，停止")
            break

    if not stop_reason:
        stop_reason = "n_epochs"  # 2026-09-25a：原来跑满时留空，见文件头第13条(a)
    if best_state is not None:
        model.load_state_dict(best_state)
    head_states = None
    if save_per_head_best:
        head_states = {}
        for h, hb in head_best.items():
            st = hb["state"] if hb["state"] is not None else best_state
            if hb["state"] is None:
                print(f"[seed{seed}] 提示：{h} 头的指标 {head_keys[h]} 在 val 上始终不是有限值"
                      "(比如某类样本缺席)，这个头改用选中的那份权重")
            if st is not None:
                head_states[h] = {k: v.cpu() for k, v in st.items()}
    best = dict(epoch=best_rec["epoch"] if best_rec else -1,
                val_index=best_rec["val_index"] if best_rec else -1,
                epoch_progress=best_rec["epoch_progress"] if best_rec else float("nan"),
                step=best_rec["step"] if best_rec else -1, metric=select_metric,
                score=float(sel_best), n_val_per_epoch=n_val, stop_reason=stop_reason,
                head_best={h: dict(value=hb["value"],
                                   val_index=hb["rec"]["val_index"] if hb["rec"] else -1,
                                   epoch_progress=hb["rec"]["epoch_progress"] if hb["rec"]
                                   else float("nan")) for h, hb in head_best.items()},
                head_states=head_states, model_kwargs=model_kwargs,
                head_a_all=dict(ha_stats) if (head_a_all and ha_stats) else None)
    if best_rec is not None:
        hb_txt = "  ".join(f"{h}@epoch{v['epoch_progress']:.2f}({head_keys[h][4:]}={v['value']:.4f})"
                           for h, v in best["head_best"].items())
        print(f"[seed{seed}] 选中的权重: val#{best['val_index']} @epoch{best['epoch_progress']:.2f}"
              f"(按{select_metric}，val_loss={best_rec['val_loss']:.4f}、r_a_gene="
              f"{best_rec['val_r_a_gene']:.4f}、r_b={best_rec['val_r_b']:.4f}、AUPRC均值="
              f"{best_rec['val_auprc_mean']:.4f})" + (f"；各头最优: {hb_txt}" if hb_txt else ""))
    del train_loader, val_loader, best_state, opt, head_best  # 关掉 persistent worker、释放优化器状态
    gc.collect()
    if head_a_all and hasattr(ds, "set_head_a_all"):  # 2026-09-28b：还原成真实样本表，test 评估/17 号导出不受影响
        ds.set_head_a_all(False, verbose=False)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model, history, best


# --------------------------------------------------------------------------
# 3. 评估：逐样本预测 + 基础指标(不含CI，CI在下面单独算)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, ds, idx, class_counts: torch.Tensor, batch_size: int = 128,
            device: str = "cpu", lambda_b: float = 1.0, lambda_c: float = 1.0,
            lambda_sign: float = 0.1, gamma: float = 2.0, beta: float = 0.999,
            num_workers: int = 0, amp: str = "off", forward_mode: str = "grouped",
            eval_batch_size: int = 512, loader=None, layout_buckets: int = 1,
            lambda_a: float = 1.0, loss_b: str = "mse", huber_delta: float = 1.0,
            lambda_bd: float = 0.0, dense_on: str = "ns", dense_delta: float = None,
            c_ignore: bool = False) -> dict:
    """跑一遍给定下标子集，收集Head A/B/C的预测+标签，算基础指标：
      Head A: 对y_a非NaN的部分算Pearson r
      Head B: 对y_b非NaN("显著")的部分算Pearson r + RMSE
      Head C: macro F1(三类F1算术平均，>96%都是ns，只看accuracy会掩盖down/up
              两个少数类的表现，信心8/10) + 整体accuracy
    返回值里同时带走逐样本预测/标签(顺序跟 idx 一致)，供外面做bootstrap CI用。

    【第二轮提速】forward_mode='grouped'时按基因整组打包前向(纯前向无dropout，结果
    跟逐样本前向一致)，算完再按 idx 顺序放回。loss 的定义不变：按 idx 顺序每
    batch_size 条切一块、各算 compute_total_loss、取平均。loader 可以传入
    _make_loader(...,'eval',...) 的返回值复用(run_one_seed 的 val 就是这样)。
    【第三轮】返回值新增 genes、r_a_gene(每个基因只算一次的 Head A r)、n_genes。
    【2026-09-24 第2批】指标部分挪进 _eval_metrics(evaluate_per_head 要对"各头各取一份权重"
    拼出来的预测复用同一套指标)；新增返回 loss_parts(同一分块口径下 l_a/l_b/l_c/l_sign 的
    均值——17号导出时才第一次看到它们，现在每次验证都记)、auroc_down/up、auprc_down/up
    (Head C 的排序能力，不受 argmax 阈值影响)；lambda_a/loss_b/huber_delta 透传给 loss。
    【2026-09-25b】batch 里有 y_bd(稠密目标)时一并收集：loss_parts 多 l_bd、指标多 r_bd/r_bd_ns
    (见 _eval_metrics)；lambda_bd/dense_on/dense_delta 透传给 loss。
    【2026-09-28b】c_ignore：透传给 loss(Head C 跳过 y_c<0 的伪样本行)；默认 False 逐位不变。"""
    model.eval()
    class_counts = class_counts.to(device)
    idx = list(idx)
    n = len(idx)
    if loader is None:
        loader = _make_loader(ds, idx, "eval", forward_mode, batch_size, 1, eval_batch_size,
                              0, num_workers, device, persistent=False,
                              layout_buckets=layout_buckets)
    loader, batches = loader
    pos_of = {g: p for p, g in enumerate(idx)}

    y_a_pred = torch.empty(n, device=device)
    y_b_pred = torch.empty(n, device=device)
    logits_c = torch.empty(n, 3, device=device)
    y_a_true = torch.empty(n)
    y_b_true = torch.empty(n)
    y_c_true = torch.empty(n, dtype=torch.long)
    y_bd_true = torch.full((n,), float("nan"))  # 2026-09-25b
    has_bd = False
    n_seen = 0
    for b_i, batch in enumerate(loader):  # 不用 zip：让 loader 迭代器自然走完，
        bidx = batches[b_i]               # persistent worker 下次复用时状态干净
        assert len(bidx) == len(batch["y_a"]), "evaluate: batch 顺序跟预定下标对不上"
        n_seen += len(bidx)
        with _autocast(device, amp):
            out = _forward_batch(model, batch, ds.tf2idx, device, forward_mode)
        pos = torch.tensor([pos_of[i] for i in bidx], dtype=torch.long)
        pos_d = pos.to(device, non_blocking=True)
        y_a_pred[pos_d] = out["y_a_pred"].float()
        y_b_pred[pos_d] = out["y_b_pred"].float()
        logits_c[pos_d] = out["logits_c"].float()
        y_a_true[pos] = batch["y_a"]
        y_b_true[pos] = batch["y_b"]
        y_c_true[pos] = batch["y_c"]
        if batch.get("y_bd") is not None:
            y_bd_true[pos] = batch["y_bd"].float()
            has_bd = True
    assert n_seen == n, f"evaluate: 只覆盖了 {n_seen}/{n} 条样本"
    genes = ds.samples["gene_id"].to_numpy()[np.asarray(idx, dtype=np.int64)] if n \
        else np.array([], dtype=object)
    loss_kw = dict(lambda_b=lambda_b, lambda_c=lambda_c, lambda_sign=lambda_sign, gamma=gamma,
                   beta=beta, lambda_a=lambda_a, loss_b=loss_b, huber_delta=huber_delta,
                   lambda_bd=lambda_bd, dense_on=dense_on, dense_delta=dense_delta)
    if c_ignore:  # 2026-09-28b：False 时 loss_kw 跟 26a 完全一样
        loss_kw["c_ignore"] = True
    return _eval_metrics(y_a_pred, y_b_pred, logits_c, y_a_true, y_b_true, y_c_true, genes,
                         class_counts, batch_size, device, loss_kw,
                         y_bd_true=y_bd_true if has_bd else None)


def _auroc_auprc(score: np.ndarray, pos: np.ndarray):
    """【第2批】AUROC(秩和公式，并列取平均秩) + AUPRC(平均精度)，跟 17 号脚本同一定义，
    手写 numpy/pandas，不引入 sklearn(项目一贯的依赖约束)。正/负类有一个为空时返回 NaN。"""
    pos = np.asarray(pos, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    auroc = float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    p = pos[np.argsort(-score, kind="mergesort")]
    prec = np.cumsum(p) / np.arange(1, len(p) + 1)
    return auroc, float(prec[p].sum() / n1)


def _eval_metrics(y_a_pred, y_b_pred, logits_c, y_a_true, y_b_true, y_c_true, genes,
                  class_counts, batch_size, device, loss_kw, y_bd_true=None) -> dict:
    """evaluate 的指标部分(第2批从 evaluate 里拆出来，数值定义一个字没改)，输入是按 idx 顺序
    排好的逐样本预测/标签。loss 仍是"按顺序每 batch_size 条切一块、各算 compute_total_loss、
    再取平均"。
    【2026-09-25b】y_bd_true(稠密目标，NaN=没有)不为 None 且有值时：loss 分项多 l_bd(λ_bd=0 时只
    监控)，指标多 r_bd(ŷ_B vs 稠密值，全部有值样本)、r_bd_ns(只在 y_b 为 NaN 的不显著样本上)、
    n_valid_bd。没有稠密值时这三个是 NaN/0，其余数值跟 25a 逐位相同。"""
    n = len(y_a_true)
    ya_d, yb_d, yc_d = y_a_true.to(device), y_b_true.to(device), y_c_true.to(device)
    yap_d, ybp_d, lgc_d = y_a_pred.to(device), y_b_pred.to(device), logits_c.to(device)
    use_bd = y_bd_true is not None and bool(torch.isfinite(y_bd_true).any())
    ybd_d = y_bd_true.to(device) if use_bd else None
    if not use_bd and loss_kw.get("lambda_bd"):
        print("提示：lambda_bd>0 但评估数据里没有稠密目标，这次评估的 loss 不含 l_bd 项")
        loss_kw = dict(loss_kw, lambda_bd=0.0)
    part_keys = ("l_a", "l_b", "l_c", "l_sign", "l_bd")
    loss_sum, n_chunks = torch.zeros((), device=device), 0
    part_sum = {k: torch.zeros((), device=device) for k in part_keys}
    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        _, parts = compute_total_loss(
            {"y_a_pred": yap_d[sl], "y_b_pred": ybp_d[sl], "logits_c": lgc_d[sl]},
            ya_d[sl], yb_d[sl], yc_d[sl], class_counts, return_tensors=True,
            y_bd=ybd_d[sl] if use_bd else None, **loss_kw)
        loss_sum += parts["total"]
        for k in part_keys:
            part_sum[k] += parts[k]
        n_chunks += 1
    loss = float(loss_sum / n_chunks) if n_chunks else float("nan")
    loss_parts = {k: (float(v / n_chunks) if n_chunks else float("nan"))
                  for k, v in part_sum.items()}

    y_a_pred, y_b_pred, logits_c = y_a_pred.cpu(), y_b_pred.cpu(), logits_c.cpu()
    y_a_true, y_b_true, y_c_true = y_a_true.cpu(), y_b_true.cpu(), y_c_true.cpu()

    mask_a = ~torch.isnan(y_a_true)
    n_valid_a = int(mask_a.sum())
    r_a = float(_pearson_r(y_a_pred[mask_a], y_a_true[mask_a])) if n_valid_a >= 2 \
        else float("nan")

    mask_b = ~torch.isnan(y_b_true)
    n_valid_b = int(mask_b.sum())
    if n_valid_b >= 2:
        r_b = float(_pearson_r(y_b_pred[mask_b], y_b_true[mask_b]))
        rmse_b = float(torch.sqrt(torch.mean((y_b_pred[mask_b] - y_b_true[mask_b]) ** 2)))
    else:
        r_b, rmse_b = float("nan"), float("nan")

    # 2026-09-28b：Head C 指标只在 y_c>=0 的行上算(第8批伪样本 y_c=-1)；全部有效时不切片，逐位不变
    lgc_c, yc_c = logits_c, y_c_true
    if n and not bool((y_c_true >= 0).all()):
        keep_c = y_c_true >= 0
        lgc_c, yc_c = logits_c[keep_c], y_c_true[keep_c]
    n_c = int(len(yc_c))
    macro_f1, f1_per_class = _macro_f1(lgc_c.argmax(-1), yc_c)
    acc_c = float((lgc_c.argmax(-1) == yc_c).float().mean()) if n_c else float("nan")
    probs = torch.softmax(lgc_c.float(), dim=-1).numpy()
    yc_np = yc_c.numpy()
    auroc_down, auprc_down = _auroc_auprc(probs[:, 0], yc_np == 0) if n_c else (float("nan"),) * 2
    auroc_up, auprc_up = _auroc_auprc(probs[:, 2], yc_np == 2) if n_c else (float("nan"),) * 2

    # 第三轮：Head A 的标签和(eval 下的)预测都只取决于基因，同一基因在这批样本里重复
    # 出现几十上百次；r_a 按样本算等于"按样本数加权的基因级 r"，r_a_gene 每个基因只算一次
    _, first_of_gene = np.unique(genes, return_index=True)
    fg = torch.from_numpy(first_of_gene.astype(np.int64))
    ya_g_pred, ya_g_true = y_a_pred[fg], y_a_true[fg]
    mg = ~torch.isnan(ya_g_true)
    r_a_gene = float(_pearson_r(ya_g_pred[mg], ya_g_true[mg])) if int(mg.sum()) >= 2 \
        else float("nan")

    r_bd, r_bd_ns, n_valid_bd = float("nan"), float("nan"), 0  # 2026-09-25b
    if use_bd:
        ybd_c = y_bd_true.cpu().float()
        m_bd = torch.isfinite(ybd_c)
        n_valid_bd = int(m_bd.sum())
        if n_valid_bd >= 2:
            r_bd = float(_pearson_r(y_b_pred[m_bd], ybd_c[m_bd]))
        m_bdn = m_bd & torch.isnan(y_b_true)
        if int(m_bdn.sum()) >= 2:
            r_bd_ns = float(_pearson_r(y_b_pred[m_bdn], ybd_c[m_bdn]))

    return dict(loss=loss, loss_parts=loss_parts, r_a=r_a, r_a_gene=r_a_gene, r_b=r_b,
                r_bd=r_bd, r_bd_ns=r_bd_ns, n_valid_bd=n_valid_bd,
                y_bd_true=y_bd_true.cpu() if use_bd else None,
                rmse_b=rmse_b, acc_c=acc_c, macro_f1_c=macro_f1, f1_per_class=f1_per_class,
                auroc_down=auroc_down, auroc_up=auroc_up, auprc_down=auprc_down,
                auprc_up=auprc_up, n_valid_a=n_valid_a, n_valid_b=n_valid_b, n_total=n,
                n_genes=int(len(first_of_gene)), y_a_pred=y_a_pred, y_a_true=y_a_true,
                y_b_pred=y_b_pred, y_b_true=y_b_true, logits_c=logits_c, y_c_true=y_c_true,
                genes=genes)


@torch.no_grad()
def evaluate_per_head(model, head_states: dict, ds, idx, class_counts: torch.Tensor,
                      batch_size: int = 128, **eval_kw) -> dict:
    """【2026-09-24 第2批】"各头各取最优权重"的组合评估：Head A 的预测来自 head_states["A"]
    那份权重、B 来自 "B"、C 来自 "C"(run_one_seed save_per_head_best=True 时记下的，按 val 上
    r_a_gene / r_b / AUPRC均值 各自选的)，三次前向后拼起来，用跟 evaluate 完全相同的
    _eval_metrics 算指标。只在 val 上选、test 上报告，不碰 test 标签。三份权重缺哪份就用
    model 当前权重(选中的那份)补。跑完把 model 恢复成调用前的权重。"""
    keep = {k: v.detach().clone() for k, v in model.state_dict().items()}
    evs = {}
    for h in ("A", "B", "C"):
        if head_states and h in head_states:
            model.load_state_dict(head_states[h])
        else:
            model.load_state_dict(keep)
        evs[h] = evaluate(model, ds, idx, class_counts, batch_size=batch_size, **eval_kw)
    model.load_state_dict(keep)
    device = eval_kw.get("device", "cpu")
    loss_kw = {k: eval_kw[k] for k in ("lambda_b", "lambda_c", "lambda_sign", "gamma", "beta",
                                        "lambda_a", "loss_b", "huber_delta", "lambda_bd",
                                        "dense_on", "dense_delta") if k in eval_kw}
    if eval_kw.get("c_ignore"):  # 2026-09-28b
        loss_kw["c_ignore"] = True
    return _eval_metrics(evs["A"]["y_a_pred"], evs["B"]["y_b_pred"], evs["C"]["logits_c"],
                         evs["A"]["y_a_true"], evs["A"]["y_b_true"], evs["A"]["y_c_true"],
                         evs["A"]["genes"], class_counts.to(device), batch_size, device, loss_kw,
                         y_bd_true=evs["A"].get("y_bd_true"))


def _macro_f1(pred_cls: torch.Tensor, true_cls: torch.Tensor):
    """三类(down=0,ns=1,up=2)的F1，手写而不依赖sklearn——这个项目到目前为止只用
    numpy/pandas/torch/tokenizers，不想为了一个F1额外加一个依赖，信心9/10
    (标准F1公式，没有含糊的地方)。返回(macro_f1, 每类F1的list，顺序down/ns/up)。"""
    f1s = []
    for cls in range(3):
        tp = int(((pred_cls == cls) & (true_cls == cls)).sum())
        fp = int(((pred_cls == cls) & (true_cls != cls)).sum())
        fn = int(((pred_cls != cls) & (true_cls == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


# --------------------------------------------------------------------------
# 4. bootstrap CI(对固定模型的测试集预测重采样，不重新训练)
# --------------------------------------------------------------------------
def bootstrap_ci(eval_result: dict, n_boot: int = 1000, ci: float = 0.95,
                 min_valid: int = 10, seed: int = 0, groups=None) -> dict:
    """对evaluate()返回的逐样本预测/标签做非参数bootstrap：有放回重采样n_boot次
    (每次重采样出跟原始子集一样多的样本)，每次重新算Head A的r、Head B的r/RMSE
    (只在这次重采样出的"非NaN/显著"子集里算，因为重采样会改变NaN比例)、Head C
    的macro F1，取(1-ci)/2和1-(1-ci)/2分位数当区间。这是衡量"测试集这批样本
    本身"带来的不确定性的标准做法，信心8/10；不是重新训练模型，跟5-seed之间的
    std是两种不同来源的不确定性，见文件头说明。

    min_valid(信心6/10，启发式阈值，不是从某个具体规则推导出来的)：一次重采样
    至少要有min_valid个非NaN点才计入r_a/r_b的bootstrap样本，不是最初写的"≥2个
    点就算"。这个改动是从16号脚本第一次真实跑出来的leave-TF-out结果里发现的
    真问题：原始8条评估样本里Head B只有1个非NaN点(点估计r_b正确地判定为NaN)，
    但≥2的阈值下，某些重采样恰好把这唯一一个点重复抽中2次以上，凑够了"2个点"，
    而"同一个点跟自己算相关系数"在_pearson_r的eps兜底下会算出≈0——这不是
    bug(每一步单独看都符合设计)，但组合起来的效果是bootstrap CI给出了一个
    看似"确定"的区间(r_b_ci=(0.0,0.0))，实际上背后只有1个真实数据点撑着，
    没有任何统计意义。min_valid=10只是把这个"至少要有几个点"的门槛抬高到一个
    不容易被"重复抽中同一个点"意外凑够的水平，本身没有理论最优值，真实数据上
    样本量大得多，这个改动的影响应该很小；如果某个评估切片(比如leave-TF-out
    某个held_out TF)本身有效点数常年低于10，说明的是这个切片数据量不够、不是
    该调这个阈值。
    groups【第三轮新增，信心：Head A 部分9/10，Head B/C 部分7/10】：传入每条样本所属的
    基因(evaluate 返回的 genes)时，改成按基因整群重采样(cluster bootstrap)——每次有放回
    地抽"跟原来一样多个基因"，被抽中的基因把它的全部样本一起带进来。原因：测试集约10万条
    样本其实只来自几百个基因(染色体 holdout 的独立单位是基因)，Head A 的标签和预测在同一
    基因的上百条样本里完全相同；按样本重采样时每个基因的权重只在 ±10% 左右晃动，相当于
    把几百个独立单位当成10万个，区间会窄一个数量级左右(约 √(每基因样本数) 倍)。Head B/C 的
    样本在同一基因内共享 cis/WT 表征，也不是独立的；同一 TF 跨基因也有相关，按基因整群只
    处理了主要那一维，所以 B/C 部分信心7/10。groups=None 时跟原来一样按样本重采样。
    整群模式下额外给 r_a_gene(每个被抽中的基因只算一次)的区间。"""
    rng = np.random.default_rng(seed)
    n = len(eval_result["y_a_true"])
    y_a_pred, y_a_true = eval_result["y_a_pred"], eval_result["y_a_true"]
    y_b_pred, y_b_true = eval_result["y_b_pred"], eval_result["y_b_true"]
    pred_c_all, y_c_true = eval_result["logits_c"].argmax(-1), eval_result["y_c_true"]

    n_valid_a_orig = int((~torch.isnan(y_a_true)).sum())
    n_valid_b_orig = int((~torch.isnan(y_b_true)).sum())
    if n_valid_a_orig < min_valid:
        print(f"提醒：这批评估样本里Head A只有{n_valid_a_orig}个非NaN标签"
             f"(<min_valid={min_valid})，r_a的bootstrap CI即使算出来了也不建议"
             "采信，样本量太小")
    if n_valid_b_orig < min_valid:
        print(f"提醒：这批评估样本里Head B只有{n_valid_b_orig}个非NaN(\"显著\")"
             f"标签(<min_valid={min_valid})，r_b的bootstrap CI即使算出来了也不"
             "建议采信，样本量太小")

    group_rows, first_rows = None, None
    if groups is not None and n:
        _, first_idx, inv = np.unique(np.asarray(groups), return_index=True,
                                      return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv)
        group_rows = np.split(order, np.cumsum(counts)[:-1])
        first_rows = first_idx
    n_units = len(group_rows) if group_rows is not None else n

    r_a_samples, r_a_gene_samples, r_b_samples, f1_samples = [], [], [], []
    for _ in range(n_boot):
        if group_rows is None:
            idx_np = rng.integers(0, n, size=n)
        else:
            chosen = rng.integers(0, n_units, size=n_units)
            idx_np = np.concatenate([group_rows[c] for c in chosen])
            fr = torch.from_numpy(first_rows[chosen].astype(np.int64))
            m_g = ~torch.isnan(y_a_true[fr])
            if m_g.sum() >= min_valid:
                r_a_gene_samples.append(float(_pearson_r(y_a_pred[fr][m_g],
                                                         y_a_true[fr][m_g])))
        idx_t = torch.from_numpy(idx_np.astype(np.int64))

        m_a = ~torch.isnan(y_a_true[idx_t])
        if m_a.sum() >= min_valid:
            r_a_samples.append(float(_pearson_r(y_a_pred[idx_t][m_a], y_a_true[idx_t][m_a])))

        m_b = ~torch.isnan(y_b_true[idx_t])
        if m_b.sum() >= min_valid:
            r_b_samples.append(float(_pearson_r(y_b_pred[idx_t][m_b], y_b_true[idx_t][m_b])))

        macro_f1, _ = _macro_f1(pred_c_all[idx_t], y_c_true[idx_t])
        f1_samples.append(macro_f1)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    def _pct(samples):
        if not samples:
            return (float("nan"), float("nan"))
        arr = np.array(samples)
        return (float(np.quantile(arr, lo_q)), float(np.quantile(arr, hi_q)))

    return dict(r_a_ci=_pct(r_a_samples), r_a_gene_ci=_pct(r_a_gene_samples),
               r_b_ci=_pct(r_b_samples), macro_f1_c_ci=_pct(f1_samples),
               unit="gene" if group_rows is not None else "sample", n_units=n_units,
               n_boot_effective=dict(r_a=len(r_a_samples), r_a_gene=len(r_a_gene_samples),
                                     r_b=len(r_b_samples), f1=len(f1_samples)),
               n_valid_orig=dict(r_a=n_valid_a_orig, r_b=n_valid_b_orig))


# --------------------------------------------------------------------------
# 5. 5-seed汇总
# --------------------------------------------------------------------------
def _eval_kwargs_from(train_kwargs: dict) -> dict:
    """把训练参数里跟 evaluate 有关的挑出来(test/leave-TF-out 评估用)。batch_size 故意
    不传：test 的 loss 仍按 evaluate 默认的 128 条一块算，跟改之前一致。
    【第2批】多透传 lambda_a/loss_b/huber_delta(loss 定义变了，test loss 要跟着变)。"""
    kw = {k: train_kwargs[k] for k in ("device", "num_workers", "amp", "forward_mode",
                                       "eval_batch_size", "lambda_b", "lambda_c",
                                       "lambda_sign", "gamma", "beta", "layout_buckets",
                                       "lambda_a", "loss_b", "huber_delta", "lambda_bd",
                                       "dense_on", "dense_delta")
          if k in train_kwargs}
    kw.setdefault("device", "cpu")
    return kw


_CONFIG_KEYS_NOT_AFFECTING_RESULT = ("device", "num_workers", "eval_batch_size", "log_every",
                                     "layout_buckets", "compile_model",
                                     "compile_cache_limit", "compile_dynamic",
                                     "compile_scope")  # 2026-09-23g，见文件头(f)
_BIG_EVAL_KEYS = ("y_a_pred", "y_a_true", "y_b_pred", "y_b_true", "logits_c", "y_c_true",
                  "genes", "y_bd_true")


def _ci_str(ci_pair) -> str:
    lo, hi = ci_pair
    return f"({lo:.4f}, {hi:.4f})"


def _print_eval_and_ci(tag: str, ev: dict, ci_gene: dict, ci_sample: dict = None):
    """第三轮：统一打印评估指标 + 两种口径的 bootstrap CI(按基因整群为主、按样本为对照)。
    【第2批】多打一行 Head C 的 AUROC/AUPRC；ci_sample=None 时不打按样本口径那行。"""
    print(f"{tag} r_a={ev['r_a']:.4f}(按样本加权，{ev['n_valid_a']} 条/{ev['n_genes']} 个基因)"
          f"  r_a_gene={ev['r_a_gene']:.4f}(每基因算一次)  "
          f"r_b={ev['r_b']:.4f}(n={ev['n_valid_b']})  rmse_b={ev['rmse_b']:.4f}  "
          f"acc_c={ev['acc_c']:.4f}  macro_f1_c={ev['macro_f1_c']:.4f}")
    if "auprc_down" in ev:
        print(f"{tag} Head C 排序能力: AUROC down/up={ev['auroc_down']:.4f}/{ev['auroc_up']:.4f}  "
              f"AUPRC down/up={ev['auprc_down']:.4f}/{ev['auprc_up']:.4f}  逐类F1(down/ns/up)="
              f"{[round(x, 4) for x in ev['f1_per_class']]}")
    if np.isfinite(ev.get("r_bd", float("nan"))):  # 2026-09-25b
        print(f"{tag} Head B vs 稠密 log2FC: r_bd={ev['r_bd']:.4f}(n={ev['n_valid_bd']})  "
              f"只看不显著样本 r_bd_ns={ev['r_bd_ns']:.4f}")
    print(f"{tag} bootstrap 95% CI【按基因整群重采样，{ci_gene['n_units']} 个基因，主口径】: "
          f"r_a={_ci_str(ci_gene['r_a_ci'])}  r_a_gene={_ci_str(ci_gene['r_a_gene_ci'])}  "
          f"r_b={_ci_str(ci_gene['r_b_ci'])}  macro_f1_c={_ci_str(ci_gene['macro_f1_c_ci'])}")
    if ci_sample is not None:
        print(f"{tag} (对照：按样本重采样的旧口径，偏窄) r_a={_ci_str(ci_sample['r_a_ci'])}  "
              f"r_b={_ci_str(ci_sample['r_b_ci'])}  "
              f"macro_f1_c={_ci_str(ci_sample['macro_f1_c_ci'])}")


def _reusable_checkpoint(ckpt_path: str, train_config: dict):
    """断点续跑用：ckpt_path 存在、code_version 跟当前代码一致、train_config(除了
    不影响结果的 device/num_workers/eval_batch_size/log_every)也一致 -> 返回 (ckpt, None)；
    否则返回 (None, 原因)。文件存在但不一致时，把它改名成 .bak_<旧版本>_<时间> 留着
    (旧 checkpoint 里的 test 指标可能还有参考价值)，不直接覆盖。"""
    if not os.path.exists(ckpt_path):
        return None, "不存在"
    try:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:  # 很老的 PyTorch 没有 weights_only 参数
            ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:  # 文件损坏(比如上次写到一半被杀)也当作不可复用
        ckpt, reason = None, f"读取失败({type(e).__name__})"
    else:
        want = {k: v for k, v in train_config.items()
                if k not in _CONFIG_KEYS_NOT_AFFECTING_RESULT}
        got = {k: v for k, v in (ckpt.get("train_config") or {}).items()
               if k not in _CONFIG_KEYS_NOT_AFFECTING_RESULT and k != "seed"}
        for k, v in _NEW_KEY_DEFAULTS.items():  # 2026-09-25a：旧 checkpoint 缺的新键按默认值比
            if k in want and k not in got:
                got[k] = v
        if ckpt.get("code_version") not in (CODE_VERSION,) + tuple(_COMPATIBLE_CODE_VERSIONS):
            reason = f"代码版本 {ckpt.get('code_version')} != 当前 {CODE_VERSION}"
        elif got != want:
            diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
            reason = f"训练配置不同: {diff}"
        else:
            return ckpt, None
    tag = str(ckpt.get("code_version") if isinstance(ckpt, dict) else "unreadable")
    bak = f"{ckpt_path}.bak_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.replace(ckpt_path, bak)
    return None, f"{reason}；旧文件已改名为 {bak}"


def run_multi_seed(ds, split_idx: dict, seeds=(42, 123, 456, 789, 2024),
                   class_counts: torch.Tensor = None, n_boot: int = 1000,
                   save_dir: str = None, resume: bool = True, **train_kwargs):
    """跑多个seed(默认5个，status文件"5-seed"的要求)，每个seed单独train+早停，
    在各自选中的权重上对test集合做evaluate()+bootstrap_ci()，最后把各
    seed的指标汇总成mean±std(跨seed的变异，衡量训练本身的不稳定性——初始化、
    数据顺序等随机性)。这跟单seed内部的bootstrap CI(衡量"这批测试样本本身"的
    不确定性)是两种不同来源的不确定性，两个都保留、分开报告，不互相替代。
    save_dir给了就把每个seed的最优权重+历史+测试指标存成.pt文件(真实训练跑
    很久，不想因为进程中断丢结果)。

    resume【第二轮提速顺带加的，信心8/10】：save_dir 里已经有同一代码版本、同一训练
    配置的 seed{s}_best.pt，就直接读它存的 history/test 指标，不重训这个 seed(每个
    seed 开头都 torch.manual_seed(seed)，各 seed 之间互不影响，所以跳过已完成的 seed
    跟一口气跑完是同一个结果)。版本/配置不一致的旧文件改名备份后重训。
    【2026-09-24 第2批】checkpoint 新增字段：model_kwargs(17号据此重建模型，不再靠猜)、
    best(选中的是第几次验证、按什么判据、各头最优在哪)、head_states(save_per_head_best 时
    三个头各自的最优权重)、test_eval_per_head/test_ci_per_head("各头各取最优权重"的 test
    指标)。写文件改成先写临时文件再 os.replace，进程在写的中途被杀不会留下半个文件。"""
    if class_counts is None:
        class_counts = _default_class_counts()
    eval_kw = _eval_kwargs_from(train_kwargs)
    train_config = {k: v for k, v in train_kwargs.items() if k != "device"}
    # 2026-09-25b(文件头第14条(c))：λ_bd=0 时稠密标签只用于监控，不影响训练结果 -> 按默认值存；
    # λ_bd>0 时记下稠密目标的指纹，21 号换了文件内容不会误复用
    if not train_config.get("lambda_bd"):
        for _k in ("dense_target", "dense_on", "dense_delta"):
            if _k in train_config:
                train_config[_k] = _NEW_KEY_DEFAULTS[_k]
    else:
        if hasattr(ds, "set_dense_target"):
            ds.set_dense_target(train_config.get("dense_target"), verbose=True)
        _v = np.asarray(getattr(ds, "_y_bd_arr", np.zeros(0)), dtype=np.float64)
        train_config["dense_target_digest"] = f"{int(np.isfinite(_v).sum())}:{float(np.nansum(_v)):.6f}"

    all_results = []
    for seed in seeds:
        if save_dir and resume:
            ckpt, why = _reusable_checkpoint(os.path.join(save_dir, f"seed{seed}_best.pt"),
                                             train_config)
            if ckpt is not None:
                print(f"[seed{seed}] 已有同版本、同配置的 checkpoint，跳过训练，直接用里面存的"
                      f"test 指标：r_a={ckpt['test_eval']['r_a']:.4f}  "
                      f"macro_f1_c={ckpt['test_eval']['macro_f1_c']:.4f}")
                all_results.append(dict(seed=seed, best_epoch=ckpt["best_epoch"],
                                        best=ckpt.get("best"), history=ckpt["history"],
                                        test_eval=ckpt["test_eval"], test_ci=ckpt["test_ci"],
                                        test_eval_per_head=ckpt.get("test_eval_per_head"),
                                        model=None))
                continue
            if why != "不存在":
                print(f"[seed{seed}] 现有 checkpoint 不能复用：{why}")
        model, history, best = run_one_seed(ds, split_idx, seed, class_counts=class_counts,
                                            **train_kwargs)
        t_test = time.time()
        test_eval = evaluate(model, ds, split_idx["test"], class_counts, **eval_kw)
        print(f"[seed{seed}] test 评估用时 {time.time() - t_test:.0f} 秒")
        test_ci = bootstrap_ci(test_eval, n_boot=n_boot, seed=seed, groups=test_eval["genes"])
        test_ci["sample_level"] = bootstrap_ci(test_eval, n_boot=n_boot, seed=seed)
        _print_eval_and_ci(f"[seed{seed}] test:", test_eval, test_ci, test_ci["sample_level"])
        test_eval_light = {k: v for k, v in test_eval.items() if k not in _BIG_EVAL_KEYS}

        ph_light, test_ci_ph = None, None
        if best.get("head_states"):
            ev_ph = evaluate_per_head(model, best["head_states"], ds, split_idx["test"],
                                      class_counts, **eval_kw)
            test_ci_ph = bootstrap_ci(ev_ph, n_boot=n_boot, seed=seed, groups=ev_ph["genes"])
            _print_eval_and_ci(f"[seed{seed}] test(各头各取最优权重):", ev_ph, test_ci_ph)
            ph_light = {k: v for k, v in ev_ph.items() if k not in _BIG_EVAL_KEYS}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"seed{seed}_best.pt")
            tmp_path = ckpt_path + ".tmp"
            torch.save({"model_state": model.state_dict(), "history": history,
                        "best_epoch": best["epoch"],
                        "best": {k: v for k, v in best.items() if k != "head_states"},
                        "model_kwargs": best["model_kwargs"],
                        "head_states": best.get("head_states"),
                        "test_eval": test_eval_light, "test_ci": test_ci,
                        "test_eval_per_head": ph_light, "test_ci_per_head": test_ci_ph,
                        "code_version": CODE_VERSION,
                        "train_config": dict(train_config, seed=seed)}, tmp_path)
            os.replace(tmp_path, ckpt_path)
            print(f"[seed{seed}] 已保存: {ckpt_path}")

        all_results.append(dict(seed=seed, best_epoch=best["epoch"],
                                best={k: v for k, v in best.items() if k != "head_states"},
                                history=history, test_eval=test_eval_light, test_ci=test_ci,
                                test_eval_per_head=ph_light, model=model))

    metrics = ("r_a", "r_a_gene", "r_b", "rmse_b", "acc_c", "macro_f1_c", "auroc_down",
               "auroc_up", "auprc_down", "auprc_up", "r_bd", "r_bd_ns")
    for key, title in (("test_eval", "选中的权重"), ("test_eval_per_head", "各头各取最优权重")):
        rows = [r[key] for r in all_results if r.get(key)]
        if not rows:
            continue
        print(f"\n跨{len(rows)}个seed汇总【{title}】(mean±std，衡量训练过程本身的不稳定性)：")
        for metric in metrics:
            vals = [r.get(metric, float("nan")) for r in rows]
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            if vals:
                print(f"  {metric}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                      f"(n={len(vals)}/{len(rows)}个seed给出了有效值)")
            elif metric in rows[0]:
                print(f"  {metric}: 所有seed都是NaN(可能是这个划分里对应的非NaN样本太少)")
    return all_results


# --------------------------------------------------------------------------
# 6. TF维度leave-TF-out验证("泛化到未见扰动"的声明依据)
# --------------------------------------------------------------------------
def run_leave_tf_out(ds, split_idx: dict, held_out_tfs, seed: int = 0,
                     class_counts: torch.Tensor = None, n_boot: int = 1000,
                     **train_kwargs):
    """从train集合里再挖掉tf_depleted∈held_out_tfs的样本(不管gene在哪个染色体
    切分里)，正常训练，然后专门在"tf_depleted∈held_out_tfs 且 gene在val/test
    染色体切分里"的样本上评估——限制在val/test染色体是为了公平：避免评估集里
    混进"本来就在train染色体里、只是这条(gene,TF)记录因为TF被摘掉了"的样本，
    那样测的是"没见过这个TF但见过这个gene"而不是纯粹的"没见过这个TF"，
    信心7/10(操作化定义见文件头第4条)。
    【第2批】ctx_mode 不是 "marker" 时打印警告：legacy/relative 下被留出 TF 的 ctx_d 是用它
    自己的耗竭实验结果算的(status 7.3②)，这时"泛化到没见过的扰动"的结论不成立。"""
    if class_counts is None:
        class_counts = _default_class_counts()
    eval_kw = _eval_kwargs_from(train_kwargs)
    if train_kwargs.get("ctx_mode", "legacy") != "marker":
        print(f"⚠ leave-TF-out 用的 ctx_mode={train_kwargs.get('ctx_mode', 'legacy')}：被留出 TF "
              "的 ctx_d 含有它自己耗竭实验测到的 mRNA 变化，等于把扰动后的测量喂给了预测扰动"
              "结果的模型——正式的 leave-TF-out 结论请用 ctx_mode=\"marker\"(见 09 号第2批说明)")

    train_keep, _ = leave_tf_out_split(ds, held_out_tfs, base_idx=split_idx["train"])
    print(f"leave-TF-out: 从train集合({len(split_idx['train'])}条)里剔除"
         f"tf_depleted∈{list(held_out_tfs)}的样本后剩{len(train_keep)}条")

    eval_pool = split_idx["val"] + split_idx["test"]
    _, heldout_eval_idx = leave_tf_out_split(ds, held_out_tfs, base_idx=eval_pool)
    print(f"评估集(val+test里tf_depleted∈held_out_tfs的样本): {len(heldout_eval_idx)}条")
    if not heldout_eval_idx:
        print("警告：val/test染色体切分里没有任何样本的tf_depleted落在held_out_tfs "
             "集合里，这次leave-TF-out实验测不出东西，换一批held_out_tfs再试(比如"
             "确认这些TF确实在val/test染色体的基因里出现过被耗竭记录)")
        return None

    modified_split = dict(train=train_keep, val=split_idx["val"], test=split_idx["test"])
    model, history, best = run_one_seed(ds, modified_split, seed,
                                        class_counts=class_counts, **train_kwargs)
    heldout_eval = evaluate(model, ds, heldout_eval_idx, class_counts, **eval_kw)
    heldout_ci = bootstrap_ci(heldout_eval, n_boot=n_boot, seed=seed,
                              groups=heldout_eval["genes"])
    heldout_ci["sample_level"] = bootstrap_ci(heldout_eval, n_boot=n_boot, seed=seed)
    _print_eval_and_ci("leave-TF-out 评估:", heldout_eval, heldout_ci,
                       heldout_ci["sample_level"])

    heldout_eval_light = {k: v for k, v in heldout_eval.items() if k not in _BIG_EVAL_KEYS}
    return dict(model=model, history=history, best_epoch=best["epoch"],
               best={k: v for k, v in best.items() if k != "head_states"},
               heldout_eval=heldout_eval_light, heldout_ci=heldout_ci)


# --------------------------------------------------------------------------
# 7. 真实训练入口(接09/08/13号脚本的真实产出)
# --------------------------------------------------------------------------
def run_full_training(layout: str = "out/tf_layout.parquet",
                      labels: str = "out/head_bc_labels.parquet",
                      head_a: str = "out/head_a_baseline_logtpm.parquet",
                      sgd: str = "data/SGD_features.tab",
                      promoter_tokens: str = "out/promoter_token_ids.parquet",
                      bpe_tokenizer: str = "out/bpe_tokenizer.json",
                      vocab_size: int = 4000, d_model: int = 256, n_heads: int = 8,
                      cis_layers: int = 6, lay_layers: int = 4, batch_size: int = 64,
                      n_epochs: int = 30, lr: float = 1e-4, weight_decay: float = 1e-5,
                      patience: int = 5, seeds=(42, 123, 456, 789, 2024),
                      lambda_b: float = 1.0,
                      lambda_c: float = 1.0, lambda_sign: float = 0.1,
                      gamma: float = 2.0, beta: float = 0.999, n_boot: int = 1000,
                      held_out_tfs=None, save_dir: str = "out/checkpoints",
                      device: str = None, num_workers: int = 4, amp: str = "bf16",
                      forward_mode: str = "grouped", tfs_per_gene: int = 4,
                      eval_batch_size: int = 512, resume: bool = True,
                      layout_buckets: int = 1, compile_model: bool = False,
                      compile_cache_limit: int = 16, compile_dynamic: str = "auto",
                      compile_scope: str = "full",
                      ctx_mode: str = "legacy", ctx_clip: float = 3.0,
                      head_c_mode: str = "delta", dropout: float = 0.1,
                      lambda_a: float = 1.0, loss_b: str = "mse", huber_delta: float = 1.0,
                      optimizer: str = "adam", grad_clip: float = 0.0,
                      lr_schedule: str = "constant", warmup_steps: int = 0,
                      min_lr_ratio: float = 0.05, val_every: float = 1.0,
                      select_metric: str = "val_loss", save_per_head_best: bool = False,
                      max_steps: int = 0, ds=None, split_idx: dict = None,
                      min_delta: float = 1e-6, dense_target: str = None, lambda_bd: float = 0.0,
                      dense_on: str = "ns", dense_delta: float = None,
                      ema_decay: float = 0.0, ablation: str = "none", head_a_all: bool = False,
                      wt_input: str = "none"):
    """接真实09号Dataset+08号切分文件，跑完整的train/val/test(5-seed+bootstrap
    CI) + 可选的leave-TF-out。vocab_size=4000要跟13号脚本训练出的bpe_tokenizer
    实际词表大小一致；d_model/n_heads/cis_layers/lay_layers默认用《方案.txt》
    写死的256/8/6/4规格。

    seeds【2026-09-23按用户要求改成42/123/456/789/2024，信心10/10——用户直接指定的值】。
    num_workers默认4，见run_one_seed的说明。
    【第二轮提速】amp/forward_mode/tfs_per_gene/eval_batch_size 见文件头第9条；这一层默认
    bf16 + grouped + K=4。完全退回改之前的行为：amp="off", forward_mode="legacy",
    tfs_per_gene=1。resume 见 run_multi_seed。
    【第三轮/第四轮(a)】layout_buckets 默认1(bench.txt ⑥实测分1段最快)。
    【第四轮~2026-09-23g】compile_* 默认关，见文件头第11条(b)(e)(f)。
    【2026-09-24 第2批】ctx_mode … max_steps 这一串参数见 run_one_seed 同名说明和文件头
    第12条，默认值全部等于原来的行为(所以 run_all.sh --train/--smoke 的老调用方式训出来的
    仍是 run1 那套配置)；第2批的新配置在 18_run_experiments.py 的预设里。
    ds/split_idx：可以传入已经建好的 TFLayoutDataset 和切分(18 号脚本连跑多个实验时复用，
    建一次 Dataset 要一两分钟)；不传就跟原来一样现建。
    【2026-09-25b】dense_target/lambda_bd/dense_on/dense_delta：Head B 稠密辅助目标，见 run_one_seed
    同名说明和文件头第14条；默认不加载、λ_bd=0，跟 25a 完全一样。
    【2026-09-26a】ema_decay：权重滑动平均，默认0=关(跟 25b 逐位相同)，见文件头第15条。
    ablation：输入消融，默认 none(跟 25b 逐位相同)，见文件头第16条。
    【2026-09-28b】head_a_all：网格外基因的 Head A 伪样本并进训练/验证，默认 False(跟 26a 逐位相同)，见文件头第17条。
    【2026-10-05a】wt_input：实测 WT 表达只进 Head B/C，默认 "none"(跟 28b 逐位相同)，见文件头第18条。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if forward_mode == "legacy" and tfs_per_gene != 1:
        print(f"提示：forward=legacy 时 tfs_per_gene={tfs_per_gene} 不生效(legacy 训练"
              "loader 就是原来的 shuffle=True)")
        tfs_per_gene = 1
    print(f"设备: {device}  DataLoader num_workers: {num_workers}  代码版本: {CODE_VERSION}")
    print(f"提速配置: forward={forward_mode}  amp={amp}  tfs_per_gene={tfs_per_gene}"
          f"(每个训练batch约 {max(1, batch_size // max(1, tfs_per_gene))} 个不同基因)  "
          f"eval_batch_size={eval_batch_size}  layout_buckets={layout_buckets}  "
          f"compile={compile_model}" +
          (f"(scope={compile_scope}  dynamic={compile_dynamic}  "
           f"cache_size_limit={compile_cache_limit}，见文件头第11条(e)(f))"
           if compile_model else ""))
    if str(device).startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}  bf16支持: "
              f"{torch.cuda.is_bf16_supported()}  PyTorch {torch.__version__}")
    if ds is None:
        ds = TFLayoutDataset(layout, labels, head_a, sgd, promoter_tokens, bpe_tokenizer,
                             ctx_mode=ctx_mode, ctx_clip=ctx_clip)
    elif hasattr(ds, "set_ctx_mode"):
        ds.set_ctx_mode(ctx_mode, ctx_clip)
    if hasattr(ds, "set_ablation"):
        ds.set_ablation(ablation)
    print(f"样本数: {len(ds)}，TF词表大小: {ds.n_tf}")
    if split_idx is None:
        split_idx = build_split_indices(ds, head_a)
    _report_need_d(ds)
    _report_chunk_gene_diversity(ds, split_idx["val"], batch_size, "val_loss(早停判据)")
    _report_chunk_gene_diversity(ds, split_idx["test"], 128, "test loss")

    print("\n=== 精确类别计数(替换15号脚本里DEFAULT_CLASS_COUNTS_APPROX的近似值) ===")
    exact_counts = ds.samples["direction_3class"].value_counts()
    class_counts = torch.tensor([float(exact_counts.get(c, 0.0))
                                 for c in ("down", "ns", "up")])
    print(f"精确计数: down={float(class_counts[0]):.0f}  "
         f"ns={float(class_counts[1]):.0f}  up={float(class_counts[2]):.0f}")
    _eff = 1.0 - np.power(float(beta), class_counts.numpy().astype(np.float64))
    _w = (1.0 - float(beta)) / np.maximum(_eff, 1e-12)
    _w = _w / _w.sum() * 3
    print(f"Head C class-balanced 权重(β={beta}，归一化到均值1) down/ns/up = "
          f"{[round(float(x), 5) for x in _w]}  focal γ={gamma}" +
          ("  ⚠ 三类权重几乎相同：1-β^n 在 n≳5/(1-β) 时饱和到1，class-balanced 这一项实际不起"
           "作用(文件头第13条(c))" if _w.max() / _w.min() < 1.05 else ""))

    common_kwargs = dict(n_tf=ds.n_tf, vocab_size=vocab_size, pad_token_id=ds.pad_id,
                         d_model=d_model, n_heads=n_heads, cis_layers=cis_layers,
                         lay_layers=lay_layers, batch_size=batch_size,
                         n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
                         patience=patience, lambda_b=lambda_b, lambda_c=lambda_c,
                         lambda_sign=lambda_sign, gamma=gamma, beta=beta, device=device,
                         num_workers=num_workers, amp=amp, forward_mode=forward_mode,
                         tfs_per_gene=tfs_per_gene, eval_batch_size=eval_batch_size,
                         layout_buckets=layout_buckets, compile_model=compile_model,
                         compile_cache_limit=compile_cache_limit,
                         compile_dynamic=compile_dynamic, compile_scope=compile_scope,
                         ctx_mode=ctx_mode, ctx_clip=ctx_clip, head_c_mode=head_c_mode,
                         dropout=dropout, lambda_a=lambda_a, loss_b=loss_b,
                         huber_delta=huber_delta, optimizer=optimizer, grad_clip=grad_clip,
                         lr_schedule=lr_schedule, warmup_steps=warmup_steps,
                         min_lr_ratio=min_lr_ratio, val_every=val_every,
                         select_metric=select_metric,
                         save_per_head_best=save_per_head_best, max_steps=max_steps,
                         min_delta=min_delta, dense_target=dense_target, lambda_bd=lambda_bd,
                         dense_on=dense_on, dense_delta=dense_delta, ema_decay=ema_decay,
                         ablation=ablation, head_a_all=bool(head_a_all), wt_input=str(wt_input))

    print("\n=== 主实验：多seed + 染色体holdout + bootstrap CI ===")
    main_results = run_multi_seed(ds, split_idx, seeds=seeds, class_counts=class_counts,
                                  n_boot=n_boot, save_dir=save_dir, resume=resume,
                                  **common_kwargs)

    lto_results = None
    if held_out_tfs:
        print(f"\n=== leave-TF-out验证：held_out_tfs={held_out_tfs} ===")
        lto_results = run_leave_tf_out(ds, split_idx, held_out_tfs, seed=seeds[0],
                                       class_counts=class_counts, n_boot=n_boot,
                                       **common_kwargs)

    return dict(ds=ds, split_idx=split_idx, class_counts=class_counts,
               main_results=main_results, lto_results=lto_results)


# --------------------------------------------------------------------------
# 8. 实测模式(--mode bench，第二轮提速新增)：真实数据上的等价性核对 + 各配置的
#    训练/评估吞吐、显存。提速到底多少以这里的实测为准。
# --------------------------------------------------------------------------
def run_benchmark(layout: str = "out/tf_layout.parquet",
                  labels: str = "out/head_bc_labels.parquet",
                  head_a: str = "out/head_a_baseline_logtpm.parquet",
                  sgd: str = "data/SGD_features.tab",
                  promoter_tokens: str = "out/promoter_token_ids.parquet",
                  bpe_tokenizer: str = "out/bpe_tokenizer.json",
                  vocab_size: int = 4000, d_model: int = 256, n_heads: int = 8,
                  cis_layers: int = 6, lay_layers: int = 4, batch_size: int = 64,
                  tfs_per_gene: int = 4, eval_batch_size: int = 512, num_workers: int = 4,
                  bench_steps: int = 30, bench_warmup: int = 5, n_eval_genes: int = 60,
                  n_epochs: int = 30, n_seeds: int = 5, lr: float = 1e-4,
                  weight_decay: float = 1e-5, device: str = None, ds=None,
                  split_idx: dict = None, class_counts: torch.Tensor = None,
                  seed: int = 0, layout_buckets: int = 1, profile_steps: int = 10,
                  compile_model: bool = False, compile_cache_limit: int = 16,
                  compile_warmup_timeout: float = 300.0,
                  bench_batch_sweep: bool = False,
                  bench_batch_sweep_sizes: str = None) -> dict:
    """不训练、只测量，九段输出(都打印出来，也放进返回的字典)：
      ① 数据形状：每基因 cis token 数、L_g 位点数、D∈L_g 样本占比(决定分组前向省多少)
      ② 真实数据上 分组前向 vs 逐样本前向 逐元素对照(eval 模式，fp32 应一致；另报
         bf16 跟 fp32 的差，给个量级)。对照前把零初始化的 FiLM 末层、ψ_corr 末层随机化，
         否则 ctx_d/ψ_corr 走错行也测不出来
      ③ 改之前的逐样本 fp32 前向里 cis / WT layout / D layout / 融合+头 各占多少时间
         (只测前向，粗略；另外单列 layout 里距离偏置 einsum 的占比，给下一轮优化定方向)
      ④ 评估速度：逐样本fp32 / 分组fp32 / 分组bf16，并核对三者预测与 loss 一致
      ⑤ DataLoader 单独吞吐(不做 GPU 计算)：如果低于⑥的训练吞吐，瓶颈又回到了数据准备
      ⑥ 训练吞吐表：逐样本fp32 K=1(≈最初版本) / 分组bf16 K 不分段(=第二轮默认) /
         分组bf16 K 分2段、3段(、layout_buckets 段)，每种先热身 bench_warmup 步再计时
         bench_steps 步，报 样本/s、峰值显存、折算的 1 个 epoch 训练分钟数。某个配置 OOM
         就跳过、继续测下一个。最后一行给出分段数的实测推荐
      ⑦【第三轮新增】对默认配置用 torch.profiler 跑 profile_steps 步：按 GPU 自身耗时列出
         最贵的算子，并用"每步 GPU 实际忙碌时间 / ⑥ 实测每步墙钟时间"估计 GPU 忙碌占比——
         占比明显低于 1 说明瓶颈在 CPU 侧(Python/kernel 启动)，下一轮该往那个方向改
      ⑧【第四轮新增，compile_model=True(即命令行 --compile)才跑，默认不跑；2026-09-23f
         加了安全网+改成三组；2026-09-23g(本次) 加第四组，见文件头第11条(e)(f)】
         torch.compile 对照：默认配置(分组+bf16+当前 layout_buckets)分别用①不编译
         ②torch.compile(scope=full,dynamic=auto)③torch.compile(scope=full,
         dynamic=True)④torch.compile(scope=submodules,dynamic=auto，2026-09-23g
         新增，只编译 cis/layout 两个子模块，见 _maybe_compile_model)四种，各自跑一遍
         训练吞吐对比。编译需要额外热身吸收一次性JIT编译耗时，用的是
         max(bench_warmup,20)而不是⑥用的bench_warmup，避免JIT编译时间被错记进
         "这个配置天生就慢"。两层安全网(都不改变任何数值结果，2026-09-23f新增)：
         (i) 编译前设 torch._dynamo.config.cache_size_limit=compile_cache_limit
         (默认16)，重编译次数超过这个数就自动退回eager，不再无限重编译；(ii) 热身
         阶段每步后检查累计用时，超过 compile_warmup_timeout(默认300秒)就主动放弃
         这一组、打印诊断，不用再靠用户手动Ctrl+Z——这是bench_--compile.txt里
         "dynamic=True编译后反复重新编译、刷屏两分多钟、最后手动^Z中止"这个真实问题
         的直接修复，详细根因分析见文件头第11条(e)。
      ⑨【2026-09-23g本次新增，bench_batch_sweep=True(即--bench-batch-sweep)才跑，
         默认不跑，见文件头第11条(f)】batch_size 扫描：默认配置(分组+bf16+当前
         K/layout_buckets)下，依次测 bench_batch_sweep_sizes 给的几个 batch_size
         (不给就自动测当前 batch_size 的 1×/2×/4×)，流程照抄⑥(热身bench_warmup步、
         计时bench_steps步)，报 样本/s、相对当前batch_size的倍数、峰值显存。动机：
         ⑦如果测出GPU忙碌占比明显低于100%，说明有一部分是固定的kernel启动开销、
         大致不随batch_size变，同样开销摊给更多样本可能有超线性收益，这是⑧节
         compile方向如果不划算时的替代路线(文件头第11条e结尾已经提过，这次接成
         可跑的代码)，只测量不建模，结论以实测为准。
    【第三轮】②④⑤⑥ 都加了 layout 分段(layout_buckets)的对照。
    ds/split_idx/class_counts 给了就直接用(self_test 用假数据走一遍这段代码)，不给就
    按 run_full_training 同样的方式读真实数据。每种配置的模型都用同一个 seed 新建，
    测的是速度，不是训练效果。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = str(device).startswith("cuda")
    bench_steps = max(1, int(bench_steps))
    bench_warmup = max(0, int(bench_warmup))

    def _sync():
        if is_cuda:
            torch.cuda.synchronize()

    def _free():
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

    def _is_oom(err):
        return "out of memory" in str(err).lower()

    print(f"=== 实测模式(bench)  代码版本 {CODE_VERSION}  设备 {device}  "
          f"PyTorch {torch.__version__} ===")
    if is_cuda:
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}  bf16支持: {torch.cuda.is_bf16_supported()}"
              f"  空闲显存 {free_b / 2**30:.1f}/{total_b / 2**30:.1f} GiB")
        if free_b < 0.8 * total_b:
            print("  ⚠ 空闲显存不到总量的 80%：大概率还有别的进程(比如之前那个正式训练)在用这块"
                  " GPU，会跟这次实测抢算力(测出来偏慢)，逐样本 fp32 配置还可能 OOM。"
                  "建议先 nvidia-smi 看 PID，确认后 kill 掉再测")
    if ds is None:
        ds = TFLayoutDataset(layout, labels, head_a, sgd, promoter_tokens, bpe_tokenizer)
        split_idx = build_split_indices(ds, head_a)
    if class_counts is None:
        exact = ds.samples["direction_3class"].value_counts()
        class_counts = torch.tensor([float(exact.get(c, 0.0)) for c in ("down", "ns", "up")])
    class_counts = class_counts.to(device)
    results = {}

    # ---------------- ① 数据形状 ----------------
    print("\n--- ① 数据形状(决定各分支的计算量) ---")
    results["frac_need_d"] = _report_need_d(ds)
    genes_all = ds.samples["gene_id"].to_numpy()
    tr_arr = np.asarray(split_idx["train"], dtype=np.int64)
    tr_genes = pd.unique(genes_all[tr_arr])
    cis_len = np.array([len(ds.gene2tokens.get(g, ())) for g in tr_genes])
    lay_len = np.array([len(ds.layout_by_gene[g]["tf_idx"]) if g in ds.layout_by_gene else 0
                        for g in tr_genes])
    print(f"train: {len(tr_arr)} 条样本 / {len(tr_genes)} 个基因 = 平均每个基因 "
          f"{len(tr_arr) / max(len(tr_genes), 1):.1f} 条(逐样本前向下，每个基因的 cis 分支"
          "每个 epoch 就要重复算这么多次)")
    print(f"每基因 cis token 数: 中位 {np.median(cis_len):.0f}  90%分位 "
          f"{np.percentile(cis_len, 90):.0f}  最大 {cis_len.max()}")
    print(f"每基因 L_g 位点数:   中位 {np.median(lay_len):.0f}  90%分位 "
          f"{np.percentile(lay_len, 90):.0f}  最大 {lay_len.max()}")
    results.update(cis_len_median=float(np.median(cis_len)),
                   lay_len_median=float(np.median(lay_len)))
    results["val_chunk_genes"] = _report_chunk_gene_diversity(ds, split_idx["val"], batch_size,
                                                              "val_loss(早停判据)")

    # ---------------- ② 真实数据逐元素对照 ----------------
    print("\n--- ② 分组前向 vs 逐样本前向：真实数据逐元素对照(eval 模式) ---")
    torch.manual_seed(seed)
    model = SiameseHeadsModel(ds.n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers,
                              pad_token_id=ds.pad_id).to(device)
    with torch.no_grad():
        for lin in (model.fusion.condition.net[-1], model.psi_corr[-1]):
            lin.weight.normal_(0.0, 0.05)
            lin.bias.normal_(0.0, 0.05)
    model.eval()
    val_arr = np.asarray(split_idx["val"], dtype=np.int64)
    val_genes = genes_all[val_arr]
    eq_idx = []
    for g in pd.unique(val_genes):  # 整基因取，凑够约 160 条(2 个基因左右)
        eq_idx.extend(val_arr[val_genes == g].tolist())
        if len(eq_idx) >= 160:
            break
    full = ds.collate_fn([ds[i] for i in eq_idx])
    grp = TFLayoutDataset.group_collated(full, ds.tf2idx, ds.pad_id)
    # 第三轮：同一批样本按长度分段(强制真的切开：chunk_penalty=0)，fp32 也应该一致
    nb = max(int(layout_buckets), 2)
    grp_b = TFLayoutDataset.group_collated(full, ds.tf2idx, ds.pad_id, layout_buckets=nb)
    grp_b["layout_wt_splits"] = _ds_mod._length_buckets(
        grp_b["layout_wt"]["mask"].sum(1).tolist(), nb, chunk_penalty=0)
    grp_b["layout_d_splits"] = _ds_mod._length_buckets(
        grp_b["layout_d"]["mask"].sum(1).tolist(), nb, chunk_penalty=0) \
        if grp_b["d_rows"].numel() else []
    o_leg = o_grp = o_bf = o_bk = None
    try:
        with torch.no_grad():
            o_leg = _forward_batch(model, full, ds.tf2idx, device, "legacy")
            o_grp = _forward_batch(model, grp, ds.tf2idx, device, "grouped")
            o_bk = _forward_batch(model, grp_b, ds.tf2idx, device, "grouped")
            with _autocast(device, "bf16"):
                o_bf = _forward_batch(model, grp, ds.tf2idx, device, "grouped")
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        print("  显存不足(OOM)，这一段没测成——多半是还有别的进程占着 GPU，请先确认")
    if o_bf is None:
        results.update(equiv_fp32_ok=None, bf16_max_rel_diff=float("nan"))
    else:
        print(f"对照 batch：{len(eq_idx)} 条样本 / {grp['cis_ids'].shape[0]} 个基因，其中 "
              f"D∈L_g(要单独算 D 侧 layout)的 {grp['d_rows'].numel()} 条；分段对照用的切分 "
              f"WT {grp_b['layout_wt_splits']}  D {grp_b['layout_d_splits']}")
        eq_ok, max_rel_bf = True, 0.0
        for k in ("y_a_pred", "y_b_pred", "logits_c", "z_wt", "z_d"):
            ref = o_leg[k].float()
            scale = float(ref.abs().max()) + 1e-12
            d32 = float((o_grp[k].float() - ref).abs().max())
            dbf = float((o_bf[k].float() - ref).abs().max())
            dbk = float((o_bk[k].float() - ref).abs().max())
            max_rel_bf = max(max_rel_bf, dbf / scale)
            eq_ok = eq_ok and d32 <= 1e-4 * max(1.0, scale) and dbk <= 1e-4 * max(1.0, scale)
            print(f"  {k:9s} 最大|值| {scale:9.4g} | fp32分组−fp32逐样本 最大差 {d32:9.3g} "
                  f"| fp32分组分段−fp32逐样本 最大差 {dbk:9.3g} | bf16分组−fp32逐样本 最大差 "
                  f"{dbf:9.3g}(相对 {dbf / scale:.2g})")
        print("  结论：fp32 分组前向(含分段)" + ("跟逐样本前向一致(阈值 1e-4) ✓" if eq_ok else
              "跟逐样本前向不一致 ✗——请把这一段贴给我；在查清之前用 --forward legacy 跑"))
        print("  (bf16 那一列只是量级参考：bf16 尾数 8 位，相对差 1e-2 量级属正常；如果到 1e-1 "
              "量级或出现 nan/inf，请贴给我，并先用 --amp off)")
        results.update(equiv_fp32_ok=eq_ok, bf16_max_rel_diff=max_rel_bf)
    del o_leg, o_grp, o_bf, o_bk, full, grp, grp_b
    _free()

    # ---------------- ③ 改之前的前向用时构成 ----------------
    print("\n--- ③ 改之前(逐样本+fp32)前向用时构成(只测前向，粗略) ---")
    rng = np.random.default_rng(seed)
    cb_idx = rng.choice(tr_arr, size=min(batch_size, len(tr_arr)), replace=False).tolist()
    cb = ds.collate_fn([ds[i] for i in cb_idx])
    cis_ids = cb["cis_ids"].to(device)
    lw, ld = _to_device(cb["layout_wt"], device), _to_device(cb["layout_d"], device)

    def _timed(fn, reps=5):
        with torch.no_grad():
            fn()
            _sync()
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            _sync()
        return (time.perf_counter() - t) / reps

    def _now():
        if is_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    try:
        t_all = _timed(lambda: _forward_batch(model, cb, ds.tf2idx, device, "legacy"))
        t_cis = _timed(lambda: model.cis(cis_ids))
        t_lw = _timed(lambda: model.layout(**lw))
        t_ld = _timed(lambda: model.layout(**ld))
        starts, spans, hooks = [], [], []
        for layer in model.layout.layers:  # 用 hook 量 layout 里距离偏置 einsum 的用时
            hooks.append(layer.dist_bias.register_forward_pre_hook(
                lambda mod, inp: starts.append(_now())))
            hooks.append(layer.dist_bias.register_forward_hook(
                lambda mod, inp, out: spans.append((starts.pop(), _now()))))
        with torch.no_grad():
            model.layout(**lw)
        _sync()
        for h in hooks:
            h.remove()
        t_bias = sum((a.elapsed_time(b) / 1e3) if is_cuda else (b - a) for a, b in spans)
        rest = max(t_all - t_cis - t_lw - t_ld, 0.0)
        print(f"  batch={len(cb_idx)}：cis padding 到 {cis_ids.shape[1]} token，L_g padding 到 "
              f"{lw['tf_idx'].shape[1]} 个位点")
        print(f"  整个前向 {t_all * 1e3:.1f} ms = cis {t_cis / t_all:.0%} + WT layout "
              f"{t_lw / t_all:.0%} + D layout {t_ld / t_all:.0%} + 融合/头/其它 {rest / t_all:.0%}")
        print(f"  其中 WT layout 里距离偏置(einsum，fp32) 约 {t_bias * 1e3:.1f} ms，占 WT layout "
              f"{t_bias / max(t_lw, 1e-12):.0%}(反向传播没算在内)")
        results.update(t_fwd_ms=t_all * 1e3, frac_cis=t_cis / t_all,
                       frac_layout=(t_lw + t_ld) / t_all,
                       frac_bias_in_layout=t_bias / max(t_lw, 1e-12))
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        print("  显存不足(OOM)，跳过这一段")
    del cb, cis_ids, lw, ld
    _free()

    # ---------------- ④ 评估速度 + 几种评估方式一致性 ----------------
    print("\n--- ④ 评估(验证/测试集)速度，含 DataLoader worker 启动开销 ---")
    keep = set(pd.unique(val_genes)[:n_eval_genes].tolist())
    ev_idx = [int(i) for i, g in zip(val_arr.tolist(), val_genes.tolist()) if g in keep]
    ev_cfgs = [("逐样本+fp32(最初版本)", "legacy", "off", 1),
               ("分组+bf16 不分段(第二轮)", "grouped", "bf16", 1)]
    if layout_buckets > 1:
        ev_cfgs.append((f"分组+bf16 分{layout_buckets}段(本轮默认)", "grouped", "bf16",
                        layout_buckets))
    ev_res = {}
    for name, fwd, amp_, nbk in ev_cfgs:
        _free()
        t0 = time.time()
        try:
            r = evaluate(model, ds, ev_idx, class_counts, batch_size=batch_size, device=device,
                         num_workers=num_workers, amp=amp_, forward_mode=fwd,
                         eval_batch_size=eval_batch_size, layout_buckets=nbk)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            print(f"  {name}: 显存不足(OOM)，跳过")
            continue
        dt = time.time() - t0
        ev_res[name] = (r, dt)
        n_val = len(split_idx["val"])
        print(f"  {name:18s}: {len(keep)} 个基因 {len(ev_idx)} 条样本 {dt:6.1f}s = "
              f"{len(ev_idx) / dt:7.0f} 样本/s → 整个 val({n_val} 条)约 "
              f"{n_val / len(ev_idx) * dt / 60:.1f} 分钟")
    names = list(ev_res)
    if len(names) >= 2:
        r0 = ev_res[names[0]][0]
        for nm in names[1:]:
            r1 = ev_res[nm][0]
            dmax = max(float((r1[k] - r0[k]).abs().max())
                       for k in ("y_a_pred", "y_b_pred", "logits_c"))
            print(f"  {nm} vs {names[0]}：预测最大差 {dmax:.3g}，val_loss {r1['loss']:.6f} vs "
                  f"{r0['loss']:.6f}，r_a {r1['r_a']:.4f} vs {r0['r_a']:.4f}")
    results["eval_sec"] = {nm: dt for nm, (_, dt) in ev_res.items()}
    results["eval_n"] = len(ev_idx)
    ev_default = ev_res.get(ev_cfgs[-1][0])
    del ev_res
    _free()

    # ---------------- ⑤ DataLoader 单独吞吐 ----------------
    print("\n--- ⑤ DataLoader 单独吞吐(只取 batch、不做 GPU 计算；有预取缓冲，数字略偏乐观) ---")
    n_dl = 4 * bench_steps
    dl_cfgs = [("legacy", 1, 1), ("grouped", tfs_per_gene, 1)]
    if layout_buckets > 1:
        dl_cfgs.append(("grouped", tfs_per_gene, layout_buckets))
    for fwd, k, nbk in dl_cfgs:
        loader, _ = _make_loader(ds, split_idx["train"], "train", fwd, batch_size, k,
                                 eval_batch_size, seed, num_workers, device, persistent=False,
                                 layout_buckets=nbk)
        it = iter(loader)
        next(it)  # 第一个 batch 含 worker 启动开销，不计时
        n_got, t0 = 0, time.time()
        for _ in range(n_dl):
            try:
                n_got += len(next(it)["y_a"])
            except StopIteration:
                break
        dt = max(time.time() - t0, 1e-9)
        print(f"  collate={'分组' if fwd == 'grouped' else '逐样本'} K={k} 分{nbk}段: "
              f"{n_got / dt:8.0f} 样本/s  (num_workers={num_workers})")
        results[f"loader_sps_{fwd}_b{nbk}"] = n_got / dt
        del it, loader
        gc.collect()

    # ---------------- ⑥ 训练吞吐表 ----------------
    print(f"\n--- ⑥ 训练吞吐(前向+反向+Adam；batch_size={batch_size}，热身 {bench_warmup} 步、"
          f"计时 {bench_steps} 步) ---")
    configs = [("逐样本+fp32 K=1(最初版本)", "legacy", "off", 1, 1, False)]
    for nbk in sorted({1, 2, 3, int(layout_buckets)}):
        tag = "(第二轮默认)" if nbk == 1 else ""
        tag += "(本轮默认)" if nbk == layout_buckets else ""
        configs.append((f"分组+bf16 K={tfs_per_gene} 分{nbk}段{tag}", "grouped", "bf16",
                        tfs_per_gene, nbk, nbk == layout_buckets))
    rows = []
    for name, fwd, amp_, k, nbk, is_def in configs:
        _free()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        m = opt = loader = it = None
        try:
            m = SiameseHeadsModel(ds.n_tf, vocab_size, d_model, n_heads, cis_layers,
                                  lay_layers, pad_token_id=ds.pad_id).to(device)
            m.train()
            opt = _make_optimizer(m, lr, weight_decay, device)
            loader, _ = _make_loader(ds, split_idx["train"], "train", fwd, batch_size, k,
                                     eval_batch_size, seed, num_workers, device,
                                     persistent=False, layout_buckets=nbk)
            it = iter(loader)
            n_s, t0, last = 0, None, None
            for step in range(bench_warmup + bench_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                if step == bench_warmup:
                    _sync()
                    t0 = time.time()
                parts, nb = _train_step(m, opt, batch, ds, device, fwd, amp_, class_counts, {})
                if step >= bench_warmup:
                    n_s += nb
                last = parts
            _sync()
            sps = n_s / max(time.time() - t0, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2**30 if is_cuda else float("nan")
            rows.append(dict(name=name, sps=sps, peak_gib=peak, buckets=nbk, grouped=fwd == "grouped",
                             is_default=is_def,
                             loss_finite=math.isfinite(float(last["total"]))))
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            rows.append(dict(name=name, sps=float("nan"), peak_gib=float("nan"), buckets=nbk,
                             grouped=fwd == "grouped", is_default=is_def, loss_finite=None))
            print(f"  {name}: 显存不足(OOM)，跳过")
        finally:
            del m, opt, loader, it
            _free()
    n_train = len(split_idx["train"])
    base = rows[0]["sps"] if rows else float("nan")
    print(f"  {'配置':<30s}| {'样本/s':>8s} | {'相对最初':>6s} | {'峰值显存GiB':>8s} | "
          f"{'1个epoch训练(分钟)':>10s} | loss有限")
    for r in rows:
        ratio = r["sps"] / base if base == base and base > 0 else float("nan")
        ep_min = n_train / r["sps"] / 60 if r["sps"] == r["sps"] and r["sps"] > 0 \
            else float("nan")
        r.update(speedup=ratio, epoch_train_min=ep_min)
        print(f"  {r['name']:<30s}| {r['sps']:8.0f} | {ratio:6.2f}x | {r['peak_gib']:8.1f} | "
              f"{ep_min:10.1f} | {r['loss_finite']}")
    results["train_rows"] = rows
    grp_rows = [r for r in rows if r["grouped"] and r["sps"] == r["sps"]]
    def_row = next((r for r in grp_rows if r["is_default"]), None)
    if grp_rows and def_row is not None:
        best = max(grp_rows, key=lambda r: r["sps"])
        if best["buckets"] == def_row["buckets"] or best["sps"] < 1.05 * def_row["sps"]:
            print(f"  分段数实测：默认的分{def_row['buckets']}段已经是最快或跟最快相差<5%，保持默认")
        else:
            print(f"  分段数实测：分{best['buckets']}段最快({best['sps']:.0f} 样本/s，默认分"
                  f"{def_row['buckets']}段 {def_row['sps']:.0f} 样本/s)，建议 run_all.sh 加 "
                  f"--layout-buckets {best['buckets']}")
        val_min = (len(split_idx["val"]) / max(results["eval_n"], 1) * ev_default[1] / 60
                   if ev_default else float("nan"))
        per_ep = def_row["epoch_train_min"] + (val_min if val_min == val_min else 0.0)
        print(f"  默认配置每个 epoch ≈ 训练 {def_row['epoch_train_min']:.1f} + 验证 {val_min:.1f}"
              f" = {per_ep:.1f} 分钟；{n_seeds} 个 seed × {n_epochs} epoch 的上限 ≈ "
              f"{per_ep * n_epochs * n_seeds / 60:.1f} 小时(有 patience 早停，实际一般更短)")

    # ---------------- ⑦ 默认配置的 GPU 耗时构成(profiler) ----------------
    print(f"\n--- ⑦ 默认配置(分组+bf16 K={tfs_per_gene} 分{layout_buckets}段)的 GPU 耗时构成"
          f"(torch.profiler，{profile_steps} 步，只做诊断) ---")
    try:
        from torch.profiler import ProfilerActivity, profile
        _free()
        torch.manual_seed(seed)
        m = SiameseHeadsModel(ds.n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers,
                              pad_token_id=ds.pad_id).to(device)
        m.train()
        opt = _make_optimizer(m, lr, weight_decay, device)
        loader, _ = _make_loader(ds, split_idx["train"], "train", "grouped", batch_size,
                                 tfs_per_gene, eval_batch_size, seed, num_workers, device,
                                 persistent=False, layout_buckets=layout_buckets)
        it = iter(loader)
        pre = [next(it) for _ in range(3 + max(1, profile_steps))]  # 先取好，排除等数据的时间
        del it, loader
        for b in pre[:3]:
            _train_step(m, opt, b, ds, device, "grouped", "bf16", class_counts, {})
        _sync()
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if is_cuda else [])
        t0 = time.time()
        with profile(activities=acts) as prof:
            for b in pre[3:]:
                _train_step(m, opt, b, ds, device, "grouped", "bf16", class_counts, {})
            _sync()
        wall_prof = (time.time() - t0) / len(pre[3:]) * 1e3
        ka = prof.key_averages()

        def _self_dev_us(e):
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                v = getattr(e, attr, None)
                if v is not None:
                    return float(v)
            return 0.0

        n_steps_p = len(pre[3:])
        kernels = [e for e in ka if str(getattr(e, "device_type", "")).endswith("CUDA")]
        gpu_ms = sum(_self_dev_us(e) for e in kernels) / 1e3 / n_steps_p
        cpu_ops = [e for e in ka if not str(getattr(e, "device_type", "")).endswith("CUDA")]
        n_calls = sum(int(e.count) for e in cpu_ops if str(e.key).startswith("aten::")) / n_steps_p
        top = sorted(cpu_ops, key=_self_dev_us, reverse=True)[:15]
        tot = sum(_self_dev_us(e) for e in cpu_ops) or 1.0
        print(f"  每步 CPU 侧记录到的 aten 算子调用约 {n_calls:.0f} 次(含嵌套调用)")
        if is_cuda:
            print(f"  {'算子(按它直接发出的 GPU kernel 耗时排序)':<44s}| 每步GPU ms | 占比")
            for e in top:
                print(f"  {str(e.key)[:44]:<44s}| {_self_dev_us(e) / 1e3 / n_steps_p:9.2f} | "
                      f"{_self_dev_us(e) / tot:5.1%}")
            wall_ms = (batch_size / def_row["sps"] * 1e3) if def_row else float("nan")
            print(f"  每步 GPU 实际忙碌 {gpu_ms:.1f} ms；⑥ 实测每步墙钟 {wall_ms:.1f} ms"
                  f"(profiler 开着时 {wall_prof:.1f} ms，含 profiler 自身开销) → GPU 忙碌占比约 "
                  f"{gpu_ms / wall_ms:.0%}" if wall_ms == wall_ms else "")
            print("  (占比接近 100% = GPU 算力是瓶颈，下一步该减计算量；明显低于 100% = CPU 侧"
                  "排 kernel/Python 开销是瓶颈，下一步该减 kernel 数量，比如 CUDA Graph/编译)")
            results.update(gpu_ms_per_step=gpu_ms, wall_ms_per_step=wall_ms)
        else:
            print("  (CPU 设备，没有 GPU 耗时可统计，只验证这段代码能跑通)")
        results["aten_calls_per_step"] = n_calls
        del m, opt, pre, prof
        _free()
    except Exception as e:  # profiler 各版本接口差异较大，失败不影响前面的结论
        print(f"  profiler 这一段失败了({type(e).__name__}: {e})，不影响 ①～⑥ 的结论")

    # ---------------- ⑧ torch.compile 对照(第四轮新增，加 --compile 才跑；
    # 2026-09-23f 加安全网+改成三组对照；2026-09-23g 加第四组(子模块粒度)，
    # 见文件头第11条(e)(f)) ----------------
    if compile_model:
        print(f"\n--- ⑧ torch.compile 对照(默认配置 分组+bf16 K={tfs_per_gene} 分"
              f"{layout_buckets}段，编译需要额外热身吸收一次性JIT编译耗时；"
              f"cache_size_limit={compile_cache_limit}，热身阶段总预算"
              f"{compile_warmup_timeout:.0f}秒，见文件头第11条(e)(f)) ---")
        compile_warmup = max(bench_warmup, 20)
        ab_sps = {}
        # 四组对照：不编译基线 + scope=full下dynamic=auto(2026-09-23f默认)/dynamic=True
        # (上一轮写法，真实触发过反复重新编译) + scope=submodules下dynamic=auto
        # (2026-09-23g新增，只编译cis/layout两个子模块，见_maybe_compile_model)。
        # 不替用户直接下结论，见文件头第11条(e)③(f)
        compile_arms = (
            ("不编译(对照)", False, None, "full"),
            ("torch.compile(full,dynamic=auto)", True, None, "full"),
            ("torch.compile(full,dynamic=True)", True, True, "full"),
            ("torch.compile(子模块,dynamic=auto)", True, None, "submodules"),
        )
        for name, use_compile, dyn, scope in compile_arms:
            _free()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(seed)
            m = opt = loader = it = None
            try:
                m = SiameseHeadsModel(ds.n_tf, vocab_size, d_model, n_heads, cis_layers,
                                      lay_layers, pad_token_id=ds.pad_id).to(device)
                m.train()
                m = _maybe_compile_model(m, use_compile, tag=f"  [{name}] ",
                                         cache_size_limit=compile_cache_limit, dynamic=dyn,
                                         scope=scope)
                opt = _make_optimizer(m, lr, weight_decay, device)
                loader, _ = _make_loader(ds, split_idx["train"], "train", "grouped", batch_size,
                                         tfs_per_gene, eval_batch_size, seed, num_workers, device,
                                         persistent=False, layout_buckets=layout_buckets)
                it = iter(loader)
                n_s, t0 = 0, None
                t_arm_start = time.time()
                for step in range(compile_warmup + bench_steps):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader)
                        batch = next(it)
                    if step == compile_warmup:
                        _sync()
                        t0 = time.time()
                    parts, nb = _train_step(m, opt, batch, ds, device, "grouped", "bf16",
                                            class_counts, {})
                    if step >= compile_warmup:
                        n_s += nb
                    # 【2026-09-23f新增安全网，见文件头第11条(e)②，取代手动Ctrl+Z】只在
                    # 还没过完热身阶段时检查，避免正常计时阶段偶尔一步变慢(比如GPU被
                    # 其它进程抢占)被误判成编译卡住。局限：只能在"步与步之间"的检查点
                    # 生效，挡不住单独一步内部真的卡住不返回的极端情况(Python没法安全
                    # 地从外部抢占一个正在跑CUDA kernel/Dynamo编译的调用)，这一点在
                    # _CompileWarmupTimeout和文件头里都写清楚了，不隐瞒。
                    if use_compile and step < compile_warmup:
                        elapsed = time.time() - t_arm_start
                        if elapsed > compile_warmup_timeout:
                            raise _CompileWarmupTimeout(
                                f"热身阶段跑到第{step + 1}/{compile_warmup}步、累计"
                                f"{elapsed:.0f}秒，超过 --compile-warmup-timeout="
                                f"{compile_warmup_timeout:.0f}秒的预算")
                _sync()
                sps = n_s / max(time.time() - t0, 1e-9)
                peak = torch.cuda.max_memory_allocated() / 2**30 if is_cuda else float("nan")
                print(f"  {name:<32s}: {sps:7.0f} 样本/s  峰值显存 {peak:5.1f} GiB "
                      f"(热身 {compile_warmup} 步不计时，只计时 {bench_steps} 步)")
                ab_sps[name] = sps
            except Exception as e:
                # 【2026-09-23e修复，见文件头(d)条；2026-09-23f新增_CompileWarmupTimeout
                # 分支，见文件头(e)条】OOM/热身超时都当已知情况处理，跳过即可。非OOM/非
                # 超时的失败：不编译(对照)这条路径已经跑通过多次(⑥⑦都在用同一条
                # forward_grouped)，这里出问题大概率是新引入的真bug，原样raise让用户
                # 看到，不能顺手吞掉。torch.compile这几条路径是本项目仍在验证的方向
                # (信心7/10全量/6/10子模块，见文件头11条(b)(e)(f))，_maybe_compile_model
                # 那层try/except只包得住"包装"这一步、包不住第一次真正调用触发的失败——
                # 上一轮的ModuleNotFoundError、之前用户实测到的"反复重新编译两分多钟"，
                # 都是从这条没接住的路径钻出来的。原则跟⑦节profiler诊断段一致(那段也是
                # except Exception+打印+不影响前面结论)：⑧节本来就是"能不能编译、
                # 编译了划不划算"的诊断，编译失败/编译太慢本身就是一种有效诊断结果，
                # 不该让失败诊断变成整个bench脚本退出码为1、连累run_all.sh。跳过后
                # ①~⑦和其它组的结论完全不受影响；如果这是还没见过的新失败，把下面
                # 这行报错贴回来。
                if isinstance(e, _CompileWarmupTimeout):
                    print(f"  [{name}] {e}，判定为torch.compile在这套动态形状下反复"
                          "重新编译、实际不可用，已自动放弃这一组(不需要再手动"
                          "Ctrl+Z)，不影响①~⑦和其它组的结论；根因分析见文件头"
                          "第11条(e)。")
                elif isinstance(e, RuntimeError) and _is_oom(e):
                    print(f"  {name}: 显存不足(OOM)，跳过")
                elif use_compile:
                    print(f"  [{name}] torch.compile 实际执行失败"
                          f"({type(e).__name__}: {e})，本组对照跳过，不影响"
                          "①~⑦和其它组的结论；如需继续排查请把这行报错贴回来。")
                else:
                    raise
            finally:
                del m, opt, loader, it
                _free()
        results["compile_ab_sps"] = ab_sps
        base = ab_sps.get("不编译(对照)")
        compile_ok = {k: v for k, v in ab_sps.items() if k != "不编译(对照)"}
        if base is None:
            print("  不编译(对照)这组也没能算出样本/s(上面已打印原因，大概率是"
                  "OOM)，没有基线可比，①~⑦的结论不受影响。")
        elif not compile_ok:
            print("  torch.compile 的三组都没能算出样本/s"
                  "(上面已打印失败原因)，跳过对比，①~⑦的结论不受影响。")
        else:
            for name, comp in compile_ok.items():
                if comp > base * 1.05:
                    print(f"  {name} 实测更快({comp:.0f} vs {base:.0f} 样本/s，"
                          f"+{(comp / base - 1):.0%})")
                elif comp < base * 0.95:
                    print(f"  {name} 实测更慢({comp:.0f} vs {base:.0f} 样本/s)，这个"
                          "模型/这台机器上这个配置不划算(也可能是热身步数/timeout"
                          "预算还没吸收完JIT编译时间，可以加大 --bench-warmup 或 "
                          "--compile-warmup-timeout 再试一次排除这个可能)")
                else:
                    print(f"  {name} 跟不编译基本打平({comp:.0f} vs {base:.0f} 样本/s)")
            best_name = max(compile_ok, key=compile_ok.get)
            if compile_ok[best_name] > base * 1.05:
                if "子模块" in best_name:
                    print(f"  建议：{best_name} 实测最快，正式 --train 时加 "
                          "--compile --compile-scope submodules"
                          "(dynamic=auto是默认值，不用额外传 --compile-dynamic，"
                          "见文件头第11条(f))")
                elif "auto" in best_name:
                    print(f"  建议：{best_name} 实测最快，正式 --train 时可以直接加 "
                          "--compile(scope=full、dynamic=auto都是默认值，不用额外传参)")
                else:
                    print(f"  建议：{best_name} 实测最快，正式 --train 时加 "
                          "--compile --compile-dynamic true")
            else:
                print("  三组 torch.compile 都没有明显更快，建议暂时放弃 --compile 这个"
                      "方向(⑦节实测GPU忙碌占比本来就有71%~85%，compile理论提速上限"
                      "有限)，改成验证调大 --batch-size 能不能把 kernel 启动开销摊得"
                      "更薄——显存还有余量，见文件头第11条(c)(e)(f)，下面 ⑨ 节可以"
                      "直接测(加 --bench-batch-sweep)")
    else:
        print("\n--- ⑧ torch.compile 对照：本次没加 --compile，跳过(想测的话在 bench 命令后加"
              " --compile，见文件头第11条(b)(e)(f)) ---")

    # ---------------- ⑨ batch_size 扫描(2026-09-23g本次新增，加 --bench-batch-sweep
    # 才跑，默认不跑；见文件头第11条(f)) ----------------
    if bench_batch_sweep:
        sizes = sorted({int(s) for s in bench_batch_sweep_sizes.split(",")}
                       if bench_batch_sweep_sizes else
                       {batch_size, batch_size * 2, batch_size * 4})
        print(f"\n--- ⑨ batch_size 扫描(默认配置 分组+bf16 K={tfs_per_gene} 分"
              f"{layout_buckets}段，固定热身{bench_warmup}步/计时{bench_steps}步、只变"
              f"batch_size；候选: {sizes}；见文件头第11条(f)) ---")
        print("  动机：⑦ 如果测出 GPU 忙碌占比明显低于100%，说明有一部分是固定的"
              "kernel启动开销、大致不随batch_size变，同样开销摊给更多样本可能有"
              "超线性收益(每样本吞吐比>1x)——这是⑧节compile方向如果不划算时风险更低"
              "的替代路线，只测量不建模，结论以下面实测为准。")
        bsw_rows = []
        for bs in sizes:
            _free()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(seed)
            m = opt = loader = it = None
            try:
                m = SiameseHeadsModel(ds.n_tf, vocab_size, d_model, n_heads, cis_layers,
                                      lay_layers, pad_token_id=ds.pad_id).to(device)
                m.train()
                opt = _make_optimizer(m, lr, weight_decay, device)
                loader, _ = _make_loader(ds, split_idx["train"], "train", "grouped", bs,
                                         tfs_per_gene, eval_batch_size, seed, num_workers,
                                         device, persistent=False, layout_buckets=layout_buckets)
                it = iter(loader)
                n_s, t0 = 0, None
                for step in range(bench_warmup + bench_steps):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader)
                        batch = next(it)
                    if step == bench_warmup:
                        _sync()
                        t0 = time.time()
                    parts, nb = _train_step(m, opt, batch, ds, device, "grouped", "bf16",
                                            class_counts, {})
                    if step >= bench_warmup:
                        n_s += nb
                _sync()
                sps = n_s / max(time.time() - t0, 1e-9)
                peak = torch.cuda.max_memory_allocated() / 2**30 if is_cuda else float("nan")
                bsw_rows.append(dict(batch_size=bs, sps=sps, peak_gib=peak))
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                bsw_rows.append(dict(batch_size=bs, sps=float("nan"), peak_gib=float("nan")))
                print(f"  batch_size={bs}: 显存不足(OOM)，跳过")
            finally:
                del m, opt, loader, it
                _free()
        results["batch_sweep_rows"] = bsw_rows
        base_row = next((r for r in bsw_rows if r["batch_size"] == batch_size), None)
        print(f"  {'batch_size':>10s} | {'样本/s':>8s} | {'相对当前':>9s} | {'每样本吞吐比':>10s} | "
              f"{'峰值显存GiB':>10s}")
        for r in bsw_rows:
            if base_row and base_row["sps"] == base_row["sps"] and base_row["sps"] > 0:
                rel = r["sps"] / base_row["sps"] if r["sps"] == r["sps"] else float("nan")
            else:
                rel = float("nan")
            size_ratio = r["batch_size"] / batch_size
            per_sample = rel / size_ratio if rel == rel and size_ratio > 0 else float("nan")
            print(f"  {r['batch_size']:>10d} | {r['sps']:8.0f} | {rel:8.2f}x | "
                  f"{per_sample:9.2f}x | {r['peak_gib']:10.1f}")
        valid = [r for r in bsw_rows if r["sps"] == r["sps"]]
        if base_row is not None and base_row["sps"] == base_row["sps"] and len(valid) > 1:
            best = max(valid, key=lambda r: r["sps"])
            if best["batch_size"] != batch_size and best["sps"] > base_row["sps"] * 1.05:
                print(f"  batch_size={best['batch_size']} 比当前默认({batch_size})快"
                      f"{(best['sps'] / base_row['sps'] - 1):.0%}"
                      f"(峰值显存 {best['peak_gib']:.1f} GiB)，值得试试调大 --batch-size"
                      "——但加大batch_size理论上可能需要相应调大lr才能保持同样收敛"
                      "速度，这段代码不会替你自动调lr，见run_all.sh文件头第7条同样"
                      "的提醒，建议配合--smoke观察loss曲线")
            else:
                print(f"  更大的 batch_size 没有带来明显提速(<5%)或显存不够，当前"
                      f"batch_size={batch_size} 保持不变即可")
        else:
            print("  没有足够的有效数据点做比较(可能全部 OOM，或候选列表里只有当前"
                  "batch_size 一个点)")
    else:
        print("\n--- ⑨ batch_size 扫描：本次没加 --bench-batch-sweep，跳过(想测的话在 bench"
              " 命令后加 --bench-batch-sweep，见文件头第11条(f)) ---")

    print("\n请把上面 ①～⑨ 的输出整段贴回来(尤其 ② 的结论行、⑥ 的表和推荐行、⑦ 的忙碌占比、"
          "⑧⑨ 的对比结论，如果跑了的话)。")
    return results


# ==========================================================================
# 自检：极简假数据集，覆盖染色体切分/训练/评估/bootstrap/5-seed/leave-TF-out
# 整条编排逻辑，不依赖任何真实文件
# ==========================================================================
class _DummyTFLayoutDataset:
    """自检用的极简假数据集，接口(tf_list/tf2idx/n_tf/class2idx/samples/
    collate_fn/pad_id/__len__/__getitem__)照抄TFLayoutDataset，好让上面的训练
    循环代码不用改一行就能同时喂真实数据集和这个假数据集(duck typing，
    DataLoader只要求对象实现__len__和__getitem__，不要求是Dataset子类)。"""

    def __init__(self, n_genes=20, n_tf=6, n_cis=30, vocab_size=40, seed=0):
        rng = np.random.default_rng(seed)
        self.tf_list = [f"TF{i}" for i in range(n_tf)]
        self.tf2idx = {t: i for i, t in enumerate(self.tf_list)}
        self.n_tf = n_tf
        self.class2idx = {"down": 0, "ns": 1, "up": 2}
        self.pad_id, self.unk_id = 0, 1

        genes = [f"GENE{i}" for i in range(n_genes)]
        # 假造染色体标签，模拟08脚本的染色体holdout(chr13/14->val, chr15/16->
        # test)，只是为了让下面的split逻辑有东西可测，数字本身没有生物学含义
        chroms = [f"chr{i % 16 + 1}" for i in range(n_genes)]
        self._gene2split = {}
        for g, c in zip(genes, chroms):
            if c in ("chr13", "chr14"):
                self._gene2split[g] = "val"
            elif c in ("chr15", "chr16"):
                self._gene2split[g] = "test"
            else:
                self._gene2split[g] = "train"

        self.layout_by_gene = {}
        for g in genes:
            n = int(rng.integers(1, 6))
            self.layout_by_gene[g] = dict(
                tf_idx=rng.integers(0, n_tf, size=n).astype(np.int64),
                pos=rng.normal(0, 500, n).astype(np.float32),
                strand=rng.integers(-1, 2, n).astype(np.float32),
                a=rng.normal(size=n).astype(np.float32),
                m=rng.normal(size=n).astype(np.float32),
                res=rng.integers(0, 2, n).astype(np.float32),
            )
        self.gene2tokens = {g: rng.integers(1, vocab_size, size=int(rng.integers(10, n_cis)))
                            .astype(np.int64) for g in genes}
        self._empty_cis = np.array([self.unk_id], dtype=np.int64)

        rows = []
        for g in genes:
            for t in self.tf_list:
                log2fc = float(rng.normal()) if rng.random() < 0.3 else float("nan")
                direction = "ns" if math.isnan(log2fc) else ("up" if log2fc > 0 else "down")
                rows.append((g, t, log2fc, direction))
        self.samples = pd.DataFrame(rows, columns=["gene_id", "tf_depleted",
                                                    "log2fc", "direction_3class"])
        self.head_a = pd.Series({g: float(rng.normal()) for g in genes})
        # ctx_d 只取决于被耗竭的 TF(跟真实 09 号脚本的 _ctx_d_cache 同一结构)，D 自己那一位
        # 置 0。第二轮提速前这里全是 0，那样"分组前向 vs 逐样本前向"的对照测不出 ctx_d
        # 用错行这类 bug，所以改成非零(只影响假数据，不影响真实训练)。
        self._ctx_d_cache = {}
        for t in self.tf_list:
            v = (0.5 * rng.normal(size=n_tf)).astype(np.float32)
            v[self.tf2idx[t]] = 0.0
            self._ctx_d_cache[t] = v
        self._y_bd_arr = np.full(len(self.samples), np.nan)  # 2026-09-25b：稠密目标，默认无

    def set_dense_target(self, path=None, column="log2fc_dense", verbose=True):
        """【2026-09-25b】假数据版：不读文件。path 非空时造一份"显著样本=log2fc、其余=小噪声、
        约5%缺失"的稠密目标(固定随机种子，可复现)，None 时清空——只为让训练循环的稠密分支在自检里
        有东西可跑。返回值结构跟 09 号同名方法一致。"""
        if not path:
            self._y_bd_arr = np.full(len(self.samples), np.nan)
        else:
            rng = np.random.default_rng(123)
            y = self.samples["log2fc"].to_numpy(np.float64)
            v = np.where(np.isfinite(y), y, 0.15 * rng.normal(size=len(y)))
            v[rng.random(len(y)) < 0.05] = np.nan
            self._y_bd_arr = v
        fin = np.isfinite(self._y_bd_arr)
        ns = ~np.isfinite(self.samples["log2fc"].to_numpy(np.float64))
        return dict(n=len(fin), n_finite=int(fin.sum()), frac=float(fin.mean()),
                    frac_ns=float(fin[ns].mean()) if ns.any() else float("nan"), by_split={})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        gid, dep_tf = row["gene_id"], row["tf_depleted"]
        lg = self.layout_by_gene[gid]
        keep = lg["tf_idx"] != self.tf2idx[dep_tf]
        lg_minus_d = {k: v[keep] for k, v in lg.items()}
        cis_ids = self.gene2tokens.get(gid, self._empty_cis)
        ctx_d = self._ctx_d_cache[dep_tf].copy()
        ctx_wt = np.zeros(self.n_tf, dtype=np.float32)
        y_b = row["log2fc"]
        return dict(
            gene_id=gid, tf_depleted=dep_tf, layout_wt=lg, layout_d=lg_minus_d,
            cis_ids=cis_ids, ctx_wt=torch.from_numpy(ctx_wt),
            ctx_d=torch.from_numpy(ctx_d),
            y_a=torch.tensor(float(self.head_a.get(gid, np.nan)), dtype=torch.float32),
            y_b=torch.tensor(float(y_b) if pd.notna(y_b) else float("nan"),
                             dtype=torch.float32),
            y_c=torch.tensor(self.class2idx[row["direction_3class"]], dtype=torch.long),
            y_bd=torch.tensor(float(self._y_bd_arr[idx]), dtype=torch.float32),
        )

    def collate_fn(self, batch):
        # 跟09_torch_dataset.py TFLayoutDataset.collate_fn实现完全一样，照抄过来
        # 是因为这是一个独立的假数据集类、不是TFLayoutDataset的子类，没法直接
        # 继承它的方法
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
            layout_wt=pad("layout_wt"), layout_d=pad("layout_d"),
            cis_ids=pad_cis(),
            ctx_wt=torch.stack([b["ctx_wt"] for b in batch]),
            ctx_d=torch.stack([b["ctx_d"] for b in batch]),
            y_a=torch.stack([b["y_a"] for b in batch]),
            y_b=torch.stack([b["y_b"] for b in batch]),
            y_c=torch.stack([b["y_c"] for b in batch]),
            y_bd=torch.stack([b["y_bd"] for b in batch]),
        )

    def collate_grouped(self, batch, layout_buckets: int = 1):
        # 分组逻辑直接复用 09 号脚本的 TFLayoutDataset.group_collated(staticmethod)，
        # 不另抄一份，保证自检测的就是真实训练用的那份代码
        return TFLayoutDataset.group_collated(self.collate_fn(batch), self.tf2idx,
                                              self.pad_id, layout_buckets)


def _dummy_split_idx(ds) -> dict:
    genes = ds.samples["gene_id"].to_numpy()
    splits = np.array([ds._gene2split[g] for g in genes])
    return {s: np.where(splits == s)[0].tolist() for s in ("train", "val", "test")}


def run_self_test(n_genes=20, n_tf=6, n_cis=30, vocab_size=40, d_model=16, n_heads=2,
                  cis_layers=1, lay_layers=1, batch_size=4, n_epochs=2, patience=2,
                  seeds=(0, 1), n_boot=50, seed=0):
    """全流程冒烟测试：假数据集→染色体切分→run_one_seed(训练+早停)→evaluate→
    bootstrap_ci→run_multi_seed(2个seed)→run_leave_tf_out。用的是极小规模假
    数据+极少epoch，只保证"整条流水线跑得通、数值有限、形状/方向正确"，不是为了
    让模型真的学到什么。"""
    # 【2026-09-23e新增回归检查】跟15号脚本run_self_test里新增的检查同一个目的：
    # 确认本文件动态加载的09/15号模块(以及15号脚本内部再动态加载的10/12/14号
    # 模块)都正确注册进了sys.modules——这正是这次真实torch.compile报
    # ModuleNotFoundError的根因，不需要GPU/torch.compile，CPU上就能覆盖，信心
    # 10/10。09/15号自己的检查在各自文件里，这里只查本文件直接加载的两个。
    for _name, _mod in (("_tflayout_09_torch_dataset", _ds_mod),
                        ("_tflayout_15_siamese_heads", _heads_mod)):
        assert importlib.import_module(_name) is _mod, (
            f"{_name} 没有正确注册进 sys.modules，torch.compile追踪全局变量时会"
            "走这条路径反查模块，查不到会报ModuleNotFoundError")
    print("动态加载的09/15号模块均已正确注册到 sys.modules（见文件头2026-09-23e）。")

    torch.manual_seed(seed)
    ds = _DummyTFLayoutDataset(n_genes, n_tf, n_cis, vocab_size, seed=seed)
    split_idx = _dummy_split_idx(ds)
    print(f"假数据集: {len(ds)}条样本, {n_genes}个基因, {n_tf}个TF")
    print(f"切分: train={len(split_idx['train'])} val={len(split_idx['val'])} "
         f"test={len(split_idx['test'])}")
    assert len(split_idx["val"]) > 0 and len(split_idx["test"]) > 0, \
        "自检用的假染色体切分没切出val/test，调大n_genes再试"

    # 假数据集类别不平衡程度跟真实178TF数据不一样，这里的class_counts只是为了
    # 让class_balanced_focal_loss这条代码路径被跑到，不代表任何真实计数
    class_counts = torch.tensor([10.0, 50.0, 10.0])
    common_kwargs = dict(n_tf=n_tf, vocab_size=vocab_size, pad_token_id=ds.pad_id,
                         d_model=d_model, n_heads=n_heads, cis_layers=cis_layers,
                         lay_layers=lay_layers, batch_size=batch_size,
                         n_epochs=n_epochs, patience=patience, log_every=0)

    model, history, best_epoch = run_one_seed(ds, split_idx, seed=seeds[0],
                                              class_counts=class_counts, **common_kwargs)
    assert len(history) >= 1, "一个epoch都没跑"
    assert all(math.isfinite(h["train_loss"]) for h in history), \
        "train_loss里有非有限值"
    print(f"单seed训练自检通过，最优epoch={best_epoch}")

    test_eval = evaluate(model, ds, split_idx["test"], class_counts,
                         batch_size=batch_size)
    print(f"test评估: r_a={test_eval['r_a']:.4f}(n={test_eval['n_valid_a']})  "
         f"r_b={test_eval['r_b']:.4f}(n={test_eval['n_valid_b']})  "
         f"acc_c={test_eval['acc_c']:.4f}  macro_f1_c={test_eval['macro_f1_c']:.4f}")
    assert math.isfinite(test_eval["loss"]), "test loss不是有限值"

    ci = bootstrap_ci(test_eval, n_boot=n_boot, seed=seed)
    print(f"bootstrap CI: {ci}")
    for k in ("r_a_ci", "r_b_ci", "macro_f1_c_ci"):
        lo, hi = ci[k]
        if not math.isnan(lo):
            assert lo <= hi, f"{k} 的区间下界比上界大，不对"
    print("bootstrap CI 自检通过：区间上下界方向正确。")

    # ---------------- 第二轮提速新增的自检 ----------------
    print("---- 第二轮提速①：GeneGroupedBatchSampler 覆盖性/可复现性 ----")
    genes_arr = ds.samples["gene_id"].to_numpy()
    tr = split_idx["train"]
    for k in (1, 2, 4):
        sp = GeneGroupedBatchSampler(genes_arr, tr, batch_size=batch_size, tfs_per_gene=k,
                                     seed=seed)
        ep1, ep2 = list(sp), list(sp)
        assert sorted(i for b in ep1 for i in b) == sorted(tr), \
            f"K={k}: 一个 epoch 没有恰好覆盖每条训练样本一次"
        assert sorted(i for b in ep2 for i in b) == sorted(tr), f"K={k}: 第二个 epoch 覆盖不对"
        assert len(ep1) == len(sp), f"K={k}: __len__ 跟实际 batch 数对不上"
        assert all(len(b) <= max(batch_size, k) for b in ep1), f"K={k}: batch 超长"
        cpb = max(1, batch_size // k)
        assert all(len(set(genes_arr[b].tolist())) <= cpb for b in ep1), \
            f"K={k}: 一个 batch 里的不同基因数超过 batch_size//K"
        sp_again = GeneGroupedBatchSampler(genes_arr, tr, batch_size=batch_size,
                                           tfs_per_gene=k, seed=seed)
        assert list(sp_again) == ep1, f"K={k}: 同一 seed 不可复现"
        print(f"  K={k}: {len(ep1)} 个 batch，覆盖/长度/可复现 ✓；两个 epoch 顺序"
              f"{'不同' if ep1 != ep2 else '相同(数据太小时可能偶然相同)'}")

    print("---- 第二轮提速②：分组训练(K=2) / 逐样本训练(legacy) 各跑 1 个 epoch ----")
    kw1 = dict(common_kwargs, n_epochs=1)
    _, h_g, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             forward_mode="grouped", tfs_per_gene=2, layout_buckets=3, **kw1)
    _, h_l, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             forward_mode="legacy", **kw1)
    assert math.isfinite(h_g[0]["train_loss"]) and math.isfinite(h_g[0]["val_loss"])
    assert math.isfinite(h_l[0]["train_loss"]) and math.isfinite(h_l[0]["val_loss"])
    print(f"  grouped K=2: train_loss={h_g[0]['train_loss']:.4f}  legacy: "
          f"train_loss={h_l[0]['train_loss']:.4f}(两者训练随机性不同，数值不要求相等) ✓")

    print("---- 第二轮提速③：evaluate 分组 vs 逐样本 逐元素一致(打乱的 idx 顺序) ----")
    with torch.no_grad():  # 零初始化末层随机化，否则 ctx_d/ψ_corr 用错行也测不出来
        for lin in (model.fusion.condition.net[-1], model.psi_corr[-1]):
            lin.weight.normal_(0.0, 0.1)
            lin.bias.normal_(0.0, 0.1)
    perm_test = [int(i) for i in np.random.default_rng(1).permutation(split_idx["test"])]
    ev_l = evaluate(model, ds, perm_test, class_counts, batch_size=batch_size,
                    forward_mode="legacy")
    ev_g = evaluate(model, ds, perm_test, class_counts, batch_size=batch_size,
                    forward_mode="grouped", eval_batch_size=5)
    ev_b = evaluate(model, ds, perm_test, class_counts, batch_size=batch_size,
                    forward_mode="grouped", eval_batch_size=20, layout_buckets=3)
    for ev_x, tag in ((ev_g, "分组"), (ev_b, "分组+分段")):
        for k in ("y_a_pred", "y_b_pred", "logits_c"):
            d = float((ev_l[k] - ev_x[k]).abs().max())
            assert d < 1e-4, f"evaluate {tag} vs 逐样本 {k} 最大差 {d}"
        assert abs(ev_l["loss"] - ev_x["loss"]) < 1e-4, (tag, ev_l["loss"], ev_x["loss"])
        assert torch.equal(ev_l["y_c_true"], ev_x["y_c_true"]), f"{tag}: 标签放回顺序不对"
        assert list(ev_x["genes"]) == list(ev_l["genes"]), f"{tag}: genes 顺序不对"
    print(f"  预测/标签/loss 一致 ✓ (loss {ev_l['loss']:.6f} vs {ev_g['loss']:.6f} vs "
          f"{ev_b['loss']:.6f})")

    print("---- 第三轮：按基因整群的 bootstrap ----")
    ci_g = bootstrap_ci(ev_l, n_boot=n_boot, seed=seed, groups=ev_l["genes"], min_valid=2)
    assert ci_g["unit"] == "gene" and ci_g["n_units"] == ev_l["n_genes"], ci_g
    assert ci_g["n_boot_effective"]["f1"] == n_boot
    for k in ("r_a_ci", "r_a_gene_ci", "r_b_ci", "macro_f1_c_ci"):
        lo, hi = ci_g[k]
        assert math.isnan(lo) or lo <= hi, f"{k} 区间方向不对"
    # 人工构造：每个基因复制很多遍的数据上，按样本重采样的区间应明显窄于按基因整群
    g_rng = np.random.default_rng(seed)
    n_g, rep_n = 30, 50
    pa = torch.from_numpy(g_rng.normal(size=n_g)).float().repeat_interleave(rep_n)
    ta = (pa + torch.from_numpy(g_rng.normal(size=n_g)).float().repeat_interleave(rep_n))
    fake = dict(y_a_pred=pa, y_a_true=ta, y_b_pred=pa.clone(),
                y_b_true=torch.full_like(pa, float("nan")),
                logits_c=torch.zeros(len(pa), 3), y_c_true=torch.ones(len(pa), dtype=torch.long))
    grp_fake = np.repeat(np.arange(n_g), rep_n)
    w_gene = np.subtract(*bootstrap_ci(fake, n_boot=200, seed=0, groups=grp_fake)["r_a_ci"][::-1])
    w_samp = np.subtract(*bootstrap_ci(fake, n_boot=200, seed=0)["r_a_ci"][::-1])
    assert w_gene > 2 * w_samp, (w_gene, w_samp)
    print(f"  整群 bootstrap ✓；构造数据(30个基因各复制50遍)上 r_a 区间宽度：按基因 "
          f"{w_gene:.3f} vs 按样本 {w_samp:.3f}(后者偏窄，这正是改口径的原因)")

    print("---- 第二轮提速④：--mode bench 代码路径(假数据、CPU、极少步数，只测跑得通) ----")
    bench = run_benchmark(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                          cis_layers=cis_layers, lay_layers=lay_layers,
                          batch_size=batch_size, tfs_per_gene=2, eval_batch_size=8,
                          num_workers=0, bench_steps=2, bench_warmup=1, n_eval_genes=2,
                          device="cpu", ds=ds, split_idx=split_idx,
                          class_counts=class_counts, seed=seed, layout_buckets=3,
                          profile_steps=2, bench_batch_sweep=True,
                          bench_batch_sweep_sizes=f"{batch_size},{batch_size * 2}")
    assert bench["equiv_fp32_ok"], "bench 里分组 vs 逐样本对照没通过"
    assert all(r["loss_finite"] for r in bench["train_rows"]), "bench 训练吞吐里 loss 非有限"
    # 【2026-09-23g新增，见文件头第11条(f)】⑨ batch_size扫描不涉及torch.compile，纯CPU
    # 可测：确认代码路径本身跑得通、产出的行数对得上传入的候选个数。不测compile相关
    # 分支(⑧节/_maybe_compile_model)——原因跟这个函数一直以来不测--compile一样，见
    # 上面"self_test 模式不使用 --compile"那条提示，编译在CPU上要么不生效要么没意义。
    assert "batch_sweep_rows" in bench and len(bench["batch_sweep_rows"]) == 2, \
        "batch_size 扫描(⑨)没有产出预期的行数"
    print("  bench 代码路径 ✓(含⑨ batch_size 扫描)")

    print("---- 多seed(5-seed协议，这里用2个seed测流程) ----")
    multi_results = run_multi_seed(ds, split_idx, seeds=seeds, class_counts=class_counts,
                                   n_boot=n_boot, **common_kwargs)
    assert len(multi_results) == len(seeds), "多seed跑出来的结果数量不对"
    print("多seed自检通过。")

    print("---- 断点续跑(resume)：同版本同配置跳过；配置变了则备份旧文件后重训 ----")
    with tempfile.TemporaryDirectory() as tmp:
        r1 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **common_kwargs)
        r2 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **common_kwargs)
        assert r2[0]["model"] is None, "同配置第二次运行应该跳过训练"
        assert r2[0]["test_eval"]["loss"] == r1[0]["test_eval"]["loss"]
        r3 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **dict(common_kwargs, lr=5e-4))
        assert r3[0]["model"] is not None, "配置变了应该重训"
        assert any(f.startswith(f"seed{seeds[0]}_best.pt.bak_") for f in os.listdir(tmp)), \
            "配置变了时旧 checkpoint 应该被改名备份"
    print("断点续跑自检通过。")

    print("---- leave-TF-out ----")
    held_out_tfs = ds.tf_list[:2]
    lto_result = run_leave_tf_out(ds, split_idx, held_out_tfs, seed=seeds[0],
                                  class_counts=class_counts, n_boot=n_boot,
                                  **common_kwargs)
    if lto_result is not None:
        assert math.isfinite(lto_result["heldout_eval"]["loss"]), \
            "leave-TF-out评估loss不是有限值"
        print("leave-TF-out自检通过。")
    else:
        print("这次假数据随机抽样正好没让held_out_tfs落进val/test，属于概率性"
             "结果，不是bug(真实178TF数据上这种概率很低，不用特别处理；但如果"
             "多次真实运行都遇到这个情况，说明held_out_tfs选得有问题，比如选到"
             "了从来没在任何val/test基因里被敲除过的TF)。")

    print("---- 第2批：子epoch验证/composite选模型/各头最优/AdamW+热身余弦/梯度裁剪/Huber/"
          "head_c/max_steps ----")
    kw2 = dict(common_kwargs, n_epochs=2, patience=10, val_every=0.5,
               select_metric="composite", save_per_head_best=True, optimizer="adamw",
               weight_decay=0.05, lr_schedule="warmup_cosine", warmup_steps=3,
               min_lr_ratio=0.1, grad_clip=1.0, loss_b="huber", huber_delta=1.0,
               head_c_mode="delta_tf_gene", dropout=0.2, lambda_a=0.5,
               forward_mode="grouped", tfs_per_gene=2)
    m2, h2, b2 = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts, **kw2)
    assert len(h2) == 4, f"val_every=0.5×2个epoch 应该验证4次，实际 {len(h2)}"
    assert [h["val_index"] for h in h2] == [0, 1, 2, 3]
    prog = [h["epoch_progress"] for h in h2]
    assert prog == sorted(prog) and abs(prog[-1] - 2.0) < 1e-6, prog
    for h in h2:
        for k in ("val_l_a", "val_l_b", "val_l_c", "val_l_sign", "train_l_b", "val_r_a_gene",
                  "val_auprc_down", "val_composite", "lr", "clip_frac"):
            assert k in h, f"history 缺字段 {k}"
        assert 0.0 <= h["clip_frac"] <= 1.0
    comps = [h["val_composite"] for h in h2]
    assert b2["val_index"] == int(np.argmax(comps)), (b2["val_index"], comps)
    assert set(b2["head_states"]) == {"A", "B", "C"}, b2["head_states"].keys()
    for hh, key in (("A", "val_r_a_gene"), ("B", "val_r_b"), ("C", "val_auprc_mean")):
        vals = [x[key] if np.isfinite(x[key]) else -np.inf for x in h2]
        if np.isfinite(max(vals)):  # 假数据的 val 里某类可能缺席、指标全 NaN，那时退回选中权重
            assert b2["head_best"][hh]["val_index"] == int(np.argmax(vals)), (hh, vals)
    lrs = [h["lr"] for h in h2]
    assert all(a > b for a, b in zip(lrs, lrs[1:])), f"热身后余弦段 lr 应单调下降: {lrs}"
    sd_before = {k: v.clone() for k, v in m2.state_dict().items()}
    ekw = dict(batch_size=batch_size, loss_b="huber", lambda_a=0.5)
    ev_ph = evaluate_per_head(m2, b2["head_states"], ds, split_idx["test"], class_counts, **ekw)
    assert all(torch.equal(sd_before[k], v) for k, v in m2.state_dict().items()), \
        "evaluate_per_head 之后模型权重应恢复原样"
    for hh, k in (("A", "y_a_pred"), ("B", "y_b_pred"), ("C", "logits_c")):
        m_h = SiameseHeadsModel(**b2["model_kwargs"])
        m_h.load_state_dict(b2["head_states"][hh], strict=True)
        ev_h = evaluate(m_h, ds, split_idx["test"], class_counts, **ekw)
        assert torch.allclose(ev_h[k], ev_ph[k], atol=1e-6), f"各头组合的 {k} 应来自 {hh} 头权重"
    assert math.isfinite(ev_ph["loss"]) and "auprc_down" in ev_ph and "loss_parts" in ev_ph
    _, h3, b3 = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             **dict(common_kwargs, n_epochs=3, max_steps=5))
    assert len(h3) == 1 and h3[0]["step"] == 5 and b3["stop_reason"] == "max_steps", h3
    assert b2["stop_reason"] == "n_epochs", f"跑满 n_epochs 时 stop_reason 应为 n_epochs: {b2['stop_reason']}"
    assert all(np.isfinite(h["grad_norm_mean"]) and h["grad_norm_mean"] > 0 for h in h2)
    _, h4, b4 = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             **dict(kw2, min_delta=10.0))
    assert b4["val_index"] == 0 and len(h4) == 4, "min_delta 很大时只有第一次验证算改善"
    # ---- 2026-10-05a 第18条：wt_input="head_bc"(实测 WT 表达=batch["y_a"] 只进 Head B/C)：训练/评估/重建/默认不写键 ----
    print("---- 第18条：wt_input=head_bc 训练、评估、按 model_kwargs 重建 ----")
    kw_wt = dict(kw2, wt_input="head_bc", n_epochs=1, val_every=1.0)
    m_wt, h_wt, b_wt = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts, **kw_wt)
    assert getattr(m_wt, "wt_input", None) == "head_bc", "run_one_seed 返回的模型应带 wt_input"
    assert b_wt["model_kwargs"].get("wt_input") == "head_bc", b_wt["model_kwargs"]
    assert b2["model_kwargs"].get("wt_input") is None, "默认 none 时 model_kwargs 里不应写 wt_input(旧 checkpoint 逐字节不变)"
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in h_wt), h_wt
    ev_wt = evaluate(m_wt, ds, split_idx["test"], class_counts, batch_size=batch_size, loss_b="huber", lambda_a=0.5)
    assert math.isfinite(ev_wt["loss"]), "wt_input=head_bc 的 test loss 应有限"
    m_wt_re = SiameseHeadsModel(**b_wt["model_kwargs"])
    m_wt_re.load_state_dict(m_wt.state_dict(), strict=True)
    print("wt_input=head_bc：训练/评估有限、model_kwargs 记录正确、按它重建能 strict 加载、默认 none 不写键 ✓")
    with tempfile.TemporaryDirectory() as tmp:
        r1 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **kw2)
        ck = torch.load(os.path.join(tmp, f"seed{seeds[0]}_best.pt"), map_location="cpu",
                        weights_only=False)
        for k in ("model_kwargs", "best", "head_states", "test_eval_per_head",
                  "test_ci_per_head", "history", "train_config"):
            assert ck.get(k) is not None, f"checkpoint 缺字段 {k}"
        assert not any(f.endswith(".tmp") for f in os.listdir(tmp)), "临时文件没被替换掉"
        m_re = SiameseHeadsModel(**ck["model_kwargs"])
        m_re.load_state_dict(ck["model_state"], strict=True)
        assert ck["train_config"].get("ctx_mode", "legacy") == "legacy" and ck["best"]["metric"] == "composite"
        r2 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **kw2)
        assert r2[0]["model"] is None and r2[0]["test_eval_per_head"] is not None, \
            "同配置第二次应跳过训练、并带回各头组合的 test 指标"
        assert r1[0]["test_eval"]["loss"] == r2[0]["test_eval"]["loss"]
        # 2026-09-25a：兼容版本(24a)存的、train_config 里没有新键 min_delta 的 checkpoint 应可复用
        pth = os.path.join(tmp, f"seed{seeds[0]}_best.pt")
        ck_old = torch.load(pth, map_location="cpu", weights_only=False)
        ck_old["code_version"] = "2026-09-24a"
        ck_old["train_config"].pop("min_delta", None)
        torch.save(ck_old, pth)
        r3 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **dict(kw2, min_delta=1e-6))
        assert r3[0]["model"] is None, "24a 版、缺 min_delta 键(=默认值)的 checkpoint 应被复用"
        r4 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **dict(kw2, min_delta=0.01))
        assert r4[0]["model"] is not None, "min_delta 不同时不应复用"
    print(f"  第2批 ✓：4次子epoch验证、composite 选中 val#{b2['val_index']}、三个头各自最优 "
          f"{ {h: v['val_index'] for h, v in b2['head_best'].items()} }、lr 热身后单调下降"
          f"({lrs[0]:.2e}→{lrs[-1]:.2e})、evaluate_per_head 组合正确且不改权重、max_steps 生效、"
          f"新 checkpoint 字段齐全可 strict 重建、断点续跑带回各头指标")

    print("---- 2026-09-25b 第3批：Head B 稠密 log2FC 辅助目标(λ_bd) ----")
    kw5 = dict(common_kwargs, n_epochs=1, val_every=0.5, forward_mode="grouped", tfs_per_gene=2)
    _, h5a, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts, **kw5)
    _, h5b, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             **dict(kw5, dense_target="<dummy>"))
    assert [h["train_loss"] for h in h5a] == [h["train_loss"] for h in h5b] and \
        [h["val_loss"] for h in h5a] == [h["val_loss"] for h in h5b], \
        "λ_bd=0 时只加载稠密目标(监控)不应改变训练/验证 loss"
    assert all(np.isfinite(h["val_r_bd"]) and np.isfinite(h["val_r_bd_ns"]) for h in h5b), h5b[-1]
    assert not any(np.isfinite(h["val_r_bd"]) for h in h5a), "没加载稠密目标时 val_r_bd 应为 NaN"
    assert all(h["train_l_bd"] == 0.0 for h in h5b), "λ_bd=0 时训练里不应计算 l_bd"
    assert all(h["val_l_bd"] > 0 for h in h5b), "加载了稠密目标时 val 上应报 l_bd"
    _, h5c, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             **dict(kw5, dense_target="<dummy>", lambda_bd=1.0))
    assert all(h["train_l_bd"] > 0 for h in h5c), "λ_bd>0 时训练 l_bd 应该>0"
    assert h5c[0]["train_loss"] != h5b[0]["train_loss"], "λ_bd>0 应改变训练 loss"
    for h in h5c:
        tot_parts = (h["train_l_a"] + h["train_l_b"] + h["train_l_c"] + 0.1 * h["train_l_sign"]
                     + 1.0 * h["train_l_bd"])
        assert abs(h["train_loss"] - tot_parts) < 1e-4, (h["train_loss"], tot_parts)
    try:
        run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                     **dict(kw5, lambda_bd=1.0))
        raise AssertionError("λ_bd>0 但没有稠密目标时应该报错")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        r1 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **kw5)
        r2 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **dict(kw5, dense_target="<dummy>"))
        assert r2[0]["model"] is None, "λ_bd=0 时 dense_target 只用于监控，不应让 checkpoint 失效"
        pth = os.path.join(tmp, f"seed{seeds[0]}_best.pt")
        ck_old = torch.load(pth, map_location="cpu", weights_only=False)
        ck_old["code_version"] = "2026-09-25a"
        for _k in ("lambda_bd", "dense_on", "dense_target", "dense_delta"):
            ck_old["train_config"].pop(_k, None)
        torch.save(ck_old, pth)
        r3 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **kw5)
        assert r3[0]["model"] is None, "25a 版、缺稠密相关键(=默认值)的 checkpoint 应被复用"
        r4 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp,
                            **dict(kw5, dense_target="<dummy>", lambda_bd=1.0))
        assert r4[0]["model"] is not None, "λ_bd 不同时不应复用"
        ck4 = torch.load(pth, map_location="cpu", weights_only=False)
        assert ck4["train_config"]["lambda_bd"] == 1.0 and ck4["train_config"]["dense_target_digest"]
        assert np.isfinite(ck4["test_eval"]["r_bd"]), "test_eval 应带 r_bd"
    ds.set_dense_target(None)
    print(f"  第3批 ✓：λ_bd=0 时加载稠密目标不改变训练(逐位)、val 上报 l_bd/r_bd/r_bd_ns"
          f"(最后一次 r_bd={h5b[-1]['val_r_bd']:.3f})、λ_bd>0 生效且 train_loss=各项加权和、缺标签报错、"
          f"断点续跑配置规则、25a 兼容")

    print("---- 2026-09-26a 第4批：权重 EMA(ema_decay) ----")
    kw6 = dict(common_kwargs, n_epochs=1, val_every=0.5, forward_mode="grouped", tfs_per_gene=2,
               ema_decay=0.0)
    _, h6a, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts, **kw6)
    _, h6b, _ = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                             **{k: v for k, v in kw6.items() if k != "ema_decay"})
    assert [h["train_loss"] for h in h6a] == [h["train_loss"] for h in h6b] and \
        [h["val_loss"] for h in h6a] == [h["val_loss"] for h in h6b], \
        "ema_decay=0(显式)应跟不传逐位相同(EMA 默认关不改变任何数)"
    m6, h6c, b6 = run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                               **dict(kw6, ema_decay=0.9))
    assert len(h6c) == len(h6a) and all(np.isfinite(h["val_loss"]) for h in h6c), h6c
    assert abs(h6c[0]["train_loss"] - h6a[0]["train_loss"]) < 1e-5, \
        "EMA 不改变训练轨迹，第一个验证点之前的 train_loss 应跟不开 EMA 一致"
    assert [h["val_loss"] for h in h6c] != [h["val_loss"] for h in h6a], \
        "EMA 权重跟训练权重不同，验证 loss 应当改变"
    for bad_decay in (-0.1, 1.0):
        try:
            run_one_seed(ds, split_idx, seed=seeds[0], class_counts=class_counts,
                         **dict(kw6, ema_decay=bad_decay))
            raise AssertionError(f"ema_decay={bad_decay} 应该报错")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                       n_boot=n_boot, save_dir=tmp, **kw6)
        pth = os.path.join(tmp, f"seed{seeds[0]}_best.pt")
        ck_old = torch.load(pth, map_location="cpu", weights_only=False)
        ck_old["code_version"] = "2026-09-25b"
        ck_old["train_config"].pop("ema_decay", None)
        torch.save(ck_old, pth)
        r2 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **kw6)
        assert r2[0]["model"] is None, "25b 版、缺 ema_decay 键(=默认0)的 checkpoint 应被复用"
        r3 = run_multi_seed(ds, split_idx, seeds=seeds[:1], class_counts=class_counts,
                            n_boot=n_boot, save_dir=tmp, **dict(kw6, ema_decay=0.9))
        assert r3[0]["model"] is not None, "ema_decay 不同时不应复用"
        ck3 = torch.load(pth, map_location="cpu", weights_only=False)
        assert ck3["train_config"]["ema_decay"] == 0.9
    print(f"  第4批 ✓：ema_decay=0 逐位不变、=0.9 时验证 loss 改变而训练 loss 不变(第一个验证点)、"
          f"非法值报错、25b 兼容(缺键按0比较)、ema_decay 不同不复用")

    print("自检通过：染色体切分、单seed训练+早停、evaluate、bootstrap CI、"
         "5-seed汇总、leave-TF-out，以及第二轮提速的分组采样/分组训练/分组评估一致性/"
         "bench 代码路径、动态加载模块的 sys.modules 注册、第2批(子epoch验证/composite/各头最优/"
         "AdamW+余弦/裁剪/Huber/head_c/max_steps/新checkpoint字段)，整条流水线都跑通了。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["self_test", "real", "bench"], default="self_test",
                    help="self_test=假数据全流程自检；real=正式训练；bench=真实数据上只测量"
                         "(等价性对照+各配置吞吐/显存，不训练，见 run_benchmark)")
    # self_test模式专用参数(只控制自检用假数据的规模，跟下面共用的几个flag无关)
    ap.add_argument("--n-genes", type=int, default=20)
    ap.add_argument("--n-tf", type=int, default=6)
    ap.add_argument("--n-cis", type=int, default=30)
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="不传时：self_test=40，real/bench=4000")
    # ↓↓↓ 下面这些flag是self_test/real两种模式共用同一个名字，但"正确默认值"完全
    # 不一样(self_test要小、real要用《方案.txt》规格)。
    # 【2026-09-22 修复一个真bug】之前这里直接把self_test的小规模数值写成这几个
    # flag的默认值，下面real分支又把它们原样传给run_full_training()——`--mode real`
    # 不显式传这几个参数时，会静默套用self_test的小数值(batch_size=4等)，而不是
    # run_full_training()自己定义的正式规格。改法：默认值全部换成None(哨兵值)，
    # 各模式分支各自在"没传"的参数上退回到自己原来该用的默认值。
    # (第二轮提速：--vocab-size 也改成同样的哨兵写法，real/bench 以前没法从命令行传)
    ap.add_argument("--d-model", type=int, default=None,
                    help="不传时：self_test=16，real/bench=256(《方案.txt》规格)")
    ap.add_argument("--n-heads", type=int, default=None,
                    help="不传时：self_test=2，real/bench=8")
    ap.add_argument("--cis-layers", type=int, default=None,
                    help="不传时：self_test=1，real/bench=6")
    ap.add_argument("--lay-layers", type=int, default=None,
                    help="不传时：self_test=1，real/bench=4")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="不传时：self_test=4，real/bench=64")
    ap.add_argument("--n-epochs", type=int, default=None,
                    help="不传时：self_test=2，real=30(bench 只用它估算总时长)")
    ap.add_argument("--patience", type=int, default=None,
                    help="不传时：self_test=2，real=5")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="不传时：self_test=50，real=1000")
    # real/bench模式额外参数
    ap.add_argument("--layout", default="out/tf_layout.parquet")
    ap.add_argument("--labels", default="out/head_bc_labels.parquet")
    ap.add_argument("--head-a", default="out/head_a_baseline_logtpm.parquet")
    ap.add_argument("--sgd", default="data/SGD_features.tab")
    ap.add_argument("--promoter-tokens", default="out/promoter_token_ids.parquet")
    ap.add_argument("--bpe-tokenizer", default="out/bpe_tokenizer.json")
    ap.add_argument("--held-out-tfs", default=None,
                    help="逗号分隔的TF名列表，比如 GAL4,MSN2；不传就不跑leave-TF-out")
    ap.add_argument("--save-dir", default="out/checkpoints",
                    help="传空字符串''可以关掉逐seed存checkpoint(比如跑smoke test时)")
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的seed列表，real模式专用，不传默认"
                        "42,123,456,789,2024(用户指定的5-seed)")
    ap.add_argument("--lr", type=float, default=None,
                    help="real/bench模式，不传默认1e-4")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader后台进程数，不传时：self_test=0，real/bench=4")
    # 第二轮提速(见文件头第9条)：不传就用 run_full_training/run_benchmark 自己的默认值
    ap.add_argument("--forward", choices=["grouped", "legacy"], default=None,
                    help="real模式：grouped=分组前向(默认)，legacy=原逐样本前向")
    ap.add_argument("--amp", choices=["bf16", "off"], default=None,
                    help="real模式：bf16=混合精度(默认)，off=纯fp32")
    ap.add_argument("--tfs-per-gene", type=int, default=None,
                    help="real/bench模式：训练batch里每个基因放几条样本，默认4；1=原来的"
                         "逐样本独立打乱(唯一改变训练动态的开关，信心6/10)")
    ap.add_argument("--eval-batch-size", type=int, default=None,
                    help="real/bench模式：分组评估时每个batch约多少条样本(整基因打包)，默认"
                         "512，只影响速度/显存，不影响结果")
    ap.add_argument("--layout-buckets", type=int, default=None,
                    help="real/bench模式【第三轮，第四轮把默认值从3改成1，见文件头第11条(a)】："
                         "layout 分支按 L_g 长度至多分几段(默认1=不分段——用户自己那份bench.txt"
                         "⑥实测分1段比默认分3段快，1=不分段)，结果不变、只影响速度")
    ap.add_argument("--compile", action="store_true",
                    help="real/bench模式【第四轮，见文件头第11条(b)；2026-09-23f加了安全网，"
                         "2026-09-23g加了--compile-scope，见第11条(e)(f)】：用torch.compile"
                         "编译模型的一部分(范围见--compile-scope)，减少每步kernel启动次数。"
                         "默认关闭，正式--train前强烈建议先跑一次 `--mode bench --compile` "
                         "看⑧节四组(不编译/full+auto/full+True/submodules+auto)对比再决定")
    ap.add_argument("--compile-cache-limit", type=int, default=16,
                    help="real/bench模式【2026-09-23f新增，见文件头第11条(e)①】：只在加了"
                         "--compile 时生效。torch._dynamo.config.cache_size_limit，某个调用点"
                         "重编译次数超过这个数就自动退回eager、不再无限期重新编译——防止上面"
                         "--compile 说明里那种'反复重编译看起来卡住'的核心安全网，不影响任何"
                         "数值结果")
    ap.add_argument("--compile-dynamic", choices=["auto", "true"], default=None,
                    help="real模式【2026-09-23f新增，见文件头第11条(e)③】：只在加了 --compile"
                         "时生效，'auto'(默认，即torch.compile的dynamic=None，只把观察到会变"
                         "的维度标成动态)或'true'(dynamic=True，上一轮的写法，从第一次调用起"
                         "全部维度都当动态量，这次bench_--compile.txt里观察到反复重新编译的"
                         "就是这个模式)。哪个更快/更稳以 `--mode bench --compile` 的⑧节实测"
                         "为准；bench模式本身固定把两种都测一遍，不受这个flag影响")
    ap.add_argument("--compile-warmup-timeout", type=float, default=300.0,
                    help="bench模式【2026-09-23f新增，见文件头第11条(e)②】：⑧节热身阶段的"
                         "总墙钟预算(秒)，超过就自动放弃当前这组编译对照、打印诊断，不用再"
                         "手动Ctrl+Z(局限：只在步与步之间的检查点生效，挡不住单独一步内部真的"
                         "卡住不返回的极端情况，见_CompileWarmupTimeout文档字符串)")
    ap.add_argument("--compile-scope", choices=["full", "submodules"], default=None,
                    help="real模式【2026-09-23g新增，见文件头第11条(f)，_maybe_compile_model】："
                         "只在加了 --compile 时生效，'full'(默认，行为跟上一轮完全一致，编译"
                         "forward/forward_grouped整个)或'submodules'(只编译cis.forward/"
                         "layout.forward两个子模块，按bench③的占比二者合计约占前向用时94%，"
                         "forward_grouped里的数据依赖分支+fusion/头留在eager，不进Dynamo trace，"
                         "更不容易触发反复重编译，但没有在真实机器上验证过)。哪个更快/更稳以"
                         "`--mode bench --compile`的⑧节实测为准(这次新增了第四组对照)；bench"
                         "模式本身固定把 full/submodules 都测一遍，不受这个flag影响")
    ap.add_argument("--no-resume", action="store_true",
                    help="real模式：不复用 save-dir 里已完成的同版本同配置 checkpoint，"
                         "全部重训(旧文件会被覆盖)")
    ap.add_argument("--bench-steps", type=int, default=30,
                    help="bench模式：每种配置计时多少个训练步")
    ap.add_argument("--bench-warmup", type=int, default=5,
                    help="bench模式：每种配置计时前先热身多少步")
    ap.add_argument("--bench-eval-genes", type=int, default=60,
                    help="bench模式：评估速度用多少个val基因(整基因)来测")
    ap.add_argument("--bench-batch-sweep", action="store_true",
                    help="bench模式【2026-09-23g新增，见文件头第11条(f)】：新增⑨节，用当前"
                         "默认配置扫几个不同的batch_size，报样本/s和峰值显存，帮判断'调大"
                         "batch_size摊薄kernel启动开销'这条路值不值得走(⑧节compile方向如果"
                         "不划算时的替代路线)。默认关闭，不影响其余①~⑧节")
    ap.add_argument("--bench-batch-sweep-sizes", type=str, default=None,
                    help="bench模式【2026-09-23g新增，见文件头第11条(f)】：只在加了"
                         "--bench-batch-sweep 时生效，逗号分隔的候选batch_size列表(如"
                         "'96,192,384')。不传就自动用当前 --batch-size 的1x/2x/4x")
    a = ap.parse_args()

    if a.mode == "self_test":
        ignored = [f for f, v in (("--forward", a.forward), ("--amp", a.amp),
                                  ("--tfs-per-gene", a.tfs_per_gene),
                                  ("--eval-batch-size", a.eval_batch_size),
                                  ("--layout-buckets", a.layout_buckets)) if v is not None]
        if ignored:
            print(f"提示：self_test 模式会自己把分组/逐样本、不同 K 都测一遍，忽略 {ignored}")
        if a.compile:
            print("提示：self_test 模式不使用 --compile(自检用的是极小规模假数据+极少epoch，"
                  "编译的一次性开销只会更慢，且自检测的是流程正确性不是速度)")
        if a.bench_batch_sweep:
            print("提示：self_test 模式的 bench 代码路径自己固定会测一次⑨节(极小规模、纯"
                  "CPU)，忽略命令行传的 --bench-batch-sweep*")
        run_self_test(a.n_genes, a.n_tf, a.n_cis,
                     a.vocab_size if a.vocab_size is not None else 40,
                     a.d_model if a.d_model is not None else 16,
                     a.n_heads if a.n_heads is not None else 2,
                     a.cis_layers if a.cis_layers is not None else 1,
                     a.lay_layers if a.lay_layers is not None else 1,
                     a.batch_size if a.batch_size is not None else 4,
                     a.n_epochs if a.n_epochs is not None else 2,
                     a.patience if a.patience is not None else 2,
                     n_boot=a.n_boot if a.n_boot is not None else 50)
    else:
        seeds = (tuple(int(s) for s in a.seeds.split(","))
                if a.seeds else (42, 123, 456, 789, 2024))
        paths = dict(layout=a.layout, labels=a.labels, head_a=a.head_a, sgd=a.sgd,
                     promoter_tokens=a.promoter_tokens, bpe_tokenizer=a.bpe_tokenizer)
        # 下面这些只有显式传了才覆盖 run_full_training()/run_benchmark() 自己的默认值；
        # 不传就完全不放进kwargs(见上面 2026-09-22 的修复说明)
        # compile_cache_limit 两种模式都支持，共用同一个默认值16，见文件头第11条(e)①
        overrides = dict(vocab_size=a.vocab_size, d_model=a.d_model, n_heads=a.n_heads,
                         cis_layers=a.cis_layers, lay_layers=a.lay_layers,
                         batch_size=a.batch_size, lr=a.lr, num_workers=a.num_workers,
                         tfs_per_gene=a.tfs_per_gene, eval_batch_size=a.eval_batch_size,
                         layout_buckets=a.layout_buckets,
                         compile_cache_limit=a.compile_cache_limit)
        if a.mode == "bench":
            kwargs = dict(paths, bench_steps=a.bench_steps, bench_warmup=a.bench_warmup,
                          n_eval_genes=a.bench_eval_genes, n_seeds=len(seeds),
                          compile_model=a.compile,
                          compile_warmup_timeout=a.compile_warmup_timeout,
                          bench_batch_sweep=a.bench_batch_sweep,
                          bench_batch_sweep_sizes=a.bench_batch_sweep_sizes)
            overrides.update(n_epochs=a.n_epochs)
            if a.forward is not None or a.amp is not None:
                print("提示：bench 模式固定把 逐样本/分组 × fp32/bf16 几种组合都测一遍，"
                      "忽略 --forward/--amp")
            if a.compile_dynamic is not None:
                print("提示：bench 模式⑧节固定把 dynamic=auto/True 两种都测一遍，"
                      "忽略 --compile-dynamic(这个flag只用于real模式实际训练)")
            if a.compile_scope is not None:
                print("提示：bench 模式⑧节固定把 scope=full/submodules 都测一遍，"
                      "忽略 --compile-scope(这个flag只用于real模式实际训练，见文件头"
                      "第11条(f))")
            kwargs.update({k: v for k, v in overrides.items() if v is not None})
            run_benchmark(**kwargs)
        else:
            held_out = a.held_out_tfs.split(",") if a.held_out_tfs else None
            save_dir = a.save_dir if a.save_dir else None
            kwargs = dict(paths, held_out_tfs=held_out, save_dir=save_dir, seeds=seeds,
                          resume=not a.no_resume, compile_model=a.compile)
            if a.compile_dynamic is not None:
                kwargs["compile_dynamic"] = a.compile_dynamic
            if a.compile_scope is not None:
                kwargs["compile_scope"] = a.compile_scope
            if a.bench_batch_sweep or a.bench_batch_sweep_sizes is not None:
                print("提示：--bench-batch-sweep* 只用于 --mode bench 的⑨节，real/smoke"
                      "模式忽略这两个flag")
            overrides.update(n_epochs=a.n_epochs, patience=a.patience, n_boot=a.n_boot,
                             amp=a.amp, forward_mode=a.forward)
            kwargs.update({k: v for k, v in overrides.items() if v is not None})
            run_full_training(**kwargs)
