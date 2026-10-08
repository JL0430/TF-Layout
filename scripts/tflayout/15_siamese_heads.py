# scripts/tflayout/15_siamese_heads.py

import argparse
import importlib.util
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_module(name: str, filename: str):
    """按文件路径动态加载模块——10_/12_/14_开头的文件名不是合法的Python标识符，
    没法直接写 `from 12_cis_transformer import CisTransformer`，只能退而求其次
    用 importlib 按路径加载。用 __file__ 的目录算路径，不依赖运行时的当前工作
    目录；假设本脚本跟10/12/14号脚本在同一目录(status文件里记录的路径都是
    scripts/tflayout/)。
    【2026-09-23e修复，见文件头说明】exec_module前必须把module注册进
    sys.modules[name]——否则模块本身能正常用，但任何要"按模块名反查回这个模块"
    的代码路径都会失败，torch.compile(TorchDynamo)追踪全局变量时走的正是这条路径，
    不注册就会报 ModuleNotFoundError(这就是这一版要修的真实bug)。顺手加一个
    "已经加载过就直接返回缓存"的判断，避免同一个 name 被 exec_module 两次、产生
    两份不同的模块/类对象。"""
    if name in sys.modules:
        return sys.modules[name]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_cis_mod = _load_module("_tflayout_12_cis_transformer", "12_cis_transformer.py")
_lay_mod = _load_module("_tflayout_10_layout_transformer", "10_layout_transformer.py")
_fusion_mod = _load_module("_tflayout_14_condition_fusion", "14_condition_fusion.py")
CisTransformer = _cis_mod.CisTransformer
LayoutTransformer = _lay_mod.LayoutTransformer
CrossModalFusion = _fusion_mod.CrossModalFusion


# 按status文件第3节08脚本报告的比例(ns=96.27%/down=2.06%/up=1.67%，基于618975条)，
# 换算到09脚本过滤后的585900条样本上的近似整数——这是用已经四舍五入过的百分比反推
# 出来的，不是精确计数(信心5/10)，只当占位参考值。正式训练前请替换成
# ds.samples["direction_3class"].value_counts() 的精确结果。顺序跟
# TFLayoutDataset.class2idx = {"down":0,"ns":1,"up":2} 对齐。
DEFAULT_CLASS_COUNTS_APPROX = {"down": 12070.0, "ns": 564057.0, "up": 9785.0}


def _pearson_r(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """batch内的皮尔逊相关系数，纯函数。eps只是防止方差为0时除0把loss污染成NaN/inf，
    不代表方差为0时返回的接近0的值是"正确"的相关系数——那种情况下相关系数本来就没有
    良好定义的数学意义，这只是一个安全的兜底。"""
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x * x).sum() * (y * y).sum()) + eps
    return num / den


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
    """target里NaN的位置不参与loss(status文件反复强调"非显著=NaN不能当0")。
    mask=None时用~isnan(target)自动推。实现细节：不能直接算(pred-target)**2再乘
    mask——NaN乘0在IEEE754里还是NaN，会把整个loss污染成NaN，所以先把target里的
    NaN替换成占位值0.0(乘mask之后这些位置反正不计入)，再算平方差，最后才乘mask、
    除以mask.sum()。mask全False(这个batch一个有效标签都没有)时返回0，而不是
    0/0出NaN——09脚本的示例batch(batch_size=8)里y_b 100%NaN是实测发生过的情况，
    不是极端假设。"""
    if mask is None:
        mask = ~torch.isnan(target)
    mask_f = mask.to(pred.dtype)
    target_filled = torch.nan_to_num(target, nan=0.0)
    diff2 = (pred - target_filled) ** 2 * mask_f
    denom = mask_f.sum().clamp(min=1.0)
    return diff2.sum() / denom


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None,
                 delta: float = 1.0) -> torch.Tensor:
    """第2批：Head B 的 Huber 版本(见文件头【2026-09-24 第2批】第1条)。"2×标准Huber"：
    |e|≤δ 时=e²(跟 masked_mse 逐位相同)，|e|>δ 时=2δ|e|−δ²，单样本梯度被限制在±2δ。
    NaN 处理、全 False mask 返回0、不触发 GPU 同步，这三点跟 masked_mse 同一写法。"""
    if mask is None:
        mask = ~torch.isnan(target)
    mask_f = mask.to(pred.dtype)
    e = pred - torch.nan_to_num(target, nan=0.0)
    ae = e.abs()
    quad = torch.clamp(ae, max=float(delta))
    h = quad * quad + 2.0 * float(delta) * (ae - quad)
    return (h * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def head_a_loss(y_a_pred: torch.Tensor, y_a_true: torch.Tensor,
                mse_weight: float = 0.5) -> torch.Tensor:
    """Head A: 0.5*MSE+0.5*(1-r)，跟status文件第4节"跟CITRA对齐"一致。这里的r是
    在当前batch内算的，不是整个validation set的r——工程选择，信心5/10，batch_size
    小(比如09脚本示例的8)时这一项噪声会很大，建议正式训练时先只监控这个指标、
    确认batch大到能稳定估计r了再让它参与梯度，或者把mse_weight调到1.0先关掉这项。
    y_a_true里的NaN(09脚本里head_a文件缺失时的占位)会被跳过，不参与这个loss。"""
    # 第三轮提速：不用布尔索引/if 判断(都会触发 GPU 同步)，改成乘 mask 的等价写法：
    # 有效点<1 时整个 loss=0、有效点=1 时 r 取0——跟原来的分支逻辑逐项对应
    mask = ~torch.isnan(y_a_true)
    m = mask.to(y_a_pred.dtype)
    n = m.sum()
    true0 = torch.nan_to_num(y_a_true, nan=0.0).to(y_a_pred.dtype)
    n_c = n.clamp(min=1.0)
    mse = (((y_a_pred - true0) ** 2) * m).sum() / n_c
    xc = (y_a_pred - (y_a_pred * m).sum() / n_c) * m
    yc = (true0 - (true0 * m).sum() / n_c) * m
    # sqrt 的输入先 clamp 到极小正数：值不变(>1e-30 时逐位相同)，但避免"方差恰为0
    # (有效点≤1、或预测全相同)时 sqrt 在0处导数为 inf、乘上 torch.where 给的0梯度得 NaN"
    ss = ((xc * xc).sum() * (yc * yc).sum()).clamp(min=1e-30)
    r = (xc * yc).sum() / (torch.sqrt(ss) + 1e-8)
    r = torch.where(n >= 2, r, torch.zeros_like(r))
    loss = mse_weight * mse + (1 - mse_weight) * (1 - r)
    return torch.where(n > 0, loss, torch.zeros_like(loss))


def class_balanced_focal_loss(logits: torch.Tensor, target: torch.Tensor,
                              class_counts: torch.Tensor, gamma: float = 2.0,
                              beta: float = 0.999, ignore_neg: bool = False) -> torch.Tensor:
    """Head C: down/ns/up三分类，class-balanced focal loss。按Cui et al. 2019
    "Class-Balanced Loss Based on Effective Number of Samples"算类别权重
    (effective_num=1-beta^n_y，weight∝1/effective_num)，再乘标准focal loss
    (Lin et al. 2017)的(1-p_t)^gamma调制项。beta=0.999/gamma=2.0是这两篇论文
    各自最常用的默认值，不是针对这批178TF数据专门调过的，信心7/10——机制上是
    标准实现，具体数值建议后续按验证集表现微调。class_counts要求是"整个训练集"
    的类别计数(不是这个batch的)，顺序跟class2idx={"down":0,"ns":1,"up":2}对齐；
    DEFAULT_CLASS_COUNTS_APPROX只是按status文件第3节比例换算出的近似占位值，
    正式训练前请换成精确计数(见文件头说明)。"""
    n_classes = logits.shape[-1]
    # 第三轮提速：底数直接用 Python float(原来先 as_tensor 到显卡上，会触发一次同步)
    effective_num = 1.0 - torch.pow(float(beta), class_counts.to(torch.float32))
    weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
    weights = weights / weights.sum() * n_classes  # 归一化到均值为1，方便和其它loss项配平

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    valid = None
    if ignore_neg:  # 2026-09-28b 第8批：y_c<0(Head A 伪样本)不参与，见文件头第8批
        valid = (target >= 0).to(logits.dtype)
        target = target.clamp(min=0)
    target_onehot = F.one_hot(target, num_classes=n_classes).to(logits.dtype)
    pt = (probs * target_onehot).sum(-1)
    focal_weight = (1 - pt).clamp(min=0.0) ** gamma
    ce = -(target_onehot * log_probs).sum(-1)
    alpha_t = weights.to(logits.device)[target]
    if valid is None:
        return (alpha_t * focal_weight * ce).mean()
    return (alpha_t * focal_weight * ce * valid).sum() / valid.sum().clamp(min=1.0)


def sign_consistency_loss(y_b_pred: torch.Tensor, mask_sig: torch.Tensor,
                          logits_c: torch.Tensor) -> torch.Tensor:
    """Head B/C的sign consistency惩罚(status文件第4节提到"B/C间有sign consistency
    惩罚"，但没给出具体公式——这整段是工程选择，信心6/10)。做法：把Head B的连续
    预测过tanh压到(-1,1)当"软符号"，把Head C的三分类概率算P(up)-P(down)当另一个
    "软符号"(down=0,ns=1,up=2，跟09_torch_dataset.py的class2idx一致)，两者的均方
    差当惩罚，只在y_b非NaN(即"显著"子集S)上算——跟Head B用同一个mask。理由：在
    ns那部分，Head B的log2FC标签本身是NaN、ŷ^FC没有监督信号锚定它的符号，如果在
    这部分也强制它跟Head C的P(up)-P(down)对齐，等于用一个没被训练过的量反过来
    污染另一个量，没有意义。"""
    mask_f = mask_sig.to(y_b_pred.dtype)
    # (第三轮提速：去掉了原来的 `if mask_f.sum()==0: return 0`——它会触发 GPU 同步；
    # mask 全 False 时下面 diff2 全是0、分母 clamp 到1，结果同样是0)
    probs = F.softmax(logits_c, dim=-1)
    soft_sign_c = probs[:, 2] - probs[:, 0]  # P(up)-P(down)，class2idx: down=0,ns=1,up=2
    soft_sign_b = torch.tanh(y_b_pred)
    diff2 = (soft_sign_b - soft_sign_c) ** 2 * mask_f
    return diff2.sum() / mask_f.sum().clamp(min=1.0)


def compute_total_loss(outputs: dict, y_a: torch.Tensor, y_b: torch.Tensor,
                       y_c: torch.Tensor, class_counts: torch.Tensor,
                       lambda_b: float = 1.0, lambda_c: float = 1.0,
                       lambda_sign: float = 0.1, gamma: float = 2.0,
                       beta: float = 0.999, return_tensors: bool = False,
                       lambda_a: float = 1.0, loss_b: str = "mse",
                       huber_delta: float = 1.0, y_bd: torch.Tensor = None,
                       lambda_bd: float = 0.0, dense_on: str = "ns",
                       dense_delta: float = None, c_ignore: bool = False) -> tuple:
    """L = L_A + λ_B·L_B + λ_C·L_C + λ_s·L_sign，status文件第4节原文公式。
    λ_B=λ_C=1.0/λ_sign=0.1是随手给的起点，没在真实数据上调过，信心4/10(单独
    标注，低于7分)——三个头的loss量纲差异比较大(L_A是回归+相关系数、L_C是分类
    交叉熵、L_B是log2FC的MSE)，正式训练时几乎一定要根据每一项loss的实际数量级
    重新配比，这里的默认值只是让代码能跑起来、且没有哪一项权重是0。
    【2026-09-25b】y_bd/lambda_bd/dense_on/dense_delta：稠密 log2FC 辅助项，见文件头第3批说明；
    y_bd=None 时 l_bd=0、total 跟原来逐位相同；lambda_bd=0 但传了 y_bd 时只算出 l_bd 供监控，
    不进 total。
    【2026-09-28b】c_ignore=True：Head C 忽略 y_c<0 的行(第8批 Head A 伪样本)；默认 False 逐位不变。"""
    l_a = head_a_loss(outputs["y_a_pred"], y_a)
    mask_sig = ~torch.isnan(y_b)
    if loss_b == "huber":  # 第2批，见文件头第1条；默认 "mse" 跟原来一样
        l_b = masked_huber(outputs["y_b_pred"], y_b, mask=mask_sig, delta=huber_delta)
    elif loss_b == "mse":
        l_b = masked_mse(outputs["y_b_pred"], y_b, mask=mask_sig)
    else:
        raise ValueError(f"loss_b 只能是 mse/huber，收到 {loss_b}")
    l_c = class_balanced_focal_loss(outputs["logits_c"], y_c, class_counts,
                                    gamma=gamma, beta=beta, ignore_neg=bool(c_ignore))  # 2026-09-28b
    l_sign = sign_consistency_loss(outputs["y_b_pred"], mask_sig, outputs["logits_c"])
    total = lambda_a * l_a + lambda_b * l_b + lambda_c * l_c + lambda_sign * l_sign
    # 第3批：稠密辅助项。Python 层面的 if 只看参数(不看张量值)，不触发 GPU 同步
    if dense_on not in ("ns", "all"):
        raise ValueError(f"dense_on 只能是 ns/all，收到 {dense_on}")
    if y_bd is not None:
        mask_bd = ~torch.isnan(y_bd)
        if dense_on == "ns":
            mask_bd = mask_bd & ~mask_sig
        l_bd = masked_huber(outputs["y_b_pred"], y_bd.to(outputs["y_b_pred"].dtype), mask=mask_bd,
                            delta=huber_delta if dense_delta is None else dense_delta)
        if lambda_bd:
            total = total + lambda_bd * l_bd
    else:
        if lambda_bd:
            raise ValueError("lambda_bd>0 但没有传 y_bd(稠密目标没加载？)")
        l_bd = torch.zeros((), dtype=l_b.dtype, device=l_b.device)
    # return_tensors=True：parts 里放 detach 后的0维张量，不在这里强制 GPU 同步
    # (第二轮提速，见文件头)；默认 False，跟原来一样返回 Python float
    parts = {"l_a": l_a.detach(), "l_b": l_b.detach(), "l_c": l_c.detach(),
             "l_sign": l_sign.detach(), "total": total.detach(), "l_bd": l_bd.detach()}
    if not return_tensors:
        parts = {k: float(v) for k, v in parts.items()}
    return total, parts


class SiameseHeadsModel(nn.Module):
    """孪生结构(WT/D两次前向共用cis分支+layout/fusion权重) + 三预测头(A/B/C)。
    对应status文件第4节Head A/B/C定义 + "孪生差分ŷ^FC=ŷ_D-ŷ_WT+ψ_corr(D,L_g)"。
    没有标准答案、按合理默认实现的地方逐条信心指数见文件头docstring，这里不重复。
    """

    def __init__(self, n_tf: int, vocab_size: int, d_model: int = 256,
                n_heads: int = 8, cis_layers: int = 6, lay_layers: int = 4,
                dropout: float = 0.1, pad_token_id: int = 0, c0_init: float = -2.0,
                head_hidden: int = None, head_c_mode: str = "delta", wt_input: str = "none"):
        super().__init__()
        if head_c_mode not in ("delta", "delta_tf", "delta_tf_gene"):
            raise ValueError(f"head_c_mode 只能是 delta/delta_tf/delta_tf_gene，收到 {head_c_mode}")
        if wt_input not in ("none", "head_bc"):  # 2026-10-05 第4批
            raise ValueError(f"wt_input 只能是 none/head_bc，收到 {wt_input}")
        self.head_c_mode = head_c_mode
        self.wt_input = wt_input
        head_hidden = head_hidden or d_model
        self.cis = CisTransformer(vocab_size, d_model, n_heads, cis_layers,
                                  dropout, pad_token_id)
        self.layout = LayoutTransformer(n_tf, d_model, n_heads, lay_layers, dropout)
        self.fusion = CrossModalFusion(n_tf, d_model, n_heads, dropout, c0_init)

        # ψ_corr(D,L_g)里"D"的表征：独立的TF embedding表，不复用layout分支里已有的
        # TF embedding(LayoutTokenEmbedding.e_tf / PairwiseDistanceBias.U,V)——三个
        # 表各自承担不同角色(token本身的身份 / 距离偏置的低秩因子 / 耗竭修正项)，
        # 没有必须共享的理由，分开更灵活，信心7/10(工程选择，不是确认过的硬性要求)。
        self.tf_embed_corr = nn.Embedding(n_tf, d_model)
        nn.init.normal_(self.tf_embed_corr.weight, std=0.02)

        self.f_reg = nn.Sequential(nn.Linear(d_model, head_hidden), nn.GELU(),
                                   nn.Linear(head_hidden, 1))
        self.psi_corr = nn.Sequential(nn.Linear(2 * d_model, head_hidden), nn.GELU(),
                                      nn.Linear(head_hidden, 1))
        # 零初始化ψ_corr最后一层，训练初期ψ_corr≈0，让ŷ^FC先退化成最简单的
        # ŷ_D-ŷ_WT，跟14号脚本ConditionFiLM"训练初期≈恒等变换"是同一个思路，
        # 信心7/10(常见惯例，没有专门针对这个任务验证过是不是真的更好训)。
        nn.init.zeros_(self.psi_corr[-1].weight)
        nn.init.zeros_(self.psi_corr[-1].bias)
        self.psi_c = nn.Sequential(nn.Linear(d_model, head_hidden), nn.GELU(),
                                   nn.Linear(head_hidden, 3))
        # 第2批(见文件头第2条)：新模块一律放在最后创建，head_c_mode="delta" 时一个都不建，
        # 其余参数的随机初始化跟原来完全一样；零初始化 -> 第0步输出跟 "delta" 逐位相同
        if head_c_mode in ("delta_tf", "delta_tf_gene"):
            self.tf_bias_c = nn.Embedding(n_tf, 3)
            nn.init.zeros_(self.tf_bias_c.weight)
        if head_c_mode == "delta_tf_gene":
            self.psi_c_gene = nn.Sequential(nn.Linear(d_model, head_hidden), nn.GELU(),
                                            nn.Linear(head_hidden, 3))
            nn.init.zeros_(self.psi_c_gene[-1].weight)
            nn.init.zeros_(self.psi_c_gene[-1].bias)
        # 第4批(2026-10-05)：实测 WT 表达只进 Head B/C。新模块一律放在最后创建，wt_input="none" 时一个都不建，
        # 其余参数的随机初始化跟原来完全一样；两个加项的最后一层零初始化 -> 第0步输出跟 "none" 逐位相同
        if wt_input != "none":
            self.expr_enc = nn.Sequential(nn.Linear(2, head_hidden), nn.GELU())  # [e, 缺失标记] -> h_e
            self.tf_embed_e = nn.Embedding(n_tf, d_model)
            nn.init.normal_(self.tf_embed_e.weight, std=0.02)
            self.psi_corr_e = nn.Sequential(nn.Linear(2 * d_model + head_hidden, head_hidden), nn.GELU(),
                                            nn.Linear(head_hidden, 1))
            self.psi_c_e = nn.Sequential(nn.Linear(3 * d_model + head_hidden, head_hidden), nn.GELU(),
                                         nn.Linear(head_hidden, 3))
            for _m in (self.psi_corr_e, self.psi_c_e):
                nn.init.zeros_(_m[-1].weight)
                nn.init.zeros_(_m[-1].bias)

    def _heads(self, z_wt: torch.Tensor, z_d: torch.Tensor,
               dep_tf_idx: torch.Tensor, wt_expr: torch.Tensor = None) -> dict:
        """三个预测头，逐样本输入 z_wt/z_d (B,d_model)。固定在 float32 里算(见文件头
        【第二轮提速】：ŷ_D−ŷ_WT 和 z_D−z_WT 都是相近数相减，bf16 会丢精度)。"""
        with torch.autocast(device_type=z_wt.device.type, enabled=False):
            z_wt, z_d = z_wt.float(), z_d.float()
            y_hat_wt = self.f_reg(z_wt).squeeze(-1)  # ŷ_WT
            y_hat_d = self.f_reg(z_d).squeeze(-1)    # ŷ_D
            y_a_pred = y_hat_wt                      # Head A = 基线WT分支预测

            corr_in = torch.cat([self.tf_embed_corr(dep_tf_idx).float(), z_wt], dim=-1)
            psi_corr = self.psi_corr(corr_in).squeeze(-1)
            y_b_pred = y_hat_d - y_hat_wt + psi_corr  # ŷ^FC = ŷ_D-ŷ_WT+ψ_corr(D,L_g)

            logits_c = self.psi_c(z_d - z_wt)         # Head C: ψ_C(Δz)
            if self.head_c_mode != "delta":           # 第2批：+b_C[D](+ψ_Cg(z_wt))
                logits_c = logits_c + self.tf_bias_c(dep_tf_idx).float()
                if self.head_c_mode == "delta_tf_gene":
                    logits_c = logits_c + self.psi_c_gene(z_wt)
            if self.wt_input != "none":  # 2026-10-05 第4批：实测 WT 表达只进 B/C，Head A(y_a_pred)不受影响
                if wt_expr is None:
                    raise ValueError("wt_input!='none' 但没有传 wt_expr(16 号 _forward_batch 会从 batch['y_a'] 传)；"
                                     "不允许悄悄退化成 wt_input='none'")
                e = wt_expr.float()
                miss = torch.isnan(e).float()
                e = torch.nan_to_num(e, nan=0.0).clamp(-6.0, 6.0)
                h_e = self.expr_enc(torch.stack([e, miss], dim=-1))
                t_e = self.tf_embed_e(dep_tf_idx).float()
                y_b_pred = y_b_pred + self.psi_corr_e(torch.cat([t_e, z_wt, h_e], dim=-1)).squeeze(-1)
                logits_c = logits_c + self.psi_c_e(torch.cat([t_e, z_wt, z_d - z_wt, h_e], dim=-1))
        return dict(y_a_pred=y_a_pred, y_b_pred=y_b_pred, logits_c=logits_c)

    def forward(self, layout_wt: dict, layout_d: dict, cis_ids: torch.Tensor,
               ctx_wt: torch.Tensor, ctx_d: torch.Tensor,
               dep_tf_idx: torch.Tensor, wt_expr: torch.Tensor = None) -> dict:
        """逐样本版前向(原写法，16号脚本 --forward legacy 用它；也是 forward_grouped
        的对照基准)。layout_wt/layout_d: 09_torch_dataset.py collate_fn产出的字典，键
        (tf_idx,pos,strand,a,m,res,mask)刚好跟LayoutTransformer.forward的参数名
        对上，可以直接**展开传。cis_ids/ctx_wt/ctx_d同样直接是09脚本collate_fn的
        输出。dep_tf_idx: (B,) long，每条样本被耗竭的TF在tf2idx里的下标——09脚本
        batch里的tf_depleted是字符串列表，要在训练循环里查一次
        ds.tf2idx[t]转成这个张量，这个模块本身不管字符串查表。"""
        h_cis, cis_pad_mask = self.cis(cis_ids)  # WT/D共用同一份cis表征
        h_lay_wt, lay_pad_mask_wt = self.layout(**layout_wt)
        h_lay_d, lay_pad_mask_d = self.layout(**layout_d)

        z_wt, aux_wt = self.fusion(h_cis, cis_pad_mask, h_lay_wt, lay_pad_mask_wt, ctx_wt)
        z_d, aux_d = self.fusion(h_cis, cis_pad_mask, h_lay_d, lay_pad_mask_d, ctx_d)

        out = self._heads(z_wt, z_d, dep_tf_idx, wt_expr)
        out.update(z_wt=z_wt, z_d=z_d, aux_wt=aux_wt, aux_d=aux_d)
        return out

    def forward_grouped(self, cis_ids: torch.Tensor, layout_wt: dict, layout_d: dict,
                        sample_gene: torch.Tensor, d_rows: torch.Tensor,
                        ctx_wt: torch.Tensor, ctx_d: torch.Tensor,
                        dep_tf_idx: torch.Tensor, layout_wt_splits=None,
                        layout_d_splits=None, wt_expr: torch.Tensor = None) -> dict:
        """分组版前向(第二轮提速，见文件头)，输入是 09 号脚本 group_collated 的输出：
          cis_ids (G,Lc) / layout_wt (G,N) / ctx_wt (G,n_tf)：每个基因一份
          sample_gene (B,)：样本->基因；layout_d (B_d,N_d) + d_rows (B_d,)：只含
          D∈L_g 的样本；ctx_d (B,n_tf)、dep_tf_idx (B,)：逐样本。
        输出跟 forward 一样是逐样本的 y_a_pred/y_b_pred/logits_c (B,…)，下游 loss/
        评估代码不用改。eval 模式下跟 forward 逐元素一致(见 run_self_test)。
        layout_wt_splits/layout_d_splits(第三轮提速)：09 号 group_collated 给的按长度
        分段，None 时不分段；分不分段结果相同，只影响速度。"""
        # ---- 每个基因只算一次：cis、WT layout、WT 融合 ----
        h_cis, cis_mask = self.cis(cis_ids)                       # (G,Lc,d)
        h_lay_wt, lay_mask_wt = self.layout.forward_chunked(layout_wt_splits,
                                                            **layout_wt)  # (G,N+1,d)
        q_lin_g, q_mean_g = self.fusion.query(h_cis, cis_mask)
        attn_wt_g, k_mean_wt_g = self.fusion.attend(q_lin_g, h_lay_wt, lay_mask_wt)
        c_wt_g = self.fusion.gate(q_mean_g, k_mean_wt_g)
        z_wt_g, _ = self.fusion.film_pool(q_lin_g, attn_wt_g, c_wt_g, cis_mask, ctx_wt)

        # ---- D 侧：默认复用 WT 的交叉注意力输出/门控(L_g\D=L_g 时两者完全相同)，
        # 只有 d_rows(D∈L_g)重新跑 layout + 交叉注意力，再用 index_copy 覆盖回去 ----
        attn_d = attn_wt_g.index_select(0, sample_gene)          # (B,Lc,d)
        c_d = c_wt_g.index_select(0, sample_gene)                # (B,)
        if d_rows.numel() > 0:  # numel 是元数据，不触发 GPU 同步
            h_lay_d, lay_mask_d = self.layout.forward_chunked(layout_d_splits,
                                                              **layout_d)  # (B_d,N_d+1,d)
            g_of_d = sample_gene.index_select(0, d_rows)
            attn_dd, k_mean_dd = self.fusion.attend(q_lin_g.index_select(0, g_of_d),
                                                    h_lay_d, lay_mask_d)
            c_dd = self.fusion.gate(q_mean_g.index_select(0, g_of_d), k_mean_dd)
            attn_d = attn_d.index_copy(0, d_rows, attn_dd.to(attn_d.dtype))
            c_d = c_d.index_copy(0, d_rows, c_dd.to(c_d.dtype))
        z_d, _ = self.fusion.film_pool(q_lin_g.index_select(0, sample_gene), attn_d, c_d,
                                       cis_mask.index_select(0, sample_gene), ctx_d)

        z_wt = z_wt_g.index_select(0, sample_gene)               # (B,d)
        out = self._heads(z_wt, z_d, dep_tf_idx, wt_expr)
        out.update(z_wt=z_wt, z_d=z_d)
        return out


def _make_dummy_layout(batch: int, n_tf: int, max_len: int, shrink: bool = False,
                       seed: int = 0) -> dict:
    """造一批随机L_g用于自检，不追求跟真实敲除逻辑一致，只为了跑出合理的shape。
    shrink=True时把长度整体压低0或1个token，粗略模拟"敲除后变短"。"""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, max_len + 1, (batch,), generator=g)
    if shrink:
        lengths = (lengths - torch.randint(0, 2, (batch,), generator=g)).clamp(min=0)
    N = int(lengths.max().clamp(min=1))
    tf_idx = torch.randint(0, n_tf, (batch, N), generator=g)
    pos = torch.randn(batch, N, generator=g) * 500
    strand = torch.randint(-1, 2, (batch, N), generator=g).float()
    a = torch.randn(batch, N, generator=g)
    m = torch.randn(batch, N, generator=g)
    res = torch.randint(0, 2, (batch, N), generator=g)
    mask = torch.arange(N).unsqueeze(0) < lengths.unsqueeze(1)
    return dict(tf_idx=tf_idx, pos=pos, strand=strand, a=a, m=m, res=res, mask=mask)


def run_self_test(n_tf=12, vocab_size=50, d_model=32, n_heads=4, cis_layers=2,
                  lay_layers=2, batch=6, n_cis=40, n_lay_max=8, seed=0):
    """随机小规模数据跑一次完整前向+反向，检查形状、梯度、以及几个边界情况/方向性。
    这是唯一的脚本入口，正式训练时改成从09_torch_dataset.py的DataLoader接数据、
    接上真实10/12/14号脚本的模块即可(这个脚本本身已经在用真实的10/12/14号脚本，
    只是这里喂的是随机数据)。"""
    # 【2026-09-23e新增回归检查】这次真实torch.compile报错的根因是_load_module
    # 没把动态加载的模块注册进sys.modules(见文件头说明)。这条检查本身不需要GPU/
    # torch.compile，纯CPU上self_test就能跑，本该在写_load_module那天就加上；
    # 只验证"注册"这一步是通的，不代表补完这个洞后compile一定能一路trace到底，
    # 信心10/10(检查的就是这次报错栈里确切失败的那一步)。
    for _name, _mod in (("_tflayout_12_cis_transformer", _cis_mod),
                        ("_tflayout_10_layout_transformer", _lay_mod),
                        ("_tflayout_14_condition_fusion", _fusion_mod)):
        assert importlib.import_module(_name) is _mod, (
            f"{_name} 没有正确注册进 sys.modules——torch.compile(TorchDynamo)追踪"
            "到这个模块里定义的函数、要解析其全局变量时就是走这条路径去反查模块，"
            "查不到会报 ModuleNotFoundError，这正是2026-09-23e修的那个真实bug")
    print("动态加载的10/12/14号模块均已正确注册到 sys.modules"
         "（torch.compile 解析模块内全局变量依赖这条路径，见文件头2026-09-23e）。")

    torch.manual_seed(seed)

    layout_wt = _make_dummy_layout(batch, n_tf, n_lay_max, shrink=False, seed=seed)
    layout_d = _make_dummy_layout(batch, n_tf, n_lay_max, shrink=True, seed=seed + 1)

    cis_lengths = torch.randint(n_cis // 2, n_cis + 1, (batch,))
    cis_ids = torch.randint(1, vocab_size, (batch, n_cis))
    for b in range(batch):
        cis_ids[b, cis_lengths[b]:] = 0  # pad_token_id=0

    ctx_wt = torch.zeros(batch, n_tf)  # 09脚本里ctx_wt恒为0(见09脚本_build_ctx说明)
    ctx_d = torch.randn(batch, n_tf)
    dep_tf_idx = torch.randint(0, n_tf, (batch,))

    y_a = torch.randn(batch)
    y_a[0] = float("nan")  # 模拟个别基因Head A缺失(09脚本head_a文件缺失时的占位)
    y_b = torch.full((batch,), float("nan"))
    n_sig = max(1, batch // 3)
    y_b[:n_sig] = torch.randn(n_sig)  # 模拟"大部分ns(NaN)、少数显著"
    y_c = torch.randint(0, 3, (batch,))

    model = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers)
    outputs = model(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)

    print(f"y_a_pred 形状: {tuple(outputs['y_a_pred'].shape)}  (应为 batch={batch},)")
    print(f"y_b_pred 形状: {tuple(outputs['y_b_pred'].shape)}")
    print(f"logits_c 形状: {tuple(outputs['logits_c'].shape)}  (应为 batch={batch}, 3)")
    assert outputs["y_a_pred"].shape == (batch,), "y_a_pred 形状不对"
    assert outputs["y_b_pred"].shape == (batch,), "y_b_pred 形状不对"
    assert outputs["logits_c"].shape == (batch, 3), "logits_c 形状不对"
    assert not torch.isnan(outputs["y_a_pred"]).any(), "y_a_pred 里有 NaN"
    assert not torch.isnan(outputs["y_b_pred"]).any(), "y_b_pred 里有 NaN"
    assert not torch.isnan(outputs["logits_c"]).any(), "logits_c 里有 NaN"

    class_counts = torch.tensor([DEFAULT_CLASS_COUNTS_APPROX["down"],
                                 DEFAULT_CLASS_COUNTS_APPROX["ns"],
                                 DEFAULT_CLASS_COUNTS_APPROX["up"]])
    total, parts = compute_total_loss(outputs, y_a, y_b, y_c, class_counts)
    print(f"loss: {parts}")
    assert math.isfinite(parts["total"]), "总loss不是有限值"

    model.zero_grad()
    total.backward()
    bad_grad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad_grad, f"以下参数梯度出现NaN/inf: {bad_grad}"
    print("反向传播梯度检查通过：没有NaN/inf梯度。")

    # ---- 边界情况：整个batch的y_b全是NaN(小batch下Head B/C常见的情况，1.txt里
    # batch_size=8时实测发生过：8条样本96.27%都是ns的概率≈74%) ----
    y_b_all_nan = torch.full((batch,), float("nan"))
    _, parts_allnan = compute_total_loss(outputs, y_a, y_b_all_nan, y_c, class_counts)
    assert parts_allnan["l_b"] == 0.0, "y_b全NaN时l_b应该是0，不是NaN"
    assert parts_allnan["l_sign"] == 0.0, "y_b全NaN时l_sign(用同一个mask)也应该是0"
    assert math.isfinite(parts_allnan["total"])
    print(f"边界情况(整batch y_b全NaN)：l_b={parts_allnan['l_b']}, "
         f"l_sign={parts_allnan['l_sign']}，total仍是有限值，符合预期。")

    # ---- sign consistency 单独校验(不依赖模型，手造数字)：方向一致时loss应该
    # 明显小于方向相反时 ----
    yb_hand = torch.tensor([2.0, -2.0])
    mask_hand = torch.tensor([True, True])
    logits_aligned = torch.tensor([[-5.0, -5.0, 5.0], [5.0, -5.0, -5.0]])   # 样本0→up,样本1→down
    logits_opposite = torch.tensor([[5.0, -5.0, -5.0], [-5.0, -5.0, 5.0]])  # 反过来
    loss_aligned = sign_consistency_loss(yb_hand, mask_hand, logits_aligned)
    loss_opposite = sign_consistency_loss(yb_hand, mask_hand, logits_opposite)
    print(f"sign consistency 校验：方向一致loss={loss_aligned:.4f}，"
         f"方向相反loss={loss_opposite:.4f}（前者应该明显更小）")
    assert loss_aligned < loss_opposite, "sign consistency loss的方向性不对"

    # ---- class-balanced权重单独校验：极端不均衡时，少数类权重应该明显更大 ----
    extreme_counts = torch.tensor([10.0, 10000.0, 8.0])  # down很少/ns很多/up最少
    eff = 1.0 - torch.pow(torch.tensor(0.999), extreme_counts)
    w = (1.0 - 0.999) / eff.clamp(min=1e-8)
    w = w / w.sum() * 3
    print(f"class-balanced权重校验(down/ns/up计数=10/10000/8)：权重={w.tolist()}")
    assert w[0] > w[1] and w[2] > w[1], "样本少的类别权重应该比样本多的类别权重大"

    # ---- 第三轮提速新增：去同步后的 loss vs 原写法(布尔索引/if 分支)，逐项一致 ----
    def _head_a_loss_old(yp, yt, mse_weight=0.5):
        msk = ~torch.isnan(yt)
        if msk.sum() == 0:
            return torch.zeros(())
        p_, t_ = yp[msk], yt[msk]
        mse_ = F.mse_loss(p_, t_)
        r_ = _pearson_r(p_, t_) if msk.sum() >= 2 else torch.zeros(())
        return mse_weight * mse_ + (1 - mse_weight) * (1 - r_)

    g_l = torch.Generator().manual_seed(seed + 11)
    for n_valid in (0, 1, 2, 5, 16):
        yp = torch.randn(16, generator=g_l)
        yt = torch.full((16,), float("nan"))
        yt[torch.randperm(16, generator=g_l)[:n_valid]] = torch.randn(n_valid, generator=g_l)
        new_v, old_v = float(head_a_loss(yp, yt)), float(_head_a_loss_old(yp, yt))
        assert abs(new_v - old_v) < 1e-5, f"head_a_loss 新旧写法不一致(n_valid={n_valid})"
        yp_g = yp.clone().requires_grad_(True)
        head_a_loss(yp_g, yt).backward()
        assert torch.isfinite(yp_g.grad).all(), f"head_a_loss 梯度出现 NaN(n_valid={n_valid})"
    cc = torch.tensor([10.0, 10000.0, 8.0])
    lg_ = torch.randn(16, 3, generator=g_l)
    tg_ = torch.randint(0, 3, (16,), generator=g_l)
    eff_old = 1.0 - torch.pow(torch.as_tensor(0.999, dtype=torch.float32), cc)
    w_old = (1.0 - 0.999) / eff_old.clamp(min=1e-8)
    w_old = w_old / w_old.sum() * 3
    lp_ = F.log_softmax(lg_, dim=-1)
    pt_ = lp_.exp().gather(-1, tg_[:, None]).squeeze(-1)
    focal_old = (w_old[tg_] * (1 - pt_).clamp(min=0) ** 2.0
                 * -lp_.gather(-1, tg_[:, None]).squeeze(-1)).mean()
    assert abs(float(class_balanced_focal_loss(lg_, tg_, cc)) - float(focal_old)) < 1e-5, \
        "class_balanced_focal_loss 新旧写法不一致"
    assert float(sign_consistency_loss(torch.randn(16), torch.zeros(16, dtype=torch.bool),
                                       lg_)) == 0.0, "mask 全 False 时 sign loss 应为0"
    print("去同步后的 loss 跟原写法一致(head_a_loss 有效点 0/1/2/5/16 五种情况、focal loss、"
          "sign loss 空 mask)。")

    # ---- 第二轮提速新增：分组前向 forward_grouped vs 逐样本前向 forward ----
    # 造一批"数据形状跟真实一致"的样本：同一基因的多条样本共享 cis/L_g，layout_d
    # 严格等于 L_g 去掉 D 的 token；D 既有在 L_g 里的、也有不在的；故意放一个 L_g
    # 为空的基因和一个"L_g 只含 D"的样本(去掉后变空)。用 09 号脚本真实的
    # collate_fn + group_collated 组 batch，两条前向在 eval 模式下应该逐元素一致。
    import types
    ds_mod = _load_module("_tflayout_09_torch_dataset", "09_torch_dataset.py")
    TFLayoutDataset = ds_mod.TFLayoutDataset
    rng = torch.Generator().manual_seed(seed + 7)
    n_genes = 5
    tf_names = [f"TF{i}" for i in range(n_tf)]
    tf2idx = {t: i for i, t in enumerate(tf_names)}
    items = []
    for gi in range(n_genes):
        n_sites = 0 if gi == 0 else int(torch.randint(1, n_lay_max + 1, (1,), generator=rng))
        lg = dict(
            tf_idx=torch.randint(0, n_tf, (n_sites,), generator=rng).numpy().astype("int64"),
            pos=(torch.randn(n_sites, generator=rng) * 500).numpy().astype("float32"),
            strand=torch.randint(-1, 2, (n_sites,), generator=rng).numpy().astype("float32"),
            a=torch.randn(n_sites, generator=rng).numpy().astype("float32"),
            m=torch.randn(n_sites, generator=rng).numpy().astype("float32"),
            res=torch.randint(0, 2, (n_sites,), generator=rng).numpy().astype("float32"))
        if gi == 1:  # 这个基因的 L_g 全是同一个TF，耗竭它之后 L_g\D 变空
            lg["tf_idx"][:] = 3
        cis_len = int(torch.randint(n_cis // 2, n_cis + 1, (1,), generator=rng))
        cis = torch.randint(2, vocab_size, (cis_len,), generator=rng).numpy().astype("int64")
        in_lg = sorted(set(lg["tf_idx"].tolist()))
        deps = in_lg[:2] + [t for t in range(n_tf) if t not in in_lg][:3]
        for d in deps:
            keep = lg["tf_idx"] != d
            items.append(dict(
                gene_id=f"G{gi}", tf_depleted=tf_names[d], layout_wt=lg,
                layout_d={k: v[keep] for k, v in lg.items()}, cis_ids=cis,
                ctx_wt=torch.zeros(n_tf), ctx_d=torch.randn(n_tf, generator=rng),
                y_a=torch.randn((), generator=rng), y_b=torch.tensor(float("nan")),
                y_c=torch.tensor(1)))
    perm = torch.randperm(len(items), generator=rng).tolist()
    items = [items[i] for i in perm]  # 打乱，让同一基因的样本在 batch 里不相邻
    fake_ds = types.SimpleNamespace(pad_id=0)  # collate_fn 只用到 self.pad_id
    legacy = TFLayoutDataset.collate_fn(fake_ds, items)
    grouped = TFLayoutDataset.group_collated(legacy, tf2idx, 0)
    dep_idx = torch.tensor([tf2idx[t] for t in legacy["tf_depleted"]])
    print(f"分组检查用的batch：{len(items)} 条样本 -> {grouped['cis_ids'].shape[0]} 个基因，"
          f"其中 D∈L_g(需要单独算D侧layout)的 {grouped['d_rows'].numel()} 条")
    assert grouped["cis_ids"].shape[0] == n_genes
    assert torch.equal(grouped["dep_tf_idx"], dep_idx)

    # ConditionFiLM 和 ψ_corr 的最后一层是零初始化(训练初期≈恒等/≈0)，不扰动的话
    # ctx_d 对 z_d 完全没影响，下面的对比就测不出"复用路径有没有正确用上 ctx_d"
    with torch.no_grad():
        for lin in (model.fusion.condition.net[-1], model.psi_corr[-1]):
            lin.weight.normal_(0.0, 0.1)
            lin.bias.normal_(0.0, 0.1)
    model.eval()
    with torch.no_grad():
        out_l = model(legacy["layout_wt"], legacy["layout_d"], legacy["cis_ids"],
                      legacy["ctx_wt"], legacy["ctx_d"], dep_idx)
        out_g = model.forward_grouped(grouped["cis_ids"], grouped["layout_wt"],
                                      grouped["layout_d"], grouped["sample_gene"],
                                      grouped["d_rows"], grouped["ctx_wt"],
                                      grouped["ctx_d"], grouped["dep_tf_idx"])
    for k in ("y_a_pred", "y_b_pred", "logits_c", "z_wt", "z_d"):
        dmax = float((out_l[k] - out_g[k]).abs().max())
        print(f"  分组 vs 逐样本 {k}: 最大绝对差 {dmax:.2e}")
        assert dmax < 1e-4, f"分组前向的 {k} 跟逐样本前向不一致"
    # 第三轮提速：同一批样本按 L_g 长度分段(layout_buckets=3，且用 chunk_penalty=0 强制
    # 真的切成多段)，结果也应该跟逐样本前向一致
    grouped_b = TFLayoutDataset.group_collated(legacy, tf2idx, 0, layout_buckets=3)
    lens_sorted = grouped_b["layout_wt"]["mask"].sum(1).tolist()
    grouped_b["layout_wt_splits"] = ds_mod._length_buckets(lens_sorted, 3, chunk_penalty=0)
    lens_d_sorted = grouped_b["layout_d"]["mask"].sum(1).tolist()
    grouped_b["layout_d_splits"] = ds_mod._length_buckets(lens_d_sorted, 3, chunk_penalty=0)
    assert lens_sorted == sorted(lens_sorted), "layout_buckets>1 时基因应按 L_g 长度升序"
    with torch.no_grad():
        out_b = model.forward_grouped(grouped_b["cis_ids"], grouped_b["layout_wt"],
                                      grouped_b["layout_d"], grouped_b["sample_gene"],
                                      grouped_b["d_rows"], grouped_b["ctx_wt"],
                                      grouped_b["ctx_d"], grouped_b["dep_tf_idx"],
                                      grouped_b["layout_wt_splits"],
                                      grouped_b["layout_d_splits"])
    for k in ("y_a_pred", "y_b_pred", "logits_c", "z_wt", "z_d"):
        dmax = float((out_l[k] - out_b[k]).abs().max())
        assert dmax < 1e-4, f"分段分组前向的 {k} 跟逐样本前向不一致({dmax})"
    print(f"  分段分组前向(WT 分段 {grouped_b['layout_wt_splits']}，D 分段 "
          f"{grouped_b['layout_d_splits']}) vs 逐样本：一致 ✓")
    # Head C 最依赖 Δz 的精度：单独确认 L_g\D=L_g 的样本 Δz 不是恰好为0(说明 ctx_d
    # 确实通过 FiLM 起作用了，而不是复用路径把 D 侧整个退化成了 WT)
    not_d = torch.ones(len(items), dtype=torch.bool)
    not_d[grouped["d_rows"]] = False
    dz = (out_g["z_d"] - out_g["z_wt"])[not_d].abs().max()
    assert float(dz) > 0, "D∉L_g 的样本 z_d 跟 z_wt 完全相同——ctx_d 的 FiLM 没生效"

    model.train()
    out_t = model.forward_grouped(grouped["cis_ids"], grouped["layout_wt"],
                                  grouped["layout_d"], grouped["sample_gene"],
                                  grouped["d_rows"], grouped["ctx_wt"],
                                  grouped["ctx_d"], grouped["dep_tf_idx"])
    y_b_g = torch.full((len(items),), float("nan"))
    y_b_g[:3] = torch.randn(3)
    y_c_g = torch.randint(0, 3, (len(items),))
    tot_g, parts_g = compute_total_loss(out_t, grouped["y_a"], y_b_g, y_c_g, class_counts,
                                        return_tensors=True)
    assert all(torch.is_tensor(v) for v in parts_g.values()), "return_tensors=True 应返回张量"
    model.zero_grad()
    tot_g.backward()
    bad_grad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad_grad, f"分组前向反传后以下参数梯度出现NaN/inf: {bad_grad}"
    print("分组前向检查通过：eval 下跟逐样本前向逐元素一致，训练模式反传梯度正常。")

    # ---- 第2批新增①：Huber 版 Head B 损失 ----
    g_h = torch.Generator().manual_seed(seed + 21)
    pr = torch.randn(64, generator=g_h) * 3
    tg = torch.randn(64, generator=g_h) * 3
    tg[::3] = float("nan")
    for dl in (0.5, 1.0, 2.0):
        ref = 2.0 * F.huber_loss(pr[~torch.isnan(tg)], tg[~torch.isnan(tg)], delta=dl)
        got = masked_huber(pr, tg, delta=dl)
        assert abs(float(ref) - float(got)) < 1e-5, f"masked_huber 跟 2×F.huber_loss 不一致(δ={dl})"
    small_p, small_t = torch.randn(20, generator=g_h) * 0.3, torch.randn(20, generator=g_h) * 0.3
    assert abs(float(masked_huber(small_p, small_t, delta=5.0))
               - float(masked_mse(small_p, small_t))) < 1e-6, "误差都 ≤δ 时 Huber 应等于 MSE"
    assert float(masked_huber(pr, torch.full_like(tg, float("nan")))) == 0.0, "全NaN应为0"
    pg = pr.clone().requires_grad_(True)
    masked_huber(pg, tg, delta=1.0).backward()
    n_valid_h = int((~torch.isnan(tg)).sum())
    assert float(pg.grad.abs().max()) <= 2.0 / n_valid_h + 1e-6, "Huber 单样本梯度应被限制在 2δ/n"
    _, parts_h = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, loss_b="huber")
    _, parts_m = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, loss_b="mse")
    assert parts_h["l_b"] <= parts_m["l_b"] + 1e-6, "Huber(2×)不应大于 MSE"
    _, parts_la = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, lambda_a=0.0)
    assert abs(parts_la["total"] - (parts_m["total"] - parts_m["l_a"])) < 1e-5, "lambda_a 没生效"
    print(f"第2批 Huber：跟 2×F.huber_loss 一致(δ=0.5/1/2)、小误差时=MSE、梯度上限 2δ/n、"
          f"lambda_a 生效 ✓ (本 batch l_b: huber={parts_h['l_b']:.4f} ≤ mse={parts_m['l_b']:.4f})")

    # ---- 第2批新增②：head_c_mode 三种模式 ----
    torch.manual_seed(seed + 5)
    m_ref = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers)
    torch.manual_seed(seed + 5)
    m_tfg = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers,
                              head_c_mode="delta_tf_gene")
    sd_ref = m_ref.state_dict()
    extra = sorted(set(m_tfg.state_dict()) - set(sd_ref))
    assert all(k.startswith(("tf_bias_c", "psi_c_gene")) for k in extra), extra
    assert all(torch.equal(sd_ref[k], m_tfg.state_dict()[k]) for k in sd_ref), \
        "新模式下原有参数的初始化应该跟 delta 模式完全相同(新模块放最后创建)"
    m_ref.eval(); m_tfg.eval()
    with torch.no_grad():
        o_ref = m_ref(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)
        o_tfg = m_tfg(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)
    assert torch.equal(o_ref["logits_c"], o_tfg["logits_c"]), "零初始化时新模式应跟 delta 逐位相同"
    m_legacy_load = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers)
    m_legacy_load.load_state_dict(sd_ref, strict=True)  # 旧 checkpoint(delta)照旧 strict 加载
    for mode in ("delta_tf", "delta_tf_gene"):
        mm = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers,
                               head_c_mode=mode)
        with torch.no_grad():
            mm.tf_bias_c.weight.normal_(0.0, 1.0)
            if mode == "delta_tf_gene":
                mm.psi_c_gene[-1].weight.normal_(0.0, 0.1)
        out_m = mm(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)
        tot_m, _ = compute_total_loss(out_m, y_a, y_b, y_c, class_counts)
        mm.zero_grad()
        tot_m.backward()
        assert mm.tf_bias_c.weight.grad is not None and \
            torch.isfinite(mm.tf_bias_c.weight.grad).all(), f"{mode}: b_C 没有梯度"
        g_rows = mm.tf_bias_c.weight.grad.abs().sum(1) > 0
        assert set(torch.nonzero(g_rows).flatten().tolist()) <= set(dep_tf_idx.tolist()), \
            f"{mode}: b_C 只有被耗竭的 TF 那几行应该有梯度"
    mm.eval()  # 分组前向 vs 逐样本前向(delta_tf_gene，b_C/ψ_Cg 已随机化)
    with torch.no_grad():
        for lin in (mm.fusion.condition.net[-1], mm.psi_corr[-1]):
            lin.weight.normal_(0.0, 0.1)
        o_l = mm(legacy["layout_wt"], legacy["layout_d"], legacy["cis_ids"], legacy["ctx_wt"],
                 legacy["ctx_d"], dep_idx)
        o_g = mm.forward_grouped(grouped["cis_ids"], grouped["layout_wt"], grouped["layout_d"],
                                 grouped["sample_gene"], grouped["d_rows"], grouped["ctx_wt"],
                                 grouped["ctx_d"], grouped["dep_tf_idx"])
    for k in ("y_a_pred", "y_b_pred", "logits_c"):
        assert float((o_l[k] - o_g[k]).abs().max()) < 1e-4, f"delta_tf_gene 分组 vs 逐样本 {k} 不一致"
    print("第2批 head_c_mode：新增参数只有 tf_bias_c/psi_c_gene、原参数初始化不变、零初始化时"
          "跟 delta 逐位相同、旧 state_dict 可 strict 加载、b_C 梯度只落在被耗竭 TF 的行、"
          "delta_tf_gene 分组前向跟逐样本一致 ✓")

    # ---- 2026-09-25b 第3批：稠密 log2FC 辅助项 l_bd ----
    tot_0, parts_0 = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, loss_b="huber")
    g_d = torch.Generator().manual_seed(seed + 31)
    y_bd = torch.randn(len(y_b), generator=g_d) * 0.2
    y_bd[0] = float("nan")
    tot_m0, parts_m0 = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, loss_b="huber",
                                          y_bd=y_bd, lambda_bd=0.0)
    assert torch.equal(tot_0, tot_m0), "lambda_bd=0 时传了 y_bd 也不应改变 total"
    assert parts_0["l_bd"] == 0.0 and parts_m0["l_bd"] >= 0.0
    ns_m = torch.isnan(y_b) & ~torch.isnan(y_bd)
    ref_bd = masked_huber(outputs["y_b_pred"], y_bd, mask=ns_m, delta=1.0)
    assert abs(float(ref_bd) - parts_m0["l_bd"]) < 1e-6, "dense_on=ns 时 l_bd 应只在不显著且有值的样本上算"
    tot_m1, parts_m1 = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, loss_b="huber",
                                          y_bd=y_bd, lambda_bd=0.7)
    assert abs(parts_m1["total"] - (parts_0["total"] + 0.7 * parts_m1["l_bd"])) < 1e-5, "lambda_bd 没生效"
    _, parts_all = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, loss_b="huber",
                                      y_bd=y_bd, lambda_bd=0.7, dense_on="all")
    ref_all = masked_huber(outputs["y_b_pred"], y_bd, delta=1.0)
    assert abs(float(ref_all) - parts_all["l_bd"]) < 1e-6, "dense_on=all 应在全部有值样本上算"
    try:
        compute_total_loss(outputs, y_a, y_b, y_c, class_counts, lambda_bd=1.0)
        raise AssertionError("lambda_bd>0 且没传 y_bd 应报错")
    except ValueError:
        pass
    y_bd_nan = torch.full_like(y_bd, float("nan"))
    tot_n, parts_n = compute_total_loss(outputs, y_a, y_b, y_c, class_counts, y_bd=y_bd_nan,
                                        lambda_bd=1.0)
    assert parts_n["l_bd"] == 0.0 and math.isfinite(parts_n["total"]), "y_bd 全 NaN 时 l_bd 应为0"
    model.zero_grad()
    out_bd = model(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)
    tot_bd, _ = compute_total_loss(out_bd, y_a, y_b, y_c, class_counts, y_bd=y_bd, lambda_bd=1.0)
    tot_bd.backward()
    assert all(torch.isfinite(p_.grad).all() for p_ in model.parameters() if p_.grad is not None), \
        "加了 l_bd 之后梯度出现 NaN/inf"
    print(f"第3批 稠密辅助项：lambda_bd=0 时 total 逐位不变、dense_on=ns/all 的掩码正确、lambda_bd 生效、"
          f"全 NaN 时为0、缺 y_bd 报错、梯度有限 ✓ (本 batch l_bd={parts_m1['l_bd']:.4f})")

    # ---- 2026-09-28b 第8批：Head C 忽略 y_c<0(c_ignore) ----
    g8 = torch.Generator().manual_seed(seed + 28)
    lg8 = torch.randn(20, 3, generator=g8)
    tg8 = torch.randint(0, 3, (20,), generator=g8)
    cc8 = torch.tensor([12410.0, 563334.0, 10156.0])
    for gm in (0.0, 2.0):
        base8 = class_balanced_focal_loss(lg8, tg8, cc8, gamma=gm)
        assert torch.equal(base8, class_balanced_focal_loss(lg8, tg8, cc8, gamma=gm, ignore_neg=False)), \
            "ignore_neg=False 时应逐位不变"
        assert abs(float(class_balanced_focal_loss(lg8, tg8, cc8, gamma=gm, ignore_neg=True)) - float(base8)) < 1e-6, \
            "全部有效时 ignore_neg=True/False 应数值一致"
        tg_m = tg8.clone()
        tg_m[::3] = -1
        keep = tg_m >= 0
        sub = class_balanced_focal_loss(lg8[keep], tg8[keep], cc8, gamma=gm)
        got = class_balanced_focal_loss(lg8, tg_m, cc8, gamma=gm, ignore_neg=True)
        assert abs(float(got) - float(sub)) < 1e-6, "部分 -1 时应等于只在有效行上算"
        lg_g = lg8.clone().requires_grad_(True)
        class_balanced_focal_loss(lg_g, tg_m, cc8, gamma=gm, ignore_neg=True).backward()
        assert torch.isfinite(lg_g.grad).all() and float(lg_g.grad[~keep].abs().sum()) == 0.0, \
            "被忽略的行梯度应为0、其余有限"
    all_neg = torch.full((20,), -1, dtype=torch.long)
    assert float(class_balanced_focal_loss(lg8, all_neg, cc8, ignore_neg=True)) == 0.0, "全部 -1 时应为0"
    out8 = {"y_a_pred": torch.randn(20, generator=g8), "y_b_pred": torch.randn(20, generator=g8), "logits_c": lg8}
    ya8, yb8 = torch.randn(20, generator=g8), torch.full((20,), float("nan"))
    t0_, p0_ = compute_total_loss(out8, ya8, yb8, tg8, cc8)
    t1_, p1_ = compute_total_loss(out8, ya8, yb8, tg8, cc8, c_ignore=False)
    assert torch.equal(t0_, t1_), "compute_total_loss c_ignore=False 应逐位不变"
    _, p2_ = compute_total_loss(out8, ya8, yb8, tg_m, cc8, c_ignore=True)
    assert abs(p2_["l_a"] - p0_["l_a"]) < 1e-7, "c_ignore 不应改变 Head A 损失"
    print("第8批 c_ignore：默认逐位不变、全部有效时一致、部分 -1 时等于有效行子集、全 -1 为0、被忽略行梯度为0 ✓")

    # ---- 第4批(2026-10-05)：wt_input —— 默认不建新模块且逐位不变；head_bc 零初始化时第0步输出跟 none 一致；
    #      Head A 不受 wt_expr 影响；B/C 受影响；缺失值有限；梯度有限且流进新模块；不传 wt_expr 要报错 ----
    torch.manual_seed(seed + 100)
    m_none = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers)
    torch.manual_seed(seed + 100)
    m_wt = SiameseHeadsModel(n_tf, vocab_size, d_model, n_heads, cis_layers, lay_layers, wt_input="head_bc")
    assert m_none.wt_input == "none" and m_wt.wt_input == "head_bc"
    assert not any(k.split(".")[0] in ("expr_enc", "tf_embed_e", "psi_corr_e", "psi_c_e")
                   for k in m_none.state_dict()), "wt_input=none 不应建任何新模块"
    miss_k, unexp_k = m_wt.load_state_dict(m_none.state_dict(), strict=False)
    assert not unexp_k and miss_k and all(k.split(".")[0] in ("expr_enc", "tf_embed_e", "psi_corr_e", "psi_c_e")
                                          for k in miss_k), (miss_k, unexp_k)
    m_none.eval()
    m_wt.eval()
    wt_in = torch.randn(batch)
    wt_in[0] = float("nan")  # 缺失值
    with torch.no_grad():
        o_n = m_none(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)
        o_w = m_wt(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx, wt_expr=wt_in)
    for k_ in ("y_a_pred", "y_b_pred", "logits_c"):
        assert torch.allclose(o_n[k_], o_w[k_], atol=1e-6), f"零初始化时 wt_input=head_bc 应跟 none 一致: {k_}"
    try:
        m_wt(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx)
    except ValueError:
        pass
    else:
        raise AssertionError("wt_input!=none 却没传 wt_expr 应该报错(防止悄悄退化成 none)")
    for _m in (m_wt.psi_corr_e, m_wt.psi_c_e):  # 解除零初始化，检查 wt_expr 真的在起作用
        nn.init.normal_(_m[-1].weight, std=0.1)
    with torch.no_grad():
        o_w1 = m_wt(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx, wt_expr=wt_in)
        o_w2 = m_wt(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx, wt_expr=wt_in + 1.0)
    assert torch.allclose(o_w1["y_a_pred"], o_w2["y_a_pred"], atol=1e-6), "Head A 不应受 wt_expr 影响"
    assert float((o_w1["y_b_pred"][1:] - o_w2["y_b_pred"][1:]).abs().max()) > 1e-6, "Head B 应随 wt_expr 变化"
    assert float((o_w1["logits_c"][1:] - o_w2["logits_c"][1:]).abs().max()) > 1e-6, "Head C 应随 wt_expr 变化"
    for k_ in ("y_a_pred", "y_b_pred", "logits_c"):
        assert torch.isfinite(o_w1[k_]).all(), f"含缺失 wt_expr 时 {k_} 应有限"
    m_wt.train()
    o_wt = m_wt(layout_wt, layout_d, cis_ids, ctx_wt, ctx_d, dep_tf_idx, wt_expr=wt_in)
    tot_wt, _ = compute_total_loss(o_wt, y_a, y_b, y_c, class_counts)
    m_wt.zero_grad()
    tot_wt.backward()
    for nm_ in ("expr_enc", "tf_embed_e", "psi_corr_e", "psi_c_e"):
        gs_ = [p.grad for n_, p in m_wt.named_parameters() if n_.startswith(nm_ + ".")]
        assert gs_ and all(g_ is not None and torch.isfinite(g_).all() for g_ in gs_), f"{nm_} 梯度应存在且有限"
    assert float(sum(p.grad.abs().sum() for n_, p in m_wt.named_parameters()
                     if n_.startswith("expr_enc."))) > 0, "实测表达编码器应收到非零梯度"
    print("第4批 wt_input：默认不建新模块、零初始化时与 none 逐位一致、Head A 不受影响、B/C 随实测表达变化、"
          "缺失值有限、梯度有限、不传 wt_expr 报错 ✓")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    print("自检通过：前向形状、梯度、loss边界情况、sign consistency方向性、"
         "class-balanced权重方向性、去同步 loss 等价性、分组(含分段)前向等价性、"
         "动态加载模块的 sys.modules 注册都正常。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tf", type=int, default=12)
    ap.add_argument("--vocab-size", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--cis-layers", type=int, default=2)
    ap.add_argument("--lay-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--n-cis", type=int, default=40)
    ap.add_argument("--n-lay-max", type=int, default=8)
    a = ap.parse_args()
    run_self_test(a.n_tf, a.vocab_size, a.d_model, a.n_heads, a.cis_layers,
                  a.lay_layers, a.batch, a.n_cis, a.n_lay_max)
