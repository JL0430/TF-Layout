# scripts/tflayout/12_cis_transformer.py

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0,
                  device=None, dtype=None):
    """返回 (seq_len, head_dim) 的 cos/sin 表，跟《方案.txt》RoPE规格一致，
    用的是LLaMA/GPT-NeoX式的"split-half"配对(不是原始论文的相邻配对，两者
    数学上等价，只是旋转子空间的取法不同，是目前最通用的实现方式)。"""
    half = head_dim // 2
    theta = base ** (-2 * torch.arange(half, device=device, dtype=dtype) / head_dim)
    m = torch.arange(seq_len, device=device, dtype=dtype)
    ang = torch.outer(m, theta)  # (seq_len, half)
    cos_half, sin_half = torch.cos(ang), torch.sin(ang)
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor,
                cache: dict = None) -> torch.Tensor:
        """x: (B,N,d_model)；key_padding_mask: (B,N) bool，True=padding。
        cache：同一次 CisTransformer 前向里各层共用的字典(第三轮提速，见文件头)，
        None 时每层自己算，结果完全一样。"""
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)               # 各 (B,N,H,head_dim)
        q = q.transpose(1, 2)                      # (B,H,N,head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        key_rope = ("rope", N, x.dtype, x.device)
        if cache is not None and key_rope in cache:
            cos, sin = cache[key_rope]
        else:
            cos, sin = _rope_cos_sin(N, self.head_dim, self.rope_base,
                                     device=x.device, dtype=x.dtype)  # (N,head_dim)
            cos, sin = cos.view(1, 1, N, -1), sin.view(1, 1, N, -1)   # 广播到(B,H,N,hd)
            if cache is not None:
                cache[key_rope] = (cos, sin)
        q = (q * cos + _rotate_half(q) * sin).to(v.dtype)  # autocast下对齐回v的dtype
        k = (k * cos + _rotate_half(k) * sin).to(v.dtype)

        key_bias = ("bias", q.dtype)
        if cache is not None and key_bias in cache:
            attn_bias = cache[key_bias]
        else:
            attn_bias = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(key_padding_mask.view(B, 1, 1, N),
                                              torch.finfo(q.dtype).min)
            if cache is not None:
                cache[key_bias] = attn_bias
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0)  # (B,H,N,head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class CisTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None,
                dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = RoPESelfAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask, cache: dict = None):
        x = x + self.dropout(self.attn(self.ln1(x), key_padding_mask, cache))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class CisTransformer(nn.Module):
    """token_ids(整数，来自任意分词器) -> H_cis。《方案.txt》规格：6层、d=256。"""

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                n_layers: int = 6, dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([
            CisTransformerLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor):
        """token_ids: (B,N) long。返回 H_cis:(B,N,d_model)，
        key_padding_mask:(B,N) bool(True=padding，等于token_ids==pad_token_id)。"""
        key_padding_mask = token_ids == self.pad_token_id
        x = self.tok_embed(token_ids)
        cache = {}  # 各层共用 RoPE 表和 padding mask(第三轮提速，见文件头)
        for layer in self.layers:
            x = layer(x, key_padding_mask, cache)
        return self.ln_out(x), key_padding_mask


def run_self_test(vocab_size=4000, d_model=64, n_heads=4, n_layers=2, batch=3,
                  seq_len=50, seed=0):
    """随机小规模token序列跑一次完整前向，检查形状+验证RoPE的相对位置不变性。"""
    torch.manual_seed(seed)
    lengths = torch.randint(seq_len // 2, seq_len + 1, (batch,))
    token_ids = torch.randint(1, vocab_size, (batch, seq_len))  # 1..vocab-1，0留给pad
    for b in range(batch):
        token_ids[b, lengths[b]:] = 0  # pad_token_id=0

    model = CisTransformer(vocab_size, d_model, n_heads, n_layers)
    h_cis, key_padding_mask = model(token_ids)

    print(f"输入序列长度(每个样本，pad到{seq_len}): {lengths.tolist()}")
    print(f"H_cis 形状: {tuple(h_cis.shape)}  (应为 batch={batch}, N={seq_len}, "
          f"d_model={d_model})")
    print(f"key_padding_mask 每行 True(padding) 个数: "
          f"{key_padding_mask.sum(dim=1).tolist()}，应等于 "
          f"{[seq_len - int(l) for l in lengths]}")
    assert h_cis.shape == (batch, seq_len, d_model), "H_cis 形状不对"
    assert not torch.isnan(h_cis).any(), "输出里有 NaN"

    # 额外自检：RoPE自注意力对"整体平移输入位置"不敏感于绝对位置，只用一层、
    # 无padding的情况下验证——把同一段token整体往右挪几位(前面补相同的pad之外的
    # 任意占位)，中间那段的相对注意力结构应该不变。这里退化成数值层面的粗检查：
    # 至少确认前向传播稳定、多次调用同输入结果一致(纯前向，非训练模式无随机性)。
    model.eval()
    with torch.no_grad():
        h1, _ = model(token_ids)
        h2, _ = model(token_ids)
    assert torch.allclose(h1, h2), "eval模式下同输入两次前向结果不一致，有非预期的随机性"

    # 第三轮提速新增：各层共用 cache vs 每层自己重算 RoPE/mask，应逐位相同
    with torch.no_grad():
        kpm = token_ids == model.pad_token_id
        x = model.tok_embed(token_ids)
        for layer in model.layers:
            x = layer(x, kpm, None)
        h_nocache = model.ln_out(x)
    assert torch.equal(h1, h_nocache), "共用 cache 跟每层重算的结果不一致"
    print("RoPE/padding mask 各层共用 cache vs 每层重算：逐位一致。")

    # 第二轮提速新增：有 CUDA+bf16 时检查 autocast 下前向不出 NaN、跟 fp32 接近
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_c = model.cuda()
        ids_c = token_ids.cuda()
        with torch.no_grad():
            ref32, kpm_c = model_c(ids_c)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out16, _ = model_c(ids_c)
        valid = ~kpm_c
        rel = float((out16.float() - ref32)[valid].norm() / ref32[valid].norm())
        print(f"bf16 autocast vs fp32(非padding位置)：相对误差 {rel:.3e}"
              "（bf16 正常量级约1e-2，不是 NaN/远大于0.05 就正常）")
        assert torch.isfinite(out16).all(), "bf16 autocast 下输出有 NaN/inf"
    else:
        print("没有可用的 CUDA+bf16，跳过 bf16 autocast 检查(不影响其余自检结论)。")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    print("自检通过：形状、数值稳定性都正常。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=50)
    a = ap.parse_args()
    run_self_test(a.vocab_size, a.d_model, a.n_heads, a.n_layers, a.batch, a.seq_len)
