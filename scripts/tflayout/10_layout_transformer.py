# scripts/tflayout/10_layout_transformer.py

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _distance_basis(delta_d: torch.Tensor) -> torch.Tensor:
    """φ(Δd)，fig.txt 图2a a1。delta_d: 任意形状 -> 多一维 13 在最后。"""
    d = delta_d
    ad = d.abs()
    const = torch.ones_like(d)
    log_term = torch.log1p(ad)
    sign_term = torch.sign(d)
    two_pi = 2 * math.pi
    helix_cos = torch.cos(two_pi * d / 10.5)
    helix_sin = torch.sin(two_pi * d / 10.5)
    nuc_cos = torch.cos(two_pi * d / 167.0)
    nuc_sin = torch.sin(two_pi * d / 167.0)
    edges = (10.0, 20.0, 40.0, 80.0, 160.0)
    bins = []
    lower = 0.0
    for e in edges:
        bins.append(((ad >= lower) & (ad < e)).to(d.dtype))
        lower = e
    bins.append((ad >= lower).to(d.dtype))  # 最后一档 160+ (fig.txt里标"320+"，泛指远距离)
    return torch.stack([const, log_term, sign_term, helix_cos, helix_sin,
                        nuc_cos, nuc_sin, *bins], dim=-1)  # (..., 13)


class PairwiseDistanceBias(nn.Module):
    """b_h(t_i,t_j,Δd) = ⟨u_h[t_i]⊙v_h[t_j], φ(Δd)⟩，图2a a2 的低秩分解。
    U/V 存成 (n_tf, n_heads*K) 的 Embedding，用 F.embedding 按 tf_idx 查表，
    避免手写高维 gather 索引出错。"""
    K = 13

    def __init__(self, n_tf: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.U = nn.Embedding(n_tf, n_heads * self.K)
        self.V = nn.Embedding(n_tf, n_heads * self.K)
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.V.weight, std=0.02)

    def forward(self, tf_idx: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """tf_idx: (B,N) long；phi: (B,N,N,K)=_distance_basis(Δd)，Δd[b,i,j]=p_j-p_i，
        由 LayoutTransformer.forward 算一次、各层共用(第二轮提速改动2)。
        返回 (B,n_heads,N,N) float32：B_h[i,j] = b_h(t_i,t_j, p_j-p_i)，跟 fig.txt a3
        的方向定义一致(j 相对 i 的偏移)。整段在 float32 里算(改动3)。"""
        B, N = tf_idx.shape
        with torch.autocast(device_type=phi.device.type, enabled=False):
            u = self.U(tf_idx).float().view(B, N, self.n_heads, self.K)  # u[b,i,h,k]
            v = self.V(tf_idx).float().view(B, N, self.n_heads, self.K)  # v[b,j,h,k]
            return torch.einsum("bihk,bjhk,bijk->bhij", u, v, phi.float())


class LayoutTokenEmbedding(nn.Module):
    """e_i = E_TF[t_i] + PE(p_i) + E_s[s_i] + MLP([a_i,m_i]) + E_l[l_i]，图1a。"""

    def __init__(self, n_tf: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.e_tf = nn.Embedding(n_tf, d_model)
        self.e_strand = nn.Embedding(3, d_model)  # -1/0/+1 -> 索引 0/1/2
        self.e_res = nn.Embedding(2, d_model)     # 0=peak-only, 1=motif-anchored
        self.am_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(),
                                    nn.Linear(d_model, d_model))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _pos_encoding(self, pos: torch.Tensor) -> torch.Tensor:
        """连续坐标(bp)的正弦位置编码，形状 (B,N) -> (B,N,d_model)。"""
        ang = pos.unsqueeze(-1) * self.inv_freq  # (B,N,d_model/2)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, tf_idx, pos, strand, a, m, res):
        """全部输入形状 (B,N)，strand ∈ {-1,0,1}(float或int均可)，res ∈ {0,1}。"""
        strand_idx = (strand.long() + 1).clamp(0, 2)
        res_idx = res.long().clamp(0, 1)
        am = torch.stack([a, m], dim=-1)  # (B,N,2)
        return (self.e_tf(tf_idx) + self._pos_encoding(pos) +
                self.e_strand(strand_idx) + self.am_mlp(am) + self.e_res(res_idx))


class LayoutTransformerLayer(nn.Module):
    """标准 Pre-LN Transformer block，注意力 logits 加成对距离偏置(图2a a3)。
    注意力手写成 qkv线性层+SDPA+out_proj(第二轮提速改动1)，数学上等价于原来的
    nn.MultiheadAttention，初始化照抄 nn.MultiheadAttention._reset_parameters。"""

    def __init__(self, d_model: int, n_heads: int, n_tf: int, d_ff: int = None,
                dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        d_ff = d_ff or 4 * d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out_proj.bias)
        self.dist_bias = PairwiseDistanceBias(n_tf, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tf_idx, phi, key_padding_mask):
        """x: (B,N,d_model)；phi: (B,N,N,13)；key_padding_mask: (B,N) bool，
        True=需要屏蔽(padding)。距离偏置和 padding 折成同一个加性 float mask：先把
        float32 的距离偏置转成 q 的 dtype(开 bf16 时是 bf16)，再把 padding 的 key
        列填成该 dtype 的有限极小值(用有限值而不是 -inf：整行都被屏蔽时 softmax 不会
        出 nan；实际上每行至少有空token这一个有效 key，不会整行屏蔽)。"""
        B, N, D = x.shape
        H, hd = self.n_heads, self.head_dim
        bias = self.dist_bias(tf_idx, phi)                     # (B,H,N,N) float32

        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, N, 3, H, hd).unbind(dim=2)  # 各 (B,N,H,hd)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_mask = bias.to(q.dtype).masked_fill(key_padding_mask.view(B, 1, 1, N),
                                                 torch.finfo(q.dtype).min)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0)  # (B,H,N,hd)
        attn = self.out_proj(attn.transpose(1, 2).reshape(B, N, D))
        x = x + self.dropout(attn)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class LayoutTransformer(nn.Module):
    """堆叠 n_layers 层，输出 H_lay(图1b)。每个 L_g 前面固定拼一个可学习的"空token"
    (图1b原文：有些基因L_g为空，交叉注意力无从计算，常见做法是加一个空token占位)，
    这样即使原始L_g是空的，序列长度也至少是1，且空token本身可以吸收"这个基因没有
    已知TF结合"这个信息，供后续交叉注意力/池化使用。"""

    def __init__(self, n_tf: int, d_model: int = 256, n_heads: int = 8,
                n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.token_embed = LayoutTokenEmbedding(n_tf, d_model)
        self.empty_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.empty_token, std=0.02)
        self.layers = nn.ModuleList([
            LayoutTransformerLayer(d_model, n_heads, n_tf, dropout=dropout)
            for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, tf_idx, pos, strand, a, m, res, mask):
        """tf_idx/pos/strand/a/m/res: (B,N)，跟09_torch_dataset.py collate_fn的
        layout_wt/layout_d字典字段一一对应；mask: (B,N) bool，True=真实token。
        返回 H_lay: (B,N+1,d_model)，key_padding_mask: (B,N+1) bool(True=padding，
        跟输入mask的语义相反，所以是 ~有效位)。"""
        B, N = tf_idx.shape
        x = self.token_embed(tf_idx, pos, strand, a, m, res)  # (B,N,d)
        empty = self.empty_token.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([empty, x], dim=1)  # (B,N+1,d)，空token放在位置0

        # 空token自己的"坐标"和"tf身份"设成0，只影响距离偏置里跟它有关的那些项，
        # 不影响其余真实token两两之间的偏置计算
        pos_full = torch.cat([torch.zeros(B, 1, device=pos.device, dtype=pos.dtype),
                              pos], dim=1)
        tf_idx_full = torch.cat([torch.zeros(B, 1, device=tf_idx.device,
                                             dtype=tf_idx.dtype), tf_idx], dim=1)
        mask_full = torch.cat([torch.ones(B, 1, device=mask.device, dtype=torch.bool),
                               mask], dim=1)  # 空token永远是"真实"位置，不参与padding
        key_padding_mask = ~mask_full  # True=padding

        # φ(Δd) 跟层无关，算一次各层共用(第二轮提速改动2)；固定 float32(改动3)
        with torch.autocast(device_type=pos_full.device.type, enabled=False):
            pf = pos_full.float()
            phi = _distance_basis(pf.unsqueeze(1) - pf.unsqueeze(2))  # (B,N+1,N+1,13)
        for layer in self.layers:
            x = layer(x, tf_idx_full, phi, key_padding_mask)
        return self.ln_out(x), key_padding_mask

    def forward_chunked(self, splits=None, **layout):
        """第三轮提速(见文件头)：按 splits=[(起,止,段内padding长度),...] 分段跑 forward 再
        拼回。要求 splits 按行号连续覆盖 [0,B)(09 号脚本 group_collated 保证)。layout 是
        forward 的全部参数(tf_idx,pos,strand,a,m,res,mask)。返回值形状、key_padding_mask
        都跟 forward(**layout) 相同。"""
        if not splits or len(splits) <= 1:
            return self.forward(**layout)
        mask = layout["mask"]
        B, N = mask.shape
        assert splits[0][0] == 0 and splits[-1][1] == B, "forward_chunked: 分段没覆盖全部行"
        outs = []
        for s0, e0, n_pad in splits:
            sub = {k: v[s0:e0, :n_pad].contiguous() for k, v in layout.items()}
            h, _ = self.forward(**sub)                     # (e0-s0, n_pad+1, d)
            if n_pad < N:
                h = F.pad(h, (0, 0, 0, N - n_pad))         # padding 位置补 0(下游屏蔽)
            outs.append(h)
        key_padding_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device),
                                      ~mask], dim=1)
        return torch.cat(outs, dim=0), key_padding_mask


def run_self_test(n_tf=178, d_model=64, n_heads=4, n_layers=2, batch=3, n_max=10,
                  seed=0):
    """随机小规模数据跑一次完整前向，检查形状对不对。这是唯一的脚本入口，
    正式训练时改成从09_torch_dataset.py的collate_fn输出接数据即可。"""
    torch.manual_seed(seed)
    lengths = torch.randint(1, n_max + 1, (batch,))
    N = int(lengths.max())
    tf_idx = torch.randint(0, n_tf, (batch, N))
    pos = torch.randn(batch, N) * 500
    strand = torch.randint(-1, 2, (batch, N)).float()
    a = torch.randn(batch, N)
    m = torch.randn(batch, N)
    res = torch.randint(0, 2, (batch, N))
    mask = torch.arange(N).unsqueeze(0) < lengths.unsqueeze(1)  # (B,N) bool

    model = LayoutTransformer(n_tf, d_model, n_heads, n_layers)
    h_lay, key_padding_mask = model(tf_idx, pos, strand, a, m, res, mask)

    print(f"输入 L_g 长度(每个样本): {lengths.tolist()}，padding 到 N={N}")
    print(f"H_lay 形状: {tuple(h_lay.shape)}  (应为 batch={batch}, N+1={N + 1}, "
          f"d_model={d_model})")
    print(f"key_padding_mask 形状: {tuple(key_padding_mask.shape)}，"
          f"每行 True(padding) 个数: {key_padding_mask.sum(dim=1).tolist()}")
    print(f"跟每个样本的(N - 真实长度)对比，应该一致: {[N - int(l) for l in lengths]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    assert h_lay.shape == (batch, N + 1, d_model), "H_lay 形状不对"
    assert not torch.isnan(h_lay).any(), "输出里有 NaN"

    # ---- 第二轮提速新增1：手写注意力 vs 旧版 nn.MultiheadAttention 写法，逐元素对比 ----
    # 把新层的 qkv/out_proj 权重原样拷进一个 nn.MultiheadAttention，按旧版
    # LayoutTransformerLayer.forward 的原始写法(偏置+padding折成(B*H,N,N)的float mask)
    # 重算一遍，两者应该只差浮点舍入。
    layer = LayoutTransformerLayer(d_model, n_heads, n_tf, dropout=0.0).eval()
    mha = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        mha.in_proj_weight.copy_(layer.qkv.weight)
        mha.in_proj_bias.copy_(layer.qkv.bias)
        mha.out_proj.weight.copy_(layer.out_proj.weight)
        mha.out_proj.bias.copy_(layer.out_proj.bias)
        x_in = torch.randn(batch, N, d_model)
        kpm = ~mask
        kpm[:, 0] = False  # 模拟空token：每行至少一个有效key
        phi = _distance_basis(pos.unsqueeze(1) - pos.unsqueeze(2))
        out_new = layer(x_in, tf_idx, phi, kpm)
        bias = layer.dist_bias(tf_idx, phi)
        pad_bias = torch.zeros(batch, N).masked_fill(kpm, torch.finfo(bias.dtype).min)
        old_mask = (bias + pad_bias.view(batch, 1, 1, N)).reshape(batch * n_heads, N, N)
        h_ref = layer.ln1(x_in)
        attn_ref, _ = mha(h_ref, h_ref, h_ref, attn_mask=old_mask, need_weights=False)
        out_ref = x_in + attn_ref
        out_ref = out_ref + layer.ffn(layer.ln2(out_ref))
    max_diff = float((out_new - out_ref).abs().max())
    print(f"手写注意力 vs 旧版 nn.MultiheadAttention 写法：最大绝对差 {max_diff:.2e}"
          "（应 < 1e-4，只是浮点舍入）")
    assert max_diff < 1e-4, "手写注意力跟旧版 nn.MultiheadAttention 写法不一致"

    # ---- 新增2：距离偏置的梯度能传回 U/V(SDPA 带 float mask 时梯度要流经 mask) ----
    model.train()
    h_lay2, _ = model(tf_idx, pos, strand, a, m, res, mask)
    h_lay2.pow(2).mean().backward()
    g_u = model.layers[0].dist_bias.U.weight.grad
    assert g_u is not None and torch.isfinite(g_u).all() and g_u.abs().sum() > 0, \
        "距离偏置 U 没有拿到有限且非零的梯度"
    print("距离偏置梯度检查通过：U/V 拿到了有限、非零的梯度。")

    # ---- 第三轮提速新增4：forward_chunked(按长度分段) vs forward，有效位置逐元素一致 ----
    model.eval()
    order = torch.sort(lengths, stable=True).indices
    lay_sorted = {k: v[order] for k, v in dict(tf_idx=tf_idx, pos=pos, strand=strand, a=a,
                                                m=m, res=res, mask=mask).items()}
    ls = lengths[order].tolist()
    cut = max(1, batch // 2)
    splits = [(0, cut, max(max(ls[:cut]), 1)), (cut, batch, max(max(ls[cut:]), 1))] \
        if batch >= 2 else [(0, batch, max(ls[-1], 1))]
    with torch.no_grad():
        h_full, kpm_full = model(**lay_sorted)
        h_chunk, kpm_chunk = model.forward_chunked(splits, **lay_sorted)
    valid = ~kpm_full
    d_chunk = float((h_full - h_chunk).abs()[valid].max())
    print(f"forward_chunked 分段 {splits} vs 不分段：有效位置最大绝对差 {d_chunk:.2e}"
          "（应 < 1e-5）")
    assert torch.equal(kpm_full, kpm_chunk), "forward_chunked 返回的 key_padding_mask 不一致"
    assert d_chunk < 1e-5, "forward_chunked 跟 forward 在有效位置上不一致"
    model.train()
    model.zero_grad()
    h_c2, _ = model.forward_chunked(splits, **lay_sorted)
    h_c2.pow(2).mean().backward()
    assert torch.isfinite(model.layers[0].dist_bias.U.weight.grad).all(), "分段反传梯度异常"
    print("forward_chunked 检查通过：有效位置一致、mask 一致、反传梯度有限。")

    # ---- 新增3(只在有 CUDA+bf16 时跑)：bf16 autocast 下前向不出 NaN、跟 fp32 接近 ----
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_c = model.cuda().eval()
        args = [t.cuda() for t in (tf_idx, pos, strand, a, m, res, mask)]
        with torch.no_grad():
            ref32, _ = model_c(*args)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out16, _ = model_c(*args)
        rel = float((out16.float() - ref32).norm() / ref32.norm())
        print(f"bf16 autocast vs fp32：相对误差 {rel:.3e}（bf16 正常量级约1e-2，"
              "只要不是 NaN/远大于0.05 就正常）")
        assert torch.isfinite(out16).all(), "bf16 autocast 下输出有 NaN/inf"
    else:
        print("没有可用的 CUDA+bf16，跳过 bf16 autocast 检查(不影响其余自检结论)。")
    print("自检通过：形状、数值、手写注意力等价性、梯度、分段前向等价性都正常。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tf", type=int, default=178)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=3)
    a = ap.parse_args()
    run_self_test(a.n_tf, a.d_model, a.n_heads, a.n_layers, a.batch)
