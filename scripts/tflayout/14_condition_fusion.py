# scripts/tflayout/14_condition_fusion.py

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionFiLM(nn.Module):
    """ctx(B,n_tf) -> MLP -> γ,β(各(B,d_model))。最后一层零初始化+
    "gamma=1+raw_gamma"，让训练初期FiLM接近恒等变换，跟c_0=-2、c≈0.12这个"训练
    初期保守"的设计思路呼应(这条呼应是自己加的初始化技巧，原文没有明确要求)。"""

    def __init__(self, n_tf: int, d_model: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d_model
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(n_tf, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx: torch.Tensor):
        raw = self.net(ctx)  # (B, 2*d_model)
        raw_gamma, beta = raw[:, :self.d_model], raw[:, self.d_model:]
        gamma = 1.0 + raw_gamma
        return gamma, beta


class CrossModalFusion(nn.Module):
    """交叉注意力(cis查询layout) + 门控c + FiLM + 残差 + attention pooling，
    对应status文件第4节"跨模态融合(4步)"。forward一次只处理一个条件(WT或D)，
    WT/D要各调一次。"""

    def __init__(self, n_tf: int, d_model: int = 256, n_heads: int = 8,
                dropout: float = 0.1, c0_init: float = -2.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.condition = ConditionFiLM(n_tf, d_model)
        self.c0 = nn.Parameter(torch.tensor(float(c0_init)))
        self.pool_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pool_query, std=0.02)

    def query(self, h_cis, cis_pad_mask):
        """第1步的Q侧：q_lin=H_cis·W_Q (B,N_cis,D)；q_mean 是在有效(非padding)token
        上的均值(B,D)，float32，给门控c用。只取决于基因，分组前向里每个基因算一次。"""
        q_lin = self.q_proj(h_cis)
        cis_valid = (~cis_pad_mask).float().unsqueeze(-1)  # (B,N_cis,1)
        q_mean = (q_lin.float() * cis_valid).sum(1) / cis_valid.sum(1).clamp(min=1)
        return q_lin, q_mean

    def attend(self, q_lin, h_lay, lay_pad_mask):
        """第1步的注意力：K/V=H_lay·W_K/W_V，返回 attn_out(过out_proj，公式里的
        "Attn"，(B,N_cis,D)) 和 k_mean(有效layout token上的均值，(B,D) float32)。
        只屏蔽layout(key/value)侧的padding；cis(query)侧padding不用在这里挡，那些
        位置算出来的attn_out之后在pooling阶段会被cis_pad_mask一并挡掉。"""
        B, N_cis, D = q_lin.shape
        N_lay = h_lay.shape[1]
        H, hd = self.n_heads, self.head_dim
        k_lin = self.k_proj(h_lay)  # (B,N_lay,D)
        v_lin = self.v_proj(h_lay)
        q = q_lin.view(B, N_cis, H, hd).transpose(1, 2)  # (B,H,N_cis,hd)
        k = k_lin.to(q.dtype).view(B, N_lay, H, hd).transpose(1, 2)
        v = v_lin.to(q.dtype).view(B, N_lay, H, hd).transpose(1, 2)
        attn_bias = torch.zeros(B, 1, 1, N_lay, device=q.device, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(lay_pad_mask.view(B, 1, 1, N_lay),
                                          torch.finfo(q.dtype).min)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0)  # (B,H,N_cis,hd)
        attn_out = attn_out.transpose(1, 2).reshape(B, N_cis, D)
        attn_out = self.out_proj(attn_out)  # 公式里的 "Attn"
        lay_valid = (~lay_pad_mask).float().unsqueeze(-1)  # (B,N_lay,1)
        k_mean = (k_lin.float() * lay_valid).sum(1) / lay_valid.sum(1).clamp(min=1)
        return attn_out, k_mean

    def gate(self, q_mean, k_mean):
        """第2步门控 c = σ(mean(Q)ᵀmean(K)/√d + c0)，每条样本一个标量，float32。
        理论上H_lay有空token兜底、H_cis每个基因都有启动子序列，不会全padding，
        上面均值里的clamp(min=1)只是防御性写法。"""
        with torch.autocast(device_type=q_mean.device.type, enabled=False):
            gate_logit = ((q_mean.float() * k_mean.float()).sum(-1)
                          / math.sqrt(self.d_model) + self.c0)  # (B,)
            return torch.sigmoid(gate_logit)

    def film_pool(self, q_lin, attn_out, c, cis_pad_mask, ctx):
        """第3、4步：FiLM(γ,β来自ctx) + 残差 z_n=Q_n+c·FiLM(Attn)_n + attention
        pooling(可学习query对token维度softmax加权，cis侧padding权重强制为0)。
        整段固定在float32里算(原因见文件头【第二轮提速】)。返回 z_pool (B,D) 和 aux。"""
        B, N_cis, D = q_lin.shape
        with torch.autocast(device_type=q_lin.device.type, enabled=False):
            q32, a32 = q_lin.float(), attn_out.float()
            gamma, beta = self.condition(ctx.float())  # 各(B,D)
            film_out = gamma.unsqueeze(1) * a32 + beta.unsqueeze(1)  # (B,N_cis,D)
            z_tok = q32 + c.float().view(B, 1, 1) * film_out  # z_n = Q_n + c·FiLM(...)_n
            scores = (z_tok @ self.pool_query.float()) / math.sqrt(D)  # (B,N_cis)
            scores = scores.masked_fill(cis_pad_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            z_pool = (z_tok * weights.unsqueeze(-1)).sum(1)  # (B,D)
        return z_pool, {"z_tok": z_tok, "gamma": gamma, "beta": beta,
                        "pool_weights": weights}

    def forward(self, h_cis, cis_pad_mask, h_lay, lay_pad_mask, ctx):
        """
        h_cis: (B,N_cis,d_model)  cis_pad_mask: (B,N_cis) bool True=padding
        h_lay: (B,N_lay,d_model)  lay_pad_mask: (B,N_lay) bool True=padding
        ctx:   (B,n_tf)
        返回 z_pool: (B,d_model)，每个(gene, 条件k)一个向量，对应原文z̄_{g,k}；
        aux里附带c/gamma/beta等中间量，方便调试，不影响下游怎么用z_pool。
        四步按原顺序串起来，跟拆分前的一口气写法数学上完全相同。
        """
        q_lin, q_mean = self.query(h_cis, cis_pad_mask)
        attn_out, k_mean = self.attend(q_lin, h_lay, lay_pad_mask)
        c = self.gate(q_mean, k_mean)
        z_pool, aux = self.film_pool(q_lin, attn_out, c, cis_pad_mask, ctx)
        aux.update({"c": c, "attn_out": attn_out, "q_lin": q_lin})
        return z_pool, aux


def run_self_test(n_tf=178, d_model=64, n_heads=4, batch=3, n_cis=50, n_lay=12,
                  seed=0):
    """随机小规模数据跑一次完整前向，检查形状、c的取值范围，以及pooling对cis侧
    padding的不变性(改padding区域的值，z_pool不应该变)。"""
    torch.manual_seed(seed)
    h_cis = torch.randn(batch, n_cis, d_model)
    cis_lengths = torch.randint(n_cis // 2, n_cis + 1, (batch,))
    cis_pad_mask = torch.arange(n_cis).unsqueeze(0) >= cis_lengths.unsqueeze(1)

    h_lay = torch.randn(batch, n_lay, d_model)
    lay_lengths = torch.randint(1, n_lay + 1, (batch,))  # 至少1(10号脚本的空token兜底)
    lay_pad_mask = torch.arange(n_lay).unsqueeze(0) >= lay_lengths.unsqueeze(1)

    ctx = torch.randn(batch, n_tf)

    model = CrossModalFusion(n_tf, d_model, n_heads, c0_init=-2.0)
    z_pool, aux = model(h_cis, cis_pad_mask, h_lay, lay_pad_mask, ctx)

    print(f"z_pool 形状: {tuple(z_pool.shape)}  (应为 batch={batch}, d_model={d_model})")
    print(f"门控 c: {[round(x, 4) for x in aux['c'].tolist()]}")
    print("  c0=-2.0时，若mean(Q)/mean(K)点积项接近0，c应接近sigmoid(-2)≈0.1192；"
          "这里Q/K来自随机初始化的线性层，点积项不会精确是0，几个值散在0.12附近"
          "属正常，不要求精确等于0.1192")
    assert z_pool.shape == (batch, d_model), "z_pool 形状不对"
    assert not torch.isnan(z_pool).any(), "z_pool 里有 NaN"
    assert torch.all((aux["c"] >= 0) & (aux["c"] <= 1)), "c 应该在(0,1)之间(sigmoid输出)"

    model.eval()
    b0 = 0
    pad_start = int(cis_lengths[b0])
    if pad_start < n_cis:
        h_cis2 = h_cis.clone()
        h_cis2[b0, pad_start:] = 1000.0
        with torch.no_grad():
            z_pool_a, _ = model(h_cis, cis_pad_mask, h_lay, lay_pad_mask, ctx)
            z_pool_b, _ = model(h_cis2, cis_pad_mask, h_lay, lay_pad_mask, ctx)
        assert torch.allclose(z_pool_a[b0], z_pool_b[b0], atol=1e-4), (
            "cis侧padding位置改了值，z_pool(该样本)却变了——pooling的mask可能没生效")
        print("padding不变性自检通过：改cis侧padding区域的值，对应样本的z_pool不变。")
    else:
        print("这次随机到的样本0没有cis侧padding，跳过不变性自检(概率性，不是bug)。")

    # ---- 第二轮提速新增1：拆成四个方法之后的 forward vs 拆分前的一口气写法 ----
    # 下面是旧版 forward 的原文(只把 self 换成 model)，逐元素对比
    with torch.no_grad():
        Bq, Nc, D = h_cis.shape
        H, hd = model.n_heads, model.head_dim
        q_lin = model.q_proj(h_cis)
        k_lin = model.k_proj(h_lay)
        v_lin = model.v_proj(h_lay)
        q = q_lin.view(Bq, Nc, H, hd).transpose(1, 2)
        k = k_lin.view(Bq, n_lay, H, hd).transpose(1, 2)
        v = v_lin.view(Bq, n_lay, H, hd).transpose(1, 2)
        ab = torch.zeros(Bq, 1, 1, n_lay).masked_fill(lay_pad_mask.view(Bq, 1, 1, n_lay),
                                                      torch.finfo(h_cis.dtype).min)
        ao = F.scaled_dot_product_attention(q, k, v, attn_mask=ab)
        ao = model.out_proj(ao.transpose(1, 2).reshape(Bq, Nc, D))
        cv = (~cis_pad_mask).float().unsqueeze(-1)
        lv = (~lay_pad_mask).float().unsqueeze(-1)
        qm = (q_lin * cv).sum(1) / cv.sum(1).clamp(min=1)
        km = (k_lin * lv).sum(1) / lv.sum(1).clamp(min=1)
        c_ref = torch.sigmoid((qm * km).sum(-1) / math.sqrt(D) + model.c0)
        g_ref, b_ref = model.condition(ctx)
        zt = q_lin + c_ref.view(Bq, 1, 1) * (g_ref.unsqueeze(1) * ao + b_ref.unsqueeze(1))
        sc = ((zt @ model.pool_query) / math.sqrt(D)).masked_fill(
            cis_pad_mask, torch.finfo(zt.dtype).min)
        z_ref = (zt * torch.softmax(sc, dim=-1).unsqueeze(-1)).sum(1)
        z_new, aux_new = model(h_cis, cis_pad_mask, h_lay, lay_pad_mask, ctx)
    d1 = float((z_new - z_ref).abs().max())
    print(f"拆分后 forward vs 拆分前一口气写法：z_pool 最大绝对差 {d1:.2e}（应 < 1e-5）")
    assert d1 < 1e-5, "拆分后的 forward 跟拆分前的数学定义不一致"

    # ---- 新增2：复用路径——layout 不变、只换 ctx 时，复用 WT 的 q_lin/attn_out/c
    # 只重做 film_pool，应该跟整条 forward 重算完全一样(15号脚本分组前向的依据) ----
    ctx2 = torch.randn(batch, n_tf)
    with torch.no_grad():
        z_full, _ = model(h_cis, cis_pad_mask, h_lay, lay_pad_mask, ctx2)
        z_reuse, _ = model.film_pool(aux_new["q_lin"], aux_new["attn_out"], aux_new["c"],
                                     cis_pad_mask, ctx2)
    d2 = float((z_full - z_reuse).abs().max())
    print(f"复用 attn_out/c 只重做 film_pool vs 整条重算：最大绝对差 {d2:.2e}（应 < 1e-6）")
    assert d2 < 1e-6, "复用路径跟整条重算不一致"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    print("自检通过：形状、数值范围、padding不变性、拆分等价性、复用路径都正常。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tf", type=int, default=178)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n-cis", type=int, default=50)
    ap.add_argument("--n-lay", type=int, default=12)
    a = ap.parse_args()
    run_self_test(a.n_tf, a.d_model, a.n_heads, a.batch, a.n_cis, a.n_lay)
