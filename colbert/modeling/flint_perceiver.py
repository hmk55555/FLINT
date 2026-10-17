import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPerceiverBottleneck(nn.Module):
    """Perceiver bottleneck shared by queries and documents."""

    def __init__(self, num_query_latents=32, num_doc_latents=32, dim=768, num_heads=8,
                 dropout=0.1, ff_ratio=4, use_ff=True, attn_impl="mha"):
        super().__init__()
        self.num_query_latents = num_query_latents
        self.num_doc_latents = num_doc_latents
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, (dim, num_heads)
        self.use_ff = use_ff
        self.attn_impl = attn_impl
        self.attn_dropout = dropout

        self.query_latents = nn.Parameter(torch.randn(num_query_latents, dim) * 0.02)
        self.doc_latents = nn.Parameter(torch.randn(num_doc_latents, dim) * 0.02)

        self.norm_latents_cross = nn.LayerNorm(dim)
        self.norm_context_cross = nn.LayerNorm(dim)
        if attn_impl == "sdpa":
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
        else:
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        if use_ff:
            ff_dim = int(dim * ff_ratio)
            self.norm_ff = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)

        if attn_impl == "sdpa":
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_latents, std=0.02)
        nn.init.normal_(self.doc_latents, std=0.02)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for ln in (self.norm_latents_cross, self.norm_context_cross):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
        if self.use_ff:
            nn.init.ones_(self.norm_ff.weight)
            nn.init.zeros_(self.norm_ff.bias)
            for m in self.ff:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, token_embeddings, key_padding_mask=None, mode='doc'):
        B = token_embeddings.size(0)

        latents_param = self.query_latents if mode == 'query' else self.doc_latents
        latents = latents_param.unsqueeze(0).expand(B, -1, -1).contiguous()

        q = self.norm_latents_cross(latents)
        kv = self.norm_context_cross(token_embeddings)

        if self.attn_impl == "sdpa":
            H, hd = self.num_heads, self.head_dim
            Lq, Lk = q.size(1), kv.size(1)
            Qp = self.q_proj(q).view(B, Lq, H, hd).transpose(1, 2)
            Kp = self.k_proj(kv).view(B, Lk, H, hd).transpose(1, 2)
            Vp = self.v_proj(kv).view(B, Lk, H, hd).transpose(1, 2)

            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = torch.zeros(B, 1, 1, Lk, dtype=Qp.dtype, device=Qp.device)
                attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

            attn_out = F.scaled_dot_product_attention(
                Qp, Kp, Vp, attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, Lq, self.dim)
            attn_out = self.out_proj(attn_out)
        else:
            attn_out, _ = self.cross_attn(
                query=q, key=kv, value=kv, key_padding_mask=key_padding_mask,
            )

        latents = latents + self.dropout(attn_out)
        if self.use_ff:
            latents = latents + self.ff(self.norm_ff(latents))

        return latents
