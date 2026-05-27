import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):

    def __init__(self, dim: int, heads: int = 8, dim_head: int = None, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert dim > 0, 'dim 必须 > 0'
        assert heads > 0, 'heads 必须 > 0'
        if dim_head is None:
            assert dim % heads == 0, 'dim 不能整除 heads，需要显式传入 dim_head'
            dim_head = dim // heads
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads


        self.to_q = nn.Linear(dim, inner_dim, bias=bias)

        self.to_k_y = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v_y = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k_z = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v_z = nn.Linear(dim, inner_dim, bias=bias)

        self.scale = dim_head ** -0.5

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, dim, bias=bias)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, nx, d = x.shape
        by, ny, dy = y.shape
        bz, nz, dz = z.shape
        assert b == by == bz, 'x / y / z 的 batch 不匹配'
        assert d == dy == dz == self.dim, f'输入特征维度不匹配: {d}, {dy}, {dz}'

        h = self.heads
        dh = self.dim_head

        # 线性映射
        q = self.to_q(x).view(b, nx, h, dh).transpose(1, 2)  # [b, h, nx, dh]
        k_y = self.to_k_y(y).view(b, ny, h, dh).transpose(1, 2)  # [b, h, ny, dh]
        v_y = self.to_v_y(y).view(b, ny, h, dh).transpose(1, 2)  # [b, h, ny, dh]
        k_z = self.to_k_z(z).view(b, nz, h, dh).transpose(1, 2)  # [b, h, nz, dh]
        v_z = self.to_v_z(z).view(b, nz, h, dh).transpose(1, 2)  # [b, h, nz, dh]

        # KV 拼接后按常规注意力计算与加权
        k = torch.cat([k_y, k_z], dim=2)  # [b, h, ny+nz, dh]
        v = torch.cat([v_y, v_z], dim=2)  # [b, h, ny+nz, dh]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, h, nx, ny+nz]

        if mask is not None:
            # 支持 [b, ny+nz] 或 [b, 1, 1, ny+nz]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # [b, 1, 1, ny+nz]
            elif mask.dim() == 4:
                assert mask.shape[1] in (1, h) or mask.shape[2] in (1, nx), 'mask 形状不符合预期'
            else:
                raise ValueError('mask 维度必须为2或4')
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [b, h, nx, dh]
        out = out.transpose(1, 2).contiguous().view(b, nx, h * dh)  # [b, nx, d]
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class JointCrossAttn(nn.Module):

    def __init__(
        self,
        dim: int,
        layers: int = 3,
        heads: int = 8,
        dim_head: int | None = None,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ):
        super().__init__()
        assert layers > 0, 'layers 必须 > 0'
        self.dim = dim
        self.layers = layers

        self.blocks = nn.ModuleList()
        self.attn_gammas = nn.ParameterList()
        self.ff_gammas = nn.ParameterList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                'ln_attn': nn.LayerNorm(dim),
                'ln_y': nn.LayerNorm(dim),
                'ln_z': nn.LayerNorm(dim),
                'attn': CrossAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                'ln_ff': nn.LayerNorm(dim),
                'ff': FeedForward(dim=dim, mult=ff_mult, dropout=dropout),
            }))
            self.attn_gammas.append(nn.Parameter(torch.zeros(1)))
            self.ff_gammas.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            ln_x = block['ln_attn'](x)
            ln_y = block['ln_y'](y)
            ln_z = block['ln_z'](z)
            x_attn = block['attn'](ln_x, ln_y, ln_z, mask=mask)
            x = x + self.attn_gammas[i] * x_attn
            # FFN 子层
            x_ff = block['ff'](block['ln_ff'](x))
            x = x + self.ff_gammas[i] * x_ff
        return x

