import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

def make_norm(kind: str, c: int) -> nn.Module:
    if kind == 'ln':
        return LayerNorm2d(c)
    if kind == 'gn':
        return nn.GroupNorm(min(8, c), c)
    if kind == 'bn':
        return nn.BatchNorm2d(c)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(kind)

class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)

        return x*self.weight[:, None, None] + self.bias[:, None, None]

class ConvNorm(nn.Sequential):
    def __init__(self, c_in, c_out, k=1, s=1, norm='ln', pad_mode='reflect'):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, padding=k//2 if s == 1 else 0,
                      padding_mode=pad_mode if k > 1 else 'zeros', bias=False),
            make_norm(norm, c_out)
        )

class ResBlockA(nn.Module):
    def __init__(self, c, dilations: Sequence[int], norm='ln', pad_mode='reflect'):
        super().__init__()

        def branch(d):
            return nn.Sequential(
                make_norm(norm, c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=d, dilation=d, padding_mode=pad_mode, bias=False),
                make_norm(norm, c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=d, dilation=d, padding_mode=pad_mode, bias=False),
            )

        self.branches = nn.ModuleList(branch(d) for d in dilations)

    def forward(self, x):
        out = x
        for b in self.branches:
            out = out + b(x)
        
        return out

class Down(nn.Module):
    def __init__(self, c_in, c_out, mode='conv2x2'):
        super().__init__()
        k = 2 if mode == 'conv2x2' else 1
        self.conv = nn.Conv2d(c_in, c_out, k, stride=2)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out, norm='ln'):
        super().__init__()
        self.proj = ConvNorm(c_in, c_out, 1, norm=norm)

    def forward(self, x):
        return self.proj(F.interpolate(x, scale_factor=2, mode='nearest'))

class Combine(nn.Module):
    def __init__(self, c, norm='ln'):
        super().__init__()
        self.proj = ConvNorm(2*c, c, 1, norm=norm)

    def forward(self, dec, enc):
        return self.proj(torch.cat([F.relu(dec), enc], 1))

class PSPPooling(nn.Module):
    def __init__(self, c, pool_sizes=(16, 8, 4, 2), mode='fixed', norm='ln'):
        super().__init__()
        assert c % len(pool_sizes) == 0
        
        self.pool_sizes, self.mode = pool_sizes, mode
        cq = c // len(pool_sizes)
        self.reduce = nn.ModuleList(ConvNorm(cq, cq, 1, norm=norm) for _ in pool_sizes)
        self.fuse = ConvNorm(2*c, c, 1, norm=norm)

    def forward(self, x):
        H, W = x.shape[-2:]
        chunks = torch.chunk(x, len(self.pool_sizes), dim=1)
        outs = []

        for xi, p, red in zip(chunks, self.pool_sizes, self.reduce):
            if self.mode == 'fixed':
                y = F.avg_pool2d(xi, kernel_size=p, stride=p, ceil_mode=True)
                y = F.interpolate(red(y), size=(H, W), mode='bilinear', align_corners=False)
            else:
                y = F.adaptive_max_pool2d(xi, p)
                y = F.interpolate(red(y), size=(H, W), mode='nearest')
            outs.append(y)

        return self.fuse(torch.cat([x] + outs, 1))

class ResUNetDenoiser(nn.Module):
    def __init__(
            self,
            n_slices: int=5,
            base: int=32,
            depth: int=4,
            dilations: Sequence[Sequence[int]] = ((1, 3, 7, 15), (1, 3, 7), (1, 3, 7), (1, 3)),
            norm: str='ln',
            pad_mode: str='reflect',
            down_mode: str='conv2x2',
            psp_mode: str='fixed',
            psp_sizes: Sequence[int] = (16, 8, 4, 2),
            final_psp: bool=False,
            residual: bool=True,
    ):
        super().__init__()
        assert len(dilations) == depth
        self.n_slices, self.residual = n_slices, residual
        ch = [base * 2**i for i in range(depth)]

        self.stem = ConvNorm(n_slices, base, 1, norm=norm)
        self.enc = nn.ModuleList(ResBlockA(c, d, norm, pad_mode) for c, d in zip(ch, dilations))
        self.down = nn.ModuleList(Down(ch[i], ch[i + 1], down_mode) for i in range(depth - 1))
        self.psp_mid = PSPPooling(ch[-1], psp_sizes, psp_mode, norm)

        self.up = nn.ModuleList(Up(ch[i + 1], ch[i], norm) for i in reversed(range(depth - 1)))
        self.comb = nn.ModuleList(Combine(ch[i], norm) for i in reversed(range(depth - 1)))
        self.dec = nn.ModuleList(ResBlockA(ch[i], dilations[i], norm, pad_mode)
                                 for i in reversed(range(depth - 1)))
        self.comb_stem = Combine(base, norm)
        self.psp_out = PSPPooling(base, psp_sizes, psp_mode, norm) if final_psp else nn.Identity()

        self.head = nn.Conv2d(base, 1, 1)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** (len(self.enc) - 1)
        assert H % m == 0 and W % m == 0, f'H, W must be multiples of {m}'

        s0 = self.stem(x)
        skips, h = [], s0
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        h = self.psp_mid(h)
        for up, comb, blk in zip(self.up, self.comb, self.dec):
            h = blk(comb(up(h), skips.pop()))
        h = self.psp_out(self.comb_stem(h, s0))
        r = self.head(h)
        if not self.residual:
            return r
        c = self.n_slices // 2
        return x[:, c:c + 1] + r