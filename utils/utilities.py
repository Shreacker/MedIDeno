import sys
import math
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from data_utils.ChestLDCT import WIN_MAX, WIN_MIN
from torch.optim.lr_scheduler import LambdaLR
from models.resunet import ResUNetDenoiser

MODELS = {'ResUNet': ResUNetDenoiser}

def setup_logging(run_dir: Path) -> None:
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'train.log', encoding='utf-8')):
        h.setFormatter(fmt)
        root.addHandler(h)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x*m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1.0)

def masked_loss(pred, tgt, mask, kind: str) -> torch.Tensor:
    d = pred - tgt
    return masked_mean(d.abs() if kind == 'l1' else d.pow(2), mask).mean()

def gaussian_window(size: int=11, sigma: float=1.5, device=None) -> torch.Tensor:
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g**2 / (2*sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]

def ssim_map(x, y, w, c1: float=0.01**2, c2: float=0.03**2) -> torch.Tensor:
    p = w.shape[-1] // 2
    f = lambda t: F.conv2d(F.pad(t, (p, p, p, p), mode='reflect'), w)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x*x) - mx**2, f(y*y) - my**2, f(x*y) - mx*my
    return ((2*mx*my + c1)*(2*sxy + c2)) / ((mx**2 + my**2 + c1)*(sxx + syy + c2))

def image_metrics(pred, tgt, mask, w) -> dict:
    pred = pred.clamp(0, 1)
    mse = masked_mean((pred - tgt)**2, mask).clamp_min(1e-12)
    return {
        'psnr': 10*torch.log10(1.0 / mse),
        'ssim': masked_mean(ssim_map(pred, tgt, w), mask),
        'rmse_hu': mse.sqrt() * (WIN_MAX - WIN_MIN),
    }

def save_ckpt(path: Path, model, opt, sched, scaler, epoch: int, best: float, args, metrics: dict) -> None:
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'sched': sched.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_psnr': best,
        'args': vars(args),
        'metrics': metrics,
    }, path)

def get_model(args, device) -> torch.nn.Module:
    if args.model not in MODELS:
        raise ValueError(f'unknown model {args.model}, choose from {list(MODELS)}')
    model = MODELS[args.model](n_slices=args.n_slices).to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print(f'No. model params: {nb_params}')
    return model

def create_optimizer(model: torch.nn.Module, args, steps_per_epoch: int):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * steps_per_epoch
    warm = args.warmup_epochs * steps_per_epoch
    sched = LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                     else 0.5*(1 + math.cos(math.pi*(s - warm) / max(1, total - warm))))
    return opt, sched
