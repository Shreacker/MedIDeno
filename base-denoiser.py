import os, sys
os.makedirs('./results', exist_ok=True)

import argparse
import csv
import json
import math
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils.ChestLDCT import get_loader
from utils.utilities import *

log = logging.getLogger('base-denoiser')
CSV_FIELDS = ['epoch', 'lr', 'train_loss', 'time_s',
              'val_psnr', 'val_ssim', 'val_rmse_hu',
              'val_ldct_psnr', 'val_ldct_ssim', 'val_ldct_rmse_hu']

###
@torch.no_grad()
def evaluate(model, loader, device, amp: bool, w) -> dict:
    model.eval()
    sums, n = {}, 0
    for b in loader:
        x = b['input'].to(device, non_blocking=True)
        y = b['target'].to(device, non_blocking=True)
        m = b['mask'].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            p = model(x)
        c = x.shape[1] // 2
        for prefix, out in (('', p.float()), ('ldct_', x[:, c:c + 1])):
            for k, v in image_metrics(out, y, m, w).items():
                sums[prefix + k] = sums.get(prefix + k, 0.0) + v.sum().item()
        n += x.shape[0]
    return {k: v / max(n, 1) for k, v in sums.items()}

###
def eval_loader(loader: DataLoader, max_items: int, batch_size: int, num_workers: int) -> DataLoader:
    ds = loader.dataset
    if max_items and len(ds) > max_items:
        ds = Subset(ds, np.linspace(0, len(ds) - 1, max_items).round().astype(int).tolist())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)

###
def main(args: argparse.Namespace) -> None:
    run_name = args.run_name or datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = Path(args.results_dir) / run_name
    ckpt_dir = Path(args.ckpt_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / f'{run_name}_best.pt', ckpt_dir / f'{run_name}_last.pt'

    setup_logging(run_dir)
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=1))
    set_seed(args.seed)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    amp = args.amp and device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    log.info('run=%s device=%s amp=%s', run_name, device, amp)

    common = dict(data_root=args.data_root, cache_dir=args.cache_dir, n_slices=args.n_slices, seed=args.seed)
    train_loader = get_loader(split='train', batch_size=args.batch_size, num_workers=args.num_workers,
                              patch_size=args.patch_size, **common)
    val_loader = eval_loader(get_loader(split='val', num_workers=0, **common),
                             args.val_max_slices, args.val_batch_size, args.num_workers)
    log.info('train batches/epoch=%d, val slices=%d', len(train_loader), len(val_loader.dataset))

    model = get_model(args, device)
    opt, sched = create_optimizer(model, args, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    w = gaussian_window(device=device)

    start, best = 1, -math.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        start, best = ck['epoch'] + 1, ck['best_psnr']
        log.info('resumed from %s at epoch %d (best val PSNR %.3f)', args.resume, ck['epoch'], best)

    csv_path = run_dir / 'metrics.csv'
    if not csv_path.exists():
        with csv_path.open('w', newline='') as f:
            csv.DictWriter(f, CSV_FIELDS).writeheader()

    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for it, b in enumerate(train_loader, 1):
            x = b['input'].to(device, non_blocking=True)
            y = b['target'].to(device, non_blocking=True)
            m = b['mask'].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                p = model(x)
            loss = masked_loss(p.float(), y, m, args.loss)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item()
            if args.log_every and it % args.log_every == 0:
                log.info('epoch %d [%d/%d] loss=%.5f lr=%.2e',
                         epoch, it, len(train_loader), running / it, opt.param_groups[0]['lr'])

        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'],
               'train_loss': running / max(len(train_loader), 1), 'time_s': time.time() - t0}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            row.update({f'val_{k}': v for k, v in evaluate(model, val_loader, device, amp, w).items()})
            if row['val_psnr'] > best:
                best = row['val_psnr']
                save_ckpt(best_path, model, opt, sched, scaler, epoch, best, args, row)
                log.info('new best val PSNR %.3f -> %s', best, best_path)
        save_ckpt(last_path, model, opt, sched, scaler, epoch, best, args, row)

        with csv_path.open('a', newline='') as f:
            csv.DictWriter(f, CSV_FIELDS, restval='').writerow(row)
        log.info('epoch %d | %s', epoch, ' '.join(f'{k}={v:.4g}' for k, v in row.items() if k != 'epoch'))

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model'])
    test_loader = eval_loader(get_loader(split='test', num_workers=0, **common),
                              0, args.val_batch_size, args.num_workers)
    test = evaluate(model, test_loader, device, amp, w)
    (run_dir / 'test_metrics.json').write_text(json.dumps(test, indent=1))
    log.info('test (%d slices) | %s', len(test_loader.dataset), ' '.join(f'{k}={v:.4g}' for k, v in test.items()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Low-dose CT scans 2.5D Base Denoiser')
    parser.add_argument('--model',
                        type=str,
                        default='ResUNet'
                    )
    parser.add_argument('--data_root',
                        type=str,
                        default='data/Chest-LDCT-data'
                    )
    parser.add_argument('--cache_dir',
                        type=str,
                        default=None
                    )
    parser.add_argument('--n_slices',
                        type=int,
                        default=5
                    )
    parser.add_argument('--patch_size',
                        type=int,
                        default=128
                    )
    parser.add_argument('--batch_size',
                        type=int,
                        default=16
                    )
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=4
                    )
    parser.add_argument('--num_workers',
                        type=int,
                        default=4
                    )
    parser.add_argument('--epochs',
                        type=int,
                        default=100
                    )
    parser.add_argument('--lr',
                        type=float,
                        default=2e-4
                    )
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4
                    )
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=2
                    )
    parser.add_argument('--loss',
                        type=str,
                        default='l1',
                        choices=['l1', 'mse']
                    )
    parser.add_argument('--grad_clip',
                        type=float,
                        default=1.0
                    )
    parser.add_argument('--amp',
                        action=argparse.BooleanOptionalAction,
                        default=True
                    )
    parser.add_argument('--val_every',
                        type=int,
                        default=1
                    )
    parser.add_argument('--val_max_slices',
                        type=int,
                        default=200
                    )
    parser.add_argument('--log_every',
                        type=int,
                        default=50
                    )
    parser.add_argument('--seed',
                        type=int,
                        default=21
                    )
    parser.add_argument('--device',
                        type=str,
                        default=None
                    )
    parser.add_argument('--run_name',
                        type=str,
                        default=None
                    )
    parser.add_argument('--results_dir',
                        type=str,
                        default='./results'
                    )
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='./models/checkpoints/ResUNet'
                    )
    parser.add_argument('--resume',
                        type=str,
                        default=None
                    )
    main(parser.parse_args())
