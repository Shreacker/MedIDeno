import os
import copy
import json
import re
import logging
from typing import Sequence, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread

log = logging.getLogger(__name__)

HU_MIN, HU_MAX = -1024, 3071
CT_IMAGE_STORAGE = '1.2.840.10008.5.1.4.1.1.2'
WIN_MIN, WIN_MAX = -1000.0, 1000.0
FOV_HU = -1024.0

def hu_to_unit(hu):
    return (np.clip(hu, WIN_MIN, WIN_MAX) - WIN_MIN) / (WIN_MAX - WIN_MIN)

def unit_to_hu(x):
    return x*(WIN_MAX - WIN_MIN) + WIN_MIN

def _rescale(ds) -> tuple[float, float]:
    return float(getattr(ds, 'RescaleSlope', 1.0)), float(getattr(ds, 'RescaleIntercept', 0.0))

class Dataset_ChestLDCT(Dataset):
    def __init__(
            self,
            cache_dir: str | os.PathLike,
            patient_ids: Sequence[str],
            n_slices: int=5,
            patch_size: Optional[int]=128,
            augment: bool=True,
            fg_prob: float=0.9,
            fg_hu: float=-500.0,
            kernels: Optional[Sequence[str]]=None,
            manufacturers: Optional[Sequence[str]]=None
    ):
        assert n_slices % 2 == 1, 'n_slices must be odd'
        self.cache = Path(cache_dir)
        self.k = n_slices // 2
        self.patch, self.augment = patch_size, augment
        self.fg_prob, self.fg_hu = fg_prob, fg_hu

        def keep(value: str, allowed):
            return allowed is None or any(a.lower() in value.lower() for a in allowed)

        self.pids, self.index = [], []
        for pid in patient_ids:
            mfile = self.cache / f'{pid}_meta.json'
            if not mfile.exists():
                log.warning('%s not in cache, skipped', pid)
                continue
            m = json.loads(mfile.read_text())
            if not (keep(m['full']['kernel'], kernels) and keep(m['full']['manufacturer'], manufacturers)):
                continue
            p = len(self.pids)
            self.pids.append(pid)
            self.index.extend((p, z) for z in range(m['n_slices']))
        self._vols: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        log.info('Dataset_ChestLDCT: %d patients, %d slices, patch=%s', len(self.pids), len(self.index), patch_size)

    def _volumes(self, p: int):
        if p not in self._vols:
            pid = self.pids[p]
            self._vols[p] = (np.load(self.cache / f'{pid}_low.npy', mmap_mode='r'),
                             np.load(self.cache / f'{pid}_full.npy', mmap_mode='r'))
        return self._vols[p]

    normalize = staticmethod(hu_to_unit)
    denormalize = staticmethod(unit_to_hu)

    def _crop_origin(self, target_hu: np.ndarray) -> tuple[int, int]:
        H, W = target_hu.shape
        P = self.patch

        if torch.rand(1).item() < self.fg_prob:
            ys, xs = np.nonzero(target_hu[::4, ::4] > self.fg_hu)
            if len(ys):
                ###
                i = torch.randint(len(ys), (1,)).item()
                cy, cx = ys[i] * 4, xs[i] * 4
                y0 = int(np.clip(cy - P//2, 0, H - P))
                x0 = int(np.clip(cx - P//2, 0, W - P))
                return y0, x0

        return (torch.randint(H - P + 1, (1,)).item(),
                torch.randint(W - P + 1, (1,)).item())

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        p, z = self.index[i]
        low, full = self._volumes(p)
        zz = np.clip(np.arange(z - self.k, z + self.k + 1), 0, low.shape[0] - 1)

        tgt = np.asarray(full[z])
        if self.patch is not None:
            y0, x0 = self._crop_origin(tgt)
            y1, x1 = y0 + self.patch, x0 + self.patch
            inp = np.asarray(low[zz, y0:y1, x0:x1])
            tgt = tgt[y0:y1, x0:x1]
        else:
            y0 = x0 = 0
            inp = np.asarray(low[zz])

        mask = (inp[self.k] > FOV_HU)[None].astype(np.float32)
        inp = self.normalize(inp.astype(np.float32))
        tgt = self.normalize(tgt.astype(np.float32))[None]

        if self.augment:
            r = torch.randint(0, 2, (3,)).tolist()
            if r[0]:
                inp, tgt, mask = inp[:, :, ::-1], tgt[:, :, ::-1], mask[:, :, ::-1]
            if r[1]:
                inp, tgt, mask = inp[:, ::-1, :], tgt[:, ::-1, :], mask[:, ::-1, :]
            if r[2]:
                inp = inp[::-1]
            k = torch.randint(0, 4, (1,)).item()
            inp, tgt, mask = np.rot90(inp, k, (1, 2)), np.rot90(tgt, k, (1, 2)), np.rot90(mask, k, (1, 2))

        return {
            'input': torch.from_numpy(np.ascontiguousarray(inp)),
            'target': torch.from_numpy(np.ascontiguousarray(tgt)),
            'mask': torch.from_numpy(np.ascontiguousarray(mask)),
            'patient_id': self.pids[p],
            'slice_idx': z,
            'origin': torch.tensor([y0, x0])
        }

@dataclass
class PatientRecord:
    patient_id: str
    full_dir: str
    low_dir: str

def _find_dose_dir(study_dir: Path, kind: str) -> Optional[Path]:
    pat = re.compile(rf'{kind}[\s_-]*dose', re.IGNORECASE)
    hits = [d for d in study_dir.iterdir() if d.is_dir() and pat.search(d.name)]
    if len(hits) > 1:
        log.warning("%s: several '%s dose' folders, using %s", study_dir, kind, hits[0].name)

    return hits[0] if hits else None

def scan_patients(data_root: str | os.PathLike, prefix: str='C') -> list[PatientRecord]:
    root = Path(data_root)
    if (root / 'ldct_and_projection_data').is_dir():
        root = root / 'ldct_and_projection_data'

    records, skipped = [], []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith(prefix)):
        found = None

        for study in sorted(d for d in pdir.iterdir() if d.is_dir()):
            full, low = _find_dose_dir(study, 'full'), _find_dose_dir(study, 'low')
            if full and low:
                found = PatientRecord(pdir.name, str(full), str(low))
                break
        if found: records.append(found)
        else: skipped.append(pdir.name)

    log.info('Found %d paired chest patients; skipped %d without low-dose: %s',
             len(records), len(skipped), skipped)
    return records

def _z_of(ds) -> float:
    if 'ImagePositionPatient' in ds:
        return float(ds.ImagePositionPatient[2])
    return float(ds.SliceLocation)

def load_series(series_dir: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, dict]:
    slices, uids, h0 = [], set(), None
    for f in Path(series_dir).iterdir():
        if not f.is_file():
            continue
        try:
            ds = dcmread(f, force=True)
        except Exception:
            continue
        if 'PixelData' not in ds or str(getattr(ds, 'SOPClassUID', '')) != CT_IMAGE_STORAGE:
            continue
        m, b = _rescale(ds)
        hu = ds.pixel_array.astype(np.float32)*m + b
        slices.append((_z_of(ds), np.clip(np.round(hu), HU_MIN, HU_MAX).astype(np.int16)))
        uids.add(getattr(ds, 'SeriesInstanceUID', None))
        if h0 is None:
            del ds.PixelData
            h0 = ds
    if not slices:
        raise RuntimeError(f'No DICOM images in {series_dir}')

    if len(uids) > 1:
        log.warning('%s contains %d series; mixing them is probably wrong', series_dir, len(uids))

    slices.sort(key=lambda t: t[0])
    z = np.array([t[0] for t in slices], dtype=np.float64)
    if len(np.unique(np.round(z, 3))) != len(z):
        log.warning('%s: duplicate z positions', series_dir)

    vol = np.stack([t[1] for t in slices])
    meta ={
        'manufacturer': str(getattr(h0, 'Manufacturer', '')),
        'model': str(getattr(h0, 'ManufacturerModelName', '')),
        'kernel': str(getattr(h0, 'ConvolutionKernel', '')),
        'slice_thickness': float(getattr(h0, 'SliceThickness', 0) or 0),
        'pixel_spacing': [float(v) for v in getattr(h0, 'PixelSpacing', [0,0])],
        'kvp': float(getattr(h0, 'KVP', 0) or 0)
    }
    return vol, z, meta

###
def load_2p5d_stack(
        dcm_paths: Sequence[str | os.PathLike],
        device: Optional[str | torch.device]=None
) -> dict:
    assert len(dcm_paths) % 2 == 1, 'number of slices must be odd'
    slices = []
    for f in dcm_paths:
        ds = dcmread(f, force=True)
        sop = str(getattr(ds, 'SOPClassUID', ''))
        if sop != CT_IMAGE_STORAGE:
            raise ValueError(f'{f}: SOPClassUID {sop} is not CT Image Storage')
        slices.append((_z_of(ds), ds))

    slices.sort(key=lambda t: t[0])
    z = np.array([t[0] for t in slices], dtype=np.float64)
    dss = [t[1] for t in slices]
    c = len(dss) // 2

    slopes, intercepts = np.array([_rescale(ds) for ds in dss], np.float32).T
    ds_center = dss[c]

    hu = [ds.pixel_array.astype(np.float32)*m + b for ds, m, b in zip(dss, slopes, intercepts)]
    if len({h.shape for h in hu}) != 1:
        raise ValueError(f'slice shapes differ: {[h.shape for h in hu]}')
    hu = np.stack(hu)

    mask = (hu[c] > FOV_HU).astype(np.float32)[None, None]
    x = hu_to_unit(hu)[None]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return {
        'x': torch.from_numpy(np.ascontiguousarray(x)).to(device),
        'mask': torch.from_numpy(mask).to(device),
        'slopes': slopes,
        'intercepts': intercepts,
        'z': z,
        'ds_center': ds_center
    }

###
def save_center_dcm(
        pred,
        ds_center,
        path: str | os.PathLike,
        keep_outside_fov: bool=True
) -> None:
    if torch.is_tensor(pred):
        pred = pred.detach().float().cpu().numpy()
    pred = np.asarray(pred, np.float32).squeeze()
    hu = unit_to_hu(pred)

    ds = copy.deepcopy(ds_center)
    m, b = _rescale(ds)
    if keep_outside_fov:
        orig = ds.pixel_array.astype(np.float32)*m + b
        hu = np.where(orig > FOV_HU, hu, orig)

    signed = int(getattr(ds, 'PixelRepresentation', 0)) == 1
    bits = int(getattr(ds, 'BitsStored', 16))
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    raw = np.clip(np.round((hu - b) / m), lo, hi).astype(np.int16 if signed else np.uint16)

    ds.set_pixel_data(raw, photometric_interpretation='MONOCHROME2', bits_stored=bits)
    ds.SeriesDescription = (str(getattr(ds, 'SeriesDescription', '')) + ' denoised').strip()
    ds.save_as(path, enforce_file_format=True)

def _pair_by_z(z_full: np.ndarray, z_low: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    j = np.searchsorted(z_low, z_full)
    j = np.clip(j, 1, len(z_low) - 1) if len(z_low) > 1 else np.zeros_like(j)
    left = np.abs(z_low[np.maximum(j - 1, 0)] - z_full)
    right = np.abs(z_low[j] - z_full)
    jl = np.where(left <= right, np.maximum(j - 1, 0), j)
    ok = np.abs(z_low[jl] - z_full) <= tol

    return np.nonzero(ok)[0], jl[ok]

def build_cache(
        records: Sequence[PatientRecord],
        cache_dir: str | os.PathLike,
        z_tol: float=0.1,
        overwrite: bool=False
) -> list[str]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    done = []

    for rec in records:
        f_full, f_low, f_meta = (cache / f"{rec.patient_id}_{s}" for s in ('full.npy', 'low.npy', 'meta.json'))
        if not overwrite and f_full.exists() and f_low.exists() and f_meta.exists():
            done.append(rec.patient_id)
            continue
        try:
            vf, zf, mf = load_series(rec.full_dir)
            vl, zl, ml = load_series(rec.low_dir)
        except Exception as e:
            log.error('%s: failed to read (%s), skipped', rec.patient_id, e)
            continue
        if vf.shape[1:] != vl.shape[1:]:
            log.error('%s: in-plane size differs %s vs %s, skipped',
                      rec.patient_id, vf.shape, vl.shape)
            continue

        i_f, i_l = _pair_by_z(zf, zl, z_tol)
        if len(i_f) < len(zf) or len(i_l) < len(zl):
            log.warning('%s: %d full / %d low slices, %d matched by z',
                        rec.patient_id, len(zf), len(zl), len(i_f))
        if len(i_f) == 0:
            log.error('%s: no slices matched by z, skipped', rec.patient_id)
            continue

        dz = np.diff(zf[i_f])
        if len(dz) and (dz.max() - dz.min()) > 0.5*max(np.median(dz), 1e-3):
            log.warning('%s: uneven slice spacing after matching (%.2f-%.2f mm)',
                        rec.patient_id, dz.min(), dz.max())

        np.save(f_full, vf[i_f])
        np.save(f_low, vl[i_l])
        meta = {
            **asdict(rec),
            'full': mf,
            'low': ml,
            'n_slices': int(len(i_f)),
            'z': zf[i_f].round(3).tolist()
        }
        f_meta.write_text(json.dumps(meta, indent=1))
        done.append(rec.patient_id)
        log.info('cached %s: %d slices, %s %s',
                 rec.patient_id, len(i_f), mf['manufacturer'], mf['kernel'])

    return done

def split_patients(patient_ids: Sequence[str], ratios=(0.8, 0.1, 0.1), seed: int=21) -> dict[str, list[str]]:
    ids = sorted(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))

    return {
        'train': sorted(ids[:n_tr]),
        'val': sorted(ids[n_tr:n_tr+n_va]),
        'test': sorted(ids[n_tr+n_va:])
    }

def get_loader(
        data_root: str | os.PathLike = 'data/Chest-LDCT-data',
        split: str='train',
        batch_size: int=16,
        num_workers: int=4,
        n_slices: int=5,
        patch_size: Optional[int]=128,
        cache_dir: Optional[str | os.PathLike]=None,
        split_ratios: Sequence[float]=(0.8, 0.1, 0.1),
        seed: int=21,
        patient_ids: Optional[Sequence[str]]=None,
        **kwargs,
) -> DataLoader:
    assert split in ('train', 'val', 'test')
    cache_dir = Path(cache_dir or Path(data_root) / 'cache_2p5d')

    split_file = cache_dir / 'splits.json'
    if patient_ids is None:
        if split_file.exists():
            splits = json.loads(split_file.read_text())
        else:
            records = scan_patients(data_root)
            cached = build_cache(records, cache_dir)
            splits = split_patients(cached, split_ratios, seed)
            split_file.write_text(json.dumps(splits, indent=1))
            log.info('splits: %s', {k: len(v) for k, v in splits.items()})
        patient_ids = splits[split]

    train = split == 'train'
    ds = Dataset_ChestLDCT(
        cache_dir, patient_ids, n_slices=n_slices,
        patch_size=patch_size if train else kwargs.pop('eval_patch_size', None),
        augment=kwargs.pop('augment', train),
        **kwargs,
    )

    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, drop_last=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )