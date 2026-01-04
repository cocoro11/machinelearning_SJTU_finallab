# -*- coding: utf-8 -*-
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _ensure_dxdy(stroke3: np.ndarray) -> np.ndarray:
    """
    Your sample is (T,3). We need (dx,dy,pen_up).
    Sometimes first two cols are absolute (x,y); sometimes already (dx,dy).
    Heuristic: if magnitude is large, treat as absolute and diff it.
    """
    s = stroke3.astype(np.float32)
    xy_or_dxy = s[:, :2]
    pen = (s[:, 2] > 0).astype(np.float32)  # pen_up in {0,1}

    if np.max(np.abs(xy_or_dxy)) > 50:  # likely absolute coords
        dxy = np.zeros_like(xy_or_dxy, dtype=np.float32)
        dxy[1:] = xy_or_dxy[1:] - xy_or_dxy[:-1]
    else:
        dxy = xy_or_dxy

    out = np.concatenate([dxy, pen.reshape(-1, 1)], axis=1).astype(np.float32)  # (T,3)
    return out


def stroke3_to_stroke5(stroke3: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    stroke-3: (dx,dy,pen_up)
    stroke-5: (dx,dy,p1,p2,p3) where:
      p1=pen_down, p2=pen_up, p3=end_of_sketch
    Returns unpadded sequence.
    """
    s3 = _ensure_dxdy(stroke3)
    # Truncate if longer than max_len
    T = min(len(s3), max_len)
    
    if T <= 0:
        # Empty sketch case
        out = np.zeros((1, 5), dtype=np.float32)
        out[0, 4] = 1.0
        mask = np.ones((1,), dtype=np.float32)
        return out, mask

    dxdy = s3[:T, 0:2]
    pen_up = s3[:T, 2]  # (T,)
    p2 = pen_up.reshape(-1, 1)
    p1 = (1.0 - pen_up).reshape(-1, 1)
    p3 = np.zeros((T, 1), dtype=np.float32)

    out = np.concatenate([dxdy, p1, p2, p3], axis=1) # (T, 5)
    
    # EOS logic: The original code set the last step to EOS. 
    # Usually in sketch-rnn, we append a specific EOS step, or set the last stroke as EOS.
    # Here we follow the original logic: force last step to be EOS p3=1
    out[T - 1, 2] = 0.0
    out[T - 1, 3] = 0.0
    out[T - 1, 4] = 1.0
    
    mask = np.ones((T,), dtype=np.float32)

    return out, mask


@dataclass
class GenConfig:
    data_dir: str = "data/QuickDraw_generation"
    classes: Optional[List[str]] = None  # list of file stem names without .npz
    split: str = "train"  # train/valid/test
    max_len: int = 200
    normalize: bool = False
    seed: int = 42
    max_samples: Optional[int] = None


class QuickDrawGenerationDataset(Dataset):
    """
    Your npz format:
      keys: ['test','train','valid']
      each split is dtype=object array of length N
      each item is np.ndarray of shape (T,3)
    Each npz file represents ONE category.
    """

    def __init__(self, cfg: GenConfig):
        super().__init__()
        self.cfg = cfg
        np.random.seed(cfg.seed)

        all_files = sorted(glob.glob(os.path.join(cfg.data_dir, "*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No .npz found under: {cfg.data_dir}")

        if cfg.classes is None:
            self.files = all_files
        else:
            wanted = set(cfg.classes)
            self.files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in wanted]
            if not self.files:
                raise FileNotFoundError(f"No matching classes in {cfg.data_dir}: {cfg.classes}")

        self.class_names = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
        self.class_to_id = {n: i for i, n in enumerate(self.class_names)}

        self._cache: Dict[int, Dict[str, Any]] = {}
        self.index: List[Tuple[int, int]] = []

        for cid in range(len(self.files)):
            d = self._load_npz(cid)
            samples = self._get_split(d, cfg.split)
            
            if cfg.max_samples is not None:
                samples = samples[:cfg.max_samples]

            for sid in range(len(samples)):
                self.index.append((cid, sid))

        if len(self.index) == 0:
            raise RuntimeError(f"Empty dataset. Check split={cfg.split} and npz keys.")

        # normalization (dxdy) estimated on this split
        self._mean = 0.0
        self._std = 1.0
        if cfg.normalize:
            self._estimate_norm()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cid, sid = self.index[idx]
        d = self._load_npz(cid)
        samples = self._get_split(d, self.cfg.split)

        raw = samples[sid]  # expected ndarray (T,3)
        if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(f"Unexpected sample type/shape: type={type(raw)} shape={getattr(raw,'shape',None)}")

        stroke5, mask = stroke3_to_stroke5(raw, self.cfg.max_len)

        if self.cfg.normalize:
            stroke5[:, 0:2] = (stroke5[:, 0:2] - self._mean) / (self._std + 1e-6)

        return {
            "seq": torch.from_numpy(stroke5),                 # (L,5)
            "mask": torch.from_numpy(mask),                   # (L,)
            "label": torch.tensor(cid, dtype=torch.long),     # ()
        }

    def _load_npz(self, cid: int):
        if cid in self._cache:
            return self._cache[cid]

        path = self.files[cid]

    # 关键：encoding="latin1" 用于兼容 python2 pickle/object-array try:
        try:
            z = np.load(path, allow_pickle=True, encoding="latin1")
        except TypeError:
        # 某些 numpy 版本不支持 encoding 参数
            z = np.load(path, allow_pickle=True)
    # 关键：不要立刻把 z[k] 全部读出来，保持懒加载
        self._cache[cid] = z
        return z


    def _get_split(self, d: Dict[str, Any], split: str):
        if split not in d:
            raise KeyError(f"Split '{split}' not in keys {list(d.keys())}. Expected train/valid/test.")
        return d[split]

    def _estimate_norm(self, n_samples: int = 2000):
        # estimate mean/std over dxdy from a subset of this dataset (current split)
        rng = np.random.default_rng(self.cfg.seed)
        choose = rng.choice(len(self.index), size=min(n_samples, len(self.index)), replace=False)
        vals = []
        for i in choose:
            cid, sid = self.index[i]
            d = self._load_npz(cid)
            raw = self._get_split(d, self.cfg.split)[sid]
            s3 = _ensure_dxdy(raw)
            dxdy = s3[:, 0:2].reshape(-1)
            vals.append(dxdy)
        vals = np.concatenate(vals, axis=0)
        self._mean = float(np.mean(vals))
        self._std = float(np.std(vals) + 1e-6)


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Dynamic padding
    seqs = [b["seq"] for b in batch]
    masks = [b["mask"] for b in batch]
    labels = [b["label"] for b in batch]

    # Pad sequences to the length of the longest sequence in this batch
    seq_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    mask_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0.0)
    
    label_stack = torch.stack(labels, dim=0)
    
    return {"seq": seq_padded, "mask": mask_padded, "label": label_stack}
