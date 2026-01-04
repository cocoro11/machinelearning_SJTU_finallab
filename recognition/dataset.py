
# -*- coding: utf-8 -*-
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import io
import matplotlib.pyplot as plt # Still needed for some fallback? No we removed it.
import PIL.ImageOps
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class RecConfig:
    data_dir: str = "data/QuickDraw_generation" # using the .npz folder
    classes: Optional[List[str]] = None
    split: str = "train"
    image_size: int = 96
    max_samples: Optional[int] = None
    seed: int = 42

def stroke3_to_image(stroke3: np.ndarray, image_size: int = 96) -> Image.Image:
    """
    Convert (dx, dy, pen_up) sequence to a PIL Image using ImageDraw (Faster than matplotlib).
    """
    # Cumsum to get absolute coordinates
    s = stroke3.astype(np.float32)
    # stroke3: (dx, dy, pen)
    
    abs_x, abs_y = 0.0, 0.0
    strokes = []
    current_stroke = []

    for i in range(len(s)):
        dx, dy, p = s[i]
        abs_x += dx
        abs_y += dy
        current_stroke.append((abs_x, abs_y))
        if p == 1: # pen up
            strokes.append(current_stroke)
            current_stroke = []
    
    if current_stroke:
        strokes.append(current_stroke)
    
    # 1. Find bounds
    all_x = []
    all_y = []
    for strk in strokes:
        for x, y in strk:
            all_x.append(x)
            all_y.append(y)
            
    if not all_x:
        return Image.new("L", (image_size, image_size), 0)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    
    # 2. Scale and Offset
    # We want to fit into image_size with some padding
    padding = image_size * 0.1
    target_size = image_size - 2 * padding
    
    scale = target_size / max(width, height)
    
    # Center the sketch
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    offset_x = (image_size / 2) - center_x * scale
    offset_y = (image_size / 2) - center_y * scale

    # 3. Draw
    # White background (255), Black strokes (0) -> Matches standard QuickDraw images
    # But for MNIST-style CNN, we usually want Black BG, White Strokes.
    # Let's draw White strokes on Black BG.
    img = Image.new("L", (image_size, image_size), 0) # Black BG
    draw = ImageDraw.Draw(img)
    
    for strk in strokes:
        if len(strk) > 1:
            # Transform coords
            points = []
            for x, y in strk:
                px = x * scale + offset_x
                py = y * scale + offset_y
                points.append((px, py))
            
            draw.line(points, fill=255, width=2)
            
    return img


class QuickDrawNpzDataset(Dataset):
    """
    Replaces QuickDraw414kImageDataset. 
    Reads .npz files (sequences) and renders them as images on-the-fly.
    """
    def __init__(self, cfg: RecConfig):
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
        
        # Load indices
        self.index: List[Tuple[int, int]] = []
        self._cache: Dict[int, Dict[str, Any]] = {}
        
        for cid in range(len(self.files)):
            d = self._load_npz(cid)
            # The .npz file has ['train', 'valid', 'test']
            # Map 'val' -> 'valid'
            split_key = cfg.split
            if split_key == 'val': split_key = 'valid'
            
            if split_key not in d:
                 # fallback if needed, or raise
                 pass
            
            samples = d[split_key]
            
            # Max samples limit
            if cfg.max_samples is not None:
                samples = samples[:cfg.max_samples]
            
            # We don't store samples in RAM if huge, just index
            # But the d is loaded in cache... 
            # With 70k samples, it is heavy. 
            # For Recognition training, we might want to lazy load or just load everything if RAM allows.
            # Given we are on desktop, 17 classes * 70k is too big.
            # But we are likely training on subset (5-10 classes).
            
            for sid in range(len(samples)):
                self.index.append((cid, sid))
                
                
        # Simple transform without torchvision
        # Image is L (grayscale), we want (1, H, W) float tensor [0,1]
        self.transform = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

    def _load_npz(self, cid: int):
        if cid in self._cache: return self._cache[cid]
        path = self.files[cid]
        try:
            z = np.load(path, allow_pickle=True, encoding="latin1")
        except:
            z = np.load(path, allow_pickle=True)
        self._cache[cid] = z
        return z

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        cid, sid = self.index[idx]
        d = self._load_npz(cid)
        
        split_key = self.cfg.split
        if split_key == 'val': split_key = 'valid'
        
        stroke3 = d[split_key][sid] 
        # Convert stroke3 (array) to Image
        img = stroke3_to_image(stroke3, self.cfg.image_size)
        
        # Manual transform
        x = self.transform(img)
        return {"x": x, "y": torch.tensor(cid, dtype=torch.long)}
