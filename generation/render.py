# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stroke5_to_xy(stroke5: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = stroke5[:, 0]
    dy = stroke5[:, 1]
    p = stroke5[:, 2:5].argmax(axis=1)  # 0/1/2
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y, p


def plot_stroke(stroke5: np.ndarray, save_path: Optional[str] = None, title: Optional[str] = None):
    x, y, p = stroke5_to_xy(stroke5)

    plt.figure()
    if title:
        plt.title(title)

    start = 0
    for i in range(len(x)):
        if p[i] == 1 or p[i] == 2:  # pen up or end
            if i >= start:
                plt.plot(x[start:i+1], -y[start:i+1])
            start = i + 1
        if p[i] == 2:
            break

    plt.axis("equal")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
        plt.close()
    else:
        return plt.gcf()
