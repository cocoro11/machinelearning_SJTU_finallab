# -*- coding: utf-8 -*-
import numpy as np


def _truncate_at_eos(stroke5: np.ndarray) -> np.ndarray:
    p = stroke5[:, 2:5].argmax(axis=1)
    eos = np.where(p == 2)[0]
    if len(eos) > 0:
        return stroke5[:eos[0] + 1]
    return stroke5


def stroke_count(stroke5: np.ndarray) -> int:
    s = _truncate_at_eos(stroke5)
    p = s[:, 2:5].argmax(axis=1)
    # count pen-up events + 1
    return int(np.sum(p == 1) + 1)


def stroke_count_error(gen: np.ndarray, ref: np.ndarray) -> int:
    return abs(stroke_count(gen) - stroke_count(ref))


def total_path_length(stroke5: np.ndarray) -> float:
    s = _truncate_at_eos(stroke5)
    d = s[:, 0:2]
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def total_path_length_error(gen: np.ndarray, ref: np.ndarray) -> float:
    return abs(total_path_length(gen) - total_path_length(ref))
