import random
from typing import List, Any

import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import exp
from dtaidistance import dtw
from typing import Literal
import numpy as np
from copy import deepcopy
import random
from typing import Literal

from numpy import ndarray, dtype


@dataclass
class Config:
    w_shift: float = 0.05
    w_warp:  float = 0.02
    noise_max: float = 0.8          # as fraction of IQR
    w_noise:  float = 0.2

    spike_max: int = 3
    w_spike:  float = 0.05
    warp_max: float = 0.5
    seg_frac_min: float = 0.2
    seg_frac_max: float = 0.4
    rng: np.random.Generator = np.random.default_rng()

def dtw_distance(ts1: np.ndarray, ts2: np.ndarray, normalize: bool = True) -> float:
    dist = dtw.distance_fast(ts1.astype(float), ts2.astype(float), penalty=0.1)
    return dist / max(len(ts1), len(ts2)) if normalize else dist

def make_dynamic_segment(length: int,
                         base_cycles: int = 2,
                         harmonics: int = 3,
                         noise_std: float = 0.85,
                         rng: np.random.Generator | None = None) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()
    t = np.linspace(0, 2 * np.pi * base_cycles, length)
    y = np.sin(t)
    for h in range(2, harmonics + 2):          # למשל h = 2 … 4
        amp   = rng.uniform(0.2, 0.6)
        phase = rng.uniform(0, 2 * np.pi)
        y += amp * np.sin(h * t + phase)

    y += rng.normal(0, noise_std, size=length)
    return y


def replace_with_average_np(
    a: np.ndarray,
    base : np.ndarray,
    idx: int,
    n: int,
    mode: Literal["backward", "forward", "around"] = "around",
) :
    """
    Collapse a block of `n` samples in `a` into their mean and return
    a new NumPy array with that block reduced to one value.
    around   – ⌊n/2⌋ on each side of idx (current element included).
    Returns
    -------
    np.ndarray  Same dtype as `a`, shorter by n-1 elements.

    """
    if not (0 <= idx < len(a)):
        raise ValueError(f"idx out of bounds :{idx}")
    if n < 1 or n > len(a):
        raise ValueError("n must be between 1 and len(a)")
    if mode not in {"backward", "forward", "around"}:
        raise ValueError("mode must be 'backward', 'forward', or 'around'")

    if mode == "backward":
        left  = max(0, idx - n + 1)
        right = idx + 1
    elif mode == "forward":
        left  = idx
        right = min(len(a), idx + n)
    else:  # "around"
        k     = n // 2

        left  = int(max(0, idx - k))
        right = int(min(len(a), idx + k + 1))

    base_seg = base[left:right]
    lost = np.abs(base_seg - base_seg.mean())
    dev = lost.mean()
    global_dev = np.abs(a - a.mean()).mean()
    p_vol = dev / (global_dev + 0.00001) # p_vol is the penalty for the volatility we lost during time wrapping

    seg_mean = a[left:right].mean(dtype=float)
    return np.concatenate((a[:left], np.array([seg_mean], dtype=a.dtype), a[right:])) ,p_vol

def stretch_time_series(x: np.ndarray, index: int, seg_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Stretch the time series by inserting new points around the given index,
    with values based on local average around that index.

    :param x: Original time series
    :param index: Index where stretching starts
    :param seg_len: Number of points to insert (stretch length)
    :param rng: Random number generator for noise
    :return: Stretched time series
    """
    # Calculate local average around the index
    left = max(0, index - seg_len // 2)
    right = min(len(x), index + seg_len // 2)
    local_segment = x[left:right]
    local_avg = local_segment.mean()

    # Generate stretched points with slight noise around local average
    noise = rng.normal(0, np.std(local_segment) * 0.05, size=seg_len)
    new_points = np.full(seg_len, local_avg) + noise

    # Insert stretched points into original time series
    stretched_x = np.concatenate([x[:index], new_points, x[index:]])

    return stretched_x



def exp_scaled(x: float, alpha: float) -> float:
    """
    Exponentially warp x in [0,1] back into [0,1].
    alpha: >0 controls how sharply it rises.
    """
    if not 0 <= x <= 1:
        raise ValueError("x must be in [0,1]")
    num   = np.exp(alpha * x) - 1
    denom = np.exp(alpha    ) - 1
    return num

def add_time_wrapping(x: np.ndarray,
                      base: np.ndarray,
                      cfg,
                      penalties: list[float],
                      rng: np.random.Generator = np.random.default_rng(),
                      seg_frac: float = 0.4,
                      eps: float = 1e-6
                     ) -> np.ndarray:
    """
    Randomly either shrink or stretch a segment of x, append the appropriate
    penalty (size + noise) to `penalties`, and return the new series.

    :param x:         the original time series
    :param cfg:       contains weight w_warp
    :param penalties: list to append penalty terms to
    :param rng:       random number generator
    :param seg_frac:  fraction of series length to distort
    :param eps:       small constant to avoid division by zero
    """
    ts_len   = len(x)
    seg_frac = rng.integers(10, 90)/100.0
    if seg_frac ==  0.01:
        print("here i s top")
    if seg_frac >= 0.5:
        index = int(ts_len /2 -1)

    else:
        index  = int(rng.integers(0, ts_len))  # where to apply
    mode      = None
    seg_len = max(2, int(ts_len * seg_frac))
    p_noise_abs = 1
    p_size = (seg_len / ts_len)
    p_size = exp_scaled(p_size,2.8)
    print(f"p_size: {p_size}")

    choices = []
    if index - seg_len >= 0:           choices.append("backward")
    if index + seg_len < ts_len:       choices.append("forward")
    if index - seg_len//2 >= 0 and index + seg_len//2 < ts_len:
        choices.append("around")
    mode = random.choice(choices)
    new_x, p = replace_with_average_np(x,base, index, seg_len, mode)
    print(f"p: {p}")
    p_size = (p_size  *(p+0.2))

    penalties.append(cfg.w_noise * p_noise_abs * p_size)

    return new_x

def inject_spikes(x, penalties, cfg, rng, mag_range=(1.5, 2.5), eps=1e-6):
    n_spikes = rng.integers(0, cfg.spike_max + 1)
    # no spikes → no change, no penalty
    if n_spikes == 0:
        return x

    n   = len(x)
    iqr = np.subtract(*np.percentile(x, [65, 35])) + eps
    mags = rng.uniform(*mag_range, size=n_spikes) * iqr
    signs= rng.choice([-1, 1], size=n_spikes)
    idx  = rng.choice(n, size=n_spikes, replace=False)

    # inject in-place
    x[idx] += signs * mags

    # penalty = weighted sum of all spike magnitudes (dimensionless)
    rel_spikes = mags.sum() / (iqr * cfg.spike_max)
    penalty    = cfg.w_spike * rel_spikes
    penalties.append(penalty)

    return x


def add_noise(penalties,rng,cfg, x2):
    iqr = np.subtract(*np.percentile(x2, [75, 25]))
    sigma = rng.uniform(0, cfg.noise_max) * iqr
    pct = 0.5
    n = x2.size
    k = int(round(pct * n))
    idx = rng.choice(n, k, replace=False)
    x_before = x2.copy()
    noise = rng.normal(0, sigma, size=k)
    x2[idx] += noise
    delta = np.abs(x2 - x_before).mean()
    penalty = cfg.w_noise * delta / (iqr if iqr else 1)
    penalties.append(penalty)
    return x2


def synth_pair(base: np.ndarray, cfg: Config):
    rng = cfg.rng
    x1 = base.copy()
    x2 = base.copy()
    penalties = []
    max_shift = int(len(x1) * 0.10)
    dt =  rng.integers(-max_shift, max_shift+1)
    dt = 0

    if dt < 0:
        x2 = x2[:dt]
    if dt > 0:
        x2 = x2[dt:]

    penalties.append(cfg.w_shift * abs(dt))


    #x2 = add_noise(penalties,rng,cfg,x2)
    #print(f"penalty for noise : {penalties[1]} penalty for time wrapping : {penalties[2]}")
    #x2= inject_spikes(x2,penalties,cfg,rng)
    x2 = add_time_wrapping(x2,x1,cfg,penalties,rng) # IMPORTANT : I might add noise and then delete all this part at time wrapping, so there is a situation the ts is been punished for nothing

    #print(f"spikes penalty: {penalty}")


    score = exp(-sum(penalties))
    dtw_score = dtw_distance(x1, x2)
    #score = score * exp(-dtw_score)


    return x1, x2, score


def build_dataset_progressive(cfg_list,
                              base_len=10,
                              n_bases=5,
                              seed=None):
    rng = np.random.default_rng(seed)
    bases, vars_, scores = [], [], []



    for i in range(n_bases):
        base = make_dynamic_segment(base_len, rng=rng)
        min_val = base.min()
        max_val = base.max()
        range_ = max_val - min_val + 1e-8
        base = (base - min_val) / range_
        bases.append(base)
        print(f"----------this is the {i} base----------")

        vars_row, scores_row = [], []
        for cfg in cfg_list:
            cfg_local = deepcopy(cfg)
            cfg_local.rng = rng                    # אותו RNG משותף
            _, var, sc = synth_pair(base, cfg_local)

            vars_row.append(var)
            scores_row.append(sc)
        vars_.append(vars_row)
        scores.append(scores_row)

    scores = np.array(scores)
    #scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    #scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    return bases, vars_, scores


def plot_progressive(bases, vars_, scores):
    """
    בכל שורה: base + חמש גרסאותיו.
    בכל פאנל הגרסה מצוירת באדום מקווקו, וה-base בשחור רציף.
    כותרת הפאנל – ציון ה-score.
    """
    n_b      = len(bases)
    n_var    = len(vars_[0])
    fig_w    = 4 * n_var
    fig_h    = 3 * n_b

    plt.figure(figsize=(fig_w, fig_h))

    counter = 1
    for i, base in enumerate(bases):
        for j, var in enumerate(vars_[i]):
            ax_idx = i * n_var + j + 1          # אינדקס תת-גרף
            ax = plt.subplot(n_b, n_var, ax_idx)

            ax.plot(base,  c="black", label="base")
            ax.plot(var,   c="tab:red", ls="--", label="variant")

            ax.set_title(f"s = {scores[i][j]:.2f}")
            counter += 1

            ax.set_xticks([]); ax.set_yticks([])
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------

# ------------------------------------------------------------
# מגדירים חמש קונפיגורציות – מהעדינה ביותר עד האגרסיבית

def make():
    base_cfg = Config()                     # כפי שהגדרת
    ladder = [
        dict(noise_max=0.10, spike_max=1,   warp_max=10),   # M1 – כמעט זהה
        dict(noise_max=0.25, spike_max=1,   warp_max=20),   # M2
        dict(noise_max=0.40, spike_max=2,   warp_max=35),   # M3
        dict(noise_max=0.60, spike_max=3,   warp_max=50),   # M4
        dict(noise_max=1.80, spike_max=4,   warp_max=70),   # M5 – הכי שונה
    ]
    cfgs = []
    for lvl in ladder:
        cfg_i = deepcopy(base_cfg)
        cfg_i.noise_max = lvl["noise_max"]
        cfg_i.spike_max = lvl["spike_max"]
        cfg_i.warp_max  = lvl["warp_max"]
        cfgs.append(cfg_i)
    bases, vars_, scrs = build_dataset_progressive(cfgs,
                                                   base_len=20,
                                                   n_bases=5,
                                                   seed=123)
    plot_progressive(bases, vars_, scrs)

make()

# cfg = Config()
# t = make_dynamic_segment(25)
# base_series = np.sin(t)
#
# x1, x2, score = synth_pair(base_series, cfg)
#
# print(f"dtw dis: {dtw_distance(x1,x2)}")
# print(f"Score: {score:.3f}")
#
# plt.figure(figsize=(9, 4))
# plt.plot(x1, label='Series 1 – base')
# plt.plot(x2, '--', label='Series 2 – transformed')
# plt.title(f"Synth Pair – Score = {score:.3f}")
# plt.legend()
# plt.tight_layout()
# plt.show()
