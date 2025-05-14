import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import exp
import numpy as np
from dtaidistance import dtw
from typing import Literal
import numpy as np
from copy import deepcopy


@dataclass
class Config:
    w_shift: float = 0.05
    w_warp:  float = 0.02
    noise_max: float = 0.8          # as fraction of IQR
    w_noise:  float = 0.7
    spike_max: int = 3
    w_spike:  float = 0.05
    warp_max: float = 1.25
    seg_frac_min: float = 0.2
    seg_frac_max: float = 0.4
    rng: np.random.Generator = np.random.default_rng()

def dtw_distance(ts1: np.ndarray, ts2: np.ndarray, normalize: bool = True) -> float:
    dist = dtw.distance_fast(ts1.astype(float), ts2.astype(float))
    return dist / max(len(ts1), len(ts2)) if normalize else dist

def make_dynamic_segment(length: int,
                         base_cycles: int = 2,
                         harmonics: int = 3,
                         noise_std: float = 0.15,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """
    יוצר סדרת-זמן "חיה" באורך `length`:
    • גל בסיסי בעל ‎base_cycles‎ מחזורים.
    • ‎harmonics‎ הרמוניות אקראיות (אמפליטודה ופאזה).
    • רעש גאוסי עם סטיית-תקן ‎noise_std‎.

    Returns
    -------
    np.ndarray  –  סדרה באורך המבוקש.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ציר זמן
    t = np.linspace(0, 2 * np.pi * base_cycles, length)

    # גל בסיסי
    y = np.sin(t)

    # הוספת הרמוניות אקראיות
    for h in range(2, harmonics + 2):          # למשל h = 2 … 4
        amp   = rng.uniform(0.2, 0.6)
        phase = rng.uniform(0, 2 * np.pi)
        y += amp * np.sin(h * t + phase)

    # רעש
    y += rng.normal(0, noise_std, size=length)
    return y


def replace_with_average_np(
    a: np.ndarray,
    idx: int,
    n: int,
    mode: Literal["backward", "forward", "around"] = "around",
) -> np.ndarray:
    """
    Collapse a block of `n` samples in `a` into their mean and return
    a new NumPy array with that block reduced to one value.
    around   – ⌊n/2⌋ on each side of idx (current element included).
    Returns
    -------
    np.ndarray  Same dtype as `a`, shorter by n-1 elements.

    """
    if not (0 < idx < len(a)):
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
        left  = max(0, idx - k)
        right = min(len(a), idx + k + 1)

    seg_mean = a[left:right].mean(dtype=float)
    return np.concatenate((a[:left], np.array([seg_mean], dtype=a.dtype), a[right:]))
# ----------  עיוות זמן ליניארי  ----------
def add_time_wrapping(x: np.ndarray,cfg,penalties,seg_frac: float = 0.4,
                      rng: np.random.Generator = np.random.default_rng()):

    n = len(x)
    seg_len = max(2, int(n * seg_frac))

    start   = rng.integers(0, n - seg_len + 1)
    end     = start + seg_len
    index = rng.integers(1, n)
    mode = random.choice(["around", "forward", "backward"])

    if seg_len + index >= n:
        mode = random.choice(["backward"])
    if seg_len - index < 0:
        mode = random.choice(["forward"])

    x = replace_with_average_np(x,index,seg_len,mode)
    p_warp =  (seg_frac * len(n) * cfg.w_warp)

    penalties.append(p_warp)


def inject_spikes(x: np.ndarray,
                  cfg,
                  penalties,
                  n_spikes: int,
                  mag_range=(1.5, 2.5),
                  rng=np.random.default_rng()):

    n_spikes = rng.integers(0, cfg.spike_max + 1)
    if n_spikes <= 0:
        return 0
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    mags = rng.uniform(*mag_range, size=n_spikes) * iqr
    signs = rng.choice([-1, 1], size=n_spikes)
    idx = rng.choice(n, size=n_spikes, replace=False)
    x[idx] += signs * mags
    penalty = mags[0] * (cfg.w_spike * n_spikes / cfg.spike_max)
    penalties.append(penalty)


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


    add_noise(penalties,rng,cfg)
    add_time_wrapping(x2,penalties,cfg,rng)
    inject_spikes(x2,penalties,cfg,rng)

    #print(f"spikes penalty: {penalty}")


    score = exp(-sum(penalties))
    return x1, x2, score


def build_dataset_progressive(cfg_list,
                              base_len=20,
                              n_bases=5,
                              seed=None):
    rng = np.random.default_rng(seed)
    bases, vars_, scores = [], [], []

    for _ in range(n_bases):
        base = make_dynamic_segment(base_len, rng=rng)
        bases.append(base)

        vars_row, scores_row = [], []
        for cfg in cfg_list:
            cfg_local = deepcopy(cfg)
            cfg_local.rng = rng                    # אותו RNG משותף
            _, var, sc = synth_pair(base, cfg_local)
            vars_row.append(var)
            scores_row.append(sc)
        vars_.append(vars_row)
        scores.append(scores_row)

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

    for i, base in enumerate(bases):
        for j, var in enumerate(vars_[i]):
            ax_idx = i * n_var + j + 1          # אינדקס תת-גרף
            ax = plt.subplot(n_b, n_var, ax_idx)

            ax.plot(base,  c="black", label="base")
            ax.plot(var,   c="tab:red", ls="--", label="variant")

            ax.set_title(f"s = {scores[i][j]:.2f}")
            ax.set_xticks([]); ax.set_yticks([])

            # להראות legend רק בפאנל הראשון
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
        dict(noise_max=0.10, spike_max=1,   warp_max=1.0),   # M1 – כמעט זהה
        dict(noise_max=0.25, spike_max=1,   warp_max=1.10),   # M2
        dict(noise_max=0.40, spike_max=2,   warp_max=1.25),   # M3
        dict(noise_max=0.60, spike_max=3,   warp_max=1.30),   # M4
        dict(noise_max=1.80, spike_max=4,   warp_max=1.55),   # M5 – הכי שונה
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
