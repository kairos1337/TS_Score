import numpy as np
from copy import deepcopy
from tqdm import tqdm

from make_data import synth_pair, make_dynamic_segment, Config
import pickle


def generate_large_dataset(cfg_list,
                           base_len=60,
                           n_bases=10_000,
                           rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    bases, variants, scores = [], [], []

    for _ in tqdm(range(n_bases), desc="Generating dataset"):
        base = make_dynamic_segment(base_len, rng=rng)
        bases.append(base)

        row_vars, row_scores = [], []

        for cfg in cfg_list:
            local_cfg = deepcopy(cfg)
            local_cfg.rng = rng
            _, var, score = synth_pair(base, local_cfg)
            row_vars.append(var)
            row_scores.append(score)

        variants.append(row_vars)
        scores.append(row_scores)

    return np.array(bases), variants, np.array(scores)


def save_dataset(bases, variants, scores, path="dataset.pkl"):
    with open(path, "wb") as f:
        pickle.dump({
            "bases": bases,
            "variants": variants,
            "scores": scores
        }, f)
    print(f"Dataset saved to: {path}")

base_cfg = Config()                     # כפי שהגדרת
ladder = [
    dict(noise_max=1.20, noise_min=1.10, spike_max=1, warp_max=0.10),  # M1 – כמעט זהה
    dict(noise_max=2.30, noise_min=2.20, spike_max=1, warp_max=0.20),  # M2
    dict(noise_max=3.40, noise_min=3.30, spike_max=2, warp_max=0.30),  # M3
    dict(noise_max=5.50, noise_min=4.4, spike_max=3, warp_max=0.40),  # M4
    dict(noise_max=8.60, noise_min=5.50, spike_max=4, warp_max=0.55),  # M5 – הכי שונה
]
cfgs = []
for lvl in ladder:
    cfg_i = deepcopy(base_cfg)
    cfg_i.noise_max = lvl["noise_max"]
    cfg_i.noise_min = lvl["noise_min"]
    cfg_i.spike_max = lvl["spike_max"]
    cfg_i.warp_max  = lvl["warp_max"]
    cfgs.append(cfg_i)
bases, vars_, scrs = generate_large_dataset(cfgs,
                                             base_len=8,
                                             n_bases=100_000,
                                             rng_seed=123)

save_dataset(bases, vars_, scrs, path="synthetic_similarity_dataset2.pkl")

