"""Run the capacity sweep in parallel processes.
python scripts/run_sweep.py --config configs/pilot_2car.yaml --bits 0 1 2 4 --seeds 0 1 2 --out experiments/runs/pilot
"""
import argparse
import copy
import sys
from multiprocessing import Pool

import yaml

sys.path.insert(0, ".")
from src.train import train  # noqa: E402


def job(args):
    cfg, b, s, out = args
    train(cfg, b, s, out, verbose=True)
    return b, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reward_mode", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.reward_mode:
        cfg["env"]["reward_mode"] = a.reward_mode
    jobs = [(copy.deepcopy(cfg), b, s, a.out) for s in a.seeds for b in a.bits]
    with Pool(a.workers) as p:
        for b, s in p.imap_unordered(job, jobs):
            print(f"done b={b} seed={s}", flush=True)
