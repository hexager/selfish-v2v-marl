# selfish-v2v-marl

Two-car intersection for studying low-bit learned communication between **individually rewarded** cars.
Independent PPO (one actor, critic and optimizer per car). Discrete channel with 2^b symbols, b ∈ {0, 1, 2, 4}.

- `SPEC.md` – environment, channel, learning and evaluation specification (with the IC3Net comparison).
- `FINDINGS.md` – what the pilot runs showed, and what to change in the methodology.

## Setup

```bash
pip install torch numpy gymnasium pettingzoo pandas matplotlib pyyaml pytest
python -m pytest tests                     # 18 tests
python scripts/ppo_cartpole_sanity.py      # PPO sanity gate (CartPole-v1 >= 195)
```

Tested with Python 3.13, torch 2.14.1, numpy (see `experiments/runs/*/manifest.json`), pettingzoo 1.27.0, gymnasium 1.4.0. CPU only.

## Run

```bash
# one run
python -m src.train --config configs/pilot_2car_c5.yaml --bits 1 --seed 0 --total_steps 2400000 --out experiments/runs/main
# sweep (2 worker processes)
python scripts/run_sweep.py --config configs/pilot_2car_c5.yaml --bits 0 1 2 4 --seeds 0 1 2 --out experiments/runs/main
# cooperative diagnostic only (never the main result)
python scripts/run_sweep.py --config configs/pilot_2car_c5.yaml --bits 1 --seeds 0 1 --reward_mode team --out experiments/runs/team_diag
# evaluation, interventions, message stats, cross-play, learning curves
python scripts/analyze.py --runs experiments/runs/main --out experiments/results/main
```

Speed on one CPU thread: about 3,500–4,000 env steps/s for 2 cars. A 2.4M-step run takes 9–12 minutes.

## Layout

```
configs/      pilot_2car.yaml (collision −2), pilot_2car_c5.yaml (collision −5, current), pilot_2car_vision1.yaml (negative control)
src/envs/     intersection.py (PettingZoo ParallelEnv)
src/agents/   scripted.py (random, always-GAS, local right-of-way, oracle)
src/rl/       ppo.py (per-car policy, GAE, PPO update)
src/          train.py, evaluate.py
scripts/      run_sweep.py, analyze.py, ppo_cartpole_sanity.py
tests/        test_env.py, test_ppo.py
experiments/  runs/<group>/b{bits}_s{seed}/ (train_log.csv, policies.pt, manifest.json), results/<group>/ (CSV + plots), *.log
```
