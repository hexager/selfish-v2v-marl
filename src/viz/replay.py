"""Record episodes and write an HTML replay (and optionally a GIF).

Scripted drivers (look at the environment before training):
  python -m src.viz.replay --config configs/tj_medium.yaml --driver oracle --episodes 5 --out replays/oracle.html
Trained run:
  python -m src.viz.replay --run experiments/runs/tj/medium_b1_s0 --episodes 5 --out replays/b1.html --gif replays/b1.gif
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from src.envs.traffic_junction import TrafficJunctionEnv
from src.viz.record import PopulationDriver, ScriptedDriver, env_meta, record_episode

TEMPLATE = os.path.join(os.path.dirname(__file__), "viewer_template.html")


def write_html(meta, episodes, out, title="Junction Replay"):
    data = json.dumps(dict(title=title, meta=meta, episodes=episodes), separators=(",", ":"))
    data = data.replace("</", "<\\/")
    html = open(TEMPLATE).read().replace("/*__DATA__*/", data)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="env config (needed for scripted drivers)")
    ap.add_argument("--run", help="trained run directory (uses its manifest config and bits)")
    ap.add_argument("--driver", default="oracle", help="random | always_gas | oracle (ignored with --run)")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000, help="first evaluation seed")
    ap.add_argument("--add_rate", type=float, default=None, help="default: end of the training curriculum")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    if a.run:
        from src.train_tj import load_policies
        cfg = json.load(open(os.path.join(a.run, "manifest.json")))["config"]
        pols, ck = load_policies(a.run)
        bits = ck["bits"]
        driver = PopulationDriver(pols, ck["pop_mode"], seed=a.seed, greedy=a.greedy)
        title = a.title or f"Junction Replay · {os.path.basename(a.run)}"
    else:
        cfg = yaml.safe_load(open(a.config))
        bits = a.bits
        driver = ScriptedDriver(a.driver, seed=a.seed)
        title = a.title or f"Junction Replay · {a.driver}"
    env = TrafficJunctionEnv(bits=bits, **cfg["env"])
    add_rate = a.add_rate if a.add_rate is not None else cfg.get("curriculum", {}).get("add_rate_end")
    eps = [record_episode(env, driver, seed=a.seed + i, add_rate=add_rate) for i in range(a.episodes)]
    meta = env_meta(env)
    print("wrote", write_html(meta, eps, a.out, title))
    for i, e in enumerate(eps):
        print(f"  episode {i + 1}: seed {e['seed']}  success {e['summary']['success']}  "
              f"cars {e['summary']['spawned']}  exited {e['summary']['completed']}  collided {e['summary']['cars_collided']}")
    if a.gif:
        from src.viz.gif import write_gif
        print("wrote", write_gif(meta, eps[0], a.gif))


if __name__ == "__main__":
    main()
