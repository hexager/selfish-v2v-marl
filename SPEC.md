# Environment and Method Specification (v0, pilot)

Status: implemented and unit-tested. Coefficients are pilot values. They are not tuned results.

## 1. Why this setup (link to prior work)

The closest prior tasks are the traffic junction (TJ) in CommNet, IC3Net and TarMAC.
We read the IC3Net code (`ic3net-envs/ic3net_envs/traffic_junction_env.py`). It uses:

| Item | IC3Net TJ | This project |
| --- | --- | --- |
| Actions | GAS / BRAKE | GAS / BRAKE (same) |
| Private info | own route id (others cannot see it) | own route id (same) |
| Vision | `--vision 0` in the README TJ command | 0 (main), 1 (negative control) |
| Reward | −0.01·τ per step, −10 per car in a collision | −0.05 per tick, −5 per car in a collision, +1 on exit |
| Reward mixing | individual (`mean_ratio = 0` for IC3Net) | individual; "team" only as a diagnostic |
| Parameters | one shared model | **one separate actor, critic and optimizer per car** |
| Message | continuous hidden state + binary gate | **discrete symbol, alphabet 2^b, b ∈ {0,1,2,4}** |
| Crash | car stays on the road | crashed car leaves the episode |

We chose **private route** (not private urgency) because it is the private variable in every TJ paper in our review.
This keeps us comparable with them. It also gives a clear capacity story: a car has 4 approaches × 3 routes = 12 possible intents.
1 bit cannot carry this. 2 bits carry part of it. 4 bits can carry all of it.
Private urgency (as in the private utilities of Cao et al. 2018) is planned as variant 2.

## 2. Environment (`src/envs/intersection.py`)

- **Grid:** 8×8. Two-lane roads in rows 3–4 and columns 3–4. Right-hand traffic. Junction box = cells (3..4, 3..4).
- **Cars:** 2 (code supports up to 4). Each episode draws distinct approaches (W, N, E, S), a route per car (straight, right, left) and a start offset 0–2 cells.
- **Routes:** straight = 8 cells, right turn = 7 cells, left turn = 9 cells. A car on its last cell that chooses GAS leaves the grid ("arrived").
- **Dynamics:** all cars move at the same time. GAS = advance one cell on its route. BRAKE = stay.
- **Collision:** two cars in the same cell after a move, or two cars that swap cells. Both cars get the collision penalty and leave the episode. Other cars are not affected. The result does not depend on agent order (tested).
- **Termination:** a car terminates on arrival or collision. All cars truncate at the horizon (20 ticks).
- **Reward of car i at each tick (while active):** −0.05, plus +1 on arrival, plus the collision penalty if car i collided. No car gets reward after it terminates.

### Observation of car i (fixed size)

| Part | Size | Notes |
| --- | --- | --- |
| own approach (one-hot) | 4 | |
| own route (one-hot) | 3 | private |
| own position (row, col) / 7 | 2 | |
| own progress along route | 1 | |
| own last action | 1 | |
| tick / horizon | 1 | |
| vision window, other cars | (2v+1)² | v = 0: only own cell |
| received messages | (n−1)·2^b | one-hot per sender, fixed slot order; **absent when b = 0** |

Other cars' route, approach and position are never in car i's observation (tested), except through the vision window.

### Channel

- Each tick, car i picks a move and a symbol m_i(t) from 2^b symbols.
- **One-tick delay:** car j sees m_i(t) at t+1 (tested). No car can react to a same-tick message.
- Before the first send, and after a car leaves, its slot shows symbol 0. There is no extra "present" flag. A flag would give b > 0 agents information that b = 0 agents do not have.
- Same frequency (every tick), same latency, same broadcast scope in every condition. Capacity = b bits per message per tick.

## 3. Learning (`src/rl/ppo.py`, `src/train.py`)

- **Independent PPO.** Each car has its own actor MLP (64-64, tanh), its own critic V_i, and its own Adam optimizer.
- Each car's loss uses only its own reward r_i and its own GAE. No pooled gradients. No shared critic.
- Factorized policy: log π_i(a, m) = log π_move + log π_msg. Separate entropy bonuses for move and message.
- CleanRL-style settings: clip 0.2, GAE λ 0.95, γ 0.99, 4 epochs, 4 minibatches, linear LR anneal, 16 parallel envs × 128 steps per update.
- Logged per update: return, collision rate, arrival rate, move and message entropy, message-head gradient norm, clip fraction.

## 4. Evaluation (`src/evaluate.py`, `scripts/analyze.py`)

- **Scenario bank:** for 2 cars the scenario space is finite: 12 ordered approach pairs × 9 route pairs × 9 offset pairs = 972. We evaluate on **all** of them (2 repeats each, 1944 episodes). This replaces a held-out sample because training sees the same finite set.
- **Scripted baselines:** random, always-GAS, local right-of-way (uses vision only), full-information oracle (upper bound, not comparable in information).
- **Metrics:** fraction of episodes with a collision, arrival rate, timeout rate, mean ticks of arrived cars, mean return per car, worst-car return, welfare (sum of returns, evaluation only).
- **Message interventions** (receiver side only): mute (all symbols → 0), random symbol, fixed permutation of the alphabet.
- **Message statistics:** symbol entropy, number of symbols used, plug-in mutual information of the message with own route, own approach and own next move.
- **Cross-play:** car_0 from seed i with car_1 from seed j.

## 5. Tests

`python -m pytest tests` (18 tests): PettingZoo `parallel_api_test` for b = 0/1/2/4, path contiguity, alphabet size 2^b, no message features at b = 0, one-tick message delay, collision attribution (only involved cars), order independence, no reward after termination, private route isolation, reset determinism, GAE against a naive reference with masks, factorized log-prob, separate parameters.

`python scripts/ppo_cartpole_sanity.py`: the same PPO code must solve CartPole-v1 (mean return ≥ 195).
