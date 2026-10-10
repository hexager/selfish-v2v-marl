# Pilot findings (10 Oct 2026)

Scope: sanity check of the methodology. Two cars, 1–2 seeds per condition. These are **pilot observations, not results**.
All evaluation numbers use the full scenario bank (972 scenarios × 2 repeats = 1,944 episodes).
Figure: `experiments/results/pilot_summary.png`. Raw numbers: `experiments/results/*/eval_all.csv`, `message_stats.csv`.

## 1. Infrastructure checks — passed

- 18 unit tests pass, including PettingZoo `parallel_api_test` for b = 0/1/2/4.
- The same PPO code solves CartPole-v1 (mean return 272 over the last 20 episodes; gate ≥ 195).
- Scripted baselines behave as expected (vision 0, collision −5): always-GAS 12.3% collision episodes, return −0.06; full-information oracle 0%, return 0.644.
- Speed: ~3,500–4,000 env steps/s per CPU thread. A 2.4M-step run = 9–12 min. A full 4-bit × 5-seed sweep ≈ 4 h on 2 cores, so the laptop/Kaggle budget is fine.

## 2. Task-design findings (these change the methodology)

**F1. With vision 1, communication is useless.** At 0 bits, independent PPO already reached the oracle (return ≈ 0.64, ≈ 0% collisions) in < 0.5M steps. Cars see each other at the junction entry and resolve the conflict by sight.
→ Main config uses **vision 0**, as in the IC3Net traffic-junction command. Vision 1 stays as a negative control (`configs/pilot_2car_vision1.yaml`, log `experiments/vision1_aborted.log`).

**F2. With collision −2, cars learn to rush.** All conditions (0/1/2/4 bits) converged near always-GAS (return ≈ 0.3, ≈ 12% collisions). Waiting was barely better than rushing in expectation, so no coordination was learned and messages were unused (mute = no change).
→ Collision penalty raised to **−5** (`configs/pilot_2car_c5.yaml`). Rushing now has return −0.06, which is clearly bad.

**F3. 0.6M steps is too short.** At 0.6M steps, the b > 0 runs were behind b = 0 (more action dimensions to explore). At 2.4M steps all selfish runs plateau. Use ≥ 2.4M steps and plot learning curves, as the handoff says.

## 3. Results at collision −5, vision 0, 2.4M steps

| Condition | Seed | Collision episodes | Return / car | car_0 | car_1 |
| --- | --- | --- | --- | --- | --- |
| selfish, b=0 | 0 | 2.6% | 0.435 | 0.501 | 0.368 |
| selfish, b=0 | 1 | 2.5% | 0.438 | 0.504 | 0.372 |
| selfish, b=1 | 0 | 2.3% | 0.435 | 0.516 | 0.354 |
| selfish, b=2 | 0 | 2.5% | 0.440 | 0.507 | 0.373 |
| selfish, b=2 | 1 | 2.3% | 0.449 | 0.516 | 0.382 |
| team diagnostic, b=1 | 0 | 0.5% | 0.317 | 0.392 | 0.242 |
| team diagnostic, b=1 | 1 | 0.4% | 0.324 | 0.132 | 0.516 |

**F4. Selfish cars learn coordination without talking, through a role convention.** All selfish runs reach ≈ 0.44, far above rushing (−0.06) but well below the oracle (0.64). car_0 does better than car_1 in every selfish run (≈ 0.51 vs ≈ 0.37). One car learns to yield more. Separate networks know their own identity, so "car_1 yields" is a free convention that needs no channel. Cross-play between seeds (car_0 from seed i, car_1 from seed j) keeps the same performance (0.435–0.441), so this convention is generic.
→ This is a real alternative to communication. It must be named in the paper. It also gives a fairness story (worst-car return).

**F5. Selfish b = 1: messages have no causal effect.** Mute, random and permuted symbols do not change any metric (all within 0.004 of normal play). Message MI with own route and next move ≈ 0.

**F6. Selfish b = 2: receivers react to messages, but the messages give no benefit.** In both seeds, random or permuted symbols raise collisions (seed 0: 2.5% → 4.5% / 5.3%; seed 1: 2.3% → 6.2% / 9.2%). But muting (all symbols → 0) changes nothing, and normal play is not better than b = 0. car_1's messages carry a small amount of information about its next move (MI 0.03–0.07 bits). car_0's messages are almost constant.
→ This is **message influence without useful information**. It is a clear example of why correlation or influence alone is not evidence of a protocol. The intervention design works as intended.

**F7. Cooperative diagnostic: communication can emerge, but not reliably.** Team b = 1, seed 0: one car (car_0) sends a 1-bit signal that tracks its own next move (MI 0.58 bits) and route (0.18 bits). Muting it raises collisions from 0.5% to 3.9% and drops return from 0.32 to 0.10. Seed 1: no protocol (message entropy stays at maximum; interventions have no effect). Cross-play between the two team seeds fails (collisions 7–10%, return < 0), so protocols are seed-specific.
→ The handoff's go/no-go gate ("comm must be learnable in an easier cooperative diagnostic") is **passed for 1 of 2 seeds**. The pipeline can detect real communication when it exists.

Note on the team diagnostic: the team reward is the mean reward of **active** cars at each tick. A car that has left does not feel its partner's later cost. Fix this before using it in the paper.

## 4. What this means for the methodology

The methodology is on the right track: the environment, independent learners, delayed channel and causal interventions all work and give interpretable answers. The pilot did surface three design problems. Fix them before the full sweep:

1. **Remove the identity convention, or measure it.** Options: (a) randomize which network controls which car each episode and hide nothing else (a network then cannot rely on "I am always car_1"); (b) keep it and report it as the no-communication solution that self-interested cars find. (a) is a one-line change in `train.py`; it does not add parameter sharing (each car keeps its own network, only the seat is random). Recommended: run both.
2. **Increase the value of information.** Ideas: more cars (3–4), more start-offset variation (timing uncertainty), or a higher time cost so that waiting is expensive. Check with the oracle-vs-b=0 gap; it is 0.20 now.
3. **Help emergence without breaking self-interest.** Independent PPO with cheap-talk channels is known to learn slowly. Options that keep individual rewards: longer runs (5–10M steps), lower message entropy bonus, or a small listener/speaker auxiliary loss on each agent's own data (positive signalling / positive listening biases, Eccles et al., 2019). Report any such bias as part of the method.

Planned main sweep after these fixes: b ∈ {0, 1, 2, 4} × 5 seeds × 2.4–5M steps, plus vision-1 and team-reward controls. Estimated cost: 4–8 h on 2 CPU cores.

## 5. What is not shown

- No claim about capacity effects. With 1–2 seeds and no significant b-to-b differences, any trend is noise.
- No evidence of deception or manipulation. F6 shows influence without benefit; it is not manipulation.
- 4-car and private-urgency variants are not implemented.
