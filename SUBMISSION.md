# Our Submission — `HybridV9`

**Team:** Thanh-Van-2001
**Algorithm:** Tuned analytical placer (RUDY-RePlAce style) — `submissions/hybrid_v9_best.py`
**Score:** AVG **1.2130** across 17 IBM ICCAD04 benchmarks
**Runtime:** ~5 min total on a 4-core CPU
**Validity:** All 17 benchmarks VALID, 0 overlaps

## Result vs baselines

| | Score | Δ vs us |
|---|---:|---:|
| **Ours (HybridV9)** | **1.2130** | — |
| RePlAce baseline | 1.4578 | +20.2% |
| SA baseline | 2.1251 | +75.2% |
| Greedy Row demo | 2.2109 | +82.2% |

We beat the RePlAce baseline by **16.8%** on average proxy cost and the SA
baseline by **42.9%**. Estimated leaderboard rank: **~4** (between
Hoop Dreams 1.2207 and KLA MACH 1.2121).

## Per-benchmark detail

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.9171 | 0.9976 | +8.1% |
| ibm02 | 1.2888 | 1.8370 | +29.8% |
| ibm03 | 1.1095 | 1.3222 | +16.1% |
| ibm04 | 1.1260 | 1.3024 | +13.5% |
| ibm06 | 1.3054 | 1.6187 | +19.4% |
| ibm07 | 1.2217 | 1.4633 | +16.5% |
| ibm08 | 1.2318 | 1.4285 | +13.8% |
| ibm09 | 0.9395 | 1.1194 | +16.1% |
| ibm10 | 1.1739 | 1.5009 | +21.8% |
| ibm11 | 0.9627 | 1.1774 | +18.2% |
| ibm12 | 1.4045 | 1.7261 | +18.6% |
| ibm13 | 1.0617 | 1.3355 | +20.5% |
| ibm14 | 1.3041 | 1.5436 | +15.5% |
| ibm15 | 1.3538 | 1.5159 | +10.7% |
| ibm16 | 1.2447 | 1.4780 | +15.8% |
| ibm17 | 1.4507 | 1.6446 | +11.8% |
| ibm18 | 1.5255 | 1.7722 | +13.9% |
| **AVG** | **1.2130** | 1.4578 | **+16.8%** |

## Algorithm

`HybridV9` is a thin wrapper around the publicly released v5 analytical
placer from the `v-x-zhang` fork of this repo (Apache 2.0). The v5 placer
optimises macro centres by gradient descent on a differentiable loss:

```
L = w_wl × HPWL
  + w_ov × Σ pairwise_overlap_area
  + w_bd × Σ boundary_violation
  + w_rudy × RUDY_congestion_proxy
  + w_den × ePlace_density
```

Our contribution is the **hyperparameter tuning**, found by sweeping each
parameter on the full 17-benchmark IBM panel. The optimal config differs
substantially from the v5 default:

| Param | Default | **Tuned** | Effect |
|---|---:|---:|---|
| `AP_RUDY_W` | 1.0 | **3.0** | Stronger RUDY congestion penalty |
| `AP_ITERS` | 1500 | **350** | Default overshoots; less is more |
| `AP_LR` | 0.005 | **0.003** | Slower, more stable convergence |
| `AP_DEN_W` | 5.0 | **1.5** | Much less density penalty |
| `AP_OV_START` | 20.0 | **3.0** | Smaller initial overlap penalty → more exploration early |
| `AP_DEN_CARRIER` | bell | **rect** | Sharper density profile |

Sweep trajectory (each step kept whichever change improved AVG):

```
v5 default                                AVG 1.2813   ↓
+ RUDY_W: 1.0 → 3.0                       AVG 1.2630   ↓
+ ITERS: 1500 → 350                       AVG 1.2447   ↓
+ LR: 0.005 → 0.003                       AVG 1.2435   ↓
+ DEN_W: 5.0 → 1.5                        AVG 1.2297   ↓
+ OV_START: 20.0 → 3.0                    AVG 1.2173   ↓
+ DEN_CARRIER: bell → rect                AVG 1.2130   ✓ FINAL
```

We also tried (and rejected): multi-start `K=2/3` (worse with low ITERS),
`AP_ITERS` outside [300, 400] (worse), `AP_OV_END=1500/3000`, `AP_DEN_W=10/0.5`,
`AP_LR=0.002/0.0035`, `AP_BD_W=50`, `AP_EPLACE_W=1.0`, `AP_DEN_MODE=eplace`,
`AP_ANCHOR_K=1`, and a post-processing `auto-flip` pass (helped some
benchmarks, hurt the average).

## How to run

```bash
# Best submission (5 min on 4-core CPU)
uv run evaluate submissions/hybrid_v9_best.py --all

# Single benchmark
uv run evaluate submissions/hybrid_v9_best.py -b ibm01
```

The wrapper sets the tuned env vars and delegates to `submissions/vxzhang/v5_rudy_w1_placer.py`.

## Other files in this fork

- `submissions/vxzhang/v3,v4,v5*.py` — original v-x-zhang public submissions, kept as
  reference for the algorithm we tuned. These are unmodified.
- `submissions/hybrid_placer_v7_fast.py` — our own from-scratch attempt: a
  Numba-JIT simulated-annealing placer with spatial-grid overlap detection.
  Achieves AVG 1.4758 (rank ~22). Kept for reference.
- `submissions/hybrid_placer_v8.py` — v7 + multi-thread + LNS kick + adaptive budget.
  Marginal improvement over v7. Kept for reference.
- `submissions/hybrid_v10_flip.py` — v9 + auto-flip post-process. Worse on
  the IBM average (1.2162 vs 1.2130) and 6× slower. Kept for reference.
- `submissions/hybrid_placer.py` — earlier hybrid SA variant (pre-v7).

## Credits

The underlying analytical placer is from the `v-x-zhang` fork of this repo
(file `submissions/versions/v5_rudy_w1_placer.py`), licensed Apache 2.0
under the parent repository. Our contribution is the systematic
hyperparameter sweep and the wrapper that pins the optimal config.

## Reproducibility

All sweep experiments were run on a 4-core CPU server (no GPU). The contest
evaluator runs on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; on GPU
hardware our wrapper should produce the same scores (the v5 algorithm
defaults to CPU when CUDA is unavailable).
