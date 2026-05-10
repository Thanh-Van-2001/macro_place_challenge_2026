# Our Submission — `HybridV12`

**Team:** Thanh-Van-2001
**Algorithm:** Tuned analytical placer + real-proxy multi-start — `submissions/hybrid_v12_multistart.py`
**Score:** AVG **1.2119** across 17 IBM ICCAD04 benchmarks
**Runtime:** ~42 min total on a 4-core CPU (≈2.5 min per benchmark, well under the 1 h/bench cap)
**Validity:** All 17 benchmarks VALID, 0 overlaps

## Result vs baselines and competitors

| | Score | Δ vs us |
|---|---:|---:|
| #1 vmallela (leaderboard) | 1.0109 | -16.6% |
| #2 Cezar (leaderboard) | 1.037 | -14.5% |
| **Ours (HybridV12)** | **1.2119** | — |
| #3 KLA MACH (leaderboard) | 1.2121 | +0.02% |
| #4 Hoop Dreams (leaderboard) | 1.2207 | +0.7% |
| RePlAce baseline | 1.4578 | +20.3% |
| SA baseline | 2.1251 | +75.3% |

We beat the RePlAce baseline by **16.9%** on average proxy cost and the SA
baseline by **43.0%**. Estimated leaderboard rank: **3** (just past KLA MACH).

## Per-benchmark detail

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.9134 | 0.9976 | +8.4% |
| ibm02 | 1.2906 | 1.8370 | +29.7% |
| ibm03 | 1.1067 | 1.3222 | +16.3% |
| ibm04 | 1.1203 | 1.3024 | +14.0% |
| ibm06 | 1.3111 | 1.6187 | +19.0% |
| ibm07 | 1.2203 | 1.4633 | +16.6% |
| ibm08 | 1.2286 | 1.4285 | +14.0% |
| ibm09 | 0.9402 | 1.1194 | +16.0% |
| ibm10 | 1.1781 | 1.5009 | +21.5% |
| ibm11 | 0.9696 | 1.1774 | +17.6% |
| ibm12 | 1.4057 | 1.7261 | +18.6% |
| ibm13 | 1.0591 | 1.3355 | +20.7% |
| ibm14 | 1.3033 | 1.5436 | +15.6% |
| ibm15 | 1.3553 | 1.5159 | +10.6% |
| ibm16 | 1.2463 | 1.4780 | +15.7% |
| ibm17 | 1.4405 | 1.6446 | +12.4% |
| ibm18 | 1.5126 | 1.7722 | +14.6% |
| **AVG** | **1.2119** | 1.4578 | **+16.9%** |

## Algorithm

`HybridV12` is a thin wrapper around the publicly released v5 analytical
placer from the `v-x-zhang` fork of this repo (Apache 2.0). The v5 placer
optimises macro centres by gradient descent on a differentiable loss:

```
L = w_wl × HPWL
  + w_ov × Σ pairwise_overlap_area
  + w_bd × Σ boundary_violation
  + w_rudy × RUDY_congestion_proxy
  + w_den × ePlace_density
```

Our contribution is two-fold:

### 1. Hyperparameter tuning (gives 1.2130 single-run)

Sweeping each parameter on the full 17-benchmark IBM panel produced:

| Param | Default | **Tuned** | Effect |
|---|---:|---:|---|
| `AP_RUDY_W` | 1.0 | **3.0** | Stronger RUDY congestion penalty |
| `AP_ITERS` | 1500 | **350** | Default overshoots; less is more |
| `AP_LR` | 0.005 | **0.003** | Slower, more stable convergence |
| `AP_DEN_W` | 5.0 | **1.5** | Much less density penalty |
| `AP_OV_START` | 20.0 | **3.0** | Smaller initial overlap penalty → more exploration early |
| `AP_DEN_CARRIER` | bell | **rect** | Sharper density profile |

Sweep trajectory:
```
v5 default                                AVG 1.2813   ↓
+ RUDY_W: 1.0 → 3.0                       AVG 1.2630   ↓
+ ITERS: 1500 → 350                       AVG 1.2447   ↓
+ LR: 0.005 → 0.003                       AVG 1.2435   ↓
+ DEN_W: 5.0 → 1.5                        AVG 1.2297   ↓
+ OV_START: 20.0 → 3.0                    AVG 1.2173   ↓
+ DEN_CARRIER: bell → rect                AVG 1.2130   ✓ V9 BEST
```

### 2. Real-proxy multi-start (gives 1.2119 final)

The v5 placer is **not bit-deterministic** on CPU even with `torch.manual_seed`
— some op order or BLAS path leaks randomness. Three runs of the placer with
identical config and seed=0 produced 0.9097, 0.9128, 0.9147 on ibm01
(spread ~0.5%).

`HybridV12` exploits this by running the placer **N=4 times** with different
seeds, then **selecting the placement with the lowest TILOS proxy cost**.

We considered v5's built-in `AP_NUM_SEEDS` ensemble, but it picks by an
INTERNAL surrogate (HPWL + density approximation + RUDY top-10%) which is
not perfectly aligned with the contest proxy — when we tried `AP_NUM_SEEDS=2/3`
during the sweep, scores went UP (1.2460/1.2582). Selecting by the actual
TILOS proxy avoids this misalignment.

## How to run

```bash
# Best submission (~42 min on 4-core CPU)
HP12_N=4 uv run evaluate submissions/hybrid_v12_multistart.py --all

# Single benchmark (~2.5 min)
HP12_N=4 uv run evaluate submissions/hybrid_v12_multistart.py -b ibm01

# Cheaper variant (no multi-start, ~5 min total)
uv run evaluate submissions/hybrid_v9_best.py --all
```

The wrapper sets the tuned env vars and delegates to
`submissions/vxzhang/v5_rudy_w1_placer.py`.

## Other files in this fork

- `submissions/hybrid_v9_best.py` — single-run version of the same tuned config
  (AVG 1.2130 in 5 min, the previous best). Kept for fast iteration.
- `submissions/hybrid_v11_cd.py` — earlier attempt at real-proxy CD post-processing
  (chuanqi-style). Net negative because run-to-run noise of v5 base dominated
  the per-macro CD gain. Kept as a record of what didn't work.
- `submissions/hybrid_v10_flip.py` — auto-flip post-processing wrapper.
  Helped some benchmarks but hurt average (1.2162). Kept for reference.
- `submissions/hybrid_placer_v7_fast.py` — own from-scratch Numba-JIT
  simulated-annealing placer with spatial-grid overlap detection.
  Achieves AVG 1.4758 (rank ~22). Kept for reference.
- `submissions/hybrid_placer_v8.py` — v7 + multi-thread + LNS kick + adaptive budget.
  Marginal improvement over v7. Kept for reference.
- `submissions/hybrid_placer.py` — earlier hybrid SA variant (pre-v7).
- `submissions/vxzhang/v3,v4,v5*.py` — original v-x-zhang public submissions, kept
  as reference for the algorithm we tuned. Unmodified.

## What we tried but rejected

- **Auto-flip post-processing** (jaydenpiao idea): apply x/y/xy reflections to
  the placement and pick the lowest-proxy variant. Helped 6 benchmarks, hurt 11.
  Net AVG worse (1.2162 vs 1.2130).
- **Real-proxy CD post-processing** (chuanqichen phase6 idea): for each macro,
  try ±delta moves at multiple scales and accept only if proxy improves.
  Too slow with the TILOS proxy as the oracle (~50 ms per call); the v5 base's
  run-to-run noise (~0.5%) swamped the CD's per-macro improvement (~0.05%).
- **`AP_NUM_SEEDS=2/3`** ensemble (v5 built-in): picks by a surrogate, not the
  real proxy. Scored worse (1.2460/1.2582) than single-seed.
- `AP_ITERS` outside [300, 400], `AP_OV_END=1500/3000`, `AP_DEN_W=10/0.5`,
  `AP_LR=0.002/0.0035`, `AP_BD_W=50`, `AP_EPLACE_W=1.0`,
  `AP_DEN_MODE=eplace`, `AP_ANCHOR_K=1` — all worse than the chosen values.

## Credits

The underlying analytical placer is from the `v-x-zhang` fork of this repo
(file `submissions/versions/v5_rudy_w1_placer.py`), licensed Apache 2.0
under the parent repository. Our contribution is the systematic
hyperparameter sweep, the real-proxy multi-start wrapper, and the analysis
that identified v5's run-to-run noise as exploitable signal.

## Reproducibility

All sweep experiments and benchmark runs were performed on a 4-core CPU
server (15 GiB RAM, no GPU). The contest evaluator runs on
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; on GPU hardware our
wrapper should produce comparable scores (the v5 algorithm defaults to CPU
when CUDA is unavailable, but a GPU should make each of the N runs faster).

The score `1.2119` reported here is the contest evaluator's output for one
specific N=4 run of the wrapper. Because the v5 base is non-deterministic,
re-running may produce a result in the range ≈ 1.211 – 1.215. Increasing
`HP12_N` (e.g. 8 or 16) further reduces the variance and may shave another
0.05–0.2% off the average, at proportional runtime cost.
