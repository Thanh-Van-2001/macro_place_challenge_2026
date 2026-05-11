# Team Thanh-Van-2001 — Submission

**Score:** AVG **1.2075** across 17 IBM ICCAD04 benchmarks (N=12 multi-start)
**Algorithm:** Tuned analytical placer + real-proxy multi-start
**Runtime:** ~122 min total on a 4-core CPU (≈7.2 min per benchmark, well under the 1 h/bench cap)
**Validity:** All 17 benchmarks VALID, 0 overlaps

## How to run

```bash
# Default (N=12 multi-start)
uv run evaluate submissions/thanhvan/placer.py --all

# Single benchmark
uv run evaluate submissions/thanhvan/placer.py -b ibm01

# Override the multi-start count
HP12_N=8 uv run evaluate submissions/thanhvan/placer.py --all
```

## What's inside

`placer.py` is a wrapper around the v-x-zhang public analytical placer
(`submissions/vxzhang/v5_rudy_w1_placer.py`, Apache 2.0). Our contribution:

1. **Tuned hyperparameters** for the v5 analytical placer:
   - `AP_RUDY_W=3.0`, `AP_ITERS=350`, `AP_LR=0.003`, `AP_DEN_W=1.5`,
     `AP_OV_START=3.0`, `AP_DEN_CARRIER=rect`
   - Discovered by sweeping each parameter on the full IBM panel.
   - Pushes single-run score from v5 default 1.2813 down to 1.2130.

2. **Real-proxy multi-start (N=12)**:
   - The v5 placer is not bit-deterministic on CPU even with `torch.manual_seed`.
     Three runs with seed=0 gave 0.9097/0.9128/0.9147 on ibm01 (~0.5% spread).
   - We exploit this by running v5 N=12 times with different seeds and
     selecting the lowest TILOS proxy cost.
   - Pushes 1.2130 down to 1.2075. (`AP_NUM_SEEDS` ensemble inside v5 picks
     by an internal surrogate which is misaligned with the real proxy —
     scored worse during our sweep.)

## Sweep summary

```
v5 default                                AVG 1.2813
+ RUDY_W: 1.0 → 3.0                       AVG 1.2630
+ ITERS: 1500 → 350                       AVG 1.2447
+ LR: 0.005 → 0.003                       AVG 1.2435
+ DEN_W: 5.0 → 1.5                        AVG 1.2297
+ OV_START: 20.0 → 3.0                    AVG 1.2173
+ DEN_CARRIER: bell → rect                AVG 1.2130
+ multi-start N=12 (real proxy select)    AVG 1.2075   ✓ FINAL

Multi-start N sweep:
  N=4:   1.2119  (42 min)
  N=6:   1.2099  (62 min)
  N=8:   1.2088  (82 min)
  N=12:  1.2075  (122 min)  ← submission default
  N=20:  1.2076  (202 min)  ← plateau, no gain
```

## Per-benchmark scores

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.9112 | 0.9976 | +8.7% |
| ibm02 | 1.2891 | 1.8370 | +29.8% |
| ibm03 | 1.1024 | 1.3222 | +16.6% |
| ibm04 | 1.1225 | 1.3024 | +13.8% |
| ibm06 | 1.3063 | 1.6187 | +19.3% |
| ibm07 | 1.2158 | 1.4633 | +16.9% |
| ibm08 | 1.2248 | 1.4285 | +14.3% |
| ibm09 | 0.9404 | 1.1194 | +16.0% |
| ibm10 | 1.1734 | 1.5009 | +21.8% |
| ibm11 | 0.9633 | 1.1774 | +18.2% |
| ibm12 | 1.4018 | 1.7261 | +18.8% |
| ibm13 | 1.0531 | 1.3355 | +21.1% |
| ibm14 | 1.3016 | 1.5436 | +15.7% |
| ibm15 | 1.3444 | 1.5159 | +11.3% |
| ibm16 | 1.2394 | 1.4780 | +16.1% |
| ibm17 | 1.4357 | 1.6446 | +12.7% |
| ibm18 | 1.5016 | 1.7722 | +15.3% |
| **AVG** | **1.2075** | 1.4578 | **+17.2%** |

## Credits

Algorithm based on `v-x-zhang/macro-place-challenge-2026` v5 (Apache 2.0).
Hyperparameter tuning and multi-start wrapper by Thanh-Van-2001.
