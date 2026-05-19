# Our Submission — `HybridV21` (team Talyxion)

**Team:** Thanh-Van-2001 (Talyxion)
**Entry point:** `submissions/talyxion/placer.py`
**Algorithm:** Multi-start analytical base → numerical-gradient descent → coordinate-descent / swap polish, all on a fast re-implementation of the contest proxy
**Score:** AVG **1.1531** across 17 IBM ICCAD04 benchmarks
**Runtime:** ≈17 min per benchmark on a 4-core CPU (well under the 1 h/bench cap)
**Validity:** all 17 benchmarks VALID, 0 overlaps

## Result vs baselines

| | Score | Δ vs us |
|---|---:|---:|
| **Ours (HybridV21)** | **1.1531** | — |
| RePlAce baseline | 1.4578 | +20.9% |
| SA baseline | 2.1251 | +45.7% |

We beat the RePlAce baseline by **20.9%** and the SA baseline by **45.7%** on
average proxy cost.

## Per-benchmark detail

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.8738 | 0.9976 | +12.4% |
| ibm02 | 1.1986 | 1.8370 | +34.8% |
| ibm03 | 1.0545 | 1.3222 | +20.2% |
| ibm04 | 1.0399 | 1.3024 | +20.2% |
| ibm06 | 1.2677 | 1.6187 | +21.7% |
| ibm07 | 1.1800 | 1.4633 | +19.4% |
| ibm08 | 1.2028 | 1.4285 | +15.8% |
| ibm09 | 0.8867 | 1.1194 | +20.8% |
| ibm10 | 1.0381 | 1.5009 | +30.8% |
| ibm11 | 0.8806 | 1.1774 | +25.2% |
| ibm12 | 1.3525 | 1.7261 | +21.6% |
| ibm13 | 0.9805 | 1.3355 | +26.6% |
| ibm14 | 1.2285 | 1.5436 | +20.4% |
| ibm15 | 1.3313 | 1.5159 | +12.2% |
| ibm16 | 1.1777 | 1.4780 | +20.3% |
| ibm17 | 1.4002 | 1.6446 | +14.9% |
| ibm18 | 1.5086 | 1.7722 | +14.9% |
| **AVG** | **1.1531** | 1.4578 | **+20.9%** |

## Algorithm

`HybridV21` is a three-stage pipeline. All refinement code is our own; only
the analytical base is reused (see Credits).

### 1. Multi-start analytical base (N=10)

The v5 analytical placer optimises macro centres by gradient descent on a
differentiable loss `HPWL + overlap + boundary + RUDY-congestion + ePlace
density`. It is not bit-deterministic on CPU, so we run it **10 times** with
different seeds and keep the placement with the lowest *true* TILOS proxy
cost. Tuned base hyper-parameters (swept on the IBM panel):
`AP_RUDY_W=6.0, AP_DEN_W=3.0, AP_ITERS=350, AP_LR=0.003,
AP_OV_START=3.0, AP_DEN_CARRIER=rect`.

### 2. Numerical-gradient descent (`numgrad_refine`)

Coordinate descent moves one macro at a time and gets stuck coordinate-wise.
Instead we estimate, for every movable hard macro, the finite-difference
gradient of the **real contest proxy** (via our fast evaluator), then step
all macros simultaneously along the negative gradient — coordinated moves CD
cannot make. Each projected step resolves overlaps and backtracks the
learning rate if the proxy did not improve.

### 3. Coordinate-descent + swap polish (`cd_polish`)

A final pass: for each movable hard macro, try K multi-scale Gaussian
candidate moves and accept the best strict improvement with zero overlap;
plus a swap phase that exchanges positions of similar-area macro pairs.

The pipeline returns the best of the three stages, scored by the **unmodified
TILOS evaluator**. The fast evaluators only change search SPEED.

### Fast proxy evaluator (`fasteval.py`, `fast_congestion.py`)

A faithful re-implementation of the TILOS `PlacementCost`: vectorized HPWL
(matches `plc.get_cost()` exactly), Numba density, and a Numba re-implementation
of the TILOS routing-congestion model. Full proxy: ~1.6 s (TILOS) → ~3 ms.
This 500× speed-up is what makes gradient/CD refinement on the real objective
feasible. Verified to match the TILOS evaluator to ~1e-3.

## How to run

```bash
uv run evaluate submissions/talyxion/placer.py --all     # all 17
uv run evaluate submissions/talyxion/placer.py -b ibm01  # single benchmark
```

Knobs (env vars, defaults shown): `HP21_N=10` multi-start count,
`HP21_GD_SEC=650` numgrad budget, `HP21_CD_SEC=750` CD budget.

## What we tried but rejected

- **Wire-mask greedy construction** (WireMask-BBO style, full proxy-cost mask):
  a constructive greedy commits each macro myopically and cannot compete with
  the analytical base — scored ~1.42.
- **DREAMPlace integration**: built CPU-only and ran it via a Bookshelf
  converter; vanilla DREAMPlace optimises HPWL+density but not the TILOS
  routing congestion, so it scored worse than ours on congestion-heavy
  benchmarks (ibm12 ≈ 1.53).
- **Hotspot-targeted congestion refinement**, SA refinement, multi-cycle
  GD↔CD, longer analytical iterations — all converge to the same local
  optimum the pipeline already reaches.

## Credits

The analytical base placer (`submissions/vxzhang/v5_rudy_w1_placer.py`) is
from the public `v-x-zhang` fork of this repository (Apache 2.0). Our own
contributions: the fast Numba proxy evaluators (`fast_congestion.py`,
`fasteval.py`), the real-proxy multi-start selection, the numerical-gradient
refinement and the coordinate-descent + swap polish (`v21_numgrad.py` =
`placer.py`).

## Reproducibility

All runs were on a 4-core CPU server (15 GiB RAM, no GPU). Because the v5
base is non-deterministic, re-running may produce a result within ≈±0.3% of
1.1531; the N=10 multi-start keeps that variance small.
