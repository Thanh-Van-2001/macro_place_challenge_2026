# Team Talyxion — Submission

**Score:** AVG **1.1531** across 17 IBM ICCAD04 benchmarks
**Algorithm:** Multi-start analytical base → numerical-gradient descent → coordinate-descent / swap polish, on a fast re-implementation of the contest proxy
**Runtime:** ≈17 min per benchmark on a 4-core CPU (well under the 1 h/bench cap)
**Validity:** all 17 benchmarks VALID, 0 overlaps

## How to run

```bash
uv run evaluate submissions/talyxion/placer.py --all
uv run evaluate submissions/talyxion/placer.py -b ibm01
```

Env knobs (defaults): `HP21_N=10` multi-start count, `HP21_GD_SEC=650`
numerical-gradient budget, `HP21_CD_SEC=750` coordinate-descent budget.

## Pipeline (`placer.py` = `HybridV21`)

1. **Multi-start analytical base (N=10).** The v5 analytical placer (RUDY +
   ePlace differentiable loss) from the public `v-x-zhang` fork (Apache 2.0),
   run with hyper-parameters we tuned on the IBM panel
   (`AP_RUDY_W=6.0, AP_DEN_W=3.0, AP_ITERS=350, AP_LR=0.003,
   AP_OV_START=3.0, AP_DEN_CARRIER=rect`). The v5 base is not bit-deterministic
   on CPU; we run it 10× and keep the lowest *true* TILOS-proxy placement.

2. **Numerical-gradient descent (`numgrad_refine`).** Finite-difference
   gradient of the real contest proxy for every movable hard macro, then a
   projected (overlap-resolved) step along the negative gradient with
   backtracking line search — coordinated moves coordinate descent cannot make.

3. **Coordinate-descent + swap polish (`cd_polish`).** Per-macro multi-scale
   Gaussian candidate moves accepted on strict improvement with zero overlap,
   plus a similar-area macro swap phase.

Our own code: `fast_congestion.py` + `fasteval.py` (a faithful Numba/numpy
re-implementation of the TILOS proxy — full proxy ~3 ms vs ~1.6 s, verified to
~1e-3), `v21_numgrad.py` (= `placer.py`, the refinement pipeline). The fast
evaluators only change search SPEED — the final placement is scored by the
unmodified TILOS evaluator.

## Per-benchmark scores

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

## Lineage / other files

`v16_cd.py`, `v19_sa.py`, `v20_cycle.py`, `v21_numgrad.py` are development
iterations kept for record. `placer.py` is a copy of `v21_numgrad.py` (the
final pipeline) with the tuned defaults baked in.

## Credits

The analytical base placer is from `v-x-zhang/macro-place-challenge-2026`
(`submissions/vxzhang/v5_rudy_w1_placer.py`, Apache 2.0). Our contributions —
the hyperparameter tuning, the real-proxy multi-start, the fast Numba proxy
evaluators, and the numerical-gradient + coordinate-descent + swap refinement
— are our own.
