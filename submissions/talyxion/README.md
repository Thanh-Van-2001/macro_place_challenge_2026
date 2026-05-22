# Team Talyxion — Submission

**Score:** AVG **0.9725** across 17 IBM ICCAD04 benchmarks
**Algorithm:** Cyclic gradient descent on a differentiable smooth approximation of the TILOS proxy, alternating with coordinate-descent, then a cold simulated-annealing final stage on the exact fast proxy
**Runtime:** ≈53 min per benchmark on a 4-core CPU (under the 1 h/bench cap)
**Validity:** all 17 benchmarks VALID, 0 overlaps

## How to run

```bash
uv run evaluate submissions/talyxion/placer.py --all
uv run evaluate submissions/talyxion/placer.py -b ibm01
```

## Pipeline (`placer.py` = `HybridV26`)

The TILOS proxy is piecewise-constant in macro coordinates (floor() cell
assignment) so it has no usable gradient. `HybridV26`:

1. **Smooth proxy (`tx_smooth.py`).** A torch-differentiable component-wise
   analog of the TILOS PlacementCost: LSE weighted-average HPWL; exact
   relu-clipped area-overlap density; L/T-route congestion with 3-pin
   T-Steiner median-row routing + macro blockage + 5-tap smoothing; power-mean
   top-K. Tracks the TILOS proxy with high rank correlation.

2. **Cyclic smooth-GD ↔ CD.** Adam GD on the smooth proxy over all movable
   macros with `gamma` annealed soft→sharp; legalize hard macros; then cycle
   coordinate-descent polish → warm-restart GD → legalize. Each GD restart
   escapes the basin CD settled into.

3. **Cold simulated-annealing final stage (`tx_sa.py`).** Single-macro
   random-displacement SA on the fast exact proxy at a very cold temperature
   (near-greedy with rare uphill escapes), filling the remaining per-benchmark
   budget. It moves the **soft** macros (cell clusters that dominate density +
   congestion, never touched by hard-macro CD) ~half the time, does millions of
   cheap random trials, re-checks zero overlaps, and only accepts strict exact
   proxy improvements. This stage took the panel from 1.0623 (v24) to 0.9725.

A validity guard falls back to an always-valid analytical base + refinement if
the smooth-GD placement cannot be legalized; the SA stage runs only on a
zero-overlap placement.

Our own code: `tx_smooth.py` (smooth proxy), `v24_smooth.py` (= the cyclic
placer in `placer.py`), `tx_sa.py` (cold-SA final stage), `fast_congestion.py`
+ `fasteval.py` (Numba/numpy re-implementation of the TILOS proxy, ~3 ms vs
~1.6 s), `v21_numgrad.py` (numerical-gradient / CD refinement). The fast
evaluators and smooth proxy only change search speed; the result is scored by
the unmodified TILOS evaluator.

## Per-benchmark scores

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.7519 | 0.9976 | +24.6% |
| ibm02 | 0.9504 | 1.8370 | +48.3% |
| ibm03 | 0.8515 | 1.3222 | +35.6% |
| ibm04 | 0.8884 | 1.3024 | +31.8% |
| ibm06 | 1.0191 | 1.6187 | +37.0% |
| ibm07 | 0.9777 | 1.4633 | +33.2% |
| ibm08 | 0.9892 | 1.4285 | +30.8% |
| ibm09 | 0.7545 | 1.1194 | +32.6% |
| ibm10 | 1.0490 | 1.5009 | +30.1% |
| ibm11 | 0.7671 | 1.1774 | +34.9% |
| ibm12 | 1.1046 | 1.7261 | +36.0% |
| ibm13 | 0.8346 | 1.3355 | +37.5% |
| ibm14 | 1.0861 | 1.5436 | +29.6% |
| ibm15 | 1.0558 | 1.5159 | +30.3% |
| ibm16 | 1.0080 | 1.4780 | +31.8% |
| ibm17 | 1.2463 | 1.6446 | +24.2% |
| ibm18 | 1.1985 | 1.7722 | +32.4% |
| **AVG** | **0.9725** | 1.4578 | **+33.3%** |

## Lineage

`v16_cd.py`, `v19_sa.py`, `v20_cycle.py`, `v21_numgrad.py` are earlier
iterations (v23 = 1.1531). `placer_v23_backup.py` is the v23 entry kept as a
fallback. `placer.py` = `v24_smooth.py` (cyclic smooth-GD) + the `tx_sa.py`
cold-SA final stage = v26, the current best.

## Credits

The analytical base placer used by the validity-fallback path is from
`v-x-zhang/macro-place-challenge-2026` (`submissions/vxzhang/v5_rudy_w1_placer.py`,
Apache 2.0). The smooth proxy, cyclic smooth-GD placer, cold-SA final stage,
fast Numba evaluators and numerical-gradient / CD refinement are our own.
