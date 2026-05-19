# Team Talyxion — Submission

**Score:** AVG **1.0623** across 17 IBM ICCAD04 benchmarks
**Algorithm:** Cyclic gradient descent on a differentiable smooth approximation of the TILOS proxy, alternating with coordinate-descent, then exact-proxy polish
**Runtime:** ≈28 min per benchmark on a 4-core CPU (under the 1 h/bench cap)
**Validity:** all 17 benchmarks VALID, 0 overlaps

## How to run

```bash
uv run evaluate submissions/talyxion/placer.py --all
uv run evaluate submissions/talyxion/placer.py -b ibm01
```

## Pipeline (`placer.py` = `HybridV24`)

The TILOS proxy is piecewise-constant in macro coordinates (floor() cell
assignment) so it has no usable gradient. `HybridV24`:

1. **Smooth proxy (`tx_smooth.py`).** A torch-differentiable component-wise
   analog of the TILOS PlacementCost: LSE weighted-average HPWL; exact
   relu-clipped area-overlap density; L/T-route congestion with 3-pin
   T-Steiner median-row routing + macro blockage + 5-tap smoothing; power-mean
   top-K. Tracks the TILOS proxy with high rank correlation.

2. **Cyclic smooth-GD ↔ CD.** Adam GD on the smooth proxy over all movable
   macros with `gamma` annealed soft→sharp; legalize hard macros; then cycle
   coordinate-descent polish → warm-restart GD → legalize. Each GD restart
   escapes the basin CD settled into.

3. **Exact-proxy polish.** Final numerical-gradient + CD/swap on the fast
   exact proxy. A validity guard falls back to an always-valid analytical
   base + refinement if the smooth-GD placement cannot be legalized.

Our own code: `tx_smooth.py` (smooth proxy), `v24_smooth.py` (= `placer.py`,
the cyclic placer), `fast_congestion.py` + `fasteval.py` (Numba/numpy
re-implementation of the TILOS proxy, ~3 ms vs ~1.6 s), `v21_numgrad.py`
(numerical-gradient / CD refinement). The fast evaluators and smooth proxy
only change search speed; the result is scored by the unmodified TILOS
evaluator.

## Per-benchmark scores

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.7818 | 0.9976 | +21.6% |
| ibm02 | 1.1107 | 1.8370 | +39.5% |
| ibm03 | 0.9398 | 1.3222 | +28.9% |
| ibm04 | 0.9995 | 1.3024 | +23.3% |
| ibm06 | 1.1638 | 1.6187 | +28.1% |
| ibm07 | 1.0494 | 1.4633 | +28.3% |
| ibm08 | 1.0885 | 1.4285 | +23.8% |
| ibm09 | 0.8255 | 1.1194 | +26.3% |
| ibm10 | 1.0511 | 1.5009 | +30.0% |
| ibm11 | 0.8232 | 1.1774 | +30.1% |
| ibm12 | 1.2152 | 1.7261 | +29.6% |
| ibm13 | 0.9176 | 1.3355 | +31.3% |
| ibm14 | 1.1614 | 1.5436 | +24.8% |
| ibm15 | 1.1838 | 1.5159 | +21.9% |
| ibm16 | 1.0994 | 1.4780 | +25.6% |
| ibm17 | 1.3147 | 1.6446 | +20.1% |
| ibm18 | 1.3329 | 1.7722 | +24.8% |
| **AVG** | **1.0623** | 1.4578 | **+27.1%** |

## Lineage

`v16_cd.py`, `v19_sa.py`, `v20_cycle.py`, `v21_numgrad.py` are earlier
iterations (v23 = 1.1531). `placer_v23_backup.py` is the v23 entry kept as a
fallback. `placer.py` = `v24_smooth.py`, the current best.

## Credits

The analytical base placer used by the validity-fallback path is from
`v-x-zhang/macro-place-challenge-2026` (`submissions/vxzhang/v5_rudy_w1_placer.py`,
Apache 2.0). The smooth proxy, cyclic smooth-GD placer, fast Numba evaluators
and numerical-gradient / CD refinement are our own.
