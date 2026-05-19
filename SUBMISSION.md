# Our Submission — `HybridV24` (team Talyxion)

**Team:** Thanh-Van-2001 (Talyxion)
**Entry point:** `submissions/talyxion/placer.py`
**Algorithm:** Cyclic gradient descent on a differentiable smooth approximation of the TILOS proxy, alternating with coordinate-descent escape, then exact-proxy polish
**Score:** AVG **1.0623** across 17 IBM ICCAD04 benchmarks
**Runtime:** ≈28 min per benchmark on a 4-core CPU (under the 1 h/bench cap)
**Validity:** all 17 benchmarks VALID, 0 overlaps

## Result vs baselines

| | Score | Δ vs us |
|---|---:|---:|
| **Ours (HybridV24)** | **1.0623** | — |
| RePlAce baseline | 1.4578 | +27.1% |
| SA baseline | 2.1251 | +50.0% |

## Per-benchmark detail

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

## Algorithm

The TILOS proxy (`1.0·WL + 0.5·density + 0.5·congestion`) is piecewise-constant
in macro coordinates — every term runs through `floor(...)` cell assignment —
so it has no usable gradient and analytical placement cannot attack it
directly. `HybridV24` builds a **torch-differentiable smooth approximation**
of the proxy and does gradient descent on that.

### 1. Smooth differentiable proxy (`tx_smooth.py`)

A component-by-component smooth analog of the TILOS `PlacementCost`:
- **WL** — pin-level weighted-average HPWL via log-sum-exp (exact as the
  sharpness `gamma` grows).
- **density** — exact per-cell area-overlap grid, relu-clipped so it is smooth
  in macro position; top-K via power-mean.
- **congestion** — L/T-route demand (driver row-band × sink column-span), with
  3-pin nets routed at their median row to match TILOS T-Steiner routing,
  plus macro routing blockage, 5-tap smoothing, and a power-mean top-5%.

Verified to track the TILOS proxy with high rank correlation, so its gradient
points in the TILOS descent direction.

### 2. Cyclic smooth-GD ↔ coordinate-descent

- Adam GD on the smooth proxy over all movable macros, with the WL sharpness
  `gamma` annealed soft → sharp so the early phase explores and the late phase
  resolves the true HPWL bounding boxes.
- Legalize hard macros (overlap resolution; stochastic jitter escape for dense
  clusters).
- Cycle: coordinate-descent polish → warm-restart GD → legalize. Each GD
  re-optimisation escapes the basin the previous CD settled into.

### 3. Exact-proxy polish

A final numerical-gradient + coordinate-descent / swap pass on the fast exact
proxy evaluator. If the smooth-GD placement cannot be legalized to zero
overlaps, a validity guard falls back to an always-valid analytical
base + refinement pipeline.

All code here is our own. The fast evaluators (`fast_congestion.py`,
`fasteval.py` — a Numba/numpy re-implementation of the TILOS PlacementCost,
~3 ms vs ~1.6 s) and the smooth proxy only change search SPEED / smoothness;
the final placement is scored by the unmodified TILOS evaluator.

## How to run

```bash
uv run evaluate submissions/talyxion/placer.py --all
uv run evaluate submissions/talyxion/placer.py -b ibm01
```

Env knobs (defaults): `HP24_DW`/`HP24_CW` smooth density/congestion weights,
`HP24_GD1_STEPS`/`HP24_GD2_STEPS` GD step counts, `HP24_CYCLES` cycle count,
`HP24_CD_SEC`/`HP24_NG_SEC`/`HP24_FCD_SEC` polish budgets.

## Progression

v12 1.2075 → v21 1.1593 → v23 1.1531 (analytical base + numerical-gradient /
CD refinement) → **v24 1.0623** (differentiable smooth-proxy cyclic GD).

## Credits

The analytical base placer used by the validity-fallback path
(`submissions/vxzhang/v5_rudy_w1_placer.py`) is from the public `v-x-zhang`
fork of this repository (Apache 2.0). The smooth proxy, the cyclic smooth-GD
placer, the fast Numba proxy evaluators and the numerical-gradient / CD
refinement are our own.

## Reproducibility

All runs were on a 4-core CPU server (15 GiB RAM, no GPU). Total runtime for
the 17-benchmark panel ≈ 8 h.
