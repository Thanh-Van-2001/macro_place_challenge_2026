# Our Submission — `HybridV26` (team Talyxion)

**Team:** Thanh-Van-2001 (Talyxion)
**Entry point:** `submissions/talyxion/placer.py`
**Algorithm:** Cyclic gradient descent on a differentiable smooth approximation of the TILOS proxy, alternating with coordinate-descent escape, then a cold simulated-annealing final stage on the exact fast proxy
**Score:** AVG **0.9725** across 17 IBM ICCAD04 benchmarks
**Runtime:** ≈53 min per benchmark on a 4-core CPU (under the 1 h/bench cap)
**Validity:** all 17 benchmarks VALID, 0 overlaps

## Result vs baselines

| | Score | Δ vs us |
|---|---:|---:|
| **Ours (HybridV26)** | **0.9725** | — |
| RePlAce baseline | 1.4578 | +33.3% |
| SA baseline | 2.1251 | +54.2% |

## Per-benchmark detail

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

## Algorithm

The TILOS proxy (`1.0·WL + 0.5·density + 0.5·congestion`) is piecewise-constant
in macro coordinates — every term runs through `floor(...)` cell assignment —
so it has no usable gradient and analytical placement cannot attack it
directly. `HybridV26` builds a **torch-differentiable smooth approximation**
of the proxy, does gradient descent on that, and then escapes the resulting
basin with a cold simulated-annealing pass on the exact fast proxy.

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

### 3. Cold simulated-annealing final stage (`tx_sa.py`)

The smooth-GD + CD pipeline converges to a strong but locked-in local optimum:
coordinate-descent only nudges hard macros one axis at a time and never touches
the **soft** macros (cell clusters) that dominate density and congestion. The
final stage runs a single-macro random-displacement simulated annealing on the
fast exact proxy, at a very cold temperature (T≈1e-6, near-greedy with rare
uphill escapes), filling whatever per-benchmark time remains under the 1 h cap:

- ~50% of moves displace a **soft** macro (no hard-overlap check needed),
  ~40% displace a hard macro (rejected if it would overlap), ~10% swap two
  same-area hard macros.
- Millions of cheap random trials (≈1400 it/s on the small benchmarks) instead
  of a fixed candidate set per macro.
- Cold Metropolis acceptance keeps it a descent with rare basin escapes; the
  result is re-checked for zero overlaps and only accepted if it strictly
  improves the exact proxy.

This stage alone moved the panel from AVG 1.0623 (v24) to **0.9725** — e.g.
ibm02 1.1107→0.9504, ibm04 0.9995→0.8884, ibm11 0.8232→0.7671.

### Validity guard

If the smooth-GD placement cannot be legalized to zero overlaps, a guard falls
back to an always-valid analytical base + refinement pipeline, and the cold-SA
stage is skipped unless the incoming placement already has zero overlaps.

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
`HP24_CD_SEC`/`HP24_NG_SEC`/`HP24_FCD_SEC` polish budgets,
`HP24_SA_CAP` (default 3200 s) total per-benchmark wall budget that the cold-SA
final stage fills.

## Progression

v12 1.2075 → v21 1.1593 → v23 1.1531 (analytical base + numerical-gradient /
CD refinement) → v24 1.0623 (differentiable smooth-proxy cyclic GD) →
**v26 0.9725** (+ cold simulated-annealing final stage).

## Credits

The analytical base placer used by the validity-fallback path
(`submissions/vxzhang/v5_rudy_w1_placer.py`) is from the public `v-x-zhang`
fork of this repository (Apache 2.0). The smooth proxy, the cyclic smooth-GD
placer, the cold-SA final stage, the fast Numba proxy evaluators and the
numerical-gradient / CD refinement are our own.

## Reproducibility

All runs were on a 4-core CPU server (15 GiB RAM, no GPU). Total runtime for
the 17-benchmark panel ≈ 15 h (the cold-SA stage fills the per-benchmark budget
up to the 1 h cap).
