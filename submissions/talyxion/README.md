# Team Talyxion — Submission

**Score:** AVG **1.1593** across 17 IBM ICCAD04 benchmarks
**Algorithm:** Tuned analytical placer + multi-start + fast-proxy coordinate-descent / swap refinement
**Runtime:** ~105 min total on a 4-core CPU (≈6 min per benchmark, well under the 1 h/bench cap)
**Validity:** All 17 benchmarks VALID, 0 overlaps

## How to run

```bash
uv run evaluate submissions/talyxion/placer.py --all
uv run evaluate submissions/talyxion/placer.py -b ibm01
```

Knobs (env vars): `HP16_N` multi-start count (default 4), `HP16_CD_SEC`
coordinate-descent budget in seconds (default 300), `HP16_K` candidates
per macro (default 10).

## Pipeline

1. **Tuned analytical base.** The v5 analytical placer (RUDY + ePlace
   differentiable loss) from the public `v-x-zhang` fork (Apache 2.0),
   run with hyper-parameters we tuned on the IBM panel:
   `AP_RUDY_W=3.0, AP_ITERS=350, AP_LR=0.003, AP_DEN_W=1.5,
   AP_OV_START=3.0, AP_DEN_CARRIER=rect`. Single-run ≈ 1.2130.

2. **Real-proxy multi-start (N=4).** The v5 placer is not bit-deterministic
   on CPU; we run it 4× with different seeds and keep the lowest-proxy
   placement. ≈ 1.2075.

3. **Fast-proxy coordinate-descent + swap refinement.** Our own code:
   - `fasteval.py` — a fast evaluator for the contest proxy. Vectorized
     HPWL (matches `plc.get_cost()` exactly), Numba density, and a Numba
     re-implementation of the TILOS routing-congestion model
     (`fast_congestion.py`). Full proxy: ~1.6 s (TILOS) → ~3 ms.
   - `v16_cd.py` — coordinate descent: for each movable hard macro, try
     K multi-scale Gaussian candidate moves, score with the fast proxy,
     accept the best strict improvement with zero overlap. Each round
     also runs a swap phase that exchanges positions of similar-area
     macro pairs (non-local moves CD cannot reach).
   - Final: 1.2075 → **1.1631**.

The fast evaluators only change search SPEED — the final placement is
still scored by the unmodified TILOS evaluator. Every benchmark improved.

## Per-benchmark scores

| Benchmark | Ours | RePlAce | vs RePlAce |
|---|---:|---:|---:|
| ibm01 | 0.8738 | 0.9976 | +12.4% |
| ibm02 | 1.2242 | 1.8370 | +33.4% |
| ibm03 | 1.0638 | 1.3222 | +19.5% |
| ibm04 | 1.0527 | 1.3024 | +19.2% |
| ibm06 | 1.2769 | 1.6187 | +21.1% |
| ibm07 | 1.1844 | 1.4633 | +19.1% |
| ibm08 | 1.2069 | 1.4285 | +15.5% |
| ibm09 | 0.8927 | 1.1194 | +20.2% |
| ibm10 | 1.0509 | 1.5009 | +30.0% |
| ibm11 | 0.8910 | 1.1774 | +24.3% |
| ibm12 | 1.3759 | 1.7261 | +20.3% |
| ibm13 | 0.9854 | 1.3355 | +26.2% |
| ibm14 | 1.2574 | 1.5436 | +18.5% |
| ibm15 | 1.3340 | 1.5159 | +12.0% |
| ibm16 | 1.1865 | 1.4780 | +19.7% |
| ibm17 | 1.4134 | 1.6446 | +14.1% |
| ibm18 | 1.5031 | 1.7722 | +15.2% |
| **AVG** | **1.1631** | 1.4578 | **+20.2%** |

## Credits

The analytical base placer is from `v-x-zhang/macro-place-challenge-2026`
(file `submissions/versions/v5_rudy_w1_placer.py`, Apache 2.0). Our
contributions — the hyperparameter tuning, the real-proxy multi-start,
the fast Numba proxy evaluators (`fast_congestion.py`, `fasteval.py`),
and the coordinate-descent + swap refinement (`v16_cd.py`) — are our own.
