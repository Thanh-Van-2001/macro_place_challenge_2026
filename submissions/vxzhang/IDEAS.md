# Ideas & next steps (post-v5 / v6 in progress)

Living scratchpad. Pick up from here on the 6700 XT box. Keep entries short.

---

## Headline numbers so far

| Variant                                    | IBM AVG | Notes                                     |
|--------------------------------------------|--------:|-------------------------------------------|
| v5 (K=1, RUDY w=1.0)                       |  1.2792 | clean baseline, all VALID                 |
| K=4 + RUDY-w sweep `0.5/1.0/1.5/2.0`       |  1.2621 | ensemble winner ranked by official proxy  |
| K=4 + RUDY rip-up refine (in progress)     |  ~1.25? | refine alone gives ~0.5–1% per bench      |

Verified competitor ceiling on the leaderboard: ~1.28 (MTK). Sub-1.2 still open.

---

## What to try first on the GPU box

The CPU-only Mac M1 was bottlenecked by:
- Each `compute_proxy_cost(...)` call (PlacementCost) ≈ 1–10 s per design.
- Refinement is pure greedy; we only afford ~40 evaluations per benchmark.
- 1500 iters × 32 macros × HPWL/density grids: ~1 min/seed on dense IBMs.

On RDNA2 with PyTorch+ROCm, the ePlace FFT density and the gradient steps
are the easy wins; PlacementCost stays on CPU (it's pure Python). So the
strategy shifts: **do many more gradient seeds**, keep the refinement
budget the same.

### Concrete experiments (ranked by expected payoff)

1. **K=16 ensemble + RUDY-w sweep + per-seed `AP_SEED` jitter**
   `AP_NUM_SEEDS=16 AP_RUDY_W_LIST="0.3,0.6,1.0,1.5,2.0,2.5"`
   Diversity dominates here. Expected AVG ~1.24.
2. **Anchor-init from port centroids** (`AP_ANCHOR_K`, `AP_ANCHOR_W`)
   Currently off. On dense IBMs the I/O-port pull early in optimization
   tends to break symmetric stuck states. Worth a sweep.
3. **Longer refine budget** — on GPU, scoring 200+ candidate moves per
   benchmark becomes cheap. Bump `AP_REFINE_MAX_EVALS=400`,
   `AP_REFINE_TIME_S=600`, `AP_REFINE_HOT_K=20`. The refine loop is the
   only component that *guarantees* monotone improvement on the official
   proxy, so spending more time here is always safe.
4. **Pair-swap moves in refinement** — current rip-up only translates one
   macro at a time. Add a swap operator (exchange two macros' centers,
   honoring size/canvas constraints), then accept on official-proxy
   improvement. Cheap to add (~30 LOC) and should help the
   high-utilization designs (ibm12/15/17) where there's no empty cell to
   move into.
5. **Simulated-annealing tail** after refine plateau. Small temperature on
   the proxy itself, not a surrogate. Stop when budget exhausted.

### Don't bother with

- More gradient iters past 2000 — sweep showed it overshoots congestion.
- Sinkhorn/OT — written up but never beat ePlace density in our hands.
- Hyperbolic init — same; novelty without gain.
- Fancier surrogates for ranking ensemble — we already use the official
  proxy directly; that's strictly better.

---

## Knobs cheat-sheet

```bash
# multi-start
AP_NUM_SEEDS=16
AP_RUDY_W_LIST="0.3,0.6,1.0,1.5,2.0,2.5"
AP_SEED=0                 # base; per-seed jitter is seed + 17*k internally

# core gradient
AP_ITERS=1500             # don't push past 2000
AP_RUDY_W=1.0             # ignored when AP_RUDY_W_LIST set
AP_DEN_CARRIER=1.0
AP_EPLACE_W=1.0
AP_LEGALIZE_ITERS=200

# refinement (RUDY rip-up greedy, official-proxy-monotone)
AP_REFINE_ITERS=50
AP_REFINE_MAX_EVALS=400
AP_REFINE_TIME_S=600
AP_REFINE_HOT_K=20
AP_REFINE_RADIUS=4
AP_REFINE_MAX_MACROS=10
AP_REFINE_CANDS=8

# anchor (off by default, try sweep)
AP_ANCHOR_K=8
AP_ANCHOR_W=5.0
AP_ANCHOR_FRAC=0.5
```

---

## 6700 XT / ROCm setup notes

- Card: RDNA2, gfx1031. Officially **unsupported** by upstream PyTorch
  ROCm wheels (which target CDNA + gfx1030/gfx1100). Workarounds:
  - Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` to spoof the runtime as gfx1030
    (Navi 21). Works for most kernels; some matmul shapes fail and fall
    back to CPU.
  - Use the `pytorch/pytorch:rocm6.x` container, override gfx version,
    smoke-test with `torch.randn(1024,1024,device="cuda").matmul(...)`
    before launching long runs.
- Linux only — ROCm doesn't ship for macOS or Windows. WSL2 ROCm is
  experimental and not worth fighting.
- Most of the placer is small-tensor PyTorch (32–512 macros). Expect
  speedups mainly on:
  - ePlace density FFT (one big complex FFT per iter)
  - dense HPWL gradient (scatter/gather over net pins)
  - parallel multi-start: launch K placers as separate processes if VRAM
    allows; or batch them along an extra leading dim.
- PlacementCost stays on CPU regardless. Don't waste time porting it.

### First-day checklist on the new box

1. `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"`
2. Run smoke: `uv run evaluate submissions/versions/v5_rudy_w1_placer.py -b ibm01`
3. Confirm proxy=0.91 ± noise on ibm01 (CPU and GPU should match within
   1e-3; any larger drift = numerics issue, investigate density grid).
4. Move device: in `analytical_placer.py` look for `device = positions.device`
   and the `torch.tensor(..., device=device)` calls. The optimizer code
   already passes a `device` through; just need `pos = pos.to("cuda")`
   at the top of `_place_once`.

---

## Open questions / leaderboard intel

- Top claims (1.037, 1.10) on the public board are unverified by us.
  Either they're (a) using a different proxy weighting, (b) cheating on
  validation, or (c) doing something we haven't thought of (transformer
  policy? RL fine-tune? hand-tuned per-design?). Worth re-reading
  challenge rules to see if per-design seeds are allowed.
- MacroPlacement's own SA reaches ~2.0 AVG; Circuit-Training-style RL
  reaches ~1.3. So 1.26 is already best-in-class for non-RL; sub-1.2
  likely requires either (a) a learned policy seeded by our analytical
  warm-start, or (b) much heavier search budget on each design.

---

## Files of interest

- [submissions/examples/analytical_placer.py](../examples/analytical_placer.py) — primary, ~1350 LOC
- [submissions/versions/v5_rudy_w1_placer.py](v5_rudy_w1_placer.py) — frozen baseline
- [macro_place/objective.py](../../macro_place/objective.py) — `compute_proxy_cost`
- [external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py](../../external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py) — official evaluator (don't touch)
