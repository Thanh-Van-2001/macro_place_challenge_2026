# Submission Versions

Archived snapshots of the analytical placer at meaningful milestones.

## v5_rudy_w1_placer.py

**IBM Score:** AVG = **1.2812** (full public panel, K=1), all 17 VALID, +12.1% vs RePlAce, +39.7% vs SA
**NG45 Score:** AVG = **0.7250**, all 4 VALID
**Date:** 2026-05-04

### What changed

Increased default `AP_RUDY_W` from 0.5 → **1.0** after sweeping it on the
congestion-heavy designs (ibm12/15/17). The H/V-split RUDY proxy now matches
the official metric closely enough that pushing it harder is genuinely
helpful.

### RUDY weight sweep (heavy-cong designs)

| | w=0.5 | w=1.0 | w=1.5 | w=2.0 |
|---|---:|---:|---:|---:|
| ibm12 | 1.5420 | 1.5106 | 1.5030 | 1.4975 |
| ibm15 | 1.3669 | 1.3736 | **1.3485** | 1.3538 |
| ibm17 | 1.4718 | 1.4641 | 1.4529 | 1.4493 |

w=1.5 was the IBM optimum (AVG=1.2731) but caused `ariane136` (NG45, 50%
utilisation) to fail push-pull and fall back to shelf-pack (proxy 1.02 vs
0.69). w=1.0 is the Pareto pick: IBM gain held, NG45 still all VALID.

### Per-design IBM impact (vs v4 default w=0.5)

Most designs improve 1–4%; ibm12 −1.5%, ibm08 +1.2%, ibm18 −2.0%, etc.

### Run

```bash
uv run evaluate submissions/versions/v5_rudy_w1_placer.py --all
```

---

## v4_robust_legalize_placer.py

**IBM Score:** AVG = **1.2927** (full public panel, K=1, 400 iters), all 17 VALID, +11.3% vs RePlAce
**NG45 Score:** AVG = **0.7134**, all 4 VALID (ariane133=0.6910, ariane136=0.6933, mempool_tile=0.7646, nvdla=0.7046)
**Date:** 2026-05-04

### What changed

Robustness fixes for highly-congested designs (NG45 commercial benchmarks).
`ariane136` (50% macro utilisation, 136 hard macros) was INVALID under v3 in
the `--ng45` batch run with 13–19 residual overlaps after legalization.

Three-tier legalization escalation in `_legalize`:

1. **Main pass** (unchanged): 400 iters of vectorized push-pull at
   cap=1.0×(w+h) per step. Resolves ~all overlaps for IBM and most NG45.
2. **Retry rescue** (new): up to 3 retries when overlaps remain.
   - Identifies "stuck" macros (those still in any overlap pair)
   - Jitters ONLY stuck macros by `0.03 × (retry+1) × canvas_diag`
     (canvas-relative, large enough to escape multi-macro lock-ups)
   - Deterministic via dedicated `torch.Generator(seed+7919)`
   - Runs 2000 push-pull iters at cap=1.0×(w+h)
3. **Shelf-pack fallback** (new): if push-pull retries still fail,
   row-pack all movable hard macros sorted by descending height. This
   guarantees a legal placement (avoids DQ) at the cost of WL on that
   single design. Empirically not triggered on the public panel after
   the retry rescue lands.

### Why canvas-scaled jitter

The earlier attempt used `0.05 × (retry+1) × (w+h)/2` (macro-fraction).
For `ariane136` the average macro is ~50µm and canvas is 1446µm — that
jitter (~5µm) is well below typical overlap depth, so the locked-in
pattern just snaps back. Canvas-scaled jitter (~43µm at retry=0) is
large enough to genuinely break the configuration.

### Rules compliance

- No per-benchmark name-based overrides
- No soft macro resizing
- No hard macro rotations
- TILOS evaluator unchanged
- Generalizes across design families (IBM ICCAD04 + NG45 commercial)

### Run

```bash
uv run evaluate submissions/versions/v4_robust_legalize_placer.py --all
uv run evaluate submissions/versions/v4_robust_legalize_placer.py --ng45
```

### Negative results

- Annealed step cap (0.5→0.05): degraded ariane133 from VALID to 13 overlaps
  in batch mode. Reverted to constant cap=1.0.
- Macro-fraction jitter (0.05×macro_size): insufficient amplitude to escape
  ariane136 lock-up.
- Rect-overlap density carrier (`AP_DEN_CARRIER=rect`): mixed (ibm01 −1.2%,
  ibm12 +1.0%, ibm17 +0.3%); not enabled by default.

---

## v3_hv_rudy_placer.py

**Score:** AVG = **1.2894** (full public panel, K=1, 1500 iters)
**Date:** 2026-05-03
**Improvement vs RePlAce baseline:** +11.6%
**Improvement vs prior version (v2 ePlace + ensemble):** -7.7% (1.3976 → 1.2894)

### What changed

The headline change is making the differentiable RUDY congestion penalty
*exactly* match the official `PlacementCost.get_congestion_cost()` metric.

Previously, RUDY:
- Used a single combined demand grid (HPWL deposited as one quantity)
- Normalized by `hcap + vcap` (combined capacity)
- Took top-10% mean of one grid
- Was off by default (`AP_RUDY_W=0.0`)

After audit of `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py`,
we found `get_congestion_cost()` actually does:

1. **Splits H/V routing demand**: each net's x-span goes to H grid, y-span to V grid
2. **Normalizes separately**: H by `hroutes_per_micron * cell_area`, V by `vroutes_per_micron * cell_area`
3. **Smooth range = 2 box filter**: V congestion spread ±2 columns, H congestion spread ±2 rows
4. **`abu(V + H, 0.05)`**: top-**5%** ABU of the *concatenated* H and V grids (2× cell count)

The fix in `analytical_placer.py`:
- Build separate `h_grid` (x-span demand) and `v_grid` (y-span demand)
- Normalize each by its own capacity
- Apply the smooth-range-2 box filter (PyTorch slicing, fully differentiable)
- `torch.cat([h_smooth, v_smooth]).topk(5%)` to score
- Default `AP_RUDY_W=0.5` since the corrected loss now genuinely tracks the metric

Same fix applied to `_score_internal` (the proxy used to rank ensemble candidates).

### Per-benchmark impact (vs baseline 1500-iter, no RUDY)

| Benchmark | Before | After | Delta |
|-----------|-------:|------:|------:|
| ibm01 | 0.9202 | 0.9211 | +0.1% |
| ibm12 | 1.8325 | 1.5372 | **−16.1%** |
| ibm17 | 1.4857 | 1.4802 | −0.4% |
| (most) | small | small | ~0% |
| **Panel AVG** | **1.3976** | **1.2894** | **−7.7%** |

ibm12 (heavily congestion-dominated) was the dominant beneficiary,
which is exactly what the math predicts: it has the worst V/H asymmetry
and benefits most from independent H/V routing modeling.

### Other ingredients (unchanged from v2)

- ePlace eDensity via DCT-II FFT Poisson solve
- Bell-shaped triangle density carrier
- Pin-offset HPWL (placer self-loads PlacementCost to extract macro pin offsets,
  no dependency on `benchmark.net_pins` field, robust to judge env)
- Vectorized push-apart legalization
- Adam optimizer, 1500 iters, overlap schedule, anchor regularizer

### Rules compliance

- No per-benchmark name-based overrides
- No soft macro resizing
- No 90°/270° hard macro rotations
- TILOS evaluator used as-is — only the *internal proxy loss* changed to model
  the official metric more faithfully

### Run

```bash
uv run evaluate submissions/versions/v3_hv_rudy_placer.py --all
```

Defaults: `AP_ITERS=1500`, `AP_RUDY_W=0.5`, `AP_NUM_SEEDS=1`.

### Negative results from this round

- `AP_NUM_SEEDS=3` ensemble with fixed RUDY: AVG=1.2899 (essentially tied,
  not worth ~3× runtime)
- `AP_ITERS=2000`: ~0.4% noise, mixed gains/losses per benchmark
- `AP_ITERS=2500`: clearly worse (congestion overshoots without enough RUDY damping)


0 /tmp/ibm_k4_diverse.log
#[ROUTES PER MICRON] Hor: 65.96, Ver: 106.96
#[CONGESTION SMOOTH RANGE] Smooth Range: 2
#[OVERLAP THRESHOLD] Threshold: 0.0040
proxy=1.6007  (wl=0.081 den=0.491 cong=2.550)  VALID  [334.80s]

--------------------------------------------------------------------------------
    Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
--------------------------------------------------------------------------------
        ibm01    0.9079    1.3166    0.9976    +31.0%       +9.0%         0
        ibm02    1.5214    1.9072    1.8370    +20.2%      +17.2%         0
        ibm03    1.1779    1.7401    1.3222    +32.3%      +10.9%         0
        ibm04    1.1598    1.5037    1.3024    +22.9%      +10.9%         0
        ibm06    1.3308    2.5057    1.6187    +46.9%      +17.8%         0
        ibm07    1.2560    2.0229    1.4633    +37.9%      +14.2%         0
        ibm08    1.2941    1.9239    1.4285    +32.7%       +9.4%         0
        ibm09    0.9697    1.3875    1.1194    +30.1%      +13.4%         0
        ibm10    1.2291    2.1108    1.5009    +41.8%      +18.1%         0
        ibm11    1.0075    1.7111    1.1774    +41.1%      +14.4%         0
        ibm12    1.4926    2.8261    1.7261    +47.2%      +13.5%         0
        ibm13    1.0924    1.9141    1.3355    +42.9%      +18.2%         0
        ibm14    1.3359    2.2750    1.5436    +41.3%      +13.5%         0
        ibm15    1.3485    2.3000    1.5159    +41.4%      +11.0%         0
        ibm16    1.2801    2.2337    1.4780    +42.7%      +13.4%         0
        ibm17    1.4519    3.6726    1.6446    +60.5%      +11.7%         0
        ibm18    1.6007    2.7755    1.7722    +42.3%       +9.7%         0
--------------------------------------------------------------------------------
          AVG    1.2621    2.1251    1.4578    +40.6%      +13.4%         0
