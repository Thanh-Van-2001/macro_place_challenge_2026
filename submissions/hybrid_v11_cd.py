"""
Hybrid Macro Placer v11 — v9 best + real-proxy CD post-processing.

After v5 produces a placement, run a coordinate-descent post-processing
pass that perturbs each movable hard macro by small offsets and accepts
ONLY if the contest's actual proxy cost improves. This is the
"exact-proxy guided" pattern from chuanqi phase6.

Key design points:
- Real proxy via TILOS PlacementCost (not a surrogate). Slow per call but
  guarantees we never accept a worse solution.
- Time-budgeted: caps post-processing at ~25 s per benchmark so total
  runtime stays within ~10 min for the full IBM panel.
- Multi-scale offsets: tries 4 distance scales × 8 directions = 32
  candidates per macro pass, but breaks early on first improvement.
- Worst-first: macros are visited in descending order of HPWL contribution
  (their bounding-box span across all incident nets), so we spend our
  budget on the macros most likely to improve.
"""

import math
import os
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, _ensure_congestion_arrays


# Tuned env defaults from v9 sweep
os.environ.setdefault("AP_RUDY_W", "3.0")
os.environ.setdefault("AP_ITERS", "350")
os.environ.setdefault("AP_LR", "0.003")
os.environ.setdefault("AP_DEN_W", "1.5")
os.environ.setdefault("AP_OV_START", "3.0")
os.environ.setdefault("AP_DEN_CARRIER", "rect")


_v5_path = Path(__file__).parent / "vxzhang" / "v5_rudy_w1_placer.py"
_spec = importlib.util.spec_from_file_location("_v5_module", str(_v5_path))
_v5 = importlib.util.module_from_spec(_spec)
sys.modules["_v5_module"] = _v5
_spec.loader.exec_module(_v5)


_BASE = None
for _name in dir(_v5):
    _obj = getattr(_v5, _name)
    if isinstance(_obj, type) and callable(getattr(_obj, "place", None)):
        _BASE = _obj
        break


def _full_proxy(plc, benchmark, placement):
    """Real contest proxy = WL + 0.5*den + 0.5*cong."""
    _set_placement(plc, placement, benchmark)
    _ensure_congestion_arrays(plc)
    return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()


def _build_pin_to_macro_map(plc, benchmark):
    """Return dict: plc_module_idx -> tensor_macro_idx."""
    name2idx = {}
    for i, pi in enumerate(benchmark.hard_macro_indices):
        name2idx[plc.modules_w_pins[pi].get_name()] = i
    for i, pi in enumerate(benchmark.soft_macro_indices):
        name2idx[plc.modules_w_pins[pi].get_name()] = benchmark.num_hard_macros + i
    pin2macro = {}
    for idx, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == 'MACRO_PIN' and hasattr(mod, 'get_macro_name'):
            mn = mod.get_macro_name()
            if mn in name2idx:
                pin2macro[idx] = name2idx[mn]
    return pin2macro


def _macro_hpwl_contribution(plc, benchmark, placement, pin2macro):
    """Approximate per-macro HPWL contribution: sum over nets containing the
    macro of (net_bbox_x_span + net_bbox_y_span)."""
    pos = placement.cpu().numpy()
    nh = benchmark.num_hard_macros
    contrib = np.zeros(nh, dtype=np.float64)
    for driver, sinks in plc.nets.items():
        macros = set()
        if driver in pin2macro:
            macros.add(pin2macro[driver])
        for s in sinks:
            if s in pin2macro:
                macros.add(pin2macro[s])
        if len(macros) < 2:
            continue
        ml = list(macros)
        xs = [pos[m, 0] for m in ml]
        ys = [pos[m, 1] for m in ml]
        span = (max(xs) - min(xs)) + (max(ys) - min(ys))
        for m in ml:
            if m < nh:
                contrib[m] += span
    return contrib


def _has_overlap_with(placement, benchmark, idx_set):
    """Check if any macro in idx_set overlaps any other hard macro."""
    pos = placement.cpu().numpy()
    sizes = benchmark.macro_sizes.cpu().numpy()
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    nh = benchmark.num_hard_macros
    for i in idx_set:
        for j in range(nh):
            if j == i:
                continue
            if abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and \
               abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]:
                return True
    return False


def _load_plc(benchmark):
    for root in [
        Path("external/MacroPlacement/Testcases/ICCAD04"),
        Path(__file__).parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
    ]:
        d = root / benchmark.name
        if d.exists():
            _, plc = load_benchmark_from_dir(d.as_posix())
            return plc
    return None


def _post_cd(base_pos, benchmark, time_budget=25.0):
    """Real-proxy coordinate-descent post-processing."""
    plc = _load_plc(benchmark)
    if plc is None:
        return base_pos

    nh = benchmark.num_hard_macros
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).cpu().numpy()
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return base_pos

    sizes = benchmark.macro_sizes.cpu().numpy()
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    cur_pos = base_pos.clone()
    try:
        cur_cost = _full_proxy(plc, benchmark, cur_pos)
    except Exception:
        return base_pos
    best_cost = cur_cost
    best_pos = cur_pos.clone()

    pin2macro = _build_pin_to_macro_map(plc, benchmark)

    # Multi-scale offsets in microns. Inspired by chuanqi phase6.
    diag = math.sqrt(cw * cw + ch * ch)
    base_step = max(0.5, diag * 0.005)
    scales = [base_step * s for s in (0.4, 0.8, 1.6, 3.0)]
    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                  (0.7, 0.7), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7)]

    t0 = time.time()
    soft_mask = benchmark.get_soft_macro_mask()

    for round_i in range(3):
        if time.time() - t0 > time_budget:
            break

        # Recompute HPWL contribution per round so worst-first stays accurate
        try:
            contrib = _macro_hpwl_contribution(plc, benchmark, cur_pos, pin2macro)
        except Exception:
            break
        # Visit movable macros in descending HPWL contribution
        order = sorted(movable_idx.tolist(), key=lambda i: -contrib[i])
        improved_round = False

        for idx in order:
            if time.time() - t0 > time_budget:
                break
            ox = float(cur_pos[idx, 0])
            oy = float(cur_pos[idx, 1])
            best_local_cost = best_cost
            best_local_pos = None
            for s in scales:
                for dx, dy in directions:
                    nx = ox + dx * s
                    ny = oy + dy * s
                    if nx < hw[idx]:
                        nx = hw[idx]
                    elif nx > cw - hw[idx]:
                        nx = cw - hw[idx]
                    if ny < hh[idx]:
                        ny = hh[idx]
                    elif ny > ch - hh[idx]:
                        ny = ch - hh[idx]
                    cand = cur_pos.clone()
                    cand[idx, 0] = nx
                    cand[idx, 1] = ny
                    # Restore fixed/soft
                    cand[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
                    cand[soft_mask] = benchmark.macro_positions[soft_mask]
                    if _has_overlap_with(cand, benchmark, [idx]):
                        continue
                    try:
                        c = _full_proxy(plc, benchmark, cand)
                    except Exception:
                        continue
                    if c < best_local_cost - 1e-6:
                        best_local_cost = c
                        best_local_pos = cand
                        # Greedy: take first improvement at smallest scale
                        break
                if best_local_pos is not None:
                    break
            if best_local_pos is not None:
                cur_pos = best_local_pos
                best_cost = best_local_cost
                best_pos = best_local_pos.clone()
                improved_round = True
        if not improved_round:
            break

    return best_pos


class HybridV11(_BASE):
    """v9 best params + real-proxy CD post-processing."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        base = super().place(benchmark)
        try:
            return _post_cd(base, benchmark, time_budget=float(os.environ.get("HP11_BUDGET", "25")))
        except Exception:
            return base
