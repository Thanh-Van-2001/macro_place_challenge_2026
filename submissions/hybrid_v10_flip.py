"""
Hybrid Macro Placer v10 — v9 best + auto-flip post-processing.

After v5 produces a placement, also evaluate flipped variants
(flip_x / flip_y / flip_xy) and keep the lowest-proxy variant.

The benchmark canvas is symmetric, so net costs are unchanged within a
clean flip. But port positions and routing topology in the ICCAD04
benchmarks are not symmetric, so one orientation often scores noticeably
better than the others.

Idea ported from the public jaydenpiao submission.
"""

import os
import sys
import importlib.util
import math
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, _ensure_congestion_arrays


# Set env defaults BEFORE loading v5 (so v5's __init__ picks them up).
os.environ.setdefault("AP_RUDY_W", "3.0")
os.environ.setdefault("AP_ITERS", "350")  # ITERS=350+rect best
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
if _BASE is None:
    raise RuntimeError("Could not find a placer class in v5 module")


def _full_proxy(plc, benchmark, placement):
    """Compute the contest's proxy = WL + 0.5*den + 0.5*cong."""
    _set_placement(plc, placement, benchmark)
    _ensure_congestion_arrays(plc)
    return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()


def _has_overlaps(placement, benchmark):
    nh = benchmark.num_hard_macros
    pos = placement.cpu().numpy()
    sizes = benchmark.macro_sizes.cpu().numpy()
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    for i in range(nh):
        for j in range(i + 1, nh):
            if abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and \
               abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]:
                return True
    return False


def _flip(placement, benchmark, mode):
    """Reflect movable hard macros along x/y/xy. Fixed and soft macros are
    pinned to original positions."""
    out = placement.clone()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    if mode == "x":
        out[:, 0] = cw - placement[:, 0]
    elif mode == "y":
        out[:, 1] = ch - placement[:, 1]
    elif mode == "xy":
        out[:, 0] = cw - placement[:, 0]
        out[:, 1] = ch - placement[:, 1]
    else:
        raise ValueError(mode)

    # Restore fixed and soft macros to their canonical positions
    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    soft_mask = benchmark.get_soft_macro_mask()
    out[soft_mask] = benchmark.macro_positions[soft_mask]
    return out


def _load_plc(benchmark):
    """Locate and load PlacementCost for the IBM benchmark."""
    for root in [
        Path("external/MacroPlacement/Testcases/ICCAD04"),
        Path(__file__).parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
    ]:
        d = root / benchmark.name
        if d.exists():
            _, plc = load_benchmark_from_dir(d.as_posix())
            return plc
    return None


class HybridV10(_BASE):
    """v9 best + auto-flip post-processing.

    Generates 4 candidates (original + 3 flips), re-evaluates the contest
    proxy on each, returns the lowest-cost one. Reverts to original if any
    flip introduces overlaps.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        base = super().place(benchmark)
        plc = _load_plc(benchmark)
        if plc is None:
            return base

        candidates = [("orig", base)]
        for mode in ("x", "y", "xy"):
            c = _flip(base, benchmark, mode)
            if not _has_overlaps(c, benchmark):
                candidates.append((mode, c))

        best_mode = "orig"
        best_cost = float("inf")
        best_pos = base
        for mode, c in candidates:
            try:
                cost = _full_proxy(plc, benchmark, c)
            except Exception:
                continue
            if cost < best_cost:
                best_cost = cost
                best_pos = c
                best_mode = mode

        return best_pos
