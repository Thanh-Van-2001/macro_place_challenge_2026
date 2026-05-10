"""
Hybrid Macro Placer v12 — v9 best + REAL-proxy multi-start.

The v5 analytical placer is not bit-deterministic on CPU even with
torch.manual_seed (some op order or BLAS path leaks randomness). Multiple
runs typically vary by ~0.5%. This wrapper exploits that by running the
placer N times with different seeds and returning the placement with the
lowest real (TILOS) proxy cost.

Why not v5's built-in AP_NUM_SEEDS? That picks the best seed by an
INTERNAL surrogate (HPWL + density approximation + RUDY top-10%), which is
not perfectly aligned with the contest proxy. We pick by the actual proxy.
"""

import os
import sys
import importlib.util
from pathlib import Path

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
    _set_placement(plc, placement, benchmark)
    _ensure_congestion_arrays(plc)
    return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()


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


class HybridV12(_BASE):
    """Run the placer N times with different seeds, pick best by real proxy."""

    def __init__(self):
        super().__init__()
        self._n_runs = int(os.environ.get("HP12_N", "6"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark)
        if plc is None:
            return super().place(benchmark)

        best_pos = None
        best_cost = float("inf")
        original_seed = self.seed
        for run_i in range(self._n_runs):
            # Different seed per run; v5 seeds torch internally before _place_once.
            self.seed = original_seed + run_i * 9973  # large prime stride
            placement = super().place(benchmark)
            try:
                cost = _full_proxy(plc, benchmark, placement)
            except Exception:
                continue
            if cost < best_cost:
                best_cost = cost
                best_pos = placement
        self.seed = original_seed

        return best_pos if best_pos is not None else super().place(benchmark)
