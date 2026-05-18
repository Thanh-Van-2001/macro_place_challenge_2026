"""
v19 — v12 multi-start base + simulated-annealing refinement on the fast proxy.

v16/v17 used greedy coordinate descent, which gets stuck in the first local
optimum (~1.16). v19 replaces the greedy accept with a Metropolis criterion:
moves that worsen the proxy are accepted with probability exp(-Δ/T), so the
search can climb out of shallow minima. Temperature anneals geometrically to
near zero, ending as pure descent.

Move set per iteration (picked at random):
  - single-macro Gaussian shift (step scaled by temperature)
  - macro-pair position swap (similar-area pairs)
Both are overlap-checked; the fast proxy (FastEval, ~3 ms) is the oracle.
The best zero-overlap placement ever seen is tracked and returned.

All code here is our own. The fast evaluator only changes search speed;
the final placement is scored by the unmodified TILOS evaluator.
"""

import math
import os
import random
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, _ensure_congestion_arrays

_TVDIR = os.path.dirname(os.path.abspath(__file__))
if _TVDIR not in sys.path:
    sys.path.insert(0, _TVDIR)
from fasteval import FastEval

os.environ.setdefault("AP_RUDY_W", "3.0")
os.environ.setdefault("AP_ITERS", "350")
os.environ.setdefault("AP_LR", "0.003")
os.environ.setdefault("AP_DEN_W", "1.5")
os.environ.setdefault("AP_OV_START", "3.0")
os.environ.setdefault("AP_DEN_CARRIER", "rect")

_v5_path = Path(__file__).parent.parent / "vxzhang" / "v5_rudy_w1_placer.py"
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


def _load_plc(benchmark):
    for root in [
        Path("external/MacroPlacement/Testcases/ICCAD04"),
        Path(__file__).parent.parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
    ]:
        d = root / benchmark.name
        if d.exists():
            _, plc = load_benchmark_from_dir(d.as_posix())
            return plc
    return None


def _real_proxy(plc, benchmark, pos_np):
    pt = torch.tensor(pos_np, dtype=torch.float32)
    pt[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _set_placement(plc, pt, benchmark)
    _ensure_congestion_arrays(plc)
    return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()


def _count_overlaps(pos, hw, hh, n_hard):
    cnt = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and \
               abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]:
                cnt += 1
    return cnt


def _overlaps_at(pos, hw, hh, idx, n_hard, nx, ny):
    hwi = hw[idx]
    hhi = hh[idx]
    for j in range(n_hard):
        if j == idx:
            continue
        if abs(nx - pos[j, 0]) < hwi + hw[j] and abs(ny - pos[j, 1]) < hhi + hh[j]:
            return True
    return False


def sa_refine(fe, plc, benchmark, base_pos, time_budget):
    """Simulated annealing on the fast proxy. Returns (best_pos, real_proxy)."""
    nh = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).cpu().numpy()
    movable_idx = np.where(movable)[0]
    nm = len(movable_idx)
    if nm == 0:
        return base_pos, _real_proxy(plc, benchmark, base_pos)

    cur = base_pos.astype(np.float64).copy()
    cur_fast, _, _, _ = fe.proxy(cur)
    best = cur.copy()
    best_fast = cur_fast

    diag = math.sqrt(cw * cw + ch * ch)
    rng = np.random.default_rng(2024)

    # Temperature: calibrate T0 from the spread of small-move deltas.
    deltas = []
    for _ in range(60):
        idx = int(movable_idx[rng.integers(nm)])
        ox, oy = cur[idx, 0], cur[idx, 1]
        nx = min(max(ox + rng.standard_normal() * diag * 0.02, hw[idx]), cw - hw[idx])
        ny = min(max(oy + rng.standard_normal() * diag * 0.02, hh[idx]), ch - hh[idx])
        if _overlaps_at(cur, hw, hh, idx, nh, nx, ny):
            continue
        cur[idx, 0] = nx
        cur[idx, 1] = ny
        f, _, _, _ = fe.proxy(cur)
        cur[idx, 0] = ox
        cur[idx, 1] = oy
        deltas.append(abs(f - cur_fast))
    T0 = (np.median(deltas) / 0.7) if deltas else 1e-3
    Tmin = T0 * 1e-3

    t0 = time.time()
    # estimate iteration budget for the anneal schedule
    iters_done = 0
    accepts = 0
    # geometric anneal recomputed against elapsed-fraction of the budget
    p_swap = 0.25

    while time.time() - t0 < time_budget:
        frac = (time.time() - t0) / time_budget
        T = max(Tmin, T0 * (Tmin / T0) ** frac)
        # block of iterations between time checks
        for _blk in range(200):
            iters_done += 1
            if rng.random() < p_swap and nm >= 2:
                # ── pair swap ──
                ii = int(movable_idx[rng.integers(nm)])
                jj = int(movable_idx[rng.integers(nm)])
                if ii == jj:
                    continue
                ai = sizes[ii, 0] * sizes[ii, 1]
                aj = sizes[jj, 0] * sizes[jj, 1]
                if aj < 0.5 * ai or aj > 2.0 * ai:
                    continue
                oxi, oyi = cur[ii, 0], cur[ii, 1]
                oxj, oyj = cur[jj, 0], cur[jj, 1]
                nxi = min(max(oxj, hw[ii]), cw - hw[ii])
                nyi = min(max(oyj, hh[ii]), ch - hh[ii])
                nxj = min(max(oxi, hw[jj]), cw - hw[jj])
                nyj = min(max(oyi, hh[jj]), ch - hh[jj])
                cur[ii, 0] = nxi; cur[ii, 1] = nyi
                cur[jj, 0] = nxj; cur[jj, 1] = nyj
                if _overlaps_at(cur, hw, hh, ii, nh, nxi, nyi) or \
                   _overlaps_at(cur, hw, hh, jj, nh, nxj, nyj):
                    cur[ii, 0] = oxi; cur[ii, 1] = oyi
                    cur[jj, 0] = oxj; cur[jj, 1] = oyj
                    continue
                f, _, _, _ = fe.proxy(cur)
                d = f - cur_fast
                if d < 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                    cur_fast = f
                    accepts += 1
                    if f < best_fast - 1e-12:
                        best_fast = f
                        best = cur.copy()
                else:
                    cur[ii, 0] = oxi; cur[ii, 1] = oyi
                    cur[jj, 0] = oxj; cur[jj, 1] = oyj
            else:
                # ── single-macro Gaussian shift ──
                idx = int(movable_idx[rng.integers(nm)])
                ox, oy = cur[idx, 0], cur[idx, 1]
                step = diag * (0.004 + 0.10 * math.sqrt(max(T / T0, 1e-4)))
                nx = ox + rng.standard_normal() * step
                ny = oy + rng.standard_normal() * step
                if nx < hw[idx]:
                    nx = hw[idx]
                elif nx > cw - hw[idx]:
                    nx = cw - hw[idx]
                if ny < hh[idx]:
                    ny = hh[idx]
                elif ny > ch - hh[idx]:
                    ny = ch - hh[idx]
                if _overlaps_at(cur, hw, hh, idx, nh, nx, ny):
                    continue
                cur[idx, 0] = nx
                cur[idx, 1] = ny
                f, _, _, _ = fe.proxy(cur)
                d = f - cur_fast
                if d < 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                    cur_fast = f
                    accepts += 1
                    if f < best_fast - 1e-12:
                        best_fast = f
                        best = cur.copy()
                else:
                    cur[idx, 0] = ox
                    cur[idx, 1] = oy
            if time.time() - t0 > time_budget:
                break

    sys.stderr.write(f"[v19] SA iters={iters_done} accepts={accepts} "
                     f"T0={T0:.4g} best_fast={best_fast:.5f}\n")
    sys.stderr.flush()
    if _count_overlaps(best, hw, hh, nh) > 0:
        best = base_pos.astype(np.float64).copy()
    return best, _real_proxy(plc, benchmark, best)


class HybridV19(_BASE):
    """v12 multi-start + simulated-annealing refinement on the fast proxy."""

    def __init__(self):
        super().__init__()
        self._n_runs = int(os.environ.get("HP19_N", "4"))
        self._sa_budget = float(os.environ.get("HP19_SA_SEC", "400"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark)
        if plc is None:
            return super().place(benchmark)

        best_pos = None
        best_cost = float("inf")
        orig_seed = self.seed
        for run_i in range(self._n_runs):
            self.seed = orig_seed + run_i * 9973
            placement = super().place(benchmark)
            pn = placement.cpu().numpy().astype(np.float64)
            cost = _real_proxy(plc, benchmark, pn)
            if cost < best_cost:
                best_cost = cost
                best_pos = pn
        self.seed = orig_seed
        if best_pos is None:
            return super().place(benchmark)

        fe = FastEval(plc, benchmark)
        base_fast, _, _, _ = fe.proxy(best_pos)
        refined, refined_cost = sa_refine(fe, plc, benchmark, best_pos, self._sa_budget)
        sys.stderr.write(
            f"[v19] base_real={best_cost:.5f} base_fast={base_fast:.5f} "
            f"refined_real={refined_cost:.5f}\n")
        sys.stderr.flush()

        final = refined if refined_cost < best_cost else best_pos
        result = torch.tensor(final, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return result
