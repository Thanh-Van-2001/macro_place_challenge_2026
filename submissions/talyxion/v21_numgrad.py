"""
v21 — numerical-gradient descent on the exact fast proxy, then CD polish.

Coordinate descent (v16/v17) moves one macro at a time and gets stuck in a
coordinate-wise local optimum. This module instead estimates, for every
movable hard macro, the finite-difference gradient of the REAL contest
proxy (via the fast evaluator), then steps all macros simultaneously along
the negative gradient — coordinated moves that CD cannot make. After a
projected (overlap-resolved) GD step it backtracks the learning rate if the
proxy did not improve. A final coordinate-descent + swap pass polishes the
result.

The gradient is of the actual TILOS proxy (computed by FastEval); this is
gradient descent on the real objective, not on a surrogate. All code here
is our own.
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
from numba import njit

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


@njit(cache=True)
def _count_overlaps(pos, hw, hh, n_hard):
    cnt = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and \
               abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]:
                cnt += 1
    return cnt


@njit(cache=True)
def _overlaps_at(pos, hw, hh, idx, n_hard, nx, ny):
    hwi = hw[idx]
    hhi = hh[idx]
    for j in range(n_hard):
        if j == idx:
            continue
        if abs(nx - pos[j, 0]) < hwi + hw[j] and abs(ny - pos[j, 1]) < hhi + hh[j]:
            return True
    return False


@njit(cache=True)
def _resolve_overlaps(pos, hw, hh, movable_mask, n_hard, cw, ch, max_passes):
    """Push overlapping macro pairs apart along the smaller-overlap axis."""
    eps = 0.02
    for _ in range(max_passes):
        found = False
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                mdx = hw[i] + hw[j] + eps
                mdy = hh[i] + hh[j] + eps
                if dx < mdx and dy < mdy:
                    found = True
                    ox = mdx - dx
                    oy = mdy - dy
                    if ox < oy:
                        sgn = 1.0 if pos[i, 0] < pos[j, 0] else -1.0
                        if movable_mask[i] and movable_mask[j]:
                            pos[i, 0] -= sgn * ox * 0.5
                            pos[j, 0] += sgn * ox * 0.5
                        elif movable_mask[i]:
                            pos[i, 0] -= sgn * ox
                        elif movable_mask[j]:
                            pos[j, 0] += sgn * ox
                    else:
                        sgn = 1.0 if pos[i, 1] < pos[j, 1] else -1.0
                        if movable_mask[i] and movable_mask[j]:
                            pos[i, 1] -= sgn * oy * 0.5
                            pos[j, 1] += sgn * oy * 0.5
                        elif movable_mask[i]:
                            pos[i, 1] -= sgn * oy
                        elif movable_mask[j]:
                            pos[j, 1] += sgn * oy
                    for k in (i, j):
                        if movable_mask[k]:
                            if pos[k, 0] < hw[k]:
                                pos[k, 0] = hw[k]
                            elif pos[k, 0] > cw - hw[k]:
                                pos[k, 0] = cw - hw[k]
                            if pos[k, 1] < hh[k]:
                                pos[k, 1] = hh[k]
                            elif pos[k, 1] > ch - hh[k]:
                                pos[k, 1] = ch - hh[k]
        if not found:
            return


def numgrad_refine(fe, plc, benchmark, base_pos, time_budget):
    """Projected numerical-gradient descent on the exact fast proxy."""
    nh = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    movable_mask = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).cpu().numpy()
    movable_idx = np.where(movable_mask)[0]
    nm = len(movable_idx)
    if nm == 0:
        return base_pos, _real_proxy(plc, benchmark, base_pos)

    cur = base_pos.astype(np.float64).copy()
    cur_f, _, _, _ = fe.proxy(cur)
    best = cur.copy()
    best_f = cur_f

    diag = math.sqrt(cw * cw + ch * ch)
    eps = diag * 0.004           # finite-difference probe
    lr = diag * 0.010            # initial step
    lr_min = diag * 0.0005

    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget and lr > lr_min:
        step += 1
        # ── finite-difference gradient for every movable macro ──
        grad = np.zeros((nh, 2), dtype=np.float64)
        for mi in movable_idx:
            mi = int(mi)
            ox = cur[mi, 0]
            oy = cur[mi, 1]
            cur[mi, 0] = ox + eps
            fxp, _, _, _ = fe.proxy(cur)
            cur[mi, 0] = ox - eps
            fxm, _, _, _ = fe.proxy(cur)
            cur[mi, 0] = ox
            cur[mi, 1] = oy + eps
            fyp, _, _, _ = fe.proxy(cur)
            cur[mi, 1] = oy - eps
            fym, _, _, _ = fe.proxy(cur)
            cur[mi, 1] = oy
            grad[mi, 0] = (fxp - fxm) / (2 * eps)
            grad[mi, 1] = (fyp - fym) / (2 * eps)
            if time.time() - t0 > time_budget:
                break
        # normalize gradient so the step size is controlled by lr
        gnorm = math.sqrt(float((grad * grad).sum())) + 1e-12
        gdir = grad / gnorm

        # ── backtracking line search along -gdir ──
        accepted = False
        for _bt in range(5):
            trial = cur.copy()
            for mi in movable_idx:
                mi = int(mi)
                nx = cur[mi, 0] - lr * gdir[mi, 0] * math.sqrt(nm)
                ny = cur[mi, 1] - lr * gdir[mi, 1] * math.sqrt(nm)
                if nx < hw[mi]:
                    nx = hw[mi]
                elif nx > cw - hw[mi]:
                    nx = cw - hw[mi]
                if ny < hh[mi]:
                    ny = hh[mi]
                elif ny > ch - hh[mi]:
                    ny = ch - hh[mi]
                trial[mi, 0] = nx
                trial[mi, 1] = ny
            _resolve_overlaps(trial, hw, hh, movable_mask, nh, cw, ch, 800)
            if _count_overlaps(trial, hw, hh, nh) > 0:
                lr *= 0.5
                continue
            f, _, _, _ = fe.proxy(trial)
            if f < cur_f - 1e-9:
                cur = trial
                cur_f = f
                accepted = True
                if f < best_f - 1e-12:
                    best_f = f
                    best = cur.copy()
                break
            lr *= 0.5
        if not accepted:
            lr *= 0.5

    sys.stderr.write(f"[v21] numgrad steps={step} best_f={best_f:.5f}\n")
    sys.stderr.flush()
    if _count_overlaps(best, hw, hh, nh) > 0:
        best = base_pos.astype(np.float64).copy()
    return best, _real_proxy(plc, benchmark, best)


# ── Coordinate-descent + swap polish (from v17) ──────────────────────────────

def cd_polish(fe, plc, benchmark, base_pos, time_budget):
    nh = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    movable_mask = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).cpu().numpy()
    movable_idx = np.where(movable_mask)[0]
    if len(movable_idx) == 0:
        return base_pos, _real_proxy(plc, benchmark, base_pos)

    cur = base_pos.astype(np.float64).copy()
    cur_fast, _, _, _ = fe.proxy(cur)
    best = cur.copy()
    best_fast = cur_fast
    diag = math.sqrt(cw * cw + ch * ch)
    scales = np.array([0.003, 0.01, 0.03, 0.08, 0.16], dtype=np.float64) * diag
    rng = random.Random(12345)
    np_rng = np.random.default_rng(777)
    t0 = time.time()
    K = 10
    for rnd in range(1000):
        if time.time() - t0 > time_budget:
            break
        order = movable_idx.tolist()
        rng.shuffle(order)
        improved = False
        for idx in order:
            if time.time() - t0 > time_budget:
                break
            ox = cur[idx, 0]
            oy = cur[idx, 1]
            bcf = cur_fast
            bx, by = ox, oy
            for _k in range(K):
                s = scales[np_rng.integers(len(scales))]
                nx = ox + np_rng.standard_normal() * s
                ny = oy + np_rng.standard_normal() * s
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
                if f < bcf - 1e-9:
                    bcf = f
                    bx, by = nx, ny
            cur[idx, 0] = bx
            cur[idx, 1] = by
            if bcf < cur_fast - 1e-9:
                cur_fast = bcf
                improved = True
        # swap phase
        n_swaps = max(300, len(movable_idx) * 5)
        for _ in range(n_swaps):
            if time.time() - t0 > time_budget:
                break
            ii = int(movable_idx[np_rng.integers(len(movable_idx))])
            jj = int(movable_idx[np_rng.integers(len(movable_idx))])
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
            if f < cur_fast - 1e-9:
                cur_fast = f
                improved = True
            else:
                cur[ii, 0] = oxi; cur[ii, 1] = oyi
                cur[jj, 0] = oxj; cur[jj, 1] = oyj
        if cur_fast < best_fast - 1e-12:
            best_fast = cur_fast
            best = cur.copy()
        if not improved:
            break
    if _count_overlaps(best, hw, hh, nh) > 0:
        best = base_pos.astype(np.float64).copy()
    return best, _real_proxy(plc, benchmark, best)


class HybridV21(_BASE):
    """Multi-start base -> numerical-gradient GD -> CD+swap polish."""

    def __init__(self):
        super().__init__()
        self._n_runs = int(os.environ.get("HP21_N", "4"))
        self._gd_budget = float(os.environ.get("HP21_GD_SEC", "200"))
        self._cd_budget = float(os.environ.get("HP21_CD_SEC", "200"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark)
        if plc is None:
            return super().place(benchmark)
        fe = FastEval(plc, benchmark)

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

        # numerical-gradient GD
        gd_pos, gd_cost = numgrad_refine(fe, plc, benchmark, best_pos, self._gd_budget)
        # CD + swap polish
        cd_pos, cd_cost = cd_polish(fe, plc, benchmark, gd_pos, self._cd_budget)
        sys.stderr.write(
            f"[v21] base={best_cost:.5f} gd={gd_cost:.5f} cd={cd_cost:.5f}\n")
        sys.stderr.flush()

        final, fcost = best_pos, best_cost
        if gd_cost < fcost:
            final, fcost = gd_pos, gd_cost
        if cd_cost < fcost:
            final, fcost = cd_pos, cd_cost

        result = torch.tensor(final, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return result
