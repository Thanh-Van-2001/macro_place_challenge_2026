"""
Hybrid Macro Placer v7 — Numba-accelerated SA + multi-seed.

Drop-in replacement for hybrid_placer.py (v6) with the same algorithmic
recipe (force-directed init, then SA with adaptive radius and full-cost
checkpoints) but with the hot loops compiled via Numba and the adjacency
stored in CSR arrays.

Targets the bee server (4-core CPU, no GPU). Runs SA across multiple seeds
sequentially in the main process; PlacementCost full-cost is evaluated
between chunks to keep the best snapshot.
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from numba import njit

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, _ensure_congestion_arrays


# ─── Numba kernels ────────────────────────────────────────────────────────────


OVERLAP_EPS = 0.01  # required gap between hard macros (in microns)


@njit(cache=False, fastmath=True)
def _has_overlap(pos, hw, hh, idx, nx, ny, n_hard, neighbors):
    """Return True if placing macro idx at (nx, ny) overlaps any other hard
    macro by more than -OVERLAP_EPS gap. Using EPS slack here makes SA produce
    placements that are robust to floating-point drift and to the strict
    contest validator (any positive overlap area = INVALID).
    """
    eps = 0.01
    n_check = neighbors.shape[0]
    hwi, hhi = hw[idx], hh[idx]
    if n_check > 0:
        for k in range(n_check):
            j = neighbors[k]
            if j == idx or j >= n_hard:
                continue
            if abs(nx - pos[j, 0]) + eps < hwi + hw[j] and abs(ny - pos[j, 1]) + eps < hhi + hh[j]:
                return True
    else:
        for j in range(n_hard):
            if j == idx:
                continue
            if abs(nx - pos[j, 0]) + eps < hwi + hw[j] and abs(ny - pos[j, 1]) + eps < hhi + hh[j]:
                return True
    return False


@njit(cache=False, fastmath=True)
def _grid_neighbors(pos, hw, hh, idx, nx, ny, n_hard, grid, grid_indptr,
                    grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows):
    """Return array of macro indices within the spatial grid cells overlapping
    the bounding box of macro idx at (nx, ny). Includes some padding for the
    'might-overlap' radius (max macro size in benchmark)."""
    hwi, hhi = hw[idx], hh[idx]
    # Bounding box of idx at new position (with a little slack equal to
    # one max half-size; caller can pre-compute padding into hw/hh if needed).
    x_lo = nx - hwi
    x_hi = nx + hwi
    y_lo = ny - hhi
    y_hi = ny + hhi
    c_lo = max(0, int((x_lo - grid_x0) / cell_w) - 1)
    c_hi = min(n_cols - 1, int((x_hi - grid_x0) / cell_w) + 1)
    r_lo = max(0, int((y_lo - grid_y0) / cell_h) - 1)
    r_hi = min(n_rows - 1, int((y_hi - grid_y0) / cell_h) + 1)

    # Count + fill in two passes
    total = 0
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            cell = r * n_cols + c
            total += grid_indptr[cell + 1] - grid_indptr[cell]
    out = np.empty(total, dtype=np.int32)
    p = 0
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            cell = r * n_cols + c
            for k in range(grid_indptr[cell], grid_indptr[cell + 1]):
                out[p] = grid[k]
                p += 1
    return out


@njit(cache=False, fastmath=True)
def _build_grid(pos, n_hard, grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows):
    """Build CSR-style spatial grid: grid_indptr[c+1] - grid_indptr[c] is the
    count of macros in cell c, grid stores macro indices."""
    n_cells = n_cols * n_rows
    counts = np.zeros(n_cells + 1, dtype=np.int32)
    cell_of = np.empty(n_hard, dtype=np.int32)
    for i in range(n_hard):
        c = int((pos[i, 0] - grid_x0) / cell_w)
        r = int((pos[i, 1] - grid_y0) / cell_h)
        if c < 0:
            c = 0
        elif c >= n_cols:
            c = n_cols - 1
        if r < 0:
            r = 0
        elif r >= n_rows:
            r = n_rows - 1
        cell = r * n_cols + c
        cell_of[i] = cell
        counts[cell + 1] += 1
    # Cumulative sum -> indptr
    for i in range(1, n_cells + 1):
        counts[i] += counts[i - 1]
    indptr = counts.copy()
    grid = np.empty(n_hard, dtype=np.int32)
    for i in range(n_hard):
        cell = cell_of[i]
        slot = indptr[cell]
        grid[slot] = i
        indptr[cell] += 1
    # indptr was incremented; rebuild from counts (counts is already cumsum)
    indptr2 = np.zeros(n_cells + 1, dtype=np.int32)
    for i in range(1, n_cells + 1):
        indptr2[i] = counts[i - 1]  # offset back
    # Actually counts stored indptr+counts after cumsum. The clean version:
    # rebuild from scratch:
    counts2 = np.zeros(n_cells + 1, dtype=np.int32)
    for i in range(n_hard):
        counts2[cell_of[i] + 1] += 1
    for i in range(1, n_cells + 1):
        counts2[i] += counts2[i - 1]
    return grid, counts2


@njit(cache=False, fastmath=True)
def _sa_chunk(pos, hw, hh, movable, n_hard,
              adj_indptr, adj_indices, adj_weights,
              cw, ch, base_disp, T_arr, n_iter,
              grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows,
              seed_state, p_random, p_anchor):
    """Run n_iter SA iterations on flat arrays, modifying pos in place.

    T_arr: per-iteration temperature schedule, shape (n_iter,)
    seed_state: int64 array length 1, holds RNG state (we use np.random)

    Returns delta_wl_total (cumulative WL surrogate change applied — useful for
    debugging) and number of accepted moves.
    """
    np.random.seed(seed_state[0])
    nm = movable.shape[0]
    accepted = 0
    delta_wl_total = 0.0

    # Build initial spatial grid; we rebuild lazily every grid_rebuild_every iters
    grid, grid_indptr = _build_grid(pos, n_hard, grid_x0, grid_y0,
                                    cell_w, cell_h, n_cols, n_rows)
    # Rebuild aggressively to avoid stale-grid false-negative overlaps.
    grid_rebuild_every = 200
    iters_since_rebuild = 0

    for it in range(n_iter):
        T = T_arr[it]
        ratio = T / T_arr[0] if T_arr[0] > 0 else 1.0
        if ratio < 0.001:
            ratio = 0.001
        disp = base_disp * math.sqrt(ratio)

        r = np.random.random()
        idx = movable[np.random.randint(0, nm)]
        ox = pos[idx, 0]
        oy = pos[idx, 1]

        if r < p_random or nm < 2:
            # ─ Random Gaussian move ─
            nx = ox + np.random.randn() * disp
            ny = oy + np.random.randn() * disp
        elif r < p_random + p_anchor:
            # ─ Anchor move toward weighted neighbor centroid ─
            wx = 0.0
            wy = 0.0
            tw = 0.0
            for e in range(adj_indptr[idx], adj_indptr[idx + 1]):
                k = adj_indices[e]
                w = adj_weights[e]
                wx += pos[k, 0] * w
                wy += pos[k, 1] * w
                tw += w
            if tw <= 0.0:
                continue
            ccx = wx / tw
            ccy = wy / tw
            sf = 0.3 * ratio
            nx = ox + sf * (ccx - ox) + np.random.randn() * disp * 0.3
            ny = oy + sf * (ccy - oy) + np.random.randn() * disp * 0.3
        else:
            # ─ Swap with another random movable ─
            jj = np.random.randint(0, nm)
            if movable[jj] == idx:
                jj = (jj + 1) % nm
            j = movable[jj]
            oxj = pos[j, 0]
            oyj = pos[j, 1]
            # Tentative swap with bounds clamp
            nxi = oxj
            nyi = oyj
            if nxi < hw[idx]:
                nxi = hw[idx]
            elif nxi > cw - hw[idx]:
                nxi = cw - hw[idx]
            if nyi < hh[idx]:
                nyi = hh[idx]
            elif nyi > ch - hh[idx]:
                nyi = ch - hh[idx]
            nxj = ox
            nyj = oy
            if nxj < hw[j]:
                nxj = hw[j]
            elif nxj > cw - hw[j]:
                nxj = cw - hw[j]
            if nyj < hh[j]:
                nyj = hh[j]
            elif nyj > ch - hh[j]:
                nyj = ch - hh[j]

            # Apply tentatively
            pos[idx, 0] = nxi
            pos[idx, 1] = nyi
            pos[j, 0] = nxj
            pos[j, 1] = nyj

            # Overlap check on both
            nbrs_i = _grid_neighbors(pos, hw, hh, idx, nxi, nyi, n_hard,
                                     grid, grid_indptr, grid_x0, grid_y0,
                                     cell_w, cell_h, n_cols, n_rows)
            if _has_overlap(pos, hw, hh, idx, nxi, nyi, n_hard, nbrs_i):
                pos[idx, 0] = ox
                pos[idx, 1] = oy
                pos[j, 0] = oxj
                pos[j, 1] = oyj
                continue
            nbrs_j = _grid_neighbors(pos, hw, hh, j, nxj, nyj, n_hard,
                                     grid, grid_indptr, grid_x0, grid_y0,
                                     cell_w, cell_h, n_cols, n_rows)
            if _has_overlap(pos, hw, hh, j, nxj, nyj, n_hard, nbrs_j):
                pos[idx, 0] = ox
                pos[idx, 1] = oy
                pos[j, 0] = oxj
                pos[j, 1] = oyj
                continue

            # Delta WL: contributions of i and j neighbors
            delta = 0.0
            for e in range(adj_indptr[idx], adj_indptr[idx + 1]):
                k = adj_indices[e]
                w = adj_weights[e]
                if k == j:
                    old_d = abs(ox - oxj) + abs(oy - oyj)
                    new_d = abs(nxi - nxj) + abs(nyi - nyj)
                else:
                    xk = pos[k, 0]
                    yk = pos[k, 1]
                    old_d = abs(ox - xk) + abs(oy - yk)
                    new_d = abs(nxi - xk) + abs(nyi - yk)
                delta += w * (new_d - old_d)
            for e in range(adj_indptr[j], adj_indptr[j + 1]):
                k = adj_indices[e]
                if k == idx:
                    continue
                w = adj_weights[e]
                xk = pos[k, 0]
                yk = pos[k, 1]
                old_d = abs(oxj - xk) + abs(oyj - yk)
                new_d = abs(nxj - xk) + abs(nyj - yk)
                delta += w * (new_d - old_d)

            # Metropolis
            if delta < 0.0 or np.random.random() < math.exp(-delta / (T if T > 1e-15 else 1e-15)):
                accepted += 1
                delta_wl_total += delta
                iters_since_rebuild += 1
                if iters_since_rebuild >= grid_rebuild_every:
                    grid, grid_indptr = _build_grid(pos, n_hard, grid_x0, grid_y0,
                                                    cell_w, cell_h, n_cols, n_rows)
                    iters_since_rebuild = 0
            else:
                pos[idx, 0] = ox
                pos[idx, 1] = oy
                pos[j, 0] = oxj
                pos[j, 1] = oyj
            continue

        # ─ Single-macro move path (random or anchor) ─
        # Clamp to canvas
        if nx < hw[idx]:
            nx = hw[idx]
        elif nx > cw - hw[idx]:
            nx = cw - hw[idx]
        if ny < hh[idx]:
            ny = hh[idx]
        elif ny > ch - hh[idx]:
            ny = ch - hh[idx]

        # Overlap check with spatial grid
        nbrs = _grid_neighbors(pos, hw, hh, idx, nx, ny, n_hard,
                               grid, grid_indptr, grid_x0, grid_y0,
                               cell_w, cell_h, n_cols, n_rows)
        if _has_overlap(pos, hw, hh, idx, nx, ny, n_hard, nbrs):
            continue

        # Delta WL using CSR adj
        delta = 0.0
        for e in range(adj_indptr[idx], adj_indptr[idx + 1]):
            k = adj_indices[e]
            w = adj_weights[e]
            xk = pos[k, 0]
            yk = pos[k, 1]
            old_d = abs(ox - xk) + abs(oy - yk)
            new_d = abs(nx - xk) + abs(ny - yk)
            delta += w * (new_d - old_d)

        if delta < 0.0 or np.random.random() < math.exp(-delta / (T if T > 1e-15 else 1e-15)):
            pos[idx, 0] = nx
            pos[idx, 1] = ny
            accepted += 1
            delta_wl_total += delta
            iters_since_rebuild += 1
            if iters_since_rebuild >= grid_rebuild_every:
                grid, grid_indptr = _build_grid(pos, n_hard, grid_x0, grid_y0,
                                                cell_w, cell_h, n_cols, n_rows)
                iters_since_rebuild = 0

    seed_state[0] = np.random.randint(0, 2**31 - 1)
    return accepted, delta_wl_total


@njit(cache=False, fastmath=True)
def _force_directed_kernel(pos, hw, hh, sizes, movable, n_hard,
                           adj_indptr, adj_indices, adj_weights,
                           cw, ch, n_steps):
    """Vectorized-but-Numba'd FD: O(steps * (movable * (avg_deg + n_hard)))."""
    vel = np.zeros_like(pos)
    cx_c = cw * 0.5
    cy_c = ch * 0.5
    nm = movable.shape[0]

    for step in range(n_steps):
        p = step / n_steps
        ka = 0.001 * (1.0 + 2.0 * p)
        kr = 2000.0 * (1.0 - 0.7 * p)
        ks = 0.0005 * (1.0 + p)

        forces = np.zeros_like(pos)
        for ii in range(nm):
            idx = movable[ii]
            xi = pos[idx, 0]
            yi = pos[idx, 1]

            # Attraction to neighbors via CSR adj
            for e in range(adj_indptr[idx], adj_indptr[idx + 1]):
                j = adj_indices[e]
                w = adj_weights[e]
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                d = math.sqrt(dx * dx + dy * dy) + 1e-6
                f = ka * w * d
                forces[idx, 0] += f * dx / d
                forces[idx, 1] += f * dy / d

            # Repulsion from all hard macros (only if close)
            for j in range(n_hard):
                if j == idx:
                    continue
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                dsq = dx * dx + dy * dy + 1.0
                d = math.sqrt(dsq)
                ms = sizes[idx, 0] + sizes[j, 0] + sizes[idx, 1] + sizes[j, 1]
                if d < ms:
                    f = kr / dsq
                    forces[idx, 0] += f * dx / d
                    forces[idx, 1] += f * dy / d

            # Pull/push from canvas center
            dx_c = xi - cx_c
            dy_c = yi - cy_c
            d_c = math.sqrt(dx_c * dx_c + dy_c * dy_c) + 1e-6
            forces[idx, 0] += ks * dx_c / d_c * sizes[idx, 0]
            forces[idx, 1] += ks * dy_c / d_c * sizes[idx, 1]

        # Integrate
        for ii in range(nm):
            idx = movable[ii]
            vel[idx, 0] = 0.85 * vel[idx, 0] + 0.3 * forces[idx, 0]
            vel[idx, 1] = 0.85 * vel[idx, 1] + 0.3 * forces[idx, 1]
            pos[idx, 0] += 0.3 * vel[idx, 0]
            pos[idx, 1] += 0.3 * vel[idx, 1]
            if pos[idx, 0] < hw[idx]:
                pos[idx, 0] = hw[idx]
            elif pos[idx, 0] > cw - hw[idx]:
                pos[idx, 0] = cw - hw[idx]
            if pos[idx, 1] < hh[idx]:
                pos[idx, 1] = hh[idx]
            elif pos[idx, 1] > ch - hh[idx]:
                pos[idx, 1] = ch - hh[idx]


@njit(cache=False, fastmath=True)
def _resolve_overlaps_kernel(pos, hw, hh, movable_mask, n_hard, cw, ch, max_passes):
    """Push overlapping pairs apart along the smaller-overlap axis.
    Uses an EPS slack so resolved pairs end up with a small but strictly
    positive gap (otherwise float drift can re-trigger the contest validator)."""
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
                    # Push along the smaller-overlap axis. If both movable,
                    # split the displacement; if only one is movable, that
                    # one absorbs the entire required separation.
                    if ox < oy:
                        sgn = 1.0 if pos[i, 0] < pos[j, 0] else -1.0
                        if movable_mask[i] and movable_mask[j]:
                            pos[i, 0] -= sgn * (ox * 0.5)
                            pos[j, 0] += sgn * (ox * 0.5)
                        elif movable_mask[i]:
                            pos[i, 0] -= sgn * ox
                        elif movable_mask[j]:
                            pos[j, 0] += sgn * ox
                    else:
                        sgn = 1.0 if pos[i, 1] < pos[j, 1] else -1.0
                        if movable_mask[i] and movable_mask[j]:
                            pos[i, 1] -= sgn * (oy * 0.5)
                            pos[j, 1] += sgn * (oy * 0.5)
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


@njit(cache=False, fastmath=True)
def _count_overlaps_kernel(pos, hw, hh, n_hard):
    cnt = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and \
               abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]:
                cnt += 1
    return cnt


# ─── Adjacency build (Python, runs once) ─────────────────────────────────────


def _build_csr_adj(plc, benchmark):
    """Build CSR adjacency: edges between any pair of (hard or soft) macros that
    share a net. Edge weight = 1/clique_size summed over shared nets."""
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

    edge = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        if driver in pin2macro:
            macros.add(pin2macro[driver])
        for s in sinks:
            if s in pin2macro:
                macros.add(pin2macro[s])
        ml = list(macros)
        if len(ml) < 2:
            continue
        w = 1.0 / len(ml)
        for a in range(len(ml)):
            for b in range(a + 1, len(ml)):
                pair_a = (ml[a], ml[b])
                pair_b = (ml[b], ml[a])
                edge[pair_a] = edge.get(pair_a, 0.0) + w
                edge[pair_b] = edge.get(pair_b, 0.0) + w

    n = benchmark.num_macros
    out_lists = [[] for _ in range(n)]
    for (i, j), w in edge.items():
        out_lists[i].append((j, w))

    indptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        indptr[i + 1] = indptr[i] + len(out_lists[i])
    n_edges = int(indptr[-1])
    indices = np.empty(n_edges, dtype=np.int32)
    weights = np.empty(n_edges, dtype=np.float64)
    for i in range(n):
        for k, (j, w) in enumerate(out_lists[i]):
            slot = indptr[i] + k
            indices[slot] = j
            weights[slot] = w
    return indptr, indices, weights


# ─── Placer class ─────────────────────────────────────────────────────────────


class HybridPlacerV7:
    def __init__(self, time_budget=60.0, n_seeds=4):
        # Default: 60s wall budget per benchmark, 4 seeds
        self.time_budget = float(os.environ.get("HP7_TIME_BUDGET", time_budget))
        self.n_seeds = int(os.environ.get("HP7_N_SEEDS", n_seeds))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_start = time.time()
        plc = self._load_plc(benchmark)
        nh = benchmark.num_hard_macros
        n = benchmark.num_macros

        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        hw = (sizes[:, 0] * 0.5).copy()
        hh = (sizes[:, 1] * 0.5).copy()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        movable_mask_np = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).numpy()
        movable = np.where(movable_mask_np)[0].astype(np.int32)

        if movable.size == 0:
            return benchmark.macro_positions.clone()

        adj_indptr, adj_indices, adj_weights = _build_csr_adj(plc, benchmark)

        # Spatial grid params (used both for SA overlap checks and FD)
        n_cols = max(8, int(math.sqrt(nh)))
        n_rows = n_cols
        cell_w = cw / n_cols
        cell_h = ch / n_rows
        grid_x0 = 0.0
        grid_y0 = 0.0

        # ─ Initial pos: ensure overlap-free start ─
        pos = benchmark.macro_positions.numpy().astype(np.float64).copy()
        if _count_overlaps_kernel(pos, hw, hh, nh) > 0:
            _resolve_overlaps_kernel(pos, hw, hh, movable_mask_np, nh, cw, ch, 1000)
            if _count_overlaps_kernel(pos, hw, hh, nh) > 0:
                pos = self._greedy_row_place(pos, sizes, movable, cw, ch)

        # ─ FD initialization ─
        fd_pos = pos.copy()
        if movable.size >= 2 and adj_indices.size > 0:
            _force_directed_kernel(fd_pos, hw, hh, sizes, movable, nh,
                                   adj_indptr, adj_indices, adj_weights,
                                   cw, ch, 200)
            _resolve_overlaps_kernel(fd_pos, hw, hh, movable_mask_np, nh, cw, ch, 200)

        # Pick the better of (initial, FD) using full proxy cost
        cost_init = self._full_cost(plc, benchmark, pos)
        if _count_overlaps_kernel(fd_pos, hw, hh, nh) == 0:
            cost_fd = self._full_cost(plc, benchmark, fd_pos)
        else:
            cost_fd = float("inf")
        sa_start = fd_pos if cost_fd < cost_init else pos

        # ─ Calibrate T0 with a quick burst (Python is fine here) ─
        T0 = self._calibrate_T0(sa_start, hw, hh, movable, adj_indptr,
                                adj_indices, adj_weights, cw, ch)
        Tmin = T0 * 1e-5
        canvas_area = cw * ch
        avg_spacing = math.sqrt(canvas_area / max(movable.size, 1))
        base_disp = avg_spacing * 0.5

        # ─ Multi-seed SA ─
        best_pos = sa_start.copy()
        best_cost = self._full_cost(plc, benchmark, best_pos)
        seeds = [42 + 137 * i for i in range(self.n_seeds)]
        time_per_seed = max(5.0, (self.time_budget - (time.time() - t_start) - 5.0) / max(1, self.n_seeds))
        # iters per chunk: target ~0.5-1s in numba → calibrate dynamically
        chunk_iters = 200_000

        for s_i, seed in enumerate(seeds):
            if time.time() - t_start > self.time_budget - 2.0:
                break
            seed_state = np.array([seed], dtype=np.int64)
            T0_s = T0 * (1.0 + 0.2 * s_i)
            sd_pos = sa_start.copy()
            seed_t_start = time.time()
            seed_best_pos = sd_pos.copy()
            seed_best_cost = self._full_cost(plc, benchmark, seed_best_pos)

            # Geometric temperature schedule across the entire budget for this seed
            # Estimate how many chunks we'll do: at least 3, at most 12
            max_chunks = 12
            T = T0_s
            chunk_idx = 0
            while time.time() - seed_t_start < time_per_seed and chunk_idx < max_chunks:
                # Build per-iter T schedule for this chunk (geometric decay)
                T_end = max(Tmin, T * 0.5)
                alpha = (T_end / max(T, Tmin * 1.01)) ** (1.0 / chunk_iters)
                T_arr = np.empty(chunk_iters, dtype=np.float64)
                Ti = T
                for k in range(chunk_iters):
                    T_arr[k] = Ti
                    Ti = max(Tmin, Ti * alpha)
                T = Ti

                _sa_chunk(sd_pos, hw, hh, movable, nh,
                          adj_indptr, adj_indices, adj_weights,
                          cw, ch, base_disp, T_arr, chunk_iters,
                          grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows,
                          seed_state, 0.55, 0.23)

                # Repair any drift overlaps before scoring
                if _count_overlaps_kernel(sd_pos, hw, hh, nh) > 0:
                    _resolve_overlaps_kernel(sd_pos, hw, hh, movable_mask_np, nh, cw, ch, 500)
                # Only checkpoint if we have a valid (overlap-free) placement
                if _count_overlaps_kernel(sd_pos, hw, hh, nh) == 0:
                    cost_now = self._full_cost(plc, benchmark, sd_pos)
                    if cost_now < seed_best_cost:
                        seed_best_cost = cost_now
                        seed_best_pos = sd_pos.copy()
                # Restart from best to avoid drift
                sd_pos = seed_best_pos.copy()
                chunk_idx += 1

            if seed_best_cost < best_cost:
                best_cost = seed_best_cost
                best_pos = seed_best_pos

        # Final safety: ensure best_pos is overlap-free. Loop a few times
        # because resolve passes can briefly create new overlaps with neighbors.
        for _ in range(20):
            n_ov = _count_overlaps_kernel(best_pos, hw, hh, nh)
            if n_ov == 0:
                break
            _resolve_overlaps_kernel(best_pos, hw, hh, movable_mask_np, nh, cw, ch, 2000)

        # Build final tensor (re-anchor fixed and soft to original)
        result = torch.tensor(best_pos, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        sm = benchmark.get_soft_macro_mask()
        result[sm] = benchmark.macro_positions[sm]
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    def _load_plc(self, benchmark):
        for root in [
            Path("external/MacroPlacement/Testcases/ICCAD04"),
            Path(__file__).parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
        ]:
            d = root / benchmark.name
            if d.exists():
                _, plc = load_benchmark_from_dir(d.as_posix())
                return plc
        raise FileNotFoundError(f"Cannot find benchmark {benchmark.name}")

    def _full_cost(self, plc, benchmark, pos_np):
        pt = torch.tensor(pos_np, dtype=torch.float32)
        # restore fixed/soft
        pt[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        sm = benchmark.get_soft_macro_mask()
        pt[sm] = benchmark.macro_positions[sm]
        _set_placement(plc, pt, benchmark)
        _ensure_congestion_arrays(plc)
        return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()

    def _greedy_row_place(self, pos, sizes, movable, cw, ch):
        sorted_mov = sorted(movable.tolist(), key=lambda i: -sizes[i, 1])
        gap = 0.01
        cx, cy, rh = 0.0, 0.0, 0.0
        for idx in sorted_mov:
            w, h = sizes[idx, 0], sizes[idx, 1]
            if cx + w > cw:
                cx = 0.0
                cy += rh + gap
                rh = 0.0
            if cy + h > ch:
                pos[idx] = [w / 2, h / 2]
                continue
            pos[idx] = [cx + w / 2, cy + h / 2]
            cx += w + gap
            rh = max(rh, h)
        return pos

    def _calibrate_T0(self, pos, hw, hh, movable, adj_indptr, adj_indices, adj_weights, cw, ch):
        """Sample some random WL deltas to set T0 ≈ median |delta| / 0.2231."""
        rng = np.random.default_rng(12345)
        nm = movable.size
        canvas_area = cw * ch
        base_disp = math.sqrt(canvas_area / max(nm, 1)) * 0.5
        deltas = []
        for _ in range(300):
            idx = movable[rng.integers(nm)]
            ox = pos[idx, 0]
            oy = pos[idx, 1]
            nx = ox + rng.standard_normal() * base_disp
            ny = oy + rng.standard_normal() * base_disp
            nx = max(hw[idx], min(cw - hw[idx], nx))
            ny = max(hh[idx], min(ch - hh[idx], ny))
            d = 0.0
            for e in range(adj_indptr[idx], adj_indptr[idx + 1]):
                k = adj_indices[e]
                w = adj_weights[e]
                xk = pos[k, 0]
                yk = pos[k, 1]
                d += w * ((abs(nx - xk) + abs(ny - yk)) - (abs(ox - xk) + abs(oy - yk)))
            if abs(d) > 1e-12:
                deltas.append(abs(d))
        if not deltas:
            return 0.01
        return max(float(np.median(deltas)) / 0.2231, 1e-6)
