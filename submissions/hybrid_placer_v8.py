"""
Hybrid Macro Placer v8 — Multi-process Numba SA with adaptive budget + LNS.

Key improvements over v7:
1. Multi-process workers via fork(): 4 seeds run truly in parallel on the
   bee server's 4 cores. Each worker has its own PlacementCost for full-cost
   checkpointing.
2. Adaptive time budget: longer for larger benchmarks (ibm17/18 with 500+
   macros need more SA time than ibm01 with 246).
3. Smaller SA chunks (50K iters) → time-check is more responsive.
4. LNS kick: when N chunks pass without improvement, displace 15% of macros
   to nearby random positions to escape local minima.
5. Tighter SA temperature schedule and slightly reduced base_disp for the
   final chunks to converge into a tight optimum.
"""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from numba import njit

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, _ensure_congestion_arrays


OVERLAP_EPS = 0.01


# ─── Numba kernels ────────────────────────────────────────────────────────────


@njit(cache=False, fastmath=True, nogil=True)
def _has_overlap(pos, hw, hh, idx, nx, ny, n_hard, neighbors):
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


@njit(cache=False, fastmath=True, nogil=True)
def _grid_neighbors(pos, hw, hh, idx, nx, ny, n_hard, grid, grid_indptr,
                    grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows, pad_cells):
    hwi, hhi = hw[idx], hh[idx]
    x_lo = nx - hwi
    x_hi = nx + hwi
    y_lo = ny - hhi
    y_hi = ny + hhi
    c_lo = max(0, int((x_lo - grid_x0) / cell_w) - pad_cells)
    c_hi = min(n_cols - 1, int((x_hi - grid_x0) / cell_w) + pad_cells)
    r_lo = max(0, int((y_lo - grid_y0) / cell_h) - pad_cells)
    r_hi = min(n_rows - 1, int((y_hi - grid_y0) / cell_h) + pad_cells)
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


@njit(cache=False, fastmath=True, nogil=True)
def _build_grid(pos, n_hard, grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows):
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
    indptr = np.zeros(n_cells + 1, dtype=np.int32)
    for i in range(1, n_cells + 1):
        indptr[i] = indptr[i - 1] + counts[i]
    grid = np.empty(n_hard, dtype=np.int32)
    fill = indptr.copy()
    for i in range(n_hard):
        cell = cell_of[i]
        grid[fill[cell]] = i
        fill[cell] += 1
    return grid, indptr


@njit(cache=False, fastmath=True, nogil=True)
def _sa_chunk(pos, hw, hh, movable, n_hard,
              adj_indptr, adj_indices, adj_weights,
              cw, ch, base_disp, T_arr, n_iter,
              grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows, pad_cells,
              seed_state, p_random, p_anchor):
    np.random.seed(seed_state[0])
    nm = movable.shape[0]
    accepted = 0

    grid, grid_indptr = _build_grid(pos, n_hard, grid_x0, grid_y0,
                                    cell_w, cell_h, n_cols, n_rows)
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
            nx = ox + np.random.randn() * disp
            ny = oy + np.random.randn() * disp
        elif r < p_random + p_anchor:
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
            jj = np.random.randint(0, nm)
            if movable[jj] == idx:
                jj = (jj + 1) % nm
            j = movable[jj]
            oxj = pos[j, 0]
            oyj = pos[j, 1]
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

            pos[idx, 0] = nxi
            pos[idx, 1] = nyi
            pos[j, 0] = nxj
            pos[j, 1] = nyj

            nbrs_i = _grid_neighbors(pos, hw, hh, idx, nxi, nyi, n_hard,
                                     grid, grid_indptr, grid_x0, grid_y0,
                                     cell_w, cell_h, n_cols, n_rows, pad_cells)
            if _has_overlap(pos, hw, hh, idx, nxi, nyi, n_hard, nbrs_i):
                pos[idx, 0] = ox
                pos[idx, 1] = oy
                pos[j, 0] = oxj
                pos[j, 1] = oyj
                continue
            nbrs_j = _grid_neighbors(pos, hw, hh, j, nxj, nyj, n_hard,
                                     grid, grid_indptr, grid_x0, grid_y0,
                                     cell_w, cell_h, n_cols, n_rows, pad_cells)
            if _has_overlap(pos, hw, hh, j, nxj, nyj, n_hard, nbrs_j):
                pos[idx, 0] = ox
                pos[idx, 1] = oy
                pos[j, 0] = oxj
                pos[j, 1] = oyj
                continue

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

            if delta < 0.0 or np.random.random() < math.exp(-delta / (T if T > 1e-15 else 1e-15)):
                accepted += 1
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

        if nx < hw[idx]:
            nx = hw[idx]
        elif nx > cw - hw[idx]:
            nx = cw - hw[idx]
        if ny < hh[idx]:
            ny = hh[idx]
        elif ny > ch - hh[idx]:
            ny = ch - hh[idx]

        nbrs = _grid_neighbors(pos, hw, hh, idx, nx, ny, n_hard,
                               grid, grid_indptr, grid_x0, grid_y0,
                               cell_w, cell_h, n_cols, n_rows, pad_cells)
        if _has_overlap(pos, hw, hh, idx, nx, ny, n_hard, nbrs):
            continue

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
            iters_since_rebuild += 1
            if iters_since_rebuild >= grid_rebuild_every:
                grid, grid_indptr = _build_grid(pos, n_hard, grid_x0, grid_y0,
                                                cell_w, cell_h, n_cols, n_rows)
                iters_since_rebuild = 0

    seed_state[0] = np.random.randint(0, 2**31 - 1)
    return accepted


@njit(cache=False, fastmath=True, nogil=True)
def _force_directed_kernel(pos, hw, hh, sizes, movable, n_hard,
                           adj_indptr, adj_indices, adj_weights,
                           cw, ch, n_steps):
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
            for e in range(adj_indptr[idx], adj_indptr[idx + 1]):
                j = adj_indices[e]
                w = adj_weights[e]
                dx = pos[j, 0] - xi
                dy = pos[j, 1] - yi
                d = math.sqrt(dx * dx + dy * dy) + 1e-6
                f = ka * w * d
                forces[idx, 0] += f * dx / d
                forces[idx, 1] += f * dy / d
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
            dx_c = xi - cx_c
            dy_c = yi - cy_c
            d_c = math.sqrt(dx_c * dx_c + dy_c * dy_c) + 1e-6
            forces[idx, 0] += ks * dx_c / d_c * sizes[idx, 0]
            forces[idx, 1] += ks * dy_c / d_c * sizes[idx, 1]
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


@njit(cache=False, fastmath=True, nogil=True)
def _resolve_overlaps_kernel(pos, hw, hh, movable_mask, n_hard, cw, ch, max_passes):
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


@njit(cache=False, fastmath=True, nogil=True)
def _count_overlaps_kernel(pos, hw, hh, n_hard):
    cnt = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and \
               abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]:
                cnt += 1
    return cnt


@njit(cache=False, fastmath=True, nogil=True)
def _lns_kick(pos, hw, hh, movable, n_hard, cw, ch, frac, seed):
    """Randomly displace `frac` of movable macros to random positions, then
    legalize. This helps escape local minima."""
    np.random.seed(seed)
    nm = movable.shape[0]
    n_kick = max(1, int(nm * frac))
    for _ in range(n_kick):
        ii = np.random.randint(0, nm)
        idx = movable[ii]
        nx = hw[idx] + np.random.random() * (cw - 2 * hw[idx])
        ny = hh[idx] + np.random.random() * (ch - 2 * hh[idx])
        pos[idx, 0] = nx
        pos[idx, 1] = ny


# ─── Adjacency build ──────────────────────────────────────────────────────────


def _build_csr_adj(plc, benchmark):
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
                pa = (ml[a], ml[b])
                pb = (ml[b], ml[a])
                edge[pa] = edge.get(pa, 0.0) + w
                edge[pb] = edge.get(pb, 0.0) + w
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


# ─── Worker (runs in subprocess via fork) ────────────────────────────────────


def _sa_worker(args):
    """Worker process: load own PlacementCost, run SA chunks with checkpoints,
    return (best_pos, best_cost)."""
    (benchmark_dir, sa_start, hw, hh, movable, movable_mask_np, n_hard,
     adj_indptr, adj_indices, adj_weights, cw, ch, base_disp, T0, Tmin,
     grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows, pad_cells,
     seed, time_budget) = args

    # Load own PlacementCost in the worker
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    seed_state = np.array([seed], dtype=np.int64)
    sd_pos = sa_start.copy()
    chunk_iters = 50_000  # small chunks → responsive time-checking
    max_chunks = 200

    best_pos = sd_pos.copy()
    best_cost = _full_cost(plc, benchmark, best_pos)

    t_start = time.time()
    T = T0
    chunk_idx = 0
    no_improve_chunks = 0
    while time.time() - t_start < time_budget and chunk_idx < max_chunks:
        # Build geometric T schedule for chunk
        T_end = max(Tmin, T * 0.7)
        alpha = (T_end / max(T, Tmin * 1.01)) ** (1.0 / chunk_iters)
        T_arr = np.empty(chunk_iters, dtype=np.float64)
        Ti = T
        for k in range(chunk_iters):
            T_arr[k] = Ti
            Ti = max(Tmin, Ti * alpha)
        T = Ti

        _sa_chunk(sd_pos, hw, hh, movable, n_hard,
                  adj_indptr, adj_indices, adj_weights,
                  cw, ch, base_disp, T_arr, chunk_iters,
                  grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows, pad_cells,
                  seed_state, 0.55, 0.23)

        # Repair drift
        if _count_overlaps_kernel(sd_pos, hw, hh, n_hard) > 0:
            _resolve_overlaps_kernel(sd_pos, hw, hh, movable_mask_np, n_hard, cw, ch, 500)
        if _count_overlaps_kernel(sd_pos, hw, hh, n_hard) == 0:
            cost_now = _full_cost(plc, benchmark, sd_pos)
            if cost_now < best_cost - 1e-6:
                best_cost = cost_now
                best_pos = sd_pos.copy()
                no_improve_chunks = 0
            else:
                no_improve_chunks += 1
        else:
            no_improve_chunks += 1

        # LNS kick if stuck for too long
        if no_improve_chunks >= 5:
            sd_pos = best_pos.copy()
            _lns_kick(sd_pos, hw, hh, movable, n_hard, cw, ch, 0.15, seed + chunk_idx * 13)
            _resolve_overlaps_kernel(sd_pos, hw, hh, movable_mask_np, n_hard, cw, ch, 1000)
            T = T0 * 0.5  # reheat partway
            no_improve_chunks = 0
        else:
            sd_pos = best_pos.copy()

        chunk_idx += 1

    return best_pos, best_cost


def _full_cost(plc, benchmark, pos_np):
    pt = torch.tensor(pos_np, dtype=torch.float32)
    pt[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    sm = benchmark.get_soft_macro_mask()
    pt[sm] = benchmark.macro_positions[sm]
    _set_placement(plc, pt, benchmark)
    _ensure_congestion_arrays(plc)
    return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()


# ─── Adaptive budget ──────────────────────────────────────────────────────────


def _adaptive_budget(n_hard, base=60.0, max_budget=180.0):
    """Scale time budget by macro count. ibm01 (~246) gets base, ibm17/18
    (~520) gets ~2x. Capped at max_budget."""
    scale = min(2.5, max(1.0, n_hard / 250.0))
    return min(max_budget, base * scale)


# ─── Placer class ─────────────────────────────────────────────────────────────


class HybridPlacerV8:
    def __init__(self, base_budget=45.0, max_budget=120.0, n_seeds=4):
        self.base_budget = float(os.environ.get("HP8_BASE", base_budget))
        self.max_budget = float(os.environ.get("HP8_MAX", max_budget))
        self.n_seeds = int(os.environ.get("HP8_N_SEEDS", n_seeds))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = self._load_plc(benchmark)
        nh = benchmark.num_hard_macros

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

        # Spatial grid
        # Use larger cells when macros are large to reduce pad_cells need.
        max_hw = float(np.max(hw))
        max_hh = float(np.max(hh))
        cell_size_x = max(cw / max(8, int(math.sqrt(nh))), 2.5 * max_hw)
        cell_size_y = max(ch / max(8, int(math.sqrt(nh))), 2.5 * max_hh)
        n_cols = max(4, int(cw / cell_size_x))
        n_rows = max(4, int(ch / cell_size_y))
        cell_w = cw / n_cols
        cell_h = ch / n_rows
        grid_x0 = 0.0
        grid_y0 = 0.0
        # Pad neighbors by 1 cell (cell is 2.5 * max half-size, so always covers)
        pad_cells = 1

        # Initial pos: ensure overlap-free
        pos = benchmark.macro_positions.numpy().astype(np.float64).copy()
        if _count_overlaps_kernel(pos, hw, hh, nh) > 0:
            _resolve_overlaps_kernel(pos, hw, hh, movable_mask_np, nh, cw, ch, 1000)
            if _count_overlaps_kernel(pos, hw, hh, nh) > 0:
                pos = self._greedy_row_place(pos, sizes, movable, cw, ch)

        # FD init
        fd_pos = pos.copy()
        if movable.size >= 2 and adj_indices.size > 0:
            _force_directed_kernel(fd_pos, hw, hh, sizes, movable, nh,
                                   adj_indptr, adj_indices, adj_weights,
                                   cw, ch, 200)
            _resolve_overlaps_kernel(fd_pos, hw, hh, movable_mask_np, nh, cw, ch, 200)

        cost_init = _full_cost(plc, benchmark, pos)
        if _count_overlaps_kernel(fd_pos, hw, hh, nh) == 0:
            cost_fd = _full_cost(plc, benchmark, fd_pos)
        else:
            cost_fd = float("inf")
        sa_start = fd_pos if cost_fd < cost_init else pos

        # Calibrate T0
        T0 = self._calibrate_T0(sa_start, hw, hh, movable, adj_indptr,
                                adj_indices, adj_weights, cw, ch)
        Tmin = T0 * 1e-5
        canvas_area = cw * ch
        avg_spacing = math.sqrt(canvas_area / max(movable.size, 1))
        base_disp = avg_spacing * 0.5

        # Adaptive budget
        budget = _adaptive_budget(nh, base=self.base_budget, max_budget=self.max_budget)

        # Locate benchmark dir for workers to reload PlacementCost
        benchmark_dir = self._find_benchmark_dir(benchmark)

        # Build worker args
        worker_args = []
        for s_i in range(self.n_seeds):
            seed = 42 + 137 * s_i
            T0_s = T0 * (1.0 + 0.15 * s_i)
            worker_args.append((
                benchmark_dir, sa_start, hw, hh, movable, movable_mask_np, nh,
                adj_indptr, adj_indices, adj_weights, cw, ch, base_disp, T0_s, Tmin,
                grid_x0, grid_y0, cell_w, cell_h, n_cols, n_rows, pad_cells,
                seed, budget,
            ))

        # Run workers in parallel via threads. Numba kernels release the GIL
        # (nogil=True), so SA loops actually run in parallel on multiple cores.
        # PlacementCost calls hold the GIL but are infrequent (1 per chunk).
        if self.n_seeds > 1:
            with ThreadPoolExecutor(max_workers=min(self.n_seeds, os.cpu_count() or 4)) as pool:
                results = list(pool.map(_sa_worker, worker_args))
        else:
            results = [_sa_worker(worker_args[0])]

        # Pick best
        best_pos = sa_start
        best_cost = _full_cost(plc, benchmark, sa_start)
        for w_pos, w_cost in results:
            # Re-evaluate in main to be fully consistent
            if _count_overlaps_kernel(w_pos, hw, hh, nh) > 0:
                _resolve_overlaps_kernel(w_pos, hw, hh, movable_mask_np, nh, cw, ch, 1000)
            if _count_overlaps_kernel(w_pos, hw, hh, nh) == 0:
                cost_main = _full_cost(plc, benchmark, w_pos)
                if cost_main < best_cost:
                    best_cost = cost_main
                    best_pos = w_pos

        # Final safety
        for _ in range(20):
            if _count_overlaps_kernel(best_pos, hw, hh, nh) == 0:
                break
            _resolve_overlaps_kernel(best_pos, hw, hh, movable_mask_np, nh, cw, ch, 2000)

        result = torch.tensor(best_pos, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        sm = benchmark.get_soft_macro_mask()
        result[sm] = benchmark.macro_positions[sm]
        return result

    # Helpers --------------------------------------------------------------

    def _load_plc(self, benchmark):
        d = self._find_benchmark_dir(benchmark)
        _, plc = load_benchmark_from_dir(d)
        return plc

    def _find_benchmark_dir(self, benchmark):
        for root in [
            Path("external/MacroPlacement/Testcases/ICCAD04"),
            Path(__file__).parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
        ]:
            d = root / benchmark.name
            if d.exists():
                return d.as_posix()
        raise FileNotFoundError(f"Cannot find benchmark {benchmark.name}")

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
