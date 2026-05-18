"""
fasteval.py — Talyxion unified fast proxy evaluator.

Combines fast wirelength (vectorized numpy), fast density (Numba), and fast
congestion (Numba) into one evaluator. The full proxy drops from ~1.6 s
(TILOS compute_proxy_cost) to ~8-15 ms, which makes coordinate-descent
refinement on the real objective feasible.

Wirelength / density / congestion are faithful re-implementations of the
TILOS PlacementCost functions (get_wirelength, get_grid_cells_density,
get_routing). They only change evaluation SPEED; the final placement is
still scored by the unmodified TILOS evaluator.

Density optimization: soft macros are fixed, so their contribution to the
density grid is precomputed once; each eval only re-adds the hard macros.
"""

import math
import numpy as np
from numba import njit

from fast_congestion import FastCongestion, compute_congestion


# ── Fast density ─────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=False)
def _add_macros_to_grid(grid, hard_mx, hard_my, hard_mw, hard_mh, n_hard,
                        n_row, n_col, grid_w, grid_h):
    """Add hard-macro overlap areas onto a (copy of the soft) density grid."""
    for m in range(n_hard):
        x_max_m = hard_mx[m] + hard_mw[m] * 0.5
        x_min_m = hard_mx[m] - hard_mw[m] * 0.5
        y_max_m = hard_my[m] + hard_mh[m] * 0.5
        y_min_m = hard_my[m] - hard_mh[m] * 0.5
        ur_r = int(math.floor(y_max_m / grid_h))
        ur_c = int(math.floor(x_max_m / grid_w))
        bl_r = int(math.floor(y_min_m / grid_h))
        bl_c = int(math.floor(x_min_m / grid_w))
        if ur_r < 0 or ur_c < 0:
            continue
        if bl_r < 0:
            bl_r = 0
        if bl_c < 0:
            bl_c = 0
        if bl_r > n_row - 1 or bl_c > n_col - 1:
            continue
        if ur_r > n_row - 1:
            ur_r = n_row - 1
        if ur_c > n_col - 1:
            ur_c = n_col - 1
        for r_i in range(bl_r, ur_r + 1):
            for c_i in range(bl_c, ur_c + 1):
                cell_xmin = c_i * grid_w
                cell_xmax = (c_i + 1) * grid_w
                cell_ymin = r_i * grid_h
                cell_ymax = (r_i + 1) * grid_h
                xd = min(x_max_m, cell_xmax) - max(x_min_m, cell_xmin)
                yd = min(y_max_m, cell_ymax) - max(y_min_m, cell_ymin)
                if xd >= 0.0 and yd >= 0.0:
                    grid[r_i * n_col + c_i] += xd * yd


@njit(cache=True, fastmath=False)
def _density_cost_from_grid(grid, n_cells, grid_area):
    """0.5 * mean of top-10%-of-all-cells, drawn from nonzero cells."""
    dens = np.empty(n_cells, dtype=np.float64)
    n_occ = 0
    for k in range(n_cells):
        v = grid[k] / grid_area
        if v != 0.0:
            dens[n_occ] = v
            n_occ += 1
    cnt = int(math.floor(n_cells * 0.1))
    occ = np.sort(dens[:n_occ])[::-1]
    if n_cells < 10:
        s = 0.0
        for k in range(n_occ):
            s += occ[k]
        return 0.5 * (s / n_occ) if n_occ > 0 else 0.0
    s = 0.0
    idx = 0
    while idx < cnt and idx < n_occ:
        s += occ[idx]
        idx += 1
    return 0.5 * (s / cnt) if cnt > 0 else 0.0


@njit(cache=True, fastmath=False)
def _wirelength(positions, flat_macro_safe, flat_offx, flat_offy,
                flat_fixedx, flat_fixedy, flat_is_fixed,
                net_start, net_npins, net_weight, denom):
    """Vectorized HPWL — Numba version matching plc.get_wirelength()."""
    n_nets = net_start.shape[0]
    total = 0.0
    for i in range(n_nets):
        s = net_start[i]
        npins = net_npins[i]
        # first pin
        if flat_is_fixed[s]:
            x = flat_fixedx[s]
            y = flat_fixedy[s]
        else:
            m = flat_macro_safe[s]
            x = positions[m, 0] + flat_offx[s]
            y = positions[m, 1] + flat_offy[s]
        xmin = x; xmax = x; ymin = y; ymax = y
        for p in range(1, npins):
            idx = s + p
            if flat_is_fixed[idx]:
                x = flat_fixedx[idx]
                y = flat_fixedy[idx]
            else:
                m = flat_macro_safe[idx]
                x = positions[m, 0] + flat_offx[idx]
                y = positions[m, 1] + flat_offy[idx]
            if x < xmin:
                xmin = x
            elif x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            elif y > ymax:
                ymax = y
        total += net_weight[i] * ((xmax - xmin) + (ymax - ymin))
    return total / denom


class FastEval:
    """Unified fast proxy = WL + 0.5*density + 0.5*congestion."""

    def __init__(self, plc, benchmark):
        self.plc = plc
        self.benchmark = benchmark
        self.fc = FastCongestion(plc, benchmark)
        self._build_wl()
        self._build_density()

    # ── WL tables ────────────────────────────────────────────────────────
    def _build_wl(self):
        plc = self.plc
        b = self.benchmark
        mods = plc.modules_w_pins
        name_to_idx = plc.mod_name_to_indices
        plc_to_bench = {}
        for bi, pi in enumerate(b.hard_macro_indices):
            plc_to_bench[pi] = bi
        nh = b.num_hard_macros
        for bi, pi in enumerate(b.soft_macro_indices):
            plc_to_bench[pi] = nh + bi

        fm, fox, foy, ffx, ffy, ffix = [], [], [], [], [], []
        nstart, nnp, nw = [], [], []
        cur = 0
        for driver_name, sinks in plc.nets.items():
            d_idx = name_to_idx[driver_name]
            w = mods[d_idx].get_weight()
            pin_idxs = [d_idx] + [name_to_idx[s] for s in sinks]
            cnt = 0
            for p_idx in pin_idxs:
                pin = mods[p_idx]
                if pin.get_type() == 'PORT':
                    px, py = pin.get_pos()
                    fm.append(0); fox.append(0.0); foy.append(0.0)
                    ffx.append(px); ffy.append(py); ffix.append(True)
                else:
                    rn = plc.get_ref_node_id(p_idx)
                    ox, oy = pin.get_offset()
                    bm = plc_to_bench.get(rn, None)
                    if bm is None:
                        rx, ry = mods[rn].get_pos()
                        fm.append(0); fox.append(0.0); foy.append(0.0)
                        ffx.append(rx + ox); ffy.append(ry + oy); ffix.append(True)
                    else:
                        fm.append(bm); fox.append(ox); foy.append(oy)
                        ffx.append(0.0); ffy.append(0.0); ffix.append(False)
                cnt += 1
            nstart.append(cur); nnp.append(cnt); nw.append(w)
            cur += cnt

        self.wl_macro = np.clip(np.array(fm, dtype=np.int64), 0, None)
        self.wl_offx = np.array(fox, dtype=np.float64)
        self.wl_offy = np.array(foy, dtype=np.float64)
        self.wl_fx = np.array(ffx, dtype=np.float64)
        self.wl_fy = np.array(ffy, dtype=np.float64)
        self.wl_fixed = np.array(ffix, dtype=np.bool_)
        self.wl_start = np.array(nstart, dtype=np.int64)
        self.wl_npins = np.array(nnp, dtype=np.int64)
        self.wl_weight = np.array(nw, dtype=np.float64)
        W, H = plc.get_canvas_width_height()
        net_cnt = plc.net_cnt if plc.net_cnt != 0 else 1
        self.wl_denom = (W + H) * net_cnt

    # ── Density tables ───────────────────────────────────────────────────
    def _build_density(self):
        plc = self.plc
        b = self.benchmark
        nh = b.num_hard_macros
        self.n_row = plc.grid_row
        self.n_col = plc.grid_col
        self.grid_w = float(plc.width) / plc.grid_col
        self.grid_h = float(plc.height) / plc.grid_row
        self.grid_area = self.grid_w * self.grid_h
        self.n_cells = self.n_row * self.n_col

        sizes = b.macro_sizes.numpy().astype(np.float64)
        # Sizes for ALL macros (hard then soft) — density sums over all.
        self.all_mw = np.ascontiguousarray(sizes[:, 0])
        self.all_mh = np.ascontiguousarray(sizes[:, 1])
        self.n_macros = b.num_macros
        self.n_hard = nh
        # hard sizes kept for the congestion macro-blockage path
        self.hard_mw = np.ascontiguousarray(sizes[:nh, 0])
        self.hard_mh = np.ascontiguousarray(sizes[:nh, 1])

    # ── Evaluators ───────────────────────────────────────────────────────
    def wirelength(self, pos_np):
        return _wirelength(pos_np, self.wl_macro, self.wl_offx, self.wl_offy,
                           self.wl_fx, self.wl_fy, self.wl_fixed,
                           self.wl_start, self.wl_npins, self.wl_weight,
                           self.wl_denom)

    def density(self, pos_np):
        # Sum over ALL macros (hard + soft) from the supplied placement —
        # no stale precompute, always consistent with pos_np.
        grid = np.zeros(self.n_cells, dtype=np.float64)
        mx = np.ascontiguousarray(pos_np[:, 0])
        my = np.ascontiguousarray(pos_np[:, 1])
        _add_macros_to_grid(grid, mx, my, self.all_mw, self.all_mh,
                            self.n_macros, self.n_row, self.n_col,
                            self.grid_w, self.grid_h)
        return _density_cost_from_grid(grid, self.n_cells, self.grid_area)

    def congestion(self, pos_np):
        return self.fc.cost(pos_np)

    def proxy(self, pos_np):
        wl = self.wirelength(pos_np)
        d = self.density(pos_np)
        c = self.congestion(pos_np)
        return wl + 0.5 * d + 0.5 * c, wl, d, c
