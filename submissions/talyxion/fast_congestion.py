"""
fast_congestion.py — Talyxion fast congestion evaluator.

A faithful, Numba-compiled re-implementation of the TILOS PlacementCost
congestion model (get_routing + get_congestion_cost). The TILOS Python
version takes 0.6 s on ibm01 and 33 s on ibm17; this version targets
~5-30 ms, which is what makes coordinate-descent refinement on the real
proxy feasible.

This is a DIRECT TRANSLATION of the TILOS routing logic (2-pin / 3-pin /
split routing, macro blockage with the partial-overlap quirk, box
smoothing, abu top-5%). The numbers it produces match the TILOS evaluator;
it only changes the speed at which the search can be guided.
"""

import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _grid_cell(x, y, grid_w, grid_h, n_row, n_col):
    """Monkeypatched __get_grid_cell_location: floor + clamp to valid range."""
    row = int(math.floor(y / grid_h))
    col = int(math.floor(x / grid_w))
    if row < 0:
        row = 0
    elif row > n_row - 1:
        row = n_row - 1
    if col < 0:
        col = 0
    elif col > n_col - 1:
        col = n_col - 1
    return row, col


@njit(cache=True, fastmath=False)
def _route_2pin(V, H, n_col, src_r, src_c, snk_r, snk_c, weight):
    row_min = src_r if src_r < snk_r else snk_r
    row_max = src_r if src_r > snk_r else snk_r
    col_min = src_c if src_c < snk_c else snk_c
    col_max = src_c if src_c > snk_c else snk_c
    # H routing on source row
    for c in range(col_min, col_max):
        H[src_r * n_col + c] += weight
    # V routing on sink col
    for r in range(row_min, row_max):
        V[r * n_col + snk_c] += weight


@njit(cache=True, fastmath=False)
def _route_3pin(V, H, n_col, gy, gx, weight):
    """3-pin routing. gy/gx are length-3 unique-cell coords.
    Mirrors TILOS __three_pin_net_routing (sort by (col,row))."""
    # sort the 3 cells by (col, row)
    order = np.array([0, 1, 2], dtype=np.int64)
    for i in range(3):
        for j in range(i + 1, 3):
            a = order[i]
            bb = order[j]
            if (gx[bb] < gx[a]) or (gx[bb] == gx[a] and gy[bb] < gy[a]):
                order[i] = bb
                order[j] = a
    y1 = gy[order[0]]; x1 = gx[order[0]]
    y2 = gy[order[1]]; x2 = gx[order[1]]
    y3 = gy[order[2]]; x3 = gx[order[2]]

    if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
        # __l_routing — sorted by (col,row) already
        for c in range(x1, x2):
            H[y1 * n_col + c] += weight
        for c in range(x2, x3):
            H[y2 * n_col + c] += weight
        for r in range(min(y1, y2), max(y1, y2)):
            V[r * n_col + x2] += weight
        for r in range(min(y2, y3), max(y2, y3)):
            V[r * n_col + x3] += weight
    elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
        for c in range(x1, x2):
            H[y1 * n_col + c] += weight
        for r in range(y1, max(y2, y3)):
            V[r * n_col + x2] += weight
    elif y2 == y3:
        for c in range(x1, x2):
            H[y1 * n_col + c] += weight
        for c in range(x2, x3):
            H[y2 * n_col + c] += weight
        for r in range(min(y2, y1), max(y2, y1)):
            V[r * n_col + x2] += weight
    else:
        # __t_routing — TILOS sorts by plain tuple (row, col)
        oy = np.array([0, 1, 2], dtype=np.int64)
        for i in range(3):
            for j in range(i + 1, 3):
                a = oy[i]
                bb = oy[j]
                if (gy[bb] < gy[a]) or (gy[bb] == gy[a] and gx[bb] < gx[a]):
                    oy[i] = bb
                    oy[j] = a
        ty1 = gy[oy[0]]; tx1 = gx[oy[0]]
        ty2 = gy[oy[1]]; tx2 = gx[oy[1]]
        ty3 = gy[oy[2]]; tx3 = gx[oy[2]]
        xmin = min(tx1, tx2, tx3)
        xmax = max(tx1, tx2, tx3)
        for c in range(xmin, xmax):
            H[ty2 * n_col + c] += weight
        for r in range(min(ty1, ty2), max(ty1, ty2)):
            V[r * n_col + tx1] += weight
        for r in range(min(ty2, ty3), max(ty2, ty3)):
            V[r * n_col + tx3] += weight


@njit(cache=True, fastmath=False)
def _macro_route(Vm, Hm, n_row, n_col, grid_w, grid_h,
                 mx, my, mw, mh, valloc, halloc):
    """__macro_route_over_grid_cell — macro blockage with partial-overlap quirk."""
    x_max_m = mx + mw * 0.5
    x_min_m = mx - mw * 0.5
    y_max_m = my + mh * 0.5
    y_min_m = my - mh * 0.5

    ur_r, ur_c = _grid_cell(x_max_m, y_max_m, grid_w, grid_h, n_row, n_col)
    bl_r, bl_c = _grid_cell(x_min_m, y_min_m, grid_w, grid_h, n_row, n_col)

    # TILOS OOB checks operate on the RAW (unclamped) corners; but with the
    # monkeypatched clamp the corners are already valid, so bl<=ur always.
    if bl_r < 0:
        bl_r = 0
    if bl_c < 0:
        bl_c = 0
    if ur_r > n_row - 1:
        ur_r = n_row - 1
    if ur_c > n_col - 1:
        ur_c = n_col - 1

    part_v = False
    part_h = False
    for r_i in range(bl_r, ur_r + 1):
        for c_i in range(bl_c, ur_c + 1):
            cell_xmin = c_i * grid_w
            cell_xmax = (c_i + 1) * grid_w
            cell_ymin = r_i * grid_h
            cell_ymax = (r_i + 1) * grid_h
            xd = min(x_max_m, cell_xmax) - max(x_min_m, cell_xmin)
            yd = min(y_max_m, cell_ymax) - max(y_min_m, cell_ymin)
            if xd <= 0.0 or yd <= 0.0:
                xd = 0.0
                yd = 0.0
            if ur_r != bl_r:
                if (r_i == bl_r and abs(yd - grid_h) > 1e-5) or \
                   (r_i == ur_r and abs(yd - grid_h) > 1e-5):
                    part_v = True
            if ur_c != bl_c:
                if (c_i == bl_c and abs(xd - grid_w) > 1e-5) or \
                   (c_i == ur_c and abs(xd - grid_w) > 1e-5):
                    part_h = True
            Vm[r_i * n_col + c_i] += xd * valloc
            Hm[r_i * n_col + c_i] += yd * halloc

    if part_v:
        r_i = ur_r
        for c_i in range(bl_c, ur_c + 1):
            cell_xmin = c_i * grid_w
            cell_xmax = (c_i + 1) * grid_w
            cell_ymin = r_i * grid_h
            cell_ymax = (r_i + 1) * grid_h
            xd = min(x_max_m, cell_xmax) - max(x_min_m, cell_xmin)
            yd = min(y_max_m, cell_ymax) - max(y_min_m, cell_ymin)
            if xd <= 0.0 or yd <= 0.0:
                xd = 0.0
            Vm[r_i * n_col + c_i] -= xd * valloc

    if part_h:
        c_i = ur_c
        for r_i in range(bl_r, ur_r + 1):
            cell_xmin = c_i * grid_w
            cell_xmax = (c_i + 1) * grid_w
            cell_ymin = r_i * grid_h
            cell_ymax = (r_i + 1) * grid_h
            xd = min(x_max_m, cell_xmax) - max(x_min_m, cell_xmin)
            yd = min(y_max_m, cell_ymax) - max(y_min_m, cell_ymin)
            if xd <= 0.0 or yd <= 0.0:
                yd = 0.0
            Hm[r_i * n_col + c_i] -= yd * halloc


@njit(cache=True, fastmath=False)
def _smooth(cong, n_row, n_col, smooth_range, is_vertical):
    """__smooth_routing_cong — box smoothing. V spreads over cols, H over rows."""
    out = np.zeros(n_row * n_col, dtype=np.float64)
    if is_vertical:
        for row in range(n_row):
            for col in range(n_col):
                lp = col - smooth_range
                if lp < 0:
                    lp = 0
                rp = col + smooth_range
                if rp >= n_col:
                    rp = n_col - 1
                cnt = rp - lp + 1
                val = cong[row * n_col + col] / cnt
                for ptr in range(lp, rp + 1):
                    out[row * n_col + ptr] += val
    else:
        for row in range(n_row):
            for col in range(n_col):
                lp = row - smooth_range
                if lp < 0:
                    lp = 0
                up = row + smooth_range
                if up >= n_row:
                    up = n_row - 1
                cnt = up - lp + 1
                val = cong[row * n_col + col] / cnt
                for ptr in range(lp, up + 1):
                    out[ptr * n_col + col] += val
    return out


@njit(cache=True, fastmath=False)
def compute_congestion(pos, pin_macro, pin_offx, pin_offy, pin_fx, pin_fy,
                       pin_is_fixed, net_start, net_weight, net_npins,
                       hard_mx, hard_my, hard_mw, hard_mh, n_hard,
                       n_row, n_col, grid_w, grid_h,
                       grid_v_routes, grid_h_routes, valloc, halloc,
                       smooth_range):
    """Full TILOS congestion: returns abu(V+H, 0.05)."""
    n_cells = n_row * n_col
    V = np.zeros(n_cells, dtype=np.float64)
    H = np.zeros(n_cells, dtype=np.float64)
    Vm = np.zeros(n_cells, dtype=np.float64)
    Hm = np.zeros(n_cells, dtype=np.float64)
    n_nets = net_start.shape[0]

    # ── Net routing ──────────────────────────────────────────────────────
    max_pins = 0
    for i in range(n_nets):
        if net_npins[i] > max_pins:
            max_pins = net_npins[i]
    cell_r = np.empty(max_pins, dtype=np.int64)
    cell_c = np.empty(max_pins, dtype=np.int64)
    uniq_r = np.empty(max_pins, dtype=np.int64)
    uniq_c = np.empty(max_pins, dtype=np.int64)

    for i in range(n_nets):
        s = net_start[i]
        npins = net_npins[i]
        w = net_weight[i]
        # pin 0 = driver/source
        for p in range(npins):
            idx = s + p
            if pin_is_fixed[idx]:
                px = pin_fx[idx]
                py = pin_fy[idx]
            else:
                m = pin_macro[idx]
                px = pos[m, 0] + pin_offx[idx]
                py = pos[m, 1] + pin_offy[idx]
            r, c = _grid_cell(px, py, grid_w, grid_h, n_row, n_col)
            cell_r[p] = r
            cell_c[p] = c
        src_r = cell_r[0]
        src_c = cell_c[0]
        # unique cells
        n_uniq = 0
        for p in range(npins):
            r = cell_r[p]
            c = cell_c[p]
            found = False
            for q in range(n_uniq):
                if uniq_r[q] == r and uniq_c[q] == c:
                    found = True
                    break
            if not found:
                uniq_r[n_uniq] = r
                uniq_c[n_uniq] = c
                n_uniq += 1
        if n_uniq == 2:
            # identify sink (the non-source unique cell)
            if uniq_r[0] == src_r and uniq_c[0] == src_c:
                sr = uniq_r[1]; sc = uniq_c[1]
            else:
                sr = uniq_r[0]; sc = uniq_c[0]
            _route_2pin(V, H, n_col, src_r, src_c, sr, sc, w)
        elif n_uniq == 3:
            gy = np.empty(3, dtype=np.int64)
            gx = np.empty(3, dtype=np.int64)
            for q in range(3):
                gy[q] = uniq_r[q]
                gx[q] = uniq_c[q]
            _route_3pin(V, H, n_col, gy, gx, w)
        elif n_uniq > 3:
            # split: each non-source unique cell paired with source
            for q in range(n_uniq):
                if uniq_r[q] == src_r and uniq_c[q] == src_c:
                    continue
                _route_2pin(V, H, n_col, src_r, src_c, uniq_r[q], uniq_c[q], w)

    # ── Macro blockage ───────────────────────────────────────────────────
    for m in range(n_hard):
        _macro_route(Vm, Hm, n_row, n_col, grid_w, grid_h,
                     hard_mx[m], hard_my[m], hard_mw[m], hard_mh[m],
                     valloc, halloc)

    # ── Normalize ────────────────────────────────────────────────────────
    for k in range(n_cells):
        V[k] = V[k] / grid_v_routes
        H[k] = H[k] / grid_h_routes
        Vm[k] = Vm[k] / grid_v_routes
        Hm[k] = Hm[k] / grid_h_routes

    # ── Smooth (routing only, not macro) ─────────────────────────────────
    Vs = _smooth(V, n_row, n_col, smooth_range, True)
    Hs = _smooth(H, n_row, n_col, smooth_range, False)

    # ── Sum routing + macro ──────────────────────────────────────────────
    combined = np.empty(2 * n_cells, dtype=np.float64)
    for k in range(n_cells):
        combined[k] = Vs[k] + Vm[k]
        combined[n_cells + k] = Hs[k] + Hm[k]

    # ── abu(combined, 0.05): mean of top 5% ──────────────────────────────
    total = combined.shape[0]
    cnt = int(math.floor(total * 0.05))
    srt = np.sort(combined)[::-1]
    if cnt == 0:
        return srt[0]
    s = 0.0
    for k in range(cnt):
        s += srt[k]
    return s / cnt


class FastCongestion:
    """Builds the flat tables once, then evaluates congestion fast."""

    def __init__(self, plc, benchmark):
        self.plc = plc
        self.benchmark = benchmark
        self._build()

    def _build(self):
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

        flat_macro = []
        flat_offx = []
        flat_offy = []
        flat_fx = []
        flat_fy = []
        flat_fixed = []
        net_start = []
        net_npins = []
        net_weight = []
        cursor = 0
        for driver_name, sinks in plc.nets.items():
            d_idx = name_to_idx[driver_name]
            w = mods[d_idx].get_weight()
            pin_idxs = [d_idx] + [name_to_idx[s] for s in sinks]
            cnt = 0
            for p_idx in pin_idxs:
                pin = mods[p_idx]
                if pin.get_type() == 'PORT':
                    px, py = pin.get_pos()
                    flat_macro.append(0)
                    flat_offx.append(0.0)
                    flat_offy.append(0.0)
                    flat_fx.append(px)
                    flat_fy.append(py)
                    flat_fixed.append(True)
                else:
                    rn = plc.get_ref_node_id(p_idx)
                    ox, oy = pin.get_offset()
                    bm = plc_to_bench.get(rn, None)
                    if bm is None:
                        rx, ry = mods[rn].get_pos()
                        flat_macro.append(0)
                        flat_offx.append(0.0)
                        flat_offy.append(0.0)
                        flat_fx.append(rx + ox)
                        flat_fy.append(ry + oy)
                        flat_fixed.append(True)
                    else:
                        flat_macro.append(bm)
                        flat_offx.append(ox)
                        flat_offy.append(oy)
                        flat_fx.append(0.0)
                        flat_fy.append(0.0)
                        flat_fixed.append(False)
                cnt += 1
            net_start.append(cursor)
            net_npins.append(cnt)
            net_weight.append(w)
            cursor += cnt

        self.pin_macro = np.array(flat_macro, dtype=np.int64)
        self.pin_offx = np.array(flat_offx, dtype=np.float64)
        self.pin_offy = np.array(flat_offy, dtype=np.float64)
        self.pin_fx = np.array(flat_fx, dtype=np.float64)
        self.pin_fy = np.array(flat_fy, dtype=np.float64)
        self.pin_is_fixed = np.array(flat_fixed, dtype=np.bool_)
        self.net_start = np.array(net_start, dtype=np.int64)
        self.net_npins = np.array(net_npins, dtype=np.int64)
        self.net_weight = np.array(net_weight, dtype=np.float64)

        # hard macro geometry
        sizes = b.macro_sizes.numpy().astype(np.float64)
        self.hard_mw = sizes[:nh, 0].copy()
        self.hard_mh = sizes[:nh, 1].copy()
        self.n_hard = nh

        # grid params
        self.n_row = plc.grid_row
        self.n_col = plc.grid_col
        self.grid_w = float(plc.width) / plc.grid_col
        self.grid_h = float(plc.height) / plc.grid_row
        self.grid_v_routes = self.grid_w * plc.vroutes_per_micron
        self.grid_h_routes = self.grid_h * plc.hroutes_per_micron
        self.valloc = plc.vrouting_alloc
        self.halloc = plc.hrouting_alloc
        self.smooth_range = int(plc.smooth_range)

    def cost(self, pos_np):
        """pos_np: [N,2] float64 macro centers (hard first). Returns congestion."""
        hard_mx = np.ascontiguousarray(pos_np[:self.n_hard, 0])
        hard_my = np.ascontiguousarray(pos_np[:self.n_hard, 1])
        return compute_congestion(
            pos_np, self.pin_macro, self.pin_offx, self.pin_offy,
            self.pin_fx, self.pin_fy, self.pin_is_fixed,
            self.net_start, self.net_weight, self.net_npins,
            hard_mx, hard_my, self.hard_mw, self.hard_mh, self.n_hard,
            self.n_row, self.n_col, self.grid_w, self.grid_h,
            self.grid_v_routes, self.grid_h_routes,
            self.valloc, self.halloc, self.smooth_range)
