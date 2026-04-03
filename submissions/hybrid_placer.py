"""
Hybrid Macro Placer - Competition Submission (v6)

Key: Fast WL-surrogate SA with adaptive radius + multi-start (2 seeds).
More frequent full-cost checkpoints to properly track density/congestion.
Force-directed init with density spreading.
"""

import math
import random
import time
import numpy as np
import torch
from pathlib import Path

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement, _ensure_congestion_arrays


class HybridPlacer:

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = self._load_plc(benchmark)
        placement = benchmark.macro_positions.clone()
        movable = self._get_movable(benchmark)

        if not movable:
            return placement

        sizes = benchmark.macro_sizes.numpy()
        nh = benchmark.num_hard_macros
        hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        movset = set(movable)

        try:
            adj = self._build_adj(plc, benchmark)
        except Exception:
            adj = {}

        # Ensure overlap-free start
        start_pos = placement.numpy().copy()
        if self._count_overlaps(start_pos, hw, hh, nh) > 0:
            self._resolve_overlaps(start_pos, hw, hh, movset, nh, cw, ch)
            if self._count_overlaps(start_pos, hw, hh, nh) > 0:
                start_pos = self._greedy_row_place(start_pos, sizes, movable, cw, ch)

        start = torch.tensor(start_pos, dtype=torch.float32)
        start[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        sm = benchmark.get_soft_macro_mask()
        start[sm] = benchmark.macro_positions[sm]

        # Force-directed initialization
        fd = self._force_directed(start.clone(), benchmark, plc, movable, adj,
                                   hw, hh, movset)
        if self._count_overlaps(fd.numpy(), hw, hh, nh) == 0:
            _set_placement(plc, fd, benchmark)
            _ensure_congestion_arrays(plc)
            fd_cost = self._full_cost(plc)
            _set_placement(plc, start, benchmark)
            _ensure_congestion_arrays(plc)
            init_cost = self._full_cost(plc)
            sa_start = fd if fd_cost < init_cost else start
        else:
            sa_start = start

        # Multi-start: 2 seeds, take best
        best_result = None
        best_cost = float('inf')

        for seed_idx in range(2):
            seed = 42 + seed_idx * 137
            random.seed(seed)
            np.random.seed(seed)

            plc_run = self._load_plc(benchmark)
            result = self._sa(sa_start.clone(), benchmark, plc_run, movable, adj,
                             hw, hh, seed_idx)

            if self._count_overlaps(result.numpy(), hw, hh, nh) > 0:
                continue

            _set_placement(plc_run, result, benchmark)
            _ensure_congestion_arrays(plc_run)
            cost = self._full_cost(plc_run)
            if cost < best_cost:
                best_cost = cost
                best_result = result

        if best_result is None:
            return sa_start
        return best_result

    # ── Helpers ──────────────────────────────────────────────────────────

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

    def _get_movable(self, benchmark):
        mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        return torch.where(mask)[0].tolist()

    def _full_cost(self, plc):
        return plc.get_cost() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()

    def _count_overlaps(self, pos, hw, hh, num_hard):
        count = 0
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                if (abs(pos[i, 0] - pos[j, 0]) < hw[i] + hw[j] and
                        abs(pos[i, 1] - pos[j, 1]) < hh[i] + hh[j]):
                    count += 1
        return count

    def _overlap(self, pos, hw, hh, idx, num_hard):
        xi, yi = pos[idx, 0], pos[idx, 1]
        dx = np.abs(pos[:num_hard, 0] - xi)
        dy = np.abs(pos[:num_hard, 1] - yi)
        ov = (dx < (hw[idx] + hw[:num_hard])) & (dy < (hh[idx] + hh[:num_hard]))
        ov[idx] = False
        return ov.any()

    def _resolve_overlaps(self, pos, hw, hh, movset, nh, cw, ch):
        for _ in range(1000):
            found = False
            for i in range(nh):
                for j in range(i + 1, nh):
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    mdx, mdy = hw[i] + hw[j], hh[i] + hh[j]
                    if dx < mdx and dy < mdy:
                        found = True
                        ox, oy = mdx - dx, mdy - dy
                        if ox < oy:
                            push = ox / 2 + 0.1
                            sign = 1 if pos[i, 0] < pos[j, 0] else -1
                            if i in movset: pos[i, 0] -= sign * push
                            if j in movset: pos[j, 0] += sign * push
                        else:
                            push = oy / 2 + 0.1
                            sign = 1 if pos[i, 1] < pos[j, 1] else -1
                            if i in movset: pos[i, 1] -= sign * push
                            if j in movset: pos[j, 1] += sign * push
                        for k in (i, j):
                            if k in movset:
                                pos[k, 0] = np.clip(pos[k, 0], hw[k], cw - hw[k])
                                pos[k, 1] = np.clip(pos[k, 1], hh[k], ch - hh[k])
            if not found:
                break

    def _greedy_row_place(self, pos, sizes, movable, cw, ch):
        sorted_mov = sorted(movable, key=lambda i: -sizes[i, 1])
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

    def _build_adj(self, plc, benchmark):
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
        adj = {}
        for driver, sinks in plc.nets.items():
            macros = set()
            if driver in pin2macro: macros.add(pin2macro[driver])
            for s in sinks:
                if s in pin2macro: macros.add(pin2macro[s])
            ml = list(macros)
            if len(ml) < 2: continue
            w = 1.0 / len(ml)
            for a in range(len(ml)):
                for b in range(a + 1, len(ml)):
                    adj.setdefault(ml[a], {})
                    adj[ml[a]][ml[b]] = adj[ml[a]].get(ml[b], 0) + w
                    adj.setdefault(ml[b], {})
                    adj[ml[b]][ml[a]] = adj[ml[b]].get(ml[a], 0) + w
        return {k: list(v.items()) for k, v in adj.items()}

    # ── Force-Directed ───────────────────────────────────────────────────

    def _force_directed(self, placement, benchmark, plc, movable, adj, hw, hh, movset):
        if len(movable) < 2 or not adj:
            return placement
        pos = placement.numpy().copy()
        sizes = benchmark.macro_sizes.numpy()
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        nh = benchmark.num_hard_macros
        vel = np.zeros_like(pos)
        cx_c, cy_c = cw / 2, ch / 2

        for step in range(300):
            p = step / 300
            forces = np.zeros_like(pos)
            ka = 0.001 * (1 + 2 * p)
            kr = 2000.0 * (1 - 0.7 * p)
            ks = 0.0005 * (1 + p)
            for idx in movable:
                xi, yi = pos[idx]
                for j, w in adj.get(idx, []):
                    dx, dy = pos[j, 0] - xi, pos[j, 1] - yi
                    d = math.sqrt(dx * dx + dy * dy) + 1e-6
                    f = ka * w * d
                    forces[idx, 0] += f * dx / d
                    forces[idx, 1] += f * dy / d
                for j in range(nh):
                    if j == idx: continue
                    dx, dy = xi - pos[j, 0], yi - pos[j, 1]
                    dsq = dx * dx + dy * dy + 1.0
                    d = math.sqrt(dsq)
                    ms = sizes[idx, 0] + sizes[j, 0] + sizes[idx, 1] + sizes[j, 1]
                    if d < ms:
                        f = kr / dsq
                        forces[idx, 0] += f * dx / d
                        forces[idx, 1] += f * dy / d
                dx_c = xi - cx_c
                dy_c = yi - cy_c
                d_c = math.sqrt(dx_c**2 + dy_c**2) + 1e-6
                forces[idx, 0] += ks * dx_c / d_c * sizes[idx, 0]
                forces[idx, 1] += ks * dy_c / d_c * sizes[idx, 1]
            for idx in movable:
                vel[idx] = 0.85 * vel[idx] + 0.3 * forces[idx]
                pos[idx] += 0.3 * vel[idx]
                pos[idx, 0] = np.clip(pos[idx, 0], hw[idx], cw - hw[idx])
                pos[idx, 1] = np.clip(pos[idx, 1], hh[idx], ch - hh[idx])

        self._resolve_overlaps(pos, hw, hh, movset, nh, cw, ch)
        result = torch.tensor(pos, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        sm = benchmark.get_soft_macro_mask()
        result[sm] = benchmark.macro_positions[sm]
        return result

    # ── SA ────────────────────────────────────────────────────────────────

    def _sa(self, placement, benchmark, plc, movable, adj, hw, hh, seed_idx):
        pos = placement.numpy().copy()
        sizes = benchmark.macro_sizes.numpy()
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        nh = benchmark.num_hard_macros
        nm = len(movable)
        canvas_area = cw * ch

        # Adaptive radius
        avg_spacing = math.sqrt(canvas_area / max(nm, 1))
        base_disp = avg_spacing * 0.5

        # Initial full cost
        _set_placement(plc, placement, benchmark)
        _ensure_congestion_arrays(plc)
        best_cost = self._full_cost(plc)
        best_pos = pos.copy()

        # Calibrate T0
        deltas = []
        for _ in range(300):
            idx = movable[random.randint(0, nm - 1)]
            ox, oy = float(pos[idx, 0]), float(pos[idx, 1])
            nx = ox + random.gauss(0, base_disp)
            ny = oy + random.gauss(0, base_disp)
            nx = max(float(hw[idx]), min(float(cw - hw[idx]), nx))
            ny = max(float(hh[idx]), min(float(ch - hh[idx]), ny))
            d = 0.0
            for k, w in adj.get(idx, []):
                xk, yk = float(pos[k, 0]), float(pos[k, 1])
                d += w * ((abs(nx - xk) + abs(ny - yk)) - (abs(ox - xk) + abs(oy - yk)))
            if abs(d) > 1e-12:
                deltas.append(abs(d))

        T0 = max(float(np.median(deltas)) / 0.2231, 1e-6) if deltas else 0.01
        Tmin = T0 * 1e-5
        T0 *= (1.0 + 0.2 * seed_idx)  # vary per seed

        time_limit = 120
        t_start = time.time()
        ipr = 500000
        full_eval_every = 250
        iteration = 0

        for rnd in range(3):
            T = T0 * (0.3 ** rnd)
            alpha = (Tmin / max(T, Tmin * 1.01)) ** (1.0 / ipr)
            pos = best_pos.copy()

            for _ in range(ipr):
                if time.time() - t_start > time_limit:
                    break
                iteration += 1
                frac = max(T / T0, 0.001)
                disp = base_disp * math.sqrt(frac)
                r = random.random()

                if r < 0.55 or nm < 2:
                    idx = movable[random.randint(0, nm - 1)]
                    ox, oy = float(pos[idx, 0]), float(pos[idx, 1])
                    nx = ox + random.gauss(0, disp)
                    ny = oy + random.gauss(0, disp)
                    nx = max(float(hw[idx]), min(float(cw - hw[idx]), nx))
                    ny = max(float(hh[idx]), min(float(ch - hh[idx]), ny))
                    pos[idx, 0], pos[idx, 1] = nx, ny
                    if self._overlap(pos, hw, hh, idx, nh):
                        pos[idx, 0], pos[idx, 1] = ox, oy
                        continue
                    delta = 0.0
                    for k, w in adj.get(idx, []):
                        xk, yk = float(pos[k, 0]), float(pos[k, 1])
                        delta += w * ((abs(nx - xk) + abs(ny - yk)) - (abs(ox - xk) + abs(oy - yk)))
                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-15)):
                        pass
                    else:
                        pos[idx, 0], pos[idx, 1] = ox, oy

                elif r < 0.78 and adj:
                    idx = movable[random.randint(0, nm - 1)]
                    neighbors = adj.get(idx, [])
                    if not neighbors: continue
                    ox, oy = float(pos[idx, 0]), float(pos[idx, 1])
                    wx, wy, tw = 0.0, 0.0, 0.0
                    for j, w in neighbors:
                        wx += pos[j, 0] * w
                        wy += pos[j, 1] * w
                        tw += w
                    if tw <= 0: continue
                    ccx, ccy = wx / tw, wy / tw
                    sf = 0.3 * frac
                    nx = ox + sf * (ccx - ox) + random.gauss(0, disp * 0.3)
                    ny = oy + sf * (ccy - oy) + random.gauss(0, disp * 0.3)
                    nx = max(float(hw[idx]), min(float(cw - hw[idx]), nx))
                    ny = max(float(hh[idx]), min(float(ch - hh[idx]), ny))
                    pos[idx, 0], pos[idx, 1] = nx, ny
                    if self._overlap(pos, hw, hh, idx, nh):
                        pos[idx, 0], pos[idx, 1] = ox, oy
                        continue
                    delta = 0.0
                    for k, w in adj.get(idx, []):
                        xk, yk = float(pos[k, 0]), float(pos[k, 1])
                        delta += w * ((abs(nx - xk) + abs(ny - yk)) - (abs(ox - xk) + abs(oy - yk)))
                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-15)):
                        pass
                    else:
                        pos[idx, 0], pos[idx, 1] = ox, oy

                else:
                    ii = random.randint(0, nm - 1)
                    jj = random.randint(0, nm - 2)
                    if jj >= ii: jj += 1
                    i, j = movable[ii], movable[jj]
                    oxi, oyi = float(pos[i, 0]), float(pos[i, 1])
                    oxj, oyj = float(pos[j, 0]), float(pos[j, 1])
                    nxi = max(float(hw[i]), min(float(cw - hw[i]), oxj))
                    nyi = max(float(hh[i]), min(float(ch - hh[i]), oyj))
                    nxj = max(float(hw[j]), min(float(cw - hw[j]), oxi))
                    nyj = max(float(hh[j]), min(float(ch - hh[j]), oyi))
                    pos[i, 0], pos[i, 1] = nxi, nyi
                    pos[j, 0], pos[j, 1] = nxj, nyj
                    if self._overlap(pos, hw, hh, i, nh) or self._overlap(pos, hw, hh, j, nh):
                        pos[i, 0], pos[i, 1] = oxi, oyi
                        pos[j, 0], pos[j, 1] = oxj, oyj
                        continue
                    delta = 0.0
                    for k, w in adj.get(i, []):
                        if k == j:
                            old_d = abs(oxi - oxj) + abs(oyi - oyj)
                            new_d = abs(nxi - nxj) + abs(nyi - nyj)
                        else:
                            xk, yk = float(pos[k, 0]), float(pos[k, 1])
                            old_d = abs(oxi - xk) + abs(oyi - yk)
                            new_d = abs(nxi - xk) + abs(nyi - yk)
                        delta += w * (new_d - old_d)
                    for k, w in adj.get(j, []):
                        if k == i: continue
                        xk, yk = float(pos[k, 0]), float(pos[k, 1])
                        old_d = abs(oxj - xk) + abs(oyj - yk)
                        new_d = abs(nxj - xk) + abs(nyj - yk)
                        delta += w * (new_d - old_d)
                    if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-15)):
                        pass
                    else:
                        pos[i, 0], pos[i, 1] = oxi, oyi
                        pos[j, 0], pos[j, 1] = oxj, oyj

                # Frequent full-cost checkpoints
                if iteration % full_eval_every == 0:
                    pt = torch.tensor(pos, dtype=torch.float32)
                    _set_placement(plc, pt, benchmark)
                    _ensure_congestion_arrays(plc)
                    fc = self._full_cost(plc)
                    if fc < best_cost:
                        best_cost = fc
                        best_pos = pos.copy()

                T = max(Tmin, T * alpha)

            if time.time() - t_start > time_limit:
                break

        # Final eval
        pt = torch.tensor(pos, dtype=torch.float32)
        _set_placement(plc, pt, benchmark)
        _ensure_congestion_arrays(plc)
        fc = self._full_cost(plc)
        if fc < best_cost:
            best_cost = fc
            best_pos = pos.copy()

        return torch.tensor(best_pos, dtype=torch.float32)
