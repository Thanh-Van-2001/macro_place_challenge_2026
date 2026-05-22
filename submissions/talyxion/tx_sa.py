"""tx_sa.py — Talyxion cold simulated annealing on the fast TILOS proxy.

Single-macro random-displacement SA with a very cold temperature (near-greedy
with rare uphill escapes), run as a final stage on top of the cyclic smooth-GD
placement. The key ingredient over the earlier coordinate-descent / numgrad
polish is that it moves SOFT macros (the cell clusters that dominate density +
congestion) ~half the time — those are never touched by the hard-macro CD
polish — and does millions of cheap random trials rather than a fixed K
candidates per macro. Cold Metropolis criterion (T~1e-6) keeps it a descent
with rare escapes.

All code is our own; the fast evaluator only changes search speed.
"""
import math
import sys
import time
import numpy as np
from v21_numgrad import _overlaps_at, _count_overlaps


def cold_sa(fe, plc, b, base_pos, time_budget,
            T_init=1e-6, T_end=1e-7, move_min=0.1, move_max=5.0,
            swap_prob=0.10, soft_prob=0.50, seed=0, n_iters=200_000_000,
            log_every=200_000):
    nh = b.num_hard_macros
    nm = b.num_macros
    sizes = b.macro_sizes.cpu().numpy().astype(np.float64)
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(b.canvas_width)
    ch = float(b.canvas_height)
    fixed = b.macro_fixed.cpu().numpy()
    rng = np.random.default_rng(seed)

    # same-area hard macro groups for swaps
    areas = np.round(sizes[:nh, 0] * sizes[:nh, 1], 3)
    groups = {}
    for i in range(nh):
        if not fixed[i]:
            groups.setdefault(areas[i], []).append(i)
    swap_groups = [np.array(g, dtype=np.int64) for g in groups.values() if len(g) >= 2]
    can_swap = len(swap_groups) > 0
    n_soft = nm - nh

    cur = base_pos.astype(np.float64).copy()
    cur_f, _, _, _ = fe.proxy(cur)
    best = cur.copy(); best_f = cur_f
    cool = (T_end / T_init) ** (1.0 / max(n_iters - 1, 1))
    T = T_init
    t0 = time.time()
    it = 0
    n_acc = 0
    while it < n_iters:
        if (it & 0xFF) == 0 and time.time() - t0 >= time_budget:
            break
        r = rng.random()
        move_kind = 0
        if can_swap and r < swap_prob:
            grp = swap_groups[rng.integers(len(swap_groups))]
            i = int(grp[rng.integers(grp.shape[0])])
            j = int(grp[rng.integers(grp.shape[0])])
            if i == j:
                it += 1; T *= cool; continue
            # swap; same area => mutual fit, but verify against others
            oix, oiy = cur[i, 0], cur[i, 1]
            ojx, ojy = cur[j, 0], cur[j, 1]
            cur[i, 0], cur[i, 1] = ojx, ojy
            cur[j, 0], cur[j, 1] = oix, oiy
            if _overlaps_at(cur, hw, hh, i, nh, ojx, ojy) or \
               _overlaps_at(cur, hw, hh, j, nh, oix, oiy):
                cur[i, 0], cur[i, 1] = oix, oiy
                cur[j, 0], cur[j, 1] = ojx, ojy
                it += 1; T *= cool; continue
            move_kind = 1
        else:
            move_soft = (n_soft > 0) and (rng.random() < soft_prob)
            if move_soft:
                i = int(nh + rng.integers(n_soft))
            else:
                i = int(rng.integers(nh))
            if fixed[i]:
                it += 1; T *= cool; continue
            ang = rng.random() * 2.0 * math.pi
            mag = move_min + rng.random() * (move_max - move_min)
            nx = cur[i, 0] + math.cos(ang) * mag
            ny = cur[i, 1] + math.sin(ang) * mag
            wi = hw[i]; hi = hh[i]
            if nx < wi or nx > cw - wi or ny < hi or ny > ch - hi:
                it += 1; T *= cool; continue
            if not move_soft and _overlaps_at(cur, hw, hh, i, nh, nx, ny):
                it += 1; T *= cool; continue
            oix, oiy = cur[i, 0], cur[i, 1]
            cur[i, 0] = nx; cur[i, 1] = ny

        f, _, _, _ = fe.proxy(cur)
        d = f - cur_f
        if d <= 0.0 or rng.random() < math.exp(-d / max(T, 1e-12)):
            cur_f = f
            n_acc += 1
            if cur_f < best_f:
                best_f = cur_f; best = cur.copy()
        else:
            if move_kind == 1:
                cur[i, 0], cur[i, 1] = oix, oiy
                cur[j, 0], cur[j, 1] = ojx, ojy
            else:
                cur[i, 0] = oix; cur[i, 1] = oiy
        it += 1
        T *= cool
        if log_every and it % log_every == 0:
            sys.stderr.write(f"[tx_sa] it={it} best={best_f:.5f} "
                             f"acc={n_acc} {it/(time.time()-t0):.0f}it/s\n")
            sys.stderr.flush()

    if _count_overlaps(best, hw, hh, nh) > 0:
        return base_pos, fe.proxy(base_pos)[0]
    return best, best_f
