"""v24_smooth.py — Talyxion HybridV24: cyclic smooth-proxy GD placer.

The TILOS proxy is piecewise-constant in macro coordinates and has no usable
gradient. `tx_smooth` builds a torch-differentiable approximation (verified to
track the TILOS proxy). This placer:

  1. Adam GD on the smooth proxy with gamma_wl annealed soft -> sharp, jointly
     over all movable macros (hard + soft);
  2. legalize hard macros (overlap resolution);
  3. cycle: coordinate-descent polish -> warm-restart GD -> legalize, x2.
     Each GD re-optimisation escapes the basin the previous CD settled into;
  4. final numerical-gradient + CD/swap polish on the exact fast proxy.

All code here is our own. The smooth proxy and fast evaluator only change
search SPEED / smoothness; the result is scored by the unmodified TILOS
evaluator. Best-of-stage by true proxy is returned.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_TVDIR = os.path.dirname(os.path.abspath(__file__))
if _TVDIR not in sys.path:
    sys.path.insert(0, _TVDIR)

import tx_smooth as TS
from fasteval import FastEval
import v21_numgrad as NG

_real_proxy = NG._real_proxy
_load_plc = NG._load_plc

# pin-count cap: bounds the [n_nets, max_pins, grid] routing tensors. Nets
# above the cap (clock / scan chains) keep their first PIN_CAP pins.
PIN_CAP = 32


def _build_pins(b):
    n_hard = b.num_hard_macros
    nets = [net for net in b.net_pin_nodes if net.shape[0] >= 2]
    P = min(PIN_CAP, max(int(n.shape[0]) for n in nets))
    N = len(nets)
    owner = torch.zeros(N, P, dtype=torch.long)
    offset = torch.zeros(N, P, 2)
    mask = torch.zeros(N, P, dtype=torch.bool)
    poff = b.macro_pin_offsets
    for i, net in enumerate(nets):
        k = min(P, int(net.shape[0]))
        owner[i, :k] = net[:k, 0].long()
        mask[i, :k] = True
        for j in range(k):
            ow = int(net[j, 0]); slot = int(net[j, 1])
            if ow < n_hard and slot < poff[ow].shape[0]:
                offset[i, j, 0] = float(poff[ow][slot, 0])
                offset[i, j, 1] = float(poff[ow][slot, 1])
    return owner, offset, mask


class _SmoothGD:
    """Adam GD on the tx_smooth proxy. Build once per benchmark, run passes."""

    def __init__(self, b, plc, dw, cw_):
        self.b = b
        self.dw = dw
        self.cw_ = cw_
        self.cw = float(b.canvas_width)
        self.ch = float(b.canvas_height)
        self.n_macros = b.num_macros
        self.n_hard = b.num_hard_macros
        self.sizes = b.macro_sizes.float()
        self.hw = self.sizes[:, 0] * 0.5
        self.hh = self.sizes[:, 1] * 0.5
        self.owner, self.offset, self.mask = _build_pins(b)
        nets = [net for net in b.net_pin_nodes if net.shape[0] >= 2]
        self.net_w = torch.ones(len(nets))
        self.net_cnt = float(plc.net_cnt if plc.net_cnt else len(nets))
        self.Gw, self.Gh = int(plc.grid_col), int(plc.grid_row)
        self.hrpm = float(plc.hroutes_per_micron)
        self.vrpm = float(plc.vroutes_per_micron)
        self.halloc = float(plc.hrouting_alloc)
        self.valloc = float(plc.vrouting_alloc)
        self.srange = int(plc.smooth_range)
        self.port_pos = b.port_positions.float()
        self.movable = b.get_movable_mask()

    def run(self, init_pos, n_steps, max_sec, g_start, g_end, lr0, lr1, seed=42):
        """Step-based Adam GD: gamma_wl + lr annealed over a FIXED n_steps so
        the step count (hence convergence behaviour) is identical on small and
        large benchmarks. A wall-clock cap stops huge benchmarks early."""
        torch.manual_seed(seed)
        init = torch.tensor(np.asarray(init_pos), dtype=torch.float32)
        init[:, 0].clamp_(self.hw, self.cw - self.hw)
        init[:, 1].clamp_(self.hh, self.ch - self.hh)
        fixed_xy = init.clone()
        pos = torch.nn.Parameter(init.clone())
        opt = torch.optim.Adam([pos], lr=lr0)
        PH = 0.5
        t0 = time.time()
        for step in range(n_steps):
            if step > 40 and time.time() - t0 > max_sec:
                break
            t = step / max(n_steps - 1, 1)
            if t < PH:
                gamma = g_start + t / PH * (2.0 - g_start)
            else:
                gamma = 2.0 * (g_end / 2.0) ** ((t - PH) / (1 - PH))
            lr = (lr0 * (step + 1) / 20 if step < 20
                  else lr0 + (step - 20) / max(n_steps - 21, 1) * (lr1 - lr0))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            pos_aug = torch.cat([pos, self.port_pos], dim=0)
            wl, d, c = TS.proxy_components(
                pos_aug, self.sizes, self.owner, self.offset, self.mask,
                self.net_w, self.net_cnt, self.Gw, self.Gh, self.cw, self.ch,
                self.hrpm, self.vrpm, self.halloc, self.valloc,
                self.n_hard, self.n_macros, gamma_wl=gamma, srange=self.srange)
            (wl + self.dw * d + self.cw_ * c).backward()
            opt.step()
            with torch.no_grad():
                pos[~self.movable] = fixed_xy[~self.movable]
                pos[:, 0].clamp_(self.hw, self.cw - self.hw)
                pos[:, 1].clamp_(self.hh, self.ch - self.hh)
        return pos.detach().numpy().astype(np.float64)


def _legalize(pos, b):
    """Resolve hard-macro overlaps. Pairwise push first; if dense clusters
    survive, jitter movable hard macros with growing amplitude and re-push."""
    nh = b.num_hard_macros
    sizes = b.macro_sizes.cpu().numpy().astype(np.float64)
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    cw = float(b.canvas_width)
    ch = float(b.canvas_height)
    movable = (b.get_movable_mask() & b.get_hard_macro_mask()).cpu().numpy()
    p = pos.astype(np.float64).copy()
    NG._resolve_overlaps(p, hw, hh, movable, nh, cw, ch, 8000)
    ov = NG._count_overlaps(p, hw, hh, nh)
    if ov == 0:
        return p, 0
    # stubborn clusters: stochastic jitter + re-resolve
    diag = (cw * cw + ch * ch) ** 0.5
    rng = np.random.default_rng(20240519)
    best_p = p.copy()
    best_ov = ov
    for rnd in range(60):
        amp = diag * (0.01 + 0.05 * (rnd / 60.0))
        q = best_p.copy()
        for i in range(nh):
            if movable[i]:
                q[i, 0] += rng.uniform(-amp, amp)
                q[i, 1] += rng.uniform(-amp, amp)
                q[i, 0] = min(max(q[i, 0], hw[i]), cw - hw[i])
                q[i, 1] = min(max(q[i, 1], hh[i]), ch - hh[i])
        NG._resolve_overlaps(q, hw, hh, movable, nh, cw, ch, 8000)
        c = NG._count_overlaps(q, hw, hh, nh)
        if c < best_ov:
            best_ov, best_p = c, q.copy()
            if c == 0:
                break
    return best_p, best_ov


class HybridV24(NG._BASE):
    """Cyclic smooth-proxy GD + numerical-gradient/CD polish."""

    def __init__(self):
        super().__init__()
        self._dw = float(os.environ.get("HP24_DW", "0.7"))
        self._cw = float(os.environ.get("HP24_CW", "1.3"))
        self._gd1_steps = int(os.environ.get("HP24_GD1_STEPS", "650"))
        self._gd1_max = float(os.environ.get("HP24_GD1_MAX", "780"))
        self._gd2_steps = int(os.environ.get("HP24_GD2_STEPS", "320"))
        self._gd2_max = float(os.environ.get("HP24_GD2_MAX", "420"))
        self._cycles = int(os.environ.get("HP24_CYCLES", "3"))
        self._cd_sec = float(os.environ.get("HP24_CD_SEC", "150"))
        self._ng_sec = float(os.environ.get("HP24_NG_SEC", "220"))
        self._fcd_sec = float(os.environ.get("HP24_FCD_SEC", "260"))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _load_plc(benchmark)
        if plc is None:
            return super().place(benchmark)
        fe = FastEval(plc, benchmark)
        gd = _SmoothGD(benchmark, plc, self._dw, self._cw)
        t0 = time.time()

        # cycle 1: full-anneal GD from the benchmark's initial placement
        pos = gd.run(benchmark.macro_positions.numpy().astype(np.float64),
                     self._gd1_steps, self._gd1_max, 0.5, 8.0, 0.2, 0.02)
        pos, _ = _legalize(pos, benchmark)
        best = pos.copy()
        best_f = _real_proxy(plc, benchmark, best)
        sys.stderr.write(f"[v24] {benchmark.name} cycle1 ={best_f:.5f} "
                         f"[{time.time()-t0:.0f}s]\n")
        sys.stderr.flush()

        # cycles 2..: CD polish -> warm-restart GD -> legalize
        for cyc in range(2, self._cycles + 1):
            cd_pos, _ = NG.cd_polish(fe, plc, benchmark, pos, self._cd_sec)
            pos = gd.run(cd_pos, self._gd2_steps, self._gd2_max,
                         4.0, 8.0, 0.05, 0.02)
            pos, ov = _legalize(pos, benchmark)
            f = _real_proxy(plc, benchmark, pos)
            if ov == 0 and f < best_f:
                best_f, best = f, pos.copy()
            sys.stderr.write(f"[v24] {benchmark.name} cycle{cyc} ={f:.5f} "
                             f"ov={ov} [{time.time()-t0:.0f}s]\n")
            sys.stderr.flush()

        # final polish on the exact fast proxy
        gd_pos, gd_c = NG.numgrad_refine(fe, plc, benchmark, best, self._ng_sec)
        if gd_c < best_f:
            best_f, best = gd_c, gd_pos
        cd_pos, cd_c = NG.cd_polish(fe, plc, benchmark, best, self._fcd_sec)
        if cd_c < best_f:
            best_f, best = cd_c, cd_pos

        # validity guard: if the smooth-GD placement cannot be legalized,
        # fall back to the always-valid v5-base + numgrad + CD pipeline.
        sz = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
        hw_ = sz[:, 0] * 0.5
        hh_ = sz[:, 1] * 0.5
        nh = benchmark.num_hard_macros
        if NG._count_overlaps(best, hw_, hh_, nh) > 0:
            sys.stderr.write(f"[v24] {benchmark.name} smooth-GD invalid "
                             f"-> v5 fallback\n")
            sys.stderr.flush()
            v5 = super().place(benchmark).cpu().numpy().astype(np.float64)
            fb, fb_f = v5, _real_proxy(plc, benchmark, v5)
            gp, gc = NG.numgrad_refine(fe, plc, benchmark, v5, self._ng_sec)
            if NG._count_overlaps(gp, hw_, hh_, nh) == 0 and gc < fb_f:
                fb, fb_f = gp, gc
            cp, cc = NG.cd_polish(fe, plc, benchmark, fb, self._fcd_sec)
            if NG._count_overlaps(cp, hw_, hh_, nh) == 0 and cc < fb_f:
                fb, fb_f = cp, cc
            best, best_f = fb, fb_f

        sys.stderr.write(f"[v24] {benchmark.name} FINAL ={best_f:.5f} "
                         f"[{time.time()-t0:.0f}s]\n")
        sys.stderr.flush()

        result = torch.tensor(best, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return result
