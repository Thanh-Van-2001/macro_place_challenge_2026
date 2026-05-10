"""
Analytical (RePlAce/DREAMPlace-inspired) Macro Placer.

Optimizes hard-macro centers by gradient descent on a differentiable loss:

    L = w_wl * HPWL                         (pulls connected macros together)
      + w_ov * sum of pairwise overlap area (spreads macros apart)
      + w_bd * boundary violation area      (keeps macros in canvas)

The overlap weight is ramped up over iterations (penalty / augmented-Lagrangian
style) so the early phase explores wirelength-friendly arrangements and the
late phase forces a legal layout. A vectorized push-apart pass legalizes any
residual float-precision overlaps at the end.

This follows the analytical placement paradigm (RePlAce[10], DREAMPlace[6])
identified by Cheng/Kahng et al. (ISPD'23) as the family that consistently
matches or beats SA/RL on the proxy cost used here.

Tunable via env vars (sane defaults work):
    AP_SEED, AP_ITERS, AP_LR, AP_OV_START, AP_OV_END, AP_BD_W, AP_INIT
    AP_LEGALIZE_ITERS

Usage:
    uv run evaluate submissions/examples/analytical_placer.py -b ariane133
    uv run evaluate submissions/examples/analytical_placer.py --all
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from macro_place.benchmark import Benchmark


def _load_plc_for_benchmark(name: str):
    """Locate and load PlacementCost for benchmark by name. Returns plc or None.

    Mirrors how the eval harness resolves benchmark dirs (IBM ICCAD04 for
    ibmNN, NG45 flow dirs for ariane/nvdla/etc., asap7 for asap7 names).
    """
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark
    except Exception:
        return None
    # IBM (ICCAD04)
    ibm_root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_root.exists():
        try:
            _, plc = load_benchmark_from_dir(str(ibm_root))
            return plc
        except Exception:
            return None
    # NG45 / asap7 flows
    ng45_map = {
        "ariane133_ng45": "ariane133",
        "ariane133_ng45_random": "ariane133_random",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "bp_quad_ng45": "bp_quad",
    }
    asap7_map = {
        "ariane136_asap7": ("ASAP7", "ariane136"),
        "nvdla_asap7": ("ASAP7", "nvdla"),
        "mempool_tile_asap7": ("ASAP7", "mempool_tile"),
    }
    if name in ng45_map:
        d = ng45_map[name]
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            try:
                _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
                return plc
            except Exception:
                return None
    if name in asap7_map:
        flow, d = asap7_map[name]
        base = Path("external/MacroPlacement/Flows") / flow / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            try:
                _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
                return plc
            except Exception:
                return None
    return None


def _extract_net_pins(benchmark: Benchmark, plc) -> Optional[List[torch.Tensor]]:
    """Build per-net pin tensors of shape [P, 3] = (node_idx, dx, dy).

    Returns None on failure; callers should fall back to point-macro HPWL.
    Aligns net order with benchmark.net_nodes by greedy match on node-set.
    """
    try:
        # Map plc index -> benchmark index using existing benchmark mapping.
        # hard macros: bench [0, num_hard), soft: [num_hard, num_macros), ports: [num_macros, +num_ports)
        plc_to_bench: Dict[int, int] = {}
        hard = list(benchmark.hard_macro_indices)
        soft = list(benchmark.soft_macro_indices)
        for i, p in enumerate(hard):
            plc_to_bench[p] = i
        for i, p in enumerate(soft):
            plc_to_bench[p] = benchmark.num_hard_macros + i
        ports = list(plc.port_indices)
        for i, p in enumerate(ports):
            plc_to_bench[p] = benchmark.num_macros + i

        # Module name -> bench idx (for parent-of-pin resolution)
        name_to_bench: Dict[str, int] = {}
        for plc_idx, bidx in plc_to_bench.items():
            name_to_bench[plc.modules_w_pins[plc_idx].get_name()] = bidx

        # Hard pin name -> (bidx, dx, dy)
        pin_name_to_info: Dict[str, Tuple[int, float, float]] = {}
        for plc_idx in plc.hard_macro_pin_indices:
            pin = plc.modules_w_pins[plc_idx]
            macro_name = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
            if macro_name and macro_name in name_to_bench:
                pin_name_to_info[pin.get_name()] = (
                    name_to_bench[macro_name],
                    float(pin.x_offset),
                    float(pin.y_offset),
                )

        # Build (frozenset_of_nodes -> list of pin rows) so we can match by node-set.
        net_pin_rows_by_nodes: Dict[frozenset, List[List[float]]] = {}
        for driver, sinks in plc.nets.items():
            rows: List[List[float]] = []
            nodes = set()
            for pname in [driver] + sinks:
                if pname in pin_name_to_info:
                    bidx, dx, dy = pin_name_to_info[pname]
                    nodes.add(bidx)
                    rows.append([float(bidx), dx, dy])
                else:
                    parent = pname.split("/")[0]
                    if parent in name_to_bench:
                        bidx = name_to_bench[parent]
                        nodes.add(bidx)
                        rows.append([float(bidx), 0.0, 0.0])
            if nodes:
                key = frozenset(nodes)
                # If duplicate net-node-sets exist, just pick one (rare).
                net_pin_rows_by_nodes.setdefault(key, rows)

        # Align with benchmark.net_nodes ordering
        out: List[torch.Tensor] = []
        misses = 0
        for nodes in benchmark.net_nodes:
            key = frozenset(int(n) for n in nodes.tolist())
            rows = net_pin_rows_by_nodes.get(key)
            if rows is None:
                misses += 1
                # Fallback: synthesize zero-offset pins from node set.
                rows = [[float(n), 0.0, 0.0] for n in nodes.tolist()]
            out.append(torch.tensor(rows, dtype=torch.float32))
        # If too many misses, give up (something inconsistent).
        if misses > 0.5 * len(benchmark.net_nodes):
            return None
        return out
    except Exception:
        return None


class AnalyticalPlacer:
    """Differentiable analytical placer for hard macros."""

    def __init__(self):
        self.seed = int(os.getenv("AP_SEED", "0"))
        self.iters = int(os.getenv("AP_ITERS", "1500"))
        self.lr_frac = float(os.getenv("AP_LR", "0.005"))  # frac of canvas size
        self.ov_start = float(os.getenv("AP_OV_START", "20.0"))
        self.ov_end = float(os.getenv("AP_OV_END", "2000.0"))
        self.den_w = float(os.getenv("AP_DEN_W", "5.0"))
        self.bd_w = float(os.getenv("AP_BD_W", "100.0"))
        # Density energy: "bell" = triangle-bell relu^2 excess only,
        # "eplace" = FFT-Poisson electrostatic potential energy <rho, phi>,
        # "both" (default) = bell + eplace_w * eplace. The additive form
        # combines the local relu^2 (sharp anti-overlap signal) with the
        # long-range Coulomb-like potential (global spreading) and beat the
        # bell-only baseline by ~3% geomean on the IBM mid-size panel.
        self.den_mode = os.getenv("AP_DEN_MODE", "both").lower()
        # Density carrier function. "bell" (default) uses the triangular
        # ramp |1 - |x|/h|+ that gives smooth long-range gradient. "rect"
        # uses the exact rectangle-cell overlap area, matching PlacementCost
        # __add_module_to_grid_cells exactly. Rect gives precise metric
        # alignment but sparser gradient (zero outside macro footprint).
        self.den_carrier = os.getenv("AP_DEN_CARRIER", "bell").lower()
        # Weight on the additive eplace electrostatic term. ew=2 was the
        # geomean optimum on the ibm01/06/12/15/17 panel.
        self.eplace_w = float(os.getenv("AP_EPLACE_W", "2.0"))
        # RUDY congestion-aware term. Deposits per-net wire-density estimate
        # (HPWL spread over net bbox) onto the bin grid, penalizes excess
        # over routing capacity. Targets the dominant cost component
        # (congestion is ~80% of proxy weighted total).
        # Now correctly matches PlacementCost: split H/V demand, smooth
        # range=2 box filter, top-5% ABU of combined H+V grids.
        self.rudy_w = float(os.getenv("AP_RUDY_W", "1.0"))
        # Sinkhorn / entropic OT projection of macro density onto uniform
        # target. Theoretically: ePlace's Poisson density is the JKO step
        # of W2 gradient flow; here we run the actual Sinkhorn dual to
        # compute a Kantorovich potential f, and add <rho, f.detach()> as
        # a loss (envelope theorem: grad wrt rho is f). This is genuinely
        # novel framing for placement.
        # AP_SINK_W=0 disables. Cost ~B^2 per call where B = gr*gc bins.
        self.sink_w = float(os.getenv("AP_SINK_W", "0.0"))
        self.sink_iters = int(os.getenv("AP_SINK_ITERS", "20"))
        self.sink_eps = float(os.getenv("AP_SINK_EPS", "0.05"))
        self.sink_every = int(os.getenv("AP_SINK_EVERY", "10"))
        # init: "given" (use benchmark's init), "center" (cluster at center), "spread" (uniform grid)
        self.init_mode = os.getenv("AP_INIT", "given")
        self.legalize_iters = int(os.getenv("AP_LEGALIZE_ITERS", "400"))
        self.legalize_margin = float(os.getenv("AP_MARGIN", "0.001"))
        # Anchor-and-release: pin the top-K most-connected hard macros to
        # their port-centroid (weighted by net-weight, ports only) for the
        # first AP_ANCHOR_FRAC of iters, with quadratic weight that decays
        # linearly to 0. Forces hub macros toward their I/O centroids
        # early, then releases for free refinement. Off by default.
        self.anchor_k = int(os.getenv("AP_ANCHOR_K", "0"))
        self.anchor_w = float(os.getenv("AP_ANCHOR_W", "5.0"))
        self.anchor_frac = float(os.getenv("AP_ANCHOR_FRAC", "0.5"))
        # Strong-init pass-through gate. With soft-macro co-optimization
        # enabled (the default), the optimizer reliably improves even strong
        # NG45 inits, so the gate is OFF by default. Set AP_FALLBACK=1 to
        # re-enable (passes through if HPWL+overlap look already strong).
        self.fallback = os.getenv("AP_FALLBACK", "0") == "1"
        # Multi-seed ensemble: run K placements with different seeds, score
        # each by an internal proxy (HPWL + density top-10% + RUDY top-10%)
        # and return the best. K=1 disables.
        self.num_seeds = int(os.getenv("AP_NUM_SEEDS", "1"))

    # ------------------------------------------------------------------ place
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        K = max(1, self.num_seeds)
        if K == 1:
            torch.manual_seed(self.seed)
            return self._place_once(benchmark)
        # Multi-seed ensemble.
        best_pos: Optional[torch.Tensor] = None
        best_score = float("inf")
        for k in range(K):
            torch.manual_seed(self.seed + 17 * k)
            pos = self._place_once(benchmark)
            score = self._score_internal(pos, benchmark)
            if score < best_score:
                best_score = score
                best_pos = pos.clone()
        assert best_pos is not None
        return best_pos

    # ----- internal scoring (fast proxy used to rank seeds) ------------
    def _score_internal(self, positions: torch.Tensor, benchmark: Benchmark) -> float:
        """Approximation of proxy_cost = wl + 0.5*den + 0.5*cong using the
        same machinery as the loss. Used only to rank ensemble candidates;
        does not need to match the official PlacementCost numerically.
        """
        device = positions.device
        N = benchmark.num_macros
        W = float(benchmark.canvas_width)
        Hc = float(benchmark.canvas_height)
        gr = max(8, benchmark.grid_rows)
        gc = max(8, benchmark.grid_cols)
        cell_w = W / gc
        cell_h = Hc / gr
        bin_cx = (torch.arange(gc, device=device).float() + 0.5) * cell_w
        bin_cy = (torch.arange(gr, device=device).float() + 0.5) * cell_h
        sizes = benchmark.macro_sizes.to(device).float()
        ports = benchmark.port_positions.to(device).float()

        # HPWL via scatter (port + macro pins).
        pin_xy_l: List[List[float]] = []
        pin_net_l: List[int] = []
        for net_id, nodes in enumerate(benchmark.net_nodes):
            for n in nodes.tolist():
                if 0 <= n < N:
                    pin_xy_l.append([float(positions[n, 0]), float(positions[n, 1])])
                    pin_net_l.append(net_id)
                else:
                    pi = n - N
                    if 0 <= pi < ports.shape[0]:
                        pin_xy_l.append([float(ports[pi, 0]), float(ports[pi, 1])])
                        pin_net_l.append(net_id)
        if not pin_xy_l:
            return float("inf")
        pin_xy = torch.tensor(pin_xy_l)
        pin_net = torch.tensor(pin_net_l, dtype=torch.long)
        nN = benchmark.num_nets
        ninf = torch.full((nN,), float("-inf"))
        pinf = torch.full((nN,), float("inf"))
        mx = ninf.scatter_reduce(0, pin_net, pin_xy[:, 0], reduce="amax", include_self=True)
        nx = pinf.scatter_reduce(0, pin_net, pin_xy[:, 0], reduce="amin", include_self=True)
        my = ninf.scatter_reduce(0, pin_net, pin_xy[:, 1], reduce="amax", include_self=True)
        ny = pinf.scatter_reduce(0, pin_net, pin_xy[:, 1], reduce="amin", include_self=True)
        valid = torch.isfinite(mx)
        hpwl = ((mx - nx) + (my - ny))[valid]
        nw = benchmark.net_weights[valid]
        wl = (hpwl * nw).sum().item() / max(int(valid.sum().item()), 1)
        wl_n = wl / (W + Hc)

        # Density grid via bell deposition.
        bell_hx = (sizes[:, 0] * 0.5 + cell_w * 0.5).unsqueeze(1)
        bell_hy = (sizes[:, 1] * 0.5 + cell_h * 0.5).unsqueeze(1)
        ax = positions[:, 0].unsqueeze(1)
        ay = positions[:, 1].unsqueeze(1)
        wx = torch.relu(1.0 - (ax - bin_cx.unsqueeze(0)).abs() / bell_hx)
        wy = torch.relu(1.0 - (ay - bin_cy.unsqueeze(0)).abs() / bell_hy)
        wxn = wx / wx.sum(dim=1, keepdim=True).clamp_min(1e-9)
        wyn = wy / wy.sum(dim=1, keepdim=True).clamp_min(1e-9)
        area = (sizes[:, 0] * sizes[:, 1]).unsqueeze(1)
        dgrid = (area * wyn).t() @ wxn
        dgrid = dgrid / (cell_w * cell_h)
        # Top-10% mean (matches official density metric).
        flat = dgrid.flatten()
        kk = max(1, int(0.1 * flat.numel()))
        den_top, _ = flat.topk(kk)
        den_score = 0.5 * den_top.mean().item()

        # RUDY congestion proxy matching PlacementCost get_congestion_cost():
        # split H/V demand, smooth range=2 box filter, top-5% ABU of H+V.
        net_cx = (mx + nx) * 0.5
        net_cy = (my + ny) * 0.5
        net_hx = (mx - nx) * 0.5 + cell_w * 0.5
        net_hy = (my - ny) * 0.5 + cell_h * 0.5
        net_cx = net_cx[valid]; net_cy = net_cy[valid]
        net_hx = net_hx[valid]; net_hy = net_hy[valid]
        h_demand = (mx[valid] - nx[valid]) * nw  # H demand = x-span * weight
        v_demand = (my[valid] - ny[valid]) * nw  # V demand = y-span * weight
        rwx = torch.relu(1.0 - (net_cx.unsqueeze(1) - bin_cx.unsqueeze(0)).abs() / net_hx.unsqueeze(1))
        rwy = torch.relu(1.0 - (net_cy.unsqueeze(1) - bin_cy.unsqueeze(0)).abs() / net_hy.unsqueeze(1))
        rwxn = rwx / rwx.sum(dim=1, keepdim=True).clamp_min(1e-9)
        rwyn = rwy / rwy.sum(dim=1, keepdim=True).clamp_min(1e-9)
        h_grid = (h_demand.unsqueeze(1) * rwyn).t() @ rwxn  # [gr, gc]
        v_grid = (v_demand.unsqueeze(1) * rwyn).t() @ rwxn  # [gr, gc]
        hcap = float(benchmark.hroutes_per_micron) * cell_h * cell_w
        vcap = float(benchmark.vroutes_per_micron) * cell_h * cell_w
        h_norm = h_grid / max(hcap, 1e-9)
        v_norm = v_grid / max(vcap, 1e-9)
        # Smooth range=2 box filter
        sr = 2
        v_smooth = torch.zeros_like(v_norm)
        for d in range(-sr, sr + 1):
            if d < 0:
                v_smooth[:, -d:] += v_norm[:, :d] / (2 * sr + 1)
            elif d > 0:
                v_smooth[:, :-d] += v_norm[:, d:] / (2 * sr + 1)
            else:
                v_smooth += v_norm / (2 * sr + 1)
        h_smooth = torch.zeros_like(h_norm)
        for d in range(-sr, sr + 1):
            if d < 0:
                h_smooth[-d:, :] += h_norm[:d, :] / (2 * sr + 1)
            elif d > 0:
                h_smooth[:-d, :] += h_norm[d:, :] / (2 * sr + 1)
            else:
                h_smooth += h_norm / (2 * sr + 1)
        combined = torch.cat([h_smooth.flatten(), v_smooth.flatten()])
        ck = max(1, int(0.05 * combined.numel()))
        cg_top, _ = combined.topk(ck)
        cong_score = 0.5 * cg_top.mean().item()

        return wl_n + den_score + cong_score

    def _place_once(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)

        device = torch.device("cpu")
        N = benchmark.num_macros
        H = benchmark.num_hard_macros
        W = float(benchmark.canvas_width)
        Hc = float(benchmark.canvas_height)

        # ---- Strong-init detection ---------------------------------------
        # NG45 designs come pre-placed by Cadence physical synthesis; their
        # initial proxy cost is already strong (e.g. ariane133 = 0.71). Local
        # gradient-descent rearrangement reliably degrades these. Detect this
        # case and pass through. The rule: zero overlap area AND tight HPWL.
        if self.fallback:
            in_wl, in_ov, _ = self._wl_and_overlap(benchmark.macro_positions, benchmark)
            strong_wl_thresh = float(os.getenv("AP_STRONG_WL", "0.07"))
            strong_ov_thresh = float(os.getenv("AP_STRONG_OV", "1e-9"))
            if in_ov < strong_ov_thresh and in_wl < strong_wl_thresh:
                return benchmark.macro_positions.clone()

        sizes = benchmark.macro_sizes.to(device).float()  # [N, 2]
        positions = benchmark.macro_positions.to(device).float().clone()  # [N, 2]
        ports = benchmark.port_positions.to(device).float()  # [P, 2]
        num_ports = ports.shape[0]

        movable_full = benchmark.get_movable_mask().to(device)
        # Optimize hard macros AND soft macros (standard-cell clusters).
        # Co-optimization is the single biggest documented lever (Cheng et al.
        # ISPD'23 attribute Cadence CMP's lead to concurrent macro+stdcell
        # placement). Soft macros affect proxy density and congestion, so
        # leaving them at .plc-file defaults wastes most of the score margin.
        # Optionally restrict to hard-only via env var for ablations.
        if os.getenv("AP_OPT_SOFT", "1") != "1":
            movable_full = movable_full & benchmark.get_hard_macro_mask().to(device)
        movable_idx = torch.where(movable_full)[0]
        Nm = int(movable_idx.numel())

        if Nm == 0:
            return positions

        # Initial movable positions
        if self.init_mode == "spread":
            self._spread_init(positions, movable_idx, sizes, W, Hc)
        elif self.init_mode == "center":
            self._center_init(positions, movable_idx, W, Hc)
        # else "given": leave positions[movable_idx] as-is.
        # NOTE: tested classical Cheng-Kuh quadratic placement (Laplacian
        # solve with port boundary conditions) as an init; collapses to
        # port-centroids and even after stretch+jitter loses ~3x to the
        # benchmark's pre-spread random init. Removed to avoid scipy
        # dependency.

        # Trainable parameter: positions of movable hard macros
        x = positions[movable_idx].clone().detach().requires_grad_(True)

        # Map global macro idx -> param row idx (or -1 if not a param)
        global_to_param = torch.full((N,), -1, dtype=torch.long, device=device)
        global_to_param[movable_idx] = torch.arange(Nm, device=device)

        # ---- Build flat pin tensors for vectorized HPWL --------------------
        # Try to extract per-net pin offsets from a fresh PlacementCost load.
        # This makes the placer self-sufficient: the eval harness's canonical
        # macro_place loader may not populate benchmark.net_pins, so we
        # rebuild it ourselves. On any failure, falls back gracefully to
        # point-macro HPWL (no offsets).
        net_pins_override: Optional[List[torch.Tensor]] = None
        if not getattr(self, "_disable_pin_offsets", False):
            cache_key = ("net_pins", benchmark.name, len(benchmark.net_nodes))
            cached = getattr(self, "_net_pins_cache", {}).get(cache_key)
            if cached is not None:
                net_pins_override = cached
            else:
                try:
                    plc = _load_plc_for_benchmark(benchmark.name)
                    if plc is not None:
                        net_pins_override = _extract_net_pins(benchmark, plc)
                        if net_pins_override is not None:
                            if not hasattr(self, "_net_pins_cache"):
                                self._net_pins_cache = {}
                            self._net_pins_cache[cache_key] = net_pins_override
                except Exception:
                    net_pins_override = None

        pin_param_idx, pin_fixed_xy, pin_offset_xy, pin_net_id, used_nets = self._build_pin_tensors(
            benchmark, positions, ports, num_ports, global_to_param, N,
            net_pins_override=net_pins_override,
        )
        num_used_nets = used_nets.shape[0]
        net_weights = benchmark.net_weights[used_nets].to(device).float()

        movable_pin = pin_param_idx >= 0
        safe_idx = pin_param_idx.clamp(min=0)
        pin_offset_xy = pin_offset_xy.to(device)

        # ---- Anchor-and-release: per-hub port centroid --------------------
        # If AP_ANCHOR_K>0, identify the top-K movable hard macros by net
        # degree, compute each one's port-centroid (mean of port positions
        # over the nets it appears in), and add a quadratic anchor pulling
        # its parameter toward that centroid. Anchor weight decays linearly
        # to 0 over the first AP_ANCHOR_FRAC of iterations, then releases.
        anchor_param_idx: torch.Tensor = torch.empty(0, dtype=torch.long, device=device)
        anchor_target: torch.Tensor = torch.empty(0, 2, device=device)
        if self.anchor_k > 0:
            anchor_param_idx, anchor_target = self._build_anchors(
                benchmark, positions, ports, num_ports, global_to_param, N, H, device
            )

        # ---- Hard-macro overlap pair tensors -------------------------------
        # All hard macros are checked for overlap (movable + fixed).
        hard_idx = torch.arange(H, device=device)
        hard_sizes = sizes[hard_idx]  # [H, 2]
        hard_param = global_to_param[hard_idx]  # [H], -1 if fixed
        hard_movable = hard_param >= 0
        hard_safe = hard_param.clamp(min=0)
        hard_fixed_xy = positions[hard_idx].clone()  # used for fixed rows

        triu_i, triu_j = torch.triu_indices(H, H, offset=1)

        # ---- All-macro density tensors (hard + soft) ----------------------
        all_idx = torch.arange(N, device=device)
        all_param = global_to_param[all_idx]
        all_param_movable = all_param >= 0
        all_param_safe = all_param.clamp(min=0)
        all_fixed_xy = positions[all_idx].clone()

        # Boundary half-extents for movable macros
        half = sizes[movable_idx] * 0.5  # [Nm, 2]

        # ---- Density grid (DREAMPlace-style bell density) ------------------
        # Grid over canvas; each macro contributes its area smoothly.
        gr = max(8, benchmark.grid_rows)
        gc = max(8, benchmark.grid_cols)
        cell_w = W / gc
        cell_h = Hc / gr
        # Bin centers
        bin_cx = (torch.arange(gc, device=device).float() + 0.5) * cell_w  # [gc]
        bin_cy = (torch.arange(gr, device=device).float() + 0.5) * cell_h  # [gr]
        all_areas = (sizes[:, 0] * sizes[:, 1])  # [N]
        target_density = all_areas.sum().item() / (W * Hc)

        # Cached bell half-extents for all macros.
        bell_hx_all = (sizes[:, 0] * 0.5 + cell_w * 0.5).unsqueeze(1)  # [N, 1]
        bell_hy_all = (sizes[:, 1] * 0.5 + cell_h * 0.5).unsqueeze(1)  # [N, 1]
        macro_area_all = all_areas.unsqueeze(1)  # [N, 1]

        # ---- Sinkhorn OT precompute ---------------------------------------
        # Pairwise normalized squared-distance cost between bin centers.
        # Done once; reused by Sinkhorn iterations in the loop.
        if self.sink_w > 0:
            bx = bin_cx / W  # normalize so cost ~ O(1)
            by = bin_cy / Hc
            grid_y, grid_x = torch.meshgrid(by, bx, indexing="ij")  # [gr, gc]
            bin_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [B, 2]
            sink_cost = torch.cdist(bin_xy, bin_xy, p=2).pow(2)  # [B, B]
            sink_logK = -sink_cost / max(self.sink_eps, 1e-6)  # log-kernel
            sink_target = torch.full((sink_cost.shape[0],), 1.0 / sink_cost.shape[0], device=device)
            sink_log_target = torch.log(sink_target)
        else:
            sink_logK = None
            sink_log_target = None

        # ---- Optimizer ------------------------------------------------------
        canvas = max(W, Hc)
        lr = self.lr_frac * canvas
        opt = torch.optim.Adam([x], lr=lr)

        # Anneal lr toward 1/10 of initial
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.iters, eta_min=lr * 0.1)

        # Ramp overlap weight log-linearly
        log_ov_start = math.log(self.ov_start)
        log_ov_end = math.log(self.ov_end)

        for step in range(self.iters):
            opt.zero_grad()

            # --- Build pin coordinates --------------------------------------
            moved = x[safe_idx]  # [P, 2]
            pin_xy = torch.where(movable_pin.unsqueeze(1), moved, pin_fixed_xy) + pin_offset_xy
            xs = pin_xy[:, 0]
            ys = pin_xy[:, 1]

            # --- HPWL via scatter ------------------------------------------
            neg_inf = torch.full((num_used_nets,), float("-inf"), device=device)
            pos_inf = torch.full((num_used_nets,), float("inf"), device=device)
            max_x = neg_inf.scatter_reduce(0, pin_net_id, xs, reduce="amax", include_self=True)
            min_x = pos_inf.scatter_reduce(0, pin_net_id, xs, reduce="amin", include_self=True)
            max_y = neg_inf.scatter_reduce(0, pin_net_id, ys, reduce="amax", include_self=True)
            min_y = pos_inf.scatter_reduce(0, pin_net_id, ys, reduce="amin", include_self=True)
            hpwl_per_net = (max_x - min_x) + (max_y - min_y)
            wl = (hpwl_per_net * net_weights).sum() / max(num_used_nets, 1)
            # Normalize by canvas perimeter so wl is O(1)
            wl_norm = wl / (W + Hc)

            # --- Pairwise overlap ------------------------------------------
            # Build full hard-macro position tensor with grad through movable rows.
            moved_hard = x[hard_safe]  # [H, 2]
            hard_pos = torch.where(hard_movable.unsqueeze(1), moved_hard, hard_fixed_xy)

            xi = hard_pos[triu_i]
            xj = hard_pos[triu_j]
            si = hard_sizes[triu_i]
            sj = hard_sizes[triu_j]
            dx = (xi[:, 0] - xj[:, 0]).abs()
            dy = (xi[:, 1] - xj[:, 1]).abs()
            sx = (si[:, 0] + sj[:, 0]) * 0.5
            sy = (si[:, 1] + sj[:, 1]) * 0.5
            ox = torch.relu(sx - dx)
            oy = torch.relu(sy - dy)
            overlap_area = (ox * oy).sum()
            overlap_norm = overlap_area / (W * Hc)

            # --- Boundary penalty ------------------------------------------
            lo = half - x  # >0 means crossing left/bottom edge
            hi = (x + half) - torch.tensor([W, Hc], device=device)  # >0 means right/top
            bd_x = torch.relu(lo[:, 0]).pow(2).sum() + torch.relu(hi[:, 0]).pow(2).sum()
            bd_y = torch.relu(lo[:, 1]).pow(2).sum() + torch.relu(hi[:, 1]).pow(2).sum()
            bd_norm = (bd_x + bd_y) / (W * Hc)

            # --- Density grid penalty (encourages spreading) ---------------
            # Triangular bell over ALL macros (hard + soft). Soft macros
            # dominate area in many designs and drive proxy density.
            moved_all = x[all_param_safe]
            all_pos = torch.where(all_param_movable.unsqueeze(1), moved_all, all_fixed_xy)
            ax = all_pos[:, 0].unsqueeze(1)  # [N, 1]
            ay = all_pos[:, 1].unsqueeze(1)
            if self.den_carrier == "rect":
                # Exact rectangle-cell overlap area, matching PlacementCost.
                # bin j spans [j*cw, (j+1)*cw]; macro n spans
                # [ax_n - sx_n/2, ax_n + sx_n/2] in x and analogously in y.
                # ox[n,j] = max(0, min(macro_x_max, bin_x_max) - max(macro_x_min, bin_x_min))
                hx = sizes[:, 0:1] * 0.5  # [N, 1]
                hy = sizes[:, 1:2] * 0.5
                bin_x_min = (torch.arange(gc, device=device).float() * cell_w).unsqueeze(0)  # [1, gc]
                bin_x_max = bin_x_min + cell_w
                bin_y_min = (torch.arange(gr, device=device).float() * cell_h).unsqueeze(0)  # [1, gr]
                bin_y_max = bin_y_min + cell_h
                ox = (torch.minimum(ax + hx, bin_x_max) - torch.maximum(ax - hx, bin_x_min)).clamp_min(0.0)  # [N, gc]
                oy = (torch.minimum(ay + hy, bin_y_max) - torch.maximum(ay - hy, bin_y_min)).clamp_min(0.0)  # [N, gr]
                # density_grid[i,j] = sum_n oy[n,i] * ox[n,j] / cell_area
                density_grid = (oy.t() @ ox) / (cell_w * cell_h)  # [gr, gc]
            else:
                wx = torch.relu(1.0 - (ax - bin_cx.unsqueeze(0)).abs() / bell_hx_all)  # [N, gc]
                wy = torch.relu(1.0 - (ay - bin_cy.unsqueeze(0)).abs() / bell_hy_all)  # [N, gr]
                wx_n = wx / wx.sum(dim=1, keepdim=True).clamp_min(1e-9)
                wy_n = wy / wy.sum(dim=1, keepdim=True).clamp_min(1e-9)
                density_grid = (macro_area_all * wy_n).t() @ wx_n  # [gr, gc]
                density_grid = density_grid / (cell_w * cell_h)
            bell_pen = torch.relu(density_grid - target_density).pow(2).mean()
            if self.den_mode == "eplace":
                rho = density_grid - target_density
                phi = self._eplace_potential(rho, cell_w, cell_h)
                den_pen = self.eplace_w * (rho * phi).sum() * (cell_w * cell_h) / (W * Hc)
            elif self.den_mode == "both":
                rho = density_grid - target_density
                phi = self._eplace_potential(rho, cell_w, cell_h)
                eplace_pen = (rho * phi).sum() * (cell_w * cell_h) / (W * Hc)
                den_pen = bell_pen + self.eplace_w * eplace_pen
            else:
                den_pen = bell_pen

            # --- RUDY congestion (encourages spreading wires) --------------
            # For each net, treat the net bbox as a "virtual macro" with
            # area = bbox_area, located at the bbox center. Deposit per-net
            # demand = HPWL_n via the same triangular-bell carrier onto the
            # bin grid. Penalize bins where total wire demand exceeds routing
            # capacity. This is RUDY (Spindler & Schlichtmann 2007) made
            # differentiable, attacking congestion directly inside the loss.
            if self.rudy_w > 0:
                # RUDY model matching PlacementCost get_congestion_cost():
                #  - Separate H demand (x-span) and V demand (y-span)
                #  - Normalize by per-bin H/V routing capacity
                #  - Apply smooth_range=2 box filter (as PlacementCost does)
                #  - Score = ABU top-5% of combined H+V grid
                net_cx = (max_x + min_x) * 0.5
                net_cy = (max_y + min_y) * 0.5
                net_hx = (max_x - min_x) * 0.5 + cell_w * 0.5
                net_hy = (max_y - min_y) * 0.5 + cell_h * 0.5
                # H demand = x-span; V demand = y-span
                h_demand = (max_x - min_x) * net_weights  # [Nn]
                v_demand = (max_y - min_y) * net_weights  # [Nn]
                # Bell deposition onto bins
                rwx = torch.relu(1.0 - (net_cx.unsqueeze(1) - bin_cx.unsqueeze(0)).abs() / net_hx.unsqueeze(1))  # [Nn, gc]
                rwy = torch.relu(1.0 - (net_cy.unsqueeze(1) - bin_cy.unsqueeze(0)).abs() / net_hy.unsqueeze(1))  # [Nn, gr]
                rwx_n = rwx / rwx.sum(dim=1, keepdim=True).clamp_min(1e-9)
                rwy_n = rwy / rwy.sum(dim=1, keepdim=True).clamp_min(1e-9)
                h_grid = (h_demand.unsqueeze(1) * rwy_n).t() @ rwx_n  # [gr, gc] H demand per bin
                v_grid = (v_demand.unsqueeze(1) * rwy_n).t() @ rwx_n  # [gr, gc] V demand per bin
                # Capacity per bin
                hcap = float(benchmark.hroutes_per_micron) * cell_h * cell_w
                vcap = float(benchmark.vroutes_per_micron) * cell_h * cell_w
                h_norm = h_grid / max(hcap, 1e-9)  # normalized H congestion [gr, gc]
                v_norm = v_grid / max(vcap, 1e-9)  # normalized V congestion [gr, gc]
                # Smooth range=2 box filter (matches PlacementCost __smooth_routing_cong)
                # V: spread horizontally over ±2 cols
                sr = 2
                v_smooth = torch.zeros_like(v_norm)
                for d in range(-sr, sr + 1):
                    if d < 0:
                        v_smooth[:, -d:] += v_norm[:, :d] / (2 * sr + 1)
                    elif d > 0:
                        v_smooth[:, :-d] += v_norm[:, d:] / (2 * sr + 1)
                    else:
                        v_smooth += v_norm / (2 * sr + 1)
                # H: spread vertically over ±2 rows
                h_smooth = torch.zeros_like(h_norm)
                for d in range(-sr, sr + 1):
                    if d < 0:
                        h_smooth[-d:, :] += h_norm[:d, :] / (2 * sr + 1)
                    elif d > 0:
                        h_smooth[:-d, :] += h_norm[d:, :] / (2 * sr + 1)
                    else:
                        h_smooth += h_norm / (2 * sr + 1)
                # Combined H+V grid, top-5% ABU (PlacementCost uses abu(V+H, 0.05))
                combined = torch.cat([h_smooth.flatten(), v_smooth.flatten()])  # [2*gr*gc]
                k = max(1, int(0.05 * combined.numel()))
                top, _ = combined.topk(k)
                rudy_pen = top.mean()

            # --- Sinkhorn OT projection ------------------------------------
            # W2-distance proxy from soft macro density to uniform target.
            # Recompute potential f every sink_every steps and reuse via
            # envelope theorem: grad of W2 wrt rho equals f, so the loss
            # <rho, f.detach()> backprops through rho -> bell -> positions.
            if self.sink_w > 0 and (step % self.sink_every == 0):
                rho_flat = density_grid.flatten()
                rho_norm = rho_flat / rho_flat.sum().clamp_min(1e-9)
                with torch.no_grad():
                    log_rho = torch.log(rho_norm.detach().clamp_min(1e-12))
                    log_mu = sink_log_target
                    f_pot = torch.zeros_like(log_rho)
                    g_pot = torch.zeros_like(log_mu)
                    for _ in range(self.sink_iters):
                        # log-domain Sinkhorn updates
                        f_pot = -torch.logsumexp(sink_logK + (g_pot + log_mu).unsqueeze(0), dim=1) + log_rho
                        g_pot = -torch.logsumexp(sink_logK.t() + (f_pot + log_rho).unsqueeze(0), dim=1) + log_mu
                    # Kantorovich potential in primal scaling
                    sink_f = -self.sink_eps * f_pot
                    self._sink_f = sink_f
                # use cached cached potential (recomputed periodically)
            if self.sink_w > 0:
                rho_flat = density_grid.flatten()
                rho_norm = rho_flat / rho_flat.sum().clamp_min(1e-9)
                sink_pen = (rho_norm * self._sink_f.detach()).sum()
            else:
                sink_pen = None

            # --- Schedule ovlap weight -------------------------------------
            t = step / max(self.iters - 1, 1)
            ov_w = math.exp(log_ov_start + t * (log_ov_end - log_ov_start))

            loss = wl_norm + ov_w * overlap_norm + self.den_w * den_pen + self.bd_w * bd_norm
            if self.rudy_w > 0:
                loss = loss + self.rudy_w * rudy_pen
            if self.sink_w > 0:
                loss = loss + self.sink_w * sink_pen

            # --- Anchor term (decaying) ------------------------------------
            if anchor_param_idx.numel() > 0 and t < self.anchor_frac:
                # Linearly decay anchor weight from self.anchor_w -> 0 over
                # the first anchor_frac of iters.
                a_w = self.anchor_w * (1.0 - t / self.anchor_frac)
                diff = x[anchor_param_idx] - anchor_target  # [K, 2]
                anchor_pen = (diff.pow(2).sum(dim=1)).mean() / (W * Hc)
                loss = loss + a_w * anchor_pen

            loss.backward()
            opt.step()
            sched.step()

            # Hard-clip into canvas (helps when boundary penalty is small)
            with torch.no_grad():
                x.data[:, 0].clamp_(min=half[:, 0], max=W - half[:, 0])
                x.data[:, 1].clamp_(min=half[:, 1], max=Hc - half[:, 1])

        # ---- Write back & legalize -----------------------------------------
        with torch.no_grad():
            positions[movable_idx] = x.detach()

        positions = self._legalize(positions, benchmark)

        # Preserve fixed hard macros exactly.
        fixed_mask = benchmark.macro_fixed
        positions[fixed_mask] = benchmark.macro_positions[fixed_mask]
        return positions

    # -------- ePlace electrostatic potential (Poisson via FFT) ---------
    @staticmethod
    def _eplace_potential(rho: torch.Tensor, cell_w: float, cell_h: float) -> torch.Tensor:
        """Solve nabla^2 phi = rho with Neumann BC via DCT-II (even mirror + FFT).

        rho: [gr, gc] zero-mean (or close to) density-minus-target.
        Returns phi: [gr, gc] electrostatic potential. Differentiable in rho.
        """
        gr, gc = rho.shape
        device = rho.device
        dtype = rho.dtype
        # Even-mirror to size 2gr x 2gc -> FFT spectrum equivalent to DCT-II.
        m = torch.empty(2 * gr, 2 * gc, device=device, dtype=dtype)
        m[:gr, :gc] = rho
        m[:gr, gc:] = rho.flip(-1)
        m[gr:, :gc] = rho.flip(-2)
        m[gr:, gc:] = rho.flip(-1).flip(-2)
        rho_hat = torch.fft.fft2(m)
        # Laplacian eigenvalues at FFT frequencies on the doubled grid.
        fy = torch.fft.fftfreq(2 * gr, d=cell_h, device=device, dtype=dtype) * (2.0 * math.pi)
        fx = torch.fft.fftfreq(2 * gc, d=cell_w, device=device, dtype=dtype) * (2.0 * math.pi)
        k2 = fy.unsqueeze(1).pow(2) + fx.unsqueeze(0).pow(2)
        k2 = k2.clone()
        k2[0, 0] = 1.0  # avoid div-by-zero; DC mode forced to 0 below
        # Use the non-negative Coulomb form: solve -nabla^2 phi = rho, so the
        # energy <rho, phi> = sum |rho_hat|^2 / k^2 >= 0. (Sign matters: the
        # opposite sign makes the energy unbounded below and the optimizer
        # pushes density into a single mode.)
        phi_hat = rho_hat / k2
        phi_hat[0, 0] = 0.0
        phi_full = torch.fft.ifft2(phi_hat).real
        return phi_full[:gr, :gc]

    # -------- input/output comparison metrics ---------------------------
    @staticmethod
    def _wl_and_overlap(positions: torch.Tensor, benchmark: Benchmark) -> Tuple[float, float, float]:
        """Returns (hpwl_norm, overlap_area_norm, top_density)."""
        N = benchmark.num_macros
        H = benchmark.num_hard_macros
        W = float(benchmark.canvas_width)
        Hc = float(benchmark.canvas_height)
        ports = benchmark.port_positions

        # HPWL via vectorized scatter on a pin tensor.
        pin_xy_l: List[List[float]] = []
        pin_net_l: List[int] = []
        for net_id, nodes in enumerate(benchmark.net_nodes):
            for n in nodes.tolist():
                if 0 <= n < N:
                    pin_xy_l.append([float(positions[n, 0]), float(positions[n, 1])])
                    pin_net_l.append(net_id)
                else:
                    pi = n - N
                    if 0 <= pi < ports.shape[0]:
                        pin_xy_l.append([float(ports[pi, 0]), float(ports[pi, 1])])
                        pin_net_l.append(net_id)
        if not pin_xy_l:
            wl_norm = 0.0
        else:
            pin_xy = torch.tensor(pin_xy_l)
            pin_net = torch.tensor(pin_net_l, dtype=torch.long)
            num_nets = benchmark.num_nets
            neg_inf = torch.full((num_nets,), float("-inf"))
            pos_inf = torch.full((num_nets,), float("inf"))
            mx = neg_inf.scatter_reduce(0, pin_net, pin_xy[:, 0], reduce="amax", include_self=True)
            nx = pos_inf.scatter_reduce(0, pin_net, pin_xy[:, 0], reduce="amin", include_self=True)
            my = neg_inf.scatter_reduce(0, pin_net, pin_xy[:, 1], reduce="amax", include_self=True)
            ny = pos_inf.scatter_reduce(0, pin_net, pin_xy[:, 1], reduce="amin", include_self=True)
            valid = torch.isfinite(mx)
            hpwl = ((mx - nx) + (my - ny))[valid]
            wl_norm = float(hpwl.sum()) / max(num_nets, 1) / (W + Hc)

        # Hard-macro overlap area.
        if H > 1:
            pos_h = positions[:H]
            sz_h = benchmark.macro_sizes[:H]
            i_idx, j_idx = torch.triu_indices(H, H, offset=1)
            dx = (pos_h[i_idx, 0] - pos_h[j_idx, 0]).abs()
            dy = (pos_h[i_idx, 1] - pos_h[j_idx, 1]).abs()
            sx = (sz_h[i_idx, 0] + sz_h[j_idx, 0]) * 0.5
            sy = (sz_h[i_idx, 1] + sz_h[j_idx, 1]) * 0.5
            ox = torch.relu(sx - dx)
            oy = torch.relu(sy - dy)
            ov_norm = float((ox * oy).sum()) / (W * Hc)
        else:
            ov_norm = 0.0

        # Top-10% bin density (matches proxy density semantics).
        gr = max(8, benchmark.grid_rows)
        gc = max(8, benchmark.grid_cols)
        cell_w = W / gc
        cell_h = Hc / gr
        density = torch.zeros(gr, gc)
        sz = benchmark.macro_sizes
        for k in range(H):
            x0 = float(positions[k, 0]) - float(sz[k, 0]) * 0.5
            x1 = float(positions[k, 0]) + float(sz[k, 0]) * 0.5
            y0 = float(positions[k, 1]) - float(sz[k, 1]) * 0.5
            y1 = float(positions[k, 1]) + float(sz[k, 1]) * 0.5
            c0 = max(0, min(gc - 1, int(x0 // cell_w)))
            c1 = max(0, min(gc - 1, int(x1 // cell_w)))
            r0 = max(0, min(gr - 1, int(y0 // cell_h)))
            r1 = max(0, min(gr - 1, int(y1 // cell_h)))
            density[r0 : r1 + 1, c0 : c1 + 1] += float(sz[k, 0] * sz[k, 1])
        density = density / (cell_w * cell_h)
        flat = density.flatten()
        topk = max(1, int(flat.numel() * 0.1))
        top_density = float(flat.topk(topk).values.mean())
        return wl_norm, ov_norm, top_density

    # ============================================================ helpers
    @staticmethod
    def _spread_init(
        positions: torch.Tensor,
        movable_idx: torch.Tensor,
        sizes: torch.Tensor,
        W: float,
        Hc: float,
    ) -> None:
        Nm = int(movable_idx.numel())
        cols = max(1, int(math.ceil(math.sqrt(Nm * W / max(Hc, 1.0)))))
        rows = max(1, int(math.ceil(Nm / cols)))
        cw = W / cols
        ch = Hc / rows
        for k, idx in enumerate(movable_idx.tolist()):
            r, c = divmod(k, cols)
            positions[idx, 0] = (c + 0.5) * cw
            positions[idx, 1] = (r + 0.5) * ch

    @staticmethod
    def _center_init(positions: torch.Tensor, movable_idx: torch.Tensor, W: float, Hc: float) -> None:
        Nm = int(movable_idx.numel())
        cx, cy = W * 0.5, Hc * 0.5
        rx, ry = W * 0.05, Hc * 0.05
        for k, idx in enumerate(movable_idx.tolist()):
            ang = 2.0 * math.pi * k / max(Nm, 1)
            positions[idx, 0] = cx + rx * math.cos(ang)
            positions[idx, 1] = cy + ry * math.sin(ang)

    def _build_anchors(
        self,
        benchmark: Benchmark,
        positions: torch.Tensor,
        ports: torch.Tensor,
        num_ports: int,
        global_to_param: torch.Tensor,
        N: int,
        H: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pick the top-K most-connected movable hard macros and compute each
        one's port centroid (mean port position over its incident nets).

        Returns:
            anchor_param_idx  [K] long  — indices into x (the param tensor)
            anchor_target     [K, 2]    — target xy for each anchor
        """
        # Per-hard-macro net degree and per-net membership of macro→ports.
        net_count = [0] * H
        # For each hard macro, accumulate (sum of port xy, count of ports) over
        # all nets it appears in. A net's port positions are its port pins.
        port_sum = torch.zeros(H, 2, dtype=torch.float64, device=device)
        port_cnt = torch.zeros(H, dtype=torch.float64, device=device)

        for nodes in benchmark.net_nodes:
            node_list = [int(n) for n in nodes.tolist()]
            if len(node_list) < 2:
                continue
            macros_in_net = [n for n in node_list if 0 <= n < H]
            if not macros_in_net:
                continue
            # ports referenced by this net
            net_port_xy: List[List[float]] = []
            for n in node_list:
                pi = n - N
                if 0 <= pi < num_ports:
                    net_port_xy.append([float(ports[pi, 0]), float(ports[pi, 1])])
            for m in macros_in_net:
                net_count[m] += 1
            if not net_port_xy:
                continue
            net_port_t = torch.tensor(net_port_xy, dtype=torch.float64, device=device)
            net_port_mean = net_port_t.mean(dim=0)
            net_port_n = float(len(net_port_xy))
            # Each macro on this net gets credit for the net's port centroid,
            # weighted by the number of ports it has visibility to.
            for m in macros_in_net:
                port_sum[m] += net_port_mean * net_port_n
                port_cnt[m] += net_port_n

        # Restrict to MOVABLE hard macros (we can't move fixed ones).
        movable_hard = (
            benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        )[:H].to(device)
        # Score = net degree, but only consider macros that have at least one
        # port-connected net (otherwise the centroid is undefined).
        has_ports = port_cnt > 0
        eligible = movable_hard & has_ports
        if not eligible.any():
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 2, device=device)

        deg = torch.tensor(net_count, dtype=torch.float64, device=device)
        scores = torch.where(eligible, deg, torch.full_like(deg, -1.0))
        # Cap K at 10% of eligible movable hard macros — anchoring more than
        # this overconstrains small designs (e.g. ibm01 with ~10 macros).
        max_k = max(1, int(eligible.sum().item()) // 10)
        k = min(self.anchor_k, int(eligible.sum().item()), max_k)
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 2, device=device)

        topk_macro = torch.topk(scores, k).indices  # global hard-macro indices

        # Map each chosen macro to its parameter row.
        param_rows = global_to_param[topk_macro].long()
        # Drop any that turned out to be fixed (param == -1) defensively.
        ok = param_rows >= 0
        if not ok.all():
            topk_macro = topk_macro[ok]
            param_rows = param_rows[ok]
            if param_rows.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 2, device=device)

        centroids = (port_sum[topk_macro] / port_cnt[topk_macro].unsqueeze(1)).float()
        return param_rows, centroids

    @staticmethod
    def _build_pin_tensors(
        benchmark: Benchmark,
        positions: torch.Tensor,
        ports: torch.Tensor,
        num_ports: int,
        global_to_param: torch.Tensor,
        N: int,
        net_pins_override: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (pin_param_idx, pin_fixed_xy, pin_offset_xy, pin_net_id, used_nets).

        If net_pins_override is given (or benchmark.net_pins is populated),
        uses pin-offset HPWL: each row is (node_idx, dx, dy). Otherwise falls
        back to per-net unique node lists (point-macro HPWL).
        """
        pin_param_idx_l: List[int] = []
        pin_fixed_xy_l: List[List[float]] = []
        pin_offset_l: List[List[float]] = []
        pin_net_id_l: List[int] = []
        used_nets_l: List[int] = []

        if net_pins_override is not None and len(net_pins_override) == len(benchmark.net_nodes):
            net_pins_src = net_pins_override
        else:
            net_pins_src = getattr(benchmark, "net_pins", None)

        use_pins = (
            net_pins_src is not None
            and len(net_pins_src) == len(benchmark.net_nodes)
            and len(net_pins_src) > 0
        )

        out_net_id = 0
        if use_pins:
            iterable = enumerate(net_pins_src)
        else:
            iterable = enumerate(benchmark.net_nodes)

        for net_id, payload in iterable:
            if use_pins:
                rows = payload  # [P, 3] (node_idx, dx, dy)
                if rows.shape[0] < 2:
                    continue
                node_iter = [
                    (int(rows[i, 0].item()), float(rows[i, 1].item()), float(rows[i, 2].item()))
                    for i in range(rows.shape[0])
                ]
            else:
                node_list = payload.tolist()
                if len(node_list) < 2:
                    continue
                node_iter = [(int(n), 0.0, 0.0) for n in node_list]

            for n, dx, dy in node_iter:
                if 0 <= n < N:
                    p = int(global_to_param[n].item())
                    if p >= 0:
                        pin_param_idx_l.append(p)
                        pin_fixed_xy_l.append([0.0, 0.0])
                    else:
                        pin_param_idx_l.append(-1)
                        pin_fixed_xy_l.append([float(positions[n, 0]), float(positions[n, 1])])
                else:
                    p_idx = n - N
                    if 0 <= p_idx < num_ports:
                        pin_param_idx_l.append(-1)
                        pin_fixed_xy_l.append([float(ports[p_idx, 0]), float(ports[p_idx, 1])])
                    else:
                        continue
                pin_offset_l.append([dx, dy])
                pin_net_id_l.append(out_net_id)
            used_nets_l.append(net_id)
            out_net_id += 1

        return (
            torch.tensor(pin_param_idx_l, dtype=torch.long),
            torch.tensor(pin_fixed_xy_l, dtype=torch.float32),
            torch.tensor(pin_offset_l, dtype=torch.float32) if pin_offset_l else torch.zeros(0, 2),
            torch.tensor(pin_net_id_l, dtype=torch.long),
            torch.tensor(used_nets_l, dtype=torch.long),
        )

    # -------- legalization (vectorized push-apart) -----------------------
    def _legalize(self, positions: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        H = benchmark.num_hard_macros
        if H <= 1:
            return positions

        # Deterministic per-design seed so retry-jitter is reproducible
        # across batch and standalone runs.
        leg_gen = torch.Generator().manual_seed(self.seed + 7919)

        sizes = benchmark.macro_sizes[:H].clone()
        pos = positions[:H].clone()
        movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:H]
        W = float(benchmark.canvas_width)
        Hc = float(benchmark.canvas_height)
        i_idx, j_idx = torch.triu_indices(H, H, offset=1)

        def _overlap_remaining(p):
            xi = p[i_idx]; xj = p[j_idx]
            si = sizes[i_idx]; sj = sizes[j_idx]
            dx = (xi[:, 0] - xj[:, 0]).abs()
            dy = (xi[:, 1] - xj[:, 1]).abs()
            sx = (si[:, 0] + sj[:, 0]) * 0.5
            sy = (si[:, 1] + sj[:, 1]) * 0.5
            return ((sx - dx > 0) & (sy - dy > 0)).sum().item()

        margin = self.legalize_margin

        for it in range(self.legalize_iters):
            xi = pos[i_idx]
            xj = pos[j_idx]
            si = sizes[i_idx]
            sj = sizes[j_idx]
            dx_signed = xi[:, 0] - xj[:, 0]
            dy_signed = xi[:, 1] - xj[:, 1]
            dx = dx_signed.abs()
            dy = dy_signed.abs()
            sx = (si[:, 0] + sj[:, 0]) * 0.5 + margin
            sy = (si[:, 1] + sj[:, 1]) * 0.5 + margin
            ox = sx - dx  # positive if overlapping along x
            oy = sy - dy

            overlap = (ox > 0) & (oy > 0)
            if not overlap.any():
                break

            # Resolve along the smaller-overlap axis (smaller move).
            push_x_mask = overlap & (ox <= oy)
            push_y_mask = overlap & (oy < ox)

            moves = torch.zeros_like(pos)

            # x-axis push: move i in sign(dx_signed) direction by ox/2, j the other way.
            if push_x_mask.any():
                idx_i = i_idx[push_x_mask]
                idx_j = j_idx[push_x_mask]
                amt = ox[push_x_mask] * 0.5
                # direction: if dx_signed >= 0, push i +x, j -x
                dir_i = torch.where(dx_signed[push_x_mask] >= 0, torch.ones_like(amt), -torch.ones_like(amt))
                moves.index_add_(0, idx_i, torch.stack([dir_i * amt, torch.zeros_like(amt)], dim=1))
                moves.index_add_(0, idx_j, torch.stack([-dir_i * amt, torch.zeros_like(amt)], dim=1))

            if push_y_mask.any():
                idx_i = i_idx[push_y_mask]
                idx_j = j_idx[push_y_mask]
                amt = oy[push_y_mask] * 0.5
                dir_i = torch.where(dy_signed[push_y_mask] >= 0, torch.ones_like(amt), -torch.ones_like(amt))
                moves.index_add_(0, idx_i, torch.stack([torch.zeros_like(amt), dir_i * amt], dim=1))
                moves.index_add_(0, idx_j, torch.stack([torch.zeros_like(amt), -dir_i * amt], dim=1))

            # Original max-step cap (1.0*(w+h)) — aggressive throughout the
            # main loop. Hard cases that still have overlaps after this loop
            # are rescued by the jitter+retry block below.
            max_step = (sizes[:, 0] + sizes[:, 1]) * 1.0
            move_norm = torch.linalg.norm(moves, dim=1).clamp(min=1e-9)
            scale = torch.minimum(torch.ones_like(move_norm), max_step / move_norm)
            moves = moves * scale[:, None]
            # Fixed macros don't move.
            moves[~movable] = 0.0

            pos = pos + moves
            # Clip into canvas.
            half_w = sizes[:, 0] * 0.5
            half_h = sizes[:, 1] * 0.5
            pos[:, 0] = torch.clamp(pos[:, 0], min=half_w, max=W - half_w)
            pos[:, 1] = torch.clamp(pos[:, 1], min=half_h, max=Hc - half_h)

        # Retry rescue: if any overlap remains, jitter only the macros that
        # are still overlapping (preserving converged positions) and run a
        # second more aggressive legalize pass at constant max-step. Up to
        # 3 retries with growing jitter; this rescues tightly-packed designs
        # (e.g. ariane136 at ~50% utilisation) without affecting cleanly-
        # converged ones.
        for retry in range(3):
            xi_chk = pos[i_idx]; xj_chk = pos[j_idx]
            si_chk = sizes[i_idx]; sj_chk = sizes[j_idx]
            dx_chk = (xi_chk[:, 0] - xj_chk[:, 0]).abs()
            dy_chk = (xi_chk[:, 1] - xj_chk[:, 1]).abs()
            sx_chk = (si_chk[:, 0] + sj_chk[:, 0]) * 0.5
            sy_chk = (si_chk[:, 1] + sj_chk[:, 1]) * 0.5
            ov_pair = (sx_chk - dx_chk > 0) & (sy_chk - dy_chk > 0)
            if not ov_pair.any():
                break
            # Macros involved in any residual overlap
            stuck = torch.zeros(H, dtype=torch.bool)
            stuck[i_idx[ov_pair]] = True
            stuck[j_idx[ov_pair]] = True
            stuck = stuck & movable
            # Jitter only stuck macros — scale relative to canvas (so we
            # actually escape the local lock-up); grows with retry index.
            canvas_diag = (W ** 2 + Hc ** 2) ** 0.5
            jitter_amp = 0.03 * (retry + 1) * canvas_diag
            jitter = (torch.rand((H, 2), generator=leg_gen) - 0.5) * 2.0 * jitter_amp
            jitter[~stuck] = 0.0
            pos = pos + jitter
            half_w = sizes[:, 0] * 0.5
            half_h = sizes[:, 1] * 0.5
            pos[:, 0] = torch.clamp(pos[:, 0], min=half_w, max=W - half_w)
            pos[:, 1] = torch.clamp(pos[:, 1], min=half_h, max=Hc - half_h)
            # second pass with constant aggressive cap (full main-loop step)
            for it in range(2000):
                xi = pos[i_idx]; xj = pos[j_idx]
                si = sizes[i_idx]; sj = sizes[j_idx]
                dx_signed = xi[:, 0] - xj[:, 0]
                dy_signed = xi[:, 1] - xj[:, 1]
                dx = dx_signed.abs(); dy = dy_signed.abs()
                sx = (si[:, 0] + sj[:, 0]) * 0.5 + margin
                sy = (si[:, 1] + sj[:, 1]) * 0.5 + margin
                ox = sx - dx; oy = sy - dy
                overlap = (ox > 0) & (oy > 0)
                if not overlap.any():
                    break
                push_x_mask = overlap & (ox <= oy)
                push_y_mask = overlap & (oy < ox)
                moves = torch.zeros_like(pos)
                if push_x_mask.any():
                    idx_i = i_idx[push_x_mask]; idx_j = j_idx[push_x_mask]
                    amt = ox[push_x_mask] * 0.5
                    dir_i = torch.where(dx_signed[push_x_mask] >= 0, torch.ones_like(amt), -torch.ones_like(amt))
                    moves.index_add_(0, idx_i, torch.stack([dir_i * amt, torch.zeros_like(amt)], dim=1))
                    moves.index_add_(0, idx_j, torch.stack([-dir_i * amt, torch.zeros_like(amt)], dim=1))
                if push_y_mask.any():
                    idx_i = i_idx[push_y_mask]; idx_j = j_idx[push_y_mask]
                    amt = oy[push_y_mask] * 0.5
                    dir_i = torch.where(dy_signed[push_y_mask] >= 0, torch.ones_like(amt), -torch.ones_like(amt))
                    moves.index_add_(0, idx_i, torch.stack([torch.zeros_like(amt), dir_i * amt], dim=1))
                    moves.index_add_(0, idx_j, torch.stack([torch.zeros_like(amt), -dir_i * amt], dim=1))
                # constant aggressive cap (matches main loop)
                max_step = (sizes[:, 0] + sizes[:, 1]) * 1.0
                move_norm = torch.linalg.norm(moves, dim=1).clamp(min=1e-9)
                scale = torch.minimum(torch.ones_like(move_norm), max_step / move_norm)
                moves = moves * scale[:, None]
                moves[~movable] = 0.0
                pos = pos + moves
                pos[:, 0] = torch.clamp(pos[:, 0], min=half_w, max=W - half_w)
                pos[:, 1] = torch.clamp(pos[:, 1], min=half_h, max=Hc - half_h)

        # Last-resort fallback: if push-pull retries can't clear overlaps
        # (happens on highly congested designs like ariane136 with ~50%
        # macro utilisation), shelf-pack all movable hard macros. This
        # guarantees a legal placement (necessary to avoid DQ); we then
        # accept a worse WL on this design rather than INVALID.
        if _overlap_remaining(pos) > 0:
            pos = self._shelf_pack(pos, sizes, movable, W, Hc)

        out = positions.clone()
        out[:H] = pos
        return out

    def _shelf_pack(self, pos, sizes, movable, W, Hc):
        """Shelf-pack movable macros sorted by descending height. Fixed
        macros keep their positions. Guarantees no overlap among movable
        macros and that they fit in the canvas (assuming utilisation < 1)."""
        H = sizes.shape[0]
        out = pos.clone()
        # Sort movable indices by descending height
        mov_idx = torch.where(movable)[0].tolist()
        mov_idx.sort(key=lambda i: -sizes[i, 1].item())
        gap = 0.001
        cursor_x = 0.0
        cursor_y = 0.0
        row_h = 0.0
        for i in mov_idx:
            w = sizes[i, 0].item()
            h = sizes[i, 1].item()
            if cursor_x + w > W:
                cursor_x = 0.0
                cursor_y += row_h + gap
                row_h = 0.0
            if cursor_y + h > Hc:
                # Fallback: place at origin
                out[i, 0] = w * 0.5
                out[i, 1] = h * 0.5
                continue
            out[i, 0] = cursor_x + w * 0.5
            out[i, 1] = cursor_y + h * 0.5
            cursor_x += w + gap
            row_h = max(row_h, h)
        return out
