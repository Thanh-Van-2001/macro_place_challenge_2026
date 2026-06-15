"""make_talyxion_gif.py — render a GIF of Talyxion HybridV26's OWN algorithm.

Shows stage 1 of HybridV26: Adam gradient descent on the team's torch-
differentiable smooth approximation of the TILOS proxy (tx_smooth), with the
WL sharpness gamma annealed soft->sharp, followed by hard-macro legalization.
Every frame is a real snapshot of the optimizer state; the title shows the
exact (TILOS-faithful) proxy via the team's FastEval.
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import imageio.v2 as imageio

REPO = "/home/thanh/macro_place_challenge_2026"
SUB = os.path.join(REPO, "submissions", "talyxion")
sys.path.insert(0, REPO)
sys.path.insert(0, SUB)

from macro_place.loader import load_benchmark_from_dir
import tx_smooth as TS
import placer as PL
from fasteval import FastEval

NAME = os.environ.get("GIF_BENCH", "ibm01")
N_STEPS = int(os.environ.get("GIF_STEPS", "650"))
MAX_SEC = float(os.environ.get("GIF_MAXSEC", "900"))
N_FRAMES = int(os.environ.get("GIF_FRAMES", "55"))
OUT = os.environ.get("GIF_OUT", os.path.join(REPO, "assets", "talyxion_ibm01.gif"))

bdir = os.path.join(REPO, "external", "MacroPlacement", "Testcases", "ICCAD04", NAME)
print(f"[gif] loading {NAME} from {bdir}")
benchmark, plc = load_benchmark_from_dir(bdir)
fe = FastEval(plc, benchmark)

dw, cw_ = 0.7, 1.3
gd = PL._SmoothGD(benchmark, plc, dw, cw_)

nh = benchmark.num_hard_macros
sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
cw = float(benchmark.canvas_width)
ch = float(benchmark.canvas_height)

# steps at which we snapshot positions (denser early where motion is large)
raw = np.unique(np.round(np.linspace(0, N_STEPS - 1, N_FRAMES) ** 1.0).astype(int))
snap_steps = set(int(s) for s in raw)

# ── run the team's smooth-GD (cycle-1 params) with frame capture ─────────────
# The GD trajectory is deterministic (seed 42) and is the slow part (~3 min),
# so cache the snapshot positions; color/style tweaks then re-render in seconds.
g_start, g_end, lr0, lr1 = 0.5, 8.0, 0.2, 0.02
CACHE = os.path.join(REPO, f".gifcache_{NAME}_{N_STEPS}_{N_FRAMES}.npz")

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    arr, labels, gammas = z["pos"], list(z["labels"]), z["gammas"]
    frames_pos = [(str(labels[i]), arr[i], float(gammas[i])) for i in range(len(labels))]
    print(f"[gif] reused {len(frames_pos)} cached GD frames from {CACHE}")
else:
    torch.manual_seed(42)
    init = torch.tensor(benchmark.macro_positions.numpy().astype(np.float64), dtype=torch.float32)
    init[:, 0].clamp_(gd.hw, gd.cw - gd.hw)
    init[:, 1].clamp_(gd.hh, gd.ch - gd.hh)
    fixed_xy = init.clone()
    pos = torch.nn.Parameter(init.clone())
    opt = torch.optim.Adam([pos], lr=lr0)
    PH = 0.5

    frames_pos = []   # (label, pos_np, gamma)
    t0 = time.time()
    print(f"[gif] running {N_STEPS} GD steps, snapshotting {len(snap_steps)} frames")
    for step in range(N_STEPS):
        if step > 40 and time.time() - t0 > MAX_SEC:
            print(f"[gif] wall cap hit at step {step}")
            break
        t = step / max(N_STEPS - 1, 1)
        if t < PH:
            gamma = g_start + t / PH * (2.0 - g_start)
        else:
            gamma = 2.0 * (g_end / 2.0) ** ((t - PH) / (1 - PH))
        lr = (lr0 * (step + 1) / 20 if step < 20
              else lr0 + (step - 20) / max(N_STEPS - 21, 1) * (lr1 - lr0))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        pos_aug = torch.cat([pos, gd.port_pos], dim=0)
        wl, d, c = TS.proxy_components(
            pos_aug, gd.sizes, gd.owner, gd.offset, gd.mask,
            gd.net_w, gd.net_cnt, gd.Gw, gd.Gh, gd.cw, gd.ch,
            gd.hrpm, gd.vrpm, gd.halloc, gd.valloc,
            gd.n_hard, gd.n_macros, gamma_wl=gamma, srange=gd.srange)
        (wl + gd.dw * d + gd.cw_ * c).backward()
        opt.step()
        with torch.no_grad():
            pos[~gd.movable] = fixed_xy[~gd.movable]
            pos[:, 0].clamp_(gd.hw, gd.cw - gd.hw)
            pos[:, 1].clamp_(gd.hh, gd.ch - gd.hh)
        if step in snap_steps:
            frames_pos.append((f"GD step {step+1}/{N_STEPS}",
                               pos.detach().numpy().astype(np.float64), gamma))

    # final GD position + legalization (resolve hard-macro overlaps -> 0 overlaps)
    gd_final = pos.detach().numpy().astype(np.float64)
    print(f"[gif] GD done in {time.time()-t0:.0f}s, legalizing...")
    leg, ov = PL._legalize(gd_final, benchmark)
    print(f"[gif] legalized, overlaps={ov}")
    frames_pos.append(("legalize (0 overlaps)", leg, g_end))

    np.savez(CACHE,
             pos=np.stack([fp[1] for fp in frames_pos]),
             labels=np.array([fp[0] for fp in frames_pos], dtype=object),
             gammas=np.array([fp[2] for fp in frames_pos]))
    print(f"[gif] cached GD frames -> {CACHE}")

# ── render ───────────────────────────────────────────────────────────────────
def proxy_of(p):
    pr, w, d, c = fe.proxy(p)
    return pr, w, d, c

fig = plt.figure(figsize=(7.6, 8.0), dpi=110)
images = []
hw = sizes[:, 0] * 0.5
hh = sizes[:, 1] * 0.5

# color hard macros by AREA RANK (one giant macro flattens any value scale),
# so the full vivid `turbo` range is used: small macros -> blue/teal,
# mid -> green/yellow, large -> orange/red. Soft macros stay a faint cloud.
area = sizes[:nh, 0] * sizes[:nh, 1]
rank01 = np.argsort(np.argsort(area)) / max(nh - 1, 1)
hcol = plt.cm.turbo(0.04 + 0.92 * rank01)

print(f"[gif] rendering {len(frames_pos)} frames")
for fi, (label, p, gamma) in enumerate(frames_pos):
    fig.clf()
    ax = fig.add_axes([0.06, 0.05, 0.90, 0.86])
    ax.set_xlim(0, cw); ax.set_ylim(0, ch); ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", lw=2))
    ax.set_xticks([]); ax.set_yticks([])

    # soft macros (cell clusters) — faint cloud
    soft_patches = [Rectangle((p[i, 0] - sizes[i, 0] / 2, p[i, 1] - sizes[i, 1] / 2),
                              sizes[i, 0], sizes[i, 1]) for i in range(nh, benchmark.num_macros)]
    ax.add_collection(PatchCollection(soft_patches, facecolor="#9db8d8",
                                      alpha=0.12, edgecolor="none", zorder=1))
    # hard macros — solid, colored by area
    hard_patches = [Rectangle((p[i, 0] - hw[i], p[i, 1] - hh[i]),
                              sizes[i, 0], sizes[i, 1]) for i in range(nh)]
    ax.add_collection(PatchCollection(hard_patches, facecolor=hcol, alpha=0.92,
                                      edgecolor="#1b2838", linewidths=0.4, zorder=3))

    pr, w, d, c = proxy_of(p)
    fig.suptitle(f"{NAME} · Talyxion HybridV26 — smooth-proxy gradient descent",
                 fontsize=13, fontweight="bold", y=0.975)
    ax.set_title(f"{label}    |    proxy = {pr:.3f}   "
                 f"(WL {w:.3f} · den {d:.3f} · cong {c:.3f})",
                 fontsize=11, pad=8)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    images.append(buf[:, :, :3].copy())

# hold the first and last frames longer
images = [images[0]] * 6 + images + [images[-1]] * 14

print(f"[gif] writing {OUT}  ({len(images)} frames)")
imageio.mimsave(OUT, images, duration=0.12, loop=0)
print(f"[gif] done: {OUT}")
