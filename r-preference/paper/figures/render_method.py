#!/usr/bin/env python3
"""Render the SPT method figure to figures/method.{pdf,png}.

Standalone matplotlib renderer (no LaTeX toolchain required). The same diagram
also exists as a TikZ source (method_fig.tex) for a camera-ready vector build.
Run from anywhere:  python render_method.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- palette ----------------------------------------------------------------
C_BB   = dict(fc="#dbe7f6", ec="#3a6ea5")   # shared backbone (blue)
C_ACT  = dict(fc="#fde8d0", ec="#d2872b")   # action expert (orange)
C_INF  = dict(fc="#d4ece9", ec="#2e8b84")   # preference inference (teal)
C_PHI  = dict(fc="#e9f4f2", ec="#5aa39b")   # pose -> token (light teal)
C_FRO  = dict(fc="#ededed", ec="#8a8a8a")   # frozen (gray)
ARR    = "#333333"
PANEL  = "#b9b9b9"

fig, ax = plt.subplots(figsize=(12.2, 4.5))
ax.set_xlim(-2.0, 17.3)
ax.set_ylim(-4.95, 3.35)
ax.axis("off")
ax.set_aspect("equal")


def box(x, y, w, h, text, col, dashed=False, fs=10):
    ls = (0, (4, 3)) if dashed else "solid"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.14",
                       fc=col["fc"], ec=col["ec"], lw=1.4, ls=ls, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def txt(x, y, text, fs=9, weight="normal", style="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, fontstyle=style, zorder=4)


def arrow(p0, p1, rad=0.0, color=ARR):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                        lw=1.3, color=color, zorder=2,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


# ---- panel separators + titles ---------------------------------------------
ax.add_patch(Rectangle((-1.7, -3.78), 9.65, 6.83, fill=False,
                       ec=PANEL, lw=1.0, ls=(0, (5, 4)), zorder=1))
ax.add_patch(Rectangle((8.25, -3.78), 8.65, 6.83, fill=False,
                       ec=PANEL, lw=1.0, ls=(0, (5, 4)), zorder=1))
txt(3.1, 2.72, "Stage A  —  Source (preference-labeled)", fs=11, weight="bold")
txt(12.55, 2.72, "Stage B  —  Target (unlabeled)", fs=11, weight="bold")

# =============================== STAGE A =====================================
box(2.4, 0.0, 1.9, 2.5, "Shared\nVLM\nbackbone", C_BB, fs=10)
txt(-0.55, 1.2, "multi-view\nclip $o$", fs=9)
box(-0.45, -1.2, 1.75, 1.0, r"pose $s\rightarrow\phi$" + "\n" + r"$\rightarrow$ token", C_PHI, fs=9)
txt(2.4, -2.62, r'$\ell$ + "Preference: $w(c)$"', fs=9)

box(5.35, 1.15, 1.75, 1.0, "Action\nexpert", C_ACT, fs=10)
box(5.35, -1.15, 1.75, 1.0, "LM head\n(pref. VQA)", C_INF, fs=10)
txt(7.25, 1.32, r"$\hat a$", fs=10)
txt(7.25, 0.95, r"$\mathcal{L}_{\mathrm{act}}$", fs=9, style="italic")
txt(7.25, -0.98, r"$\hat c$", fs=10)
txt(7.25, -1.35, r"$\mathcal{L}_{\mathrm{inf}}$", fs=9, style="italic")

arrow((0.15, 1.2), (1.42, 0.82))
arrow((0.45, -1.2), (1.42, -0.82))
arrow((2.4, -2.32), (2.4, -1.27))
arrow((3.37, 1.0), (4.45, 1.15)); txt(3.9, 1.55, "act on $c$", fs=8.5)
arrow((3.37, -1.0), (4.45, -1.15)); txt(3.9, -1.58, "infer $c$", fs=8.5)
arrow((6.25, 1.15), (6.95, 1.15))
arrow((6.25, -1.15), (6.95, -1.15))
txt(3.0, -3.45, r"$\mathcal{L}_A=\mathcal{L}_{\mathrm{act}}+\lambda\,\mathcal{L}_{\mathrm{inf}}$", fs=10, style="italic")

# =============================== STAGE B =====================================
txt(10.05, 2.0, "(1) self-label", fs=9, weight="bold")
txt(9.3, 1.0, "target demo\nclip $o$, pose $s$", fs=9)
box(11.6, 1.0, 2.05, 1.1, "Frozen $M_A$\n(infer pref.)", C_FRO, dashed=True, fs=9.5)
txt(14.25, 1.0, r"$\tilde{c}\;\Rightarrow\;(\ell_T,\,w(\tilde{c}))$", fs=9.5)
arrow((10.15, 1.0), (10.58, 1.0))
arrow((12.63, 1.0), (13.25, 1.0))

txt(10.05, -0.3, "(2) adapt", fs=9, weight="bold")
box(11.7, -1.5, 2.15, 1.0, "Action expert\n(warm-start copy)", C_ACT, fs=9.5)
txt(14.45, -1.5, r"$\mathcal{L}_B=\mathcal{L}^{\mathcal{T}}_{\mathrm{act}}$", fs=9.5, style="italic")
arrow((14.25, 0.68), (11.9, -1.0), rad=-0.28)
arrow((12.78, -1.5), (13.75, -1.5))
txt(12.55, -3.35, "target labels never read (firewall);\nno preference loss in Stage B", fs=8.5)

# =============================== DEPLOY ======================================
txt(7.55, -4.45,
    r'Test: user states $c$;  policy executes "$\ell$ + Preference: $w(c)$";  inference branch discarded.',
    fs=8.5)

for ext in ("pdf", "png"):
    out = os.path.join(HERE, f"method.{ext}")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06,
                dpi=(200 if ext == "png" else None))
    print("wrote", out)
