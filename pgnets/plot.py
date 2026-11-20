"""One-page summary figure for a PGNets prediction."""

from __future__ import annotations

import numpy as np

from .models import MASS_LABELS, MASS_RATIOS, M_JUP_IN_MSUN, Prediction
from .preprocess import HALF_WIDTH_RP

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
BAR, BAR_TOP, VGG, REG = "#9ec5e8", "#2a6fb0", "#c2185b", "#e08a00"
REG_SIGMA_DEX = 0.16  # 1-sigma of the regression on the test set (Zhang, Zhu & Kang 2022)


def summary_figure(x64, pred: Prediction, cam=None, title=None, path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(12, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.28, left=0.03, right=0.98,
                          top=0.745, bottom=0.14)
    ext = (-HALF_WIDTH_RP, HALF_WIDTH_RP, -HALF_WIDTH_RP, HALF_WIDTH_RP)
    t = np.linspace(0, 2 * np.pi, 200)

    ax = fig.add_subplot(gs[0])
    ax.imshow(x64, origin="lower", cmap="inferno", extent=ext, vmin=0, vmax=1)
    ax.plot(np.cos(t), np.sin(t), ls="--", lw=0.8, color="w", alpha=0.6)
    ax.text(0.71, 0.71, "r$_p$", color="w", alpha=0.8, ha="left", va="bottom")
    ax.set_title("(a) network input (64×64, log-scaled)", fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1])
    ax.imshow(x64, origin="lower", cmap="gray", extent=ext, vmin=0, vmax=1)
    if cam is not None:
        rgba = plt.get_cmap("jet")(cam)
        rgba[..., 3] = 0.75 * cam ** 1.5  # opacity follows importance
        ax.imshow(rgba, origin="lower", extent=ext)
    ax.set_title("(b) Grad-CAM: what drives the Mp estimate", fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2])
    logq = np.log10(MASS_RATIOS)
    colors = [BAR_TOP if i == pred.resnet_class else BAR for i in range(5)]
    ax.bar(logq, pred.resnet_probs, width=0.32, color=colors, label="ResNet classifier", zorder=2)
    ax.plot(logq, pred.vgg_probs, "o", ms=8, mfc="none", mec=VGG, mew=1.5, label="VGG classifier", zorder=3)
    lq = np.log10(pred.q)
    ax.axvspan(lq - REG_SIGMA_DEX, lq + REG_SIGMA_DEX, color=REG, alpha=0.18, lw=0, zorder=1)
    ax.axvline(lq, color=REG, lw=2, label=f"ResNet regression (±{REG_SIGMA_DEX} dex)", zorder=4)
    ax.set_xticks(logq, [f"{q:.1e}\n{lab}" for q, lab in zip(MASS_RATIOS, MASS_LABELS)], fontsize=8)
    ax.set_xlim(logq[0] - 0.4, logq[-1] + 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("class probability")
    ax.set_xlabel("planet-to-star mass ratio  Mp/M*   (labels for M* = 1 M$_\\odot$)", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(0.98, 0.99), handlelength=1.4)
    ax.set_title("(c) predicted planet mass", fontsize=10, loc="left")

    if pred.mstar is not None:
        mp = f"Mp = {pred.q * pred.mstar / M_JUP_IN_MSUN:.2f} M$_J$ (M* = {pred.mstar} M$_\\odot$)"
    else:
        mp = f"Mp/M* = {pred.q:.2e}"
    headline = f"{mp}     α = {pred.alpha:.1e}"
    note = f"network error ±{REG_SIGMA_DEX} dex in Mp, ±0.23 dex in α"
    if pred.extra.get("n_orientations", 1) > 1:
        note = (f"mean of {pred.extra['n_orientations']} image orientations "
                f"(scatter {pred.extra['logq_std']:.2f} dex);  " + note)
    fig.text(0.03, 0.97, title or "PGNets prediction", fontsize=13, weight="bold", va="top")
    fig.text(0.03, 0.905, headline, fontsize=11.5, va="top", color=INK)
    fig.text(0.03, 0.845, note, fontsize=9, va="top", color=MUTED)
    if path:
        fig.savefig(path, dpi=150)
    return fig
