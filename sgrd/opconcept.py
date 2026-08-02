"""Fig. 1: the operational concept, drawn rather than imported.

The figure has to track the design, and the design moved: the protected boundary
is drawn around *execution* and not around storage, and the navigation kernel
sits outside it on purpose. A stale diagram is worse than none, so this is
generated from the same run that produces the quantitative figures.

Nothing here is computed -- it is a schematic. It lives in its own module for
that reason, so that :mod:`sgrd.figures` stays a place where every mark is
traceable to a number.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .figures import GREY, MUTED, use_paper_style

DEAD = "#eaeaea"   # inert after denial
LIVE = "#f7f7f7"   # still meaningful
QUOR = "#d9d9d9"   # the t holders that answered


def _box(ax, x, y, w, h, text, *, fc=LIVE, ec="0.35", ls="-", lw=0.8,
         fs=5.6, tc="0.15", va="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linestyle=ls, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va=va,
            fontsize=fs, color=tc, zorder=3, linespacing=1.35)


def _arrow(ax, xy, xytext, *, ls="-", lw=0.8, color="0.3"):
    ax.add_patch(FancyArrowPatch(
        xytext, xy, arrowstyle="-|>", mutation_scale=6,
        linewidth=lw, linestyle=ls, color=color,
        shrinkA=0, shrinkB=0, zorder=4))


def figure_opconcept(outdir: Path) -> Path:
    use_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 2.44))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- on board ---------------------------------------------------------
    _box(ax, 0.012, 0.375, 0.475, 0.545, "", fc="#fcfcfc", ec="0.55", lw=0.9)
    ax.text(0.25, 0.945, "crewless vessel, after denial", ha="center",
            fontsize=6.2, color="0.1", weight="bold")

    # the protected boundary is around execution, not storage
    _box(ax, 0.030, 0.560, 0.215, 0.310, "", fc="none", ec="0.35", ls="--",
         lw=1.1)
    ax.text(0.1375, 0.892, "memory-encrypted execution domain", ha="center",
            fontsize=5.0, color=MUTED, style="italic")
    _box(ax, 0.042, 0.735, 0.191, 0.105,
         "$A_{cpi}$ execution\nworking memory discarded", fc=DEAD, tc="0.45")
    _box(ax, 0.042, 0.580, 0.191, 0.130,
         "secure element\nPUF · RBG · $pk_r$\n$K_e$ zeroised at denial",
         fc=DEAD, tc="0.45")

    _box(ax, 0.263, 0.560, 0.212, 0.310,
         "external storage\n\n$Enc(A_{cpi})$\n$W=Wrap_{K_e}(K_w)$\n"
         "$RC=Enc_{pk_r}(K_e\\Vert epoch)$\n\ninert without $K_e$")

    _box(ax, 0.030, 0.408, 0.445, 0.120,
         "navigation kernel — positioning, steering, propulsion, distress\n"
         "denial does not reach it; holds a minimum-risk condition")

    # ---- ashore -----------------------------------------------------------
    _box(ax, 0.585, 0.375, 0.403, 0.545, "", fc="#fcfcfc", ec="0.55", lw=0.9)
    ax.text(0.7865, 0.945, "ashore", ha="center", fontsize=6.2, color="0.1",
            weight="bold")

    holders = [("ROC", True), ("class", True), ("owner", False),
               ("escrow", False), ("flag", False)]
    for k, (name, in_quorum) in enumerate(holders):
        _box(ax, 0.600 + k * 0.078, 0.815, 0.068, 0.058, name,
             fc=QUOR if in_quorum else LIVE,
             lw=1.3 if in_quorum else 0.8, fs=5.2)
    ax.text(0.7865, 0.783, "$t$ of $n$, organisationally separated",
            ha="center", fontsize=5.0, color=MUTED, style="italic")

    _box(ax, 0.600 + 0.0, 0.612, 0.372, 0.145,
         "$\\tau = Sig_{t\\mathrm{-of-}n}(h_{puf}\\Vert\\nu\\Vert c{+}1"
         "\\Vert h_{safe}\\Vert h_{att}\\Vert h_{cust})$\n\n"
         "$+\\ Enc_{pk_{eph}}(K_e\\Vert epoch)$")
    _box(ax, 0.600, 0.408, 0.372, 0.120,
         "custody attestation $A_{cust}$ — port state or flag\n"
         "a second, disjoint authority; $h_{cust}=H(A_{cust})$")

    # ---- traffic ----------------------------------------------------------
    _arrow(ax, (0.583, 0.720), (0.489, 0.720))
    ax.text(0.536, 0.734, "M1 request", ha="center", fontsize=5.0,
            color=MUTED, style="italic")
    _arrow(ax, (0.489, 0.640), (0.583, 0.640))
    ax.text(0.536, 0.600, "token +\ncapsule opening", ha="center",
            fontsize=5.0, color=MUTED, style="italic", linespacing=1.3)

    # ---- ordered return ---------------------------------------------------
    stages = ["R0 admission", "R1 re-attestation", "R2 supervised nav",
              "R3 full capability"]
    for k, name in enumerate(stages):
        _box(ax, 0.012 + k * 0.196, 0.175, 0.170, 0.082, name)
        if k:
            _arrow(ax, (0.012 + k * 0.196, 0.216),
                   (0.012 + (k - 1) * 0.196 + 0.170, 0.216))
    _box(ax, 0.836, 0.175, 0.152, 0.082, "denial", fc=DEAD, tc="0.45")

    _arrow(ax, (0.097, 0.259), (0.097, 0.373))
    for k in (1, 2, 3):
        x = 0.012 + k * 0.196 + 0.085
        ax.plot([x, x, 0.912, 0.912], [0.175, 0.108, 0.108, 0.175],
                lw=0.7, ls=":", color="0.45", zorder=1)
    _arrow(ax, (0.912, 0.174), (0.912, 0.120), ls=":", lw=0.7, color="0.45")
    ax.text(0.47, 0.078,
            "attestation mismatch, or the safe-state predicate lapsing at any "
            "stage, returns the platform to denial",
            ha="center", fontsize=5.0, color=MUTED, style="italic")
    ax.text(0.47, 0.028,
            "each stage is admitted only while that predicate holds",
            ha="center", fontsize=5.0, color=MUTED, style="italic")

    fig.tight_layout(pad=0.15)
    out = outdir / "sgrd-op-concept.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out
