"""Figure generation for Sections `sec:results` (Figs. `fig:regimes`, `fig:frontier`).

Two design decisions worth stating, because both are places the obvious chart
would mislead:

**No dual axis, anywhere.** The frontier figure carries three measures on
incompatible scales -- an unauthorised-restoration probability spanning five
orders of magnitude, a loss probability in [0, 1], and a latency in hours.
Overlaying them on twin y-axes lets the reader infer a crossing point that is an
artefact of where the two scales were pinned. They are drawn as stacked small
multiples on a shared ``t`` axis instead, so a crossing is only ever read within
one panel.

**No log-scale bars.** P(ext) for B0 runs from 0.51 to 0.90 across regimes, a
factor of thirty. Putting both on one log-scaled bar axis breaks the one thing
bars encode -- length from zero. They are drawn as two panels with their own
linear axes and the ratio stated in the annotation, which is the comparison the
text actually makes.

Colours are the Okabe-Ito set, which is safe across deuteranopia, protanopia and
tritanopia; the series also carry distinct hatching so the figures survive
greyscale printing, where identity must not rest on hue alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import ctmc
from . import params as P
from .threshold import p_unauthorised_restoration

# Okabe-Ito: identity is assigned in fixed order, never cycled.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#4D4D4D"
MUTED = "#8A8A8A"

VARIANT_STYLE = {
    "B0": dict(color=VERMILLION, hatch="///"),
    "H1": dict(color=BLUE, hatch=""),
}


def use_paper_style() -> None:
    """Match the IEEE Access two-column body text; recessive grid and axes."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.6,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        # Enough pad that outboard endpoint labels are not shaved by the crop.
        "savefig.pad_inches": 0.06,
        "hatch.linewidth": 0.5,
    })


def _regime_series(adversary: str = "A2"):
    """P(ext) across the pooled and stratified possession regimes.

    Only priced variants appear. H1 is rated rather than priced (Section
    `sec:ap`), so there is no curve for it and the figure does not invent one.
    """
    regimes = [("Pooled", P.POSSESSION_POOLED_H)]
    regimes += sorted(P.POSSESSION_STRATA_H.items(), key=lambda kv: kv[1])
    labels = [name.replace(" (", "\n(") for name, _ in regimes]
    data = {
        v: [ctmc.extraction_probability(v, adversary, hours) for _, hours in regimes]
        for v in sorted(P.TIME_TO_EXTRACTION_H)
    }
    return labels, [h for _, h in regimes], data


def figure_regimes(outdir: Path, adversary: str = "A2") -> Path:
    """Fig. `fig:regimes`: extraction probability across the empirical regimes."""
    labels, hours, data = _regime_series(adversary)
    values = data["B0"]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    style = VARIANT_STYLE["B0"]
    bars = ax.bar(list(x), values, width=0.58,
                  color=style["color"], hatch=style["hatch"],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for rect, value in zip(bars, values):
        ax.annotate(f"{value:.3f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    textcoords="offset points", xytext=(0, 2),
                    ha="center", va="bottom", fontsize=7, color=GREY)

    # No titles, no in-axes commentary, no footer: at 3.5 in wide there is no
    # room for prose, and the caption is where prose belongs. An earlier version
    # carried a suptitle, an axes title, a two-line annotation and a footnote,
    # and they collided.
    ax.set_xticks(list(x))
    # Short tick labels. At 3.5 in the full stratum names run into each other;
    # the caption carries them instead.
    short = {"Pooled": "Pooled"}
    ax.set_xticklabels(
        [short.get(l, l.split(chr(10))[0].replace("NAVAREA ", "NAVAREA "))
         for l in labels], fontsize=7)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel(f"P(extracted), adversary {adversary}", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, outdir, "sgrd-regimes")


def figure_frontier(outdir: Path, adversary: str = "A2") -> Path:
    """Fig. `fig:frontier`: the threshold trade-off over the five-holder set."""
    ts = list(range(1, len(P.HOLDERS) + 1))
    rows = [ctmc.evaluate(P.Configuration(f"{t}-of-5", t, P.HOLDERS, "H1"),
                          adversary=adversary) for t in ts]
    p_u = [p_unauthorised_restoration(P.HOLDERS, t) for t in ts]
    p_loss = [r.p_loss_given_custody for r in rows]
    e_tc = [r.mean_time_to_restore_given_custody for r in rows]

    viable = [t for t in ts if p_u[t - 1] < 1e-2 and p_loss[t - 1] < 0.2]

    # Constrained layout rather than tight_layout: it accounts for the suptitle
    # when sizing the axes, so the panels do not get pushed down by half an inch
    # of reserved-but-unused space.
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 5.2), sharex=True,
                             layout="constrained")
    panels = (
        (axes[0], p_u, VERMILLION, "o",
         "$P_u$: adversary assembles authority", True, None),
        (axes[1], p_loss, BLUE, "s",
         "$P(\\mathrm{loss}\\mid c)$: recovered, then written off", False, (0, 1)),
        (axes[2], e_tc, GREEN, "^",
         "$E[T_c]$ (h): custody regained $\\rightarrow$ restored", False, None),
    )

    for ax, series, colour, marker, title, logy, ylim in panels:
        if viable:
            ax.axvspan(min(viable) - 0.35, max(viable) + 0.35,
                       color="#F0F0F0", zorder=0)
        ax.plot(ts, series, color=colour, linewidth=2.0, marker=marker,
                markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                zorder=3, clip_on=False)
        if logy:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", pad=4)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    # Direct-label the endpoints only. A number on every point would restate the
    # axis; the endpoints are what the finding is about. They sit horizontally
    # outboard of the first and last markers, in the margin the widened xlim
    # opens up: the line runs strictly between them, so no label can collide
    # with the series or with a panel title.
    for ax, series, fmt in ((axes[0], p_u, "{:.1e}"),
                            (axes[1], p_loss, "{:.3f}"),
                            (axes[2], e_tc, "{:.1f}")):
        ax.set_xlim(0.42, 5.58)
        for i, (dx, ha) in ((0, (-6, "right")), (len(ts) - 1, (6, "left"))):
            ax.annotate(fmt.format(series[i]), (ts[i], series[i]),
                        textcoords="offset points", xytext=(dx, 0),
                        ha=ha, va="center", fontsize=7, color=GREY,
                        annotation_clip=False)

    axes[2].set_xlabel("recovery threshold $t$ (over $n=5$ holders)")
    axes[2].set_xticks(ts)
    if viable:
        # Sit the band label low inside the shaded span: the P_u series descends
        # steeply across it, so anything above mid-panel would cross the line.
        axes[0].annotate(
            f"viable band $t \\in \\{{{min(viable)},{max(viable)}\\}}$",
            xy=((min(viable) + max(viable)) / 2, 0.22),
            xycoords=("data", "axes fraction"),
            ha="center", va="center", fontsize=7, color=GREY,
        )

    fig.suptitle("Raising $t$ trades unauthorised restoration\n"
                 "against permanent loss and latency", fontsize=8.5)
    return _save(fig, outdir, "sgrd-frontier")


def figure_finding1(outdir: Path, adversary: str = "A2") -> Path:
    """Supplementary: Finding 1, widening n against raising t.

    Not a figure in the paper. It is included because Finding 1 is stated as a
    dominance claim on three axes at once, and a dominance claim is far easier to
    falsify by eye than by reading a table row.
    """
    pairs = [("2-of-3", 2, ("Q1", "Q2", "Q3")),
             ("2-of-4", 2, ("Q1", "Q2", "Q3", "Q4")),
             ("3-of-4", 3, ("Q1", "Q2", "Q3", "Q4")),
             ("3-of-5", 3, ("Q1", "Q2", "Q3", "Q4", "Q5"))]
    rows = [(name, ctmc.evaluate(
        P.Configuration(name, t, P.holder_set(*tags), "H1"), adversary=adversary))
        for name, t, tags in pairs]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    metrics = (
        ("$P(\\mathrm{rest}\\mid c)$", [r.p_restored_given_custody for _, r in rows],
         BLUE, "higher is better"),
        ("$P(\\mathrm{loss}\\mid c)$", [r.p_loss_given_custody for _, r in rows],
         VERMILLION, "lower is better"),
        ("$E[T_c]$ (h)", [r.mean_time_to_restore_given_custody for _, r in rows],
         GREEN, "lower is better"),
    )
    names = [n for n, _ in rows]
    for ax, (title, values, colour, hint) in zip(axes, metrics):
        bars = ax.bar(names, values, width=0.6, color=colour,
                      edgecolor="white", linewidth=0.8, zorder=3)
        for rect, v in zip(bars, values):
            ax.annotate(f"{v:.3f}" if v < 10 else f"{v:.1f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=7, color=GREY)
        ax.set_title(f"{title}  ({hint})", loc="left", pad=4)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Finding 1: adding a holder improves every post-custody axis; "
                 "raising the threshold does not", fontsize=8.5, y=1.04)
    fig.tight_layout()
    return _save(fig, outdir, "sgrd-finding1")


def _save(fig, outdir: Path, stem: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(pdf)
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    plt.close(fig)
    return pdf


def generate_all(outdir: Path, adversary: str = "A2") -> list[Path]:
    use_paper_style()
    return [figure_regimes(outdir, adversary),
            figure_frontier(outdir, adversary),
            figure_finding1(outdir, adversary)]
