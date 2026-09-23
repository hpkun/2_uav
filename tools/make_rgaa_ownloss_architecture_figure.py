"""Generate the paper-ready RGAA + own-loss HAPPO architecture figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures"

COLORS = {
    "ink": "#233142",
    "muted": "#607386",
    "execution": "#2E6F95",
    "execution_fill": "#EAF3F8",
    "actor_fill": "#DDECF5",
    "team": "#526D82",
    "team_fill": "#EEF2F5",
    "aux": "#C56A32",
    "aux_fill": "#FFF1E7",
    "fusion": "#4F7C69",
    "fusion_fill": "#E8F2ED",
    "panel": "#FAFBFC",
    "border": "#CBD5DE",
    "white": "#FFFFFF",
}


def rounded_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    size: float = 8.2,
    weight: str = "normal",
    linestyle: str = "-",
    linewidth: float = 1.35,
    radius: float = 0.012,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=size, color=COLORS["ink"],
        fontweight=weight, linespacing=1.25, zorder=zorder + 1,
    )
    return patch


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.35,
    connectionstyle: str = "arc3,rad=0",
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=11,
        linewidth=linewidth, color=color, linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=2, zorder=zorder,
    )
    axis.add_patch(patch)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axis.text(
            midpoint[0] + label_offset[0], midpoint[1] + label_offset[1], label,
            ha="center", va="center", fontsize=6.8, color=color,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.94},
            zorder=zorder + 2,
        )
    return patch


def panel(axis, x, y, width, height, title, subtitle, color):
    rounded_box(
        axis, x, y, width, height, "", face=COLORS["panel"], edge=COLORS["border"],
        linewidth=1.15, radius=0.016, zorder=0,
    )
    axis.text(x + 0.016, y + height - 0.032, title, fontsize=12.2,
              fontweight="bold", color=color, va="center", zorder=5)
    axis.text(x + 0.016, y + height - 0.061, subtitle, fontsize=7.5,
              color=COLORS["muted"], va="center", zorder=5)


def make_figure(output_dir: Path, dpi: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "rgaa_ownloss_happo_architecture.svg"
    png_path = output_dir / "rgaa_ownloss_happo_architecture.png"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
    figure, axis = plt.subplots(figsize=(17.0, 10.4), facecolor="white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5, 0.972, "Overview of RGAA + Own-Loss HAPPO",
        ha="center", va="center", fontsize=18, fontweight="bold", color=COLORS["ink"],
    )
    axis.text(
        0.5, 0.944,
        "Role-guided auxiliary advantage with centralized training and decentralized execution",
        ha="center", va="center", fontsize=9.2, color=COLORS["muted"],
    )

    # ------------------------------------------------------------------
    # Decentralized execution
    # ------------------------------------------------------------------
    panel(
        axis, 0.025, 0.675, 0.95, 0.245,
        "A   Decentralized Execution",
        "Solid lines denote the online interaction path; no critic is evaluated at inference.",
        COLORS["execution"],
    )
    rounded_box(
        axis, 0.055, 0.724, 0.135, 0.124,
        "4v4 Environment\n1 MAV + 3 UAV\nvs. 4 Blue aircraft",
        face=COLORS["execution_fill"], edge=COLORS["execution"], size=8.5, weight="bold",
    )
    row_y = (0.826, 0.795, 0.764, 0.733)
    observation_labels = ("MAV observation  $o_M$", "UAV1 observation  $o_{U1}$",
                          "UAV2 observation  $o_{U2}$", "UAV3 observation  $o_{U3}$")
    actor_labels = ("MAV actor  $\\pi_M$", "UAV1 actor  $\\pi_{U1}$",
                    "UAV2 actor  $\\pi_{U2}$", "UAV3 actor  $\\pi_{U3}$")
    action_labels = ("$a_M$", "$a_{U1}$", "$a_{U2}$", "$a_{U3}$")
    # Four separate rows make decentralization explicit: no actor consumes another
    # actor's observation or action during deployment.
    axis.plot([0.21, 0.21], [0.745, 0.838], color=COLORS["execution"], linewidth=1.3, zorder=2)
    arrow(axis, (0.19, 0.786), (0.21, 0.786), color=COLORS["execution"], linewidth=1.7)
    for y, observation_label, actor_label, action_label in zip(
        row_y, observation_labels, actor_labels, action_labels
    ):
        rounded_box(
            axis, 0.235, y - 0.012, 0.155, 0.024, observation_label,
            face=COLORS["white"], edge=COLORS["execution"], size=6.9,
        )
        rounded_box(
            axis, 0.455, y - 0.012, 0.125, 0.024, actor_label,
            face=COLORS["actor_fill"], edge=COLORS["execution"], size=6.9, weight="bold",
        )
        axis.plot([0.21, 0.235], [y, y], color=COLORS["execution"], linewidth=1.15, zorder=2)
        arrow(axis, (0.39, y), (0.455, y), color=COLORS["execution"], linewidth=1.2)
        arrow(
            axis, (0.58, y), (0.69, y), color=COLORS["execution"], linewidth=1.2,
            label=action_label, label_offset=(0.0, 0.009),
        )
    # The four actions are collected only at the environment interface.
    axis.plot([0.69, 0.69], [0.733, 0.826], color=COLORS["execution"], linewidth=1.45, zorder=2)
    axis.text(0.706, 0.780, "joint action  $\\mathbf{a}$", fontsize=7.2,
              color=COLORS["execution"], rotation=90, ha="center", va="center")
    arrow(
        axis, (0.69, 0.733), (0.12, 0.724), color=COLORS["execution"], linewidth=1.7,
        connectionstyle="arc3,rad=-0.22",
    )
    rounded_box(
        axis, 0.83, 0.855, 0.12, 0.038,
        "",
        face=COLORS["execution"], edge=COLORS["execution"], size=7.0, weight="bold",
    )
    axis.text(0.89, 0.874, "INFERENCE: ACTORS ONLY", color="white", ha="center", va="center",
              fontsize=7.0, fontweight="bold", zorder=8)

    # ------------------------------------------------------------------
    # Centralized training
    # ------------------------------------------------------------------
    panel(
        axis, 0.025, 0.045, 0.95, 0.60,
        "B   Centralized Training",
        "Dashed modules and paths are used only for learning; the four policy networks remain independent.",
        COLORS["aux"],
    )

    # Environment outputs shared by the two learning branches.
    rounded_box(
        axis, 0.045, 0.345, 0.13, 0.185,
        "Environment rollout\n\nGlobal state  $s$\nShared reward  $r_{team}$\nLocal observations  $o_i$\nInfo: process rewards\nand death_causes",
        face="#F5F7F9", edge=COLORS["muted"], size=7.5, weight="bold",
    )

    # Team branch.
    axis.text(0.205, 0.558, "TEAM BRANCH (VANILLA HAPPO, PRESERVED)", fontsize=7.5,
              fontweight="bold", color=COLORS["team"])
    rounded_box(
        axis, 0.205, 0.455, 0.14, 0.075,
        "Centralized Team Critic\n$V_{team}(s)$",
        face=COLORS["team_fill"], edge=COLORS["team"], size=8.1,
        linestyle="--",
    )
    rounded_box(
        axis, 0.39, 0.455, 0.125, 0.075,
        "Team GAE\n$A_{team}$",
        face=COLORS["team_fill"], edge=COLORS["team"], size=8.3,
        linestyle="--",
    )
    rounded_box(
        axis, 0.555, 0.455, 0.115, 0.075,
        "Shared source\n$\\mathrm{Norm}(A_{team})$",
        face=COLORS["team_fill"], edge=COLORS["team"], size=7.8,
        linestyle="--",
    )
    arrow(axis, (0.175, 0.49), (0.205, 0.49), color=COLORS["team"], linestyle="--", label="$s$")
    arrow(axis, (0.345, 0.492), (0.39, 0.492), color=COLORS["team"], linestyle="--")
    arrow(axis, (0.515, 0.492), (0.555, 0.492), color=COLORS["team"], linestyle="--")
    arrow(
        axis, (0.175, 0.455), (0.39, 0.47), color=COLORS["team"], linestyle="--",
        connectionstyle="arc3,rad=-0.13", label="$r_{team}$", label_offset=(0.0, -0.008),
    )

    # Auxiliary reward decomposition.
    axis.text(0.205, 0.405, "RGAA AUXILIARY BRANCH (ADDED)", fontsize=7.5,
              fontweight="bold", color=COLORS["aux"])
    rounded_box(
        axis, 0.205, 0.275, 0.19, 0.11,
        "Own-loss-aware Auxiliary Reward\n\n$r_{aux,i}=r_{process,i}+r_{\\mathrm{own-loss},i}$\n"
        "death_causes $\\rightarrow$ matching agent only\nMAV: mav_loss  |  UAV$_i$: uav_loss",
        face=COLORS["aux_fill"], edge=COLORS["aux"], size=7.4,
        linestyle="--",
    )
    arrow(
        axis, (0.175, 0.405), (0.205, 0.34), color=COLORS["aux"], linestyle="--",
        connectionstyle="arc3,rad=0.16", label="info", label_offset=(0.0, 0.005),
    )

    # Two role critics: one independent MAV network and one network shared by all UAVs.
    rounded_box(
        axis, 0.435, 0.322, 0.15, 0.067,
        "MAV Auxiliary Critic\n$V_{aux,M}(o_M)$",
        face=COLORS["aux_fill"], edge=COLORS["aux"], size=7.9,
        linestyle="--",
    )
    rounded_box(
        axis, 0.435, 0.218, 0.15, 0.084,
        "Shared UAV Auxiliary Critic\n(one network shared across UAV1/UAV2/UAV3)\n"
        "$V_{aux,U}(o_{U1}), V_{aux,U}(o_{U2}), V_{aux,U}(o_{U3})$",
        face=COLORS["aux_fill"], edge=COLORS["aux"], size=7.4,
        linestyle="--",
    )
    # Observation inputs are explicit in the value-function labels above.  The
    # orange reward path below is separate and feeds GAE, not the critics.

    rounded_box(
        axis, 0.62, 0.247, 0.135, 0.11,
        "Individual-death\nAuxiliary GAE\n\nnext-active mask\nterminates each $A_{aux,i}$",
        face=COLORS["aux_fill"], edge=COLORS["aux"], size=7.6,
        linestyle="--",
    )
    arrow(axis, (0.585, 0.355), (0.62, 0.325), color=COLORS["aux"], linestyle="--")
    arrow(axis, (0.585, 0.26), (0.62, 0.282), color=COLORS["aux"], linestyle="--")
    axis.plot([0.30, 0.30, 0.60], [0.275, 0.202, 0.202], color=COLORS["aux"],
              linewidth=1.25, linestyle="--", zorder=2)
    arrow(axis, (0.60, 0.202), (0.665, 0.247), color=COLORS["aux"], linestyle="--")
    axis.text(0.43, 0.211, "$r_{aux,M},\;r_{aux,U1:U3}$", fontsize=6.5,
              color=COLORS["aux"], ha="center", va="bottom",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})

    # Advantage fusion and sequential update.
    rounded_box(
        axis, 0.715, 0.405, 0.16, 0.125,
        "Advantage Fusion\n\n$A_i=\\mathrm{Norm}(A_{team})$\n$+\\;\\lambda\\,\\mathrm{Norm}(A_{aux,i})$\n$\\lambda=0.5$",
        face=COLORS["fusion_fill"], edge=COLORS["fusion"], size=8.2, weight="bold",
        linestyle="--", linewidth=1.6,
    )
    rounded_box(
        axis, 0.82, 0.235, 0.135, 0.115,
        "Sequential HAPPO Update\n\nRandom agent order\none actor at a time\n$A_i^{eff}=F_i\\,A_i$",
        face=COLORS["fusion_fill"], edge=COLORS["fusion"], size=7.8, weight="bold",
        linestyle="--", linewidth=1.6,
    )
    arrow(axis, (0.67, 0.492), (0.715, 0.492), color=COLORS["team"], linestyle="--")
    arrow(
        axis, (0.755, 0.302), (0.77, 0.405), color=COLORS["aux"], linestyle="--",
        label="$A_{aux,M}, A_{aux,U1:U3}$", label_offset=(0.045, 0.0),
    )
    arrow(
        axis, (0.82, 0.42), (0.865, 0.35), color=COLORS["fusion"], linestyle="--",
        label="preceding factor $F_i$", label_offset=(0.035, 0.015),
    )
    arrow(
        axis, (0.91, 0.35), (0.535, 0.718), color=COLORS["fusion"], linestyle="--",
        connectionstyle="arc3,rad=-0.22", label="updates each $\\pi_i$ sequentially",
        label_offset=(0.045, -0.006),
        linewidth=1.55,
    )

    # Increment over vanilla HAPPO and visual legend.
    rounded_box(
        axis, 0.055, 0.082, 0.31, 0.105,
        "Added components over Vanilla HAPPO\n"
        "1. MAV auxiliary critic     2. Shared UAV auxiliary critic\n"
        "3. Own-loss-aware auxiliary reward\n"
        "4. Role-guided advantage fusion",
        face="#FFF8F2", edge=COLORS["aux"], size=7.25, weight="bold",
    )
    axis.plot([0.405, 0.46], [0.126, 0.126], color=COLORS["execution"], linewidth=1.8)
    axis.text(0.47, 0.126, "execution / environment interaction", va="center",
              fontsize=6.8, color=COLORS["muted"])
    axis.plot([0.405, 0.46], [0.094, 0.094], color=COLORS["aux"], linewidth=1.5, linestyle="--")
    axis.text(0.47, 0.094, "training-only dataflow or module", va="center",
              fontsize=6.8, color=COLORS["muted"])
    axis.text(
        0.955, 0.075,
        "Critics are discarded after training.\nDeployment retains $\\pi_M, \\pi_{U1}, \\pi_{U2}, \\pi_{U3}$ only.",
        ha="right", va="bottom", fontsize=7.1, color=COLORS["execution"], fontweight="bold",
    )

    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return svg_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 150:
        raise ValueError("dpi must be at least 150 for a paper preview")
    svg_path, png_path = make_figure(args.output_dir, args.dpi)
    print(f"SVG: {svg_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
