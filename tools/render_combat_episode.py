"""Render a fixed-camera scientific MP4 and preview from a recorded trace."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

from tools.combat_visualization import (ENTITY_IDS, STYLES, death_records, episode_cube_ranges,
                                        interpolate_trace_for_visualization, load_trace)


def render_episode(
    input_dir: Path, output: Path | None = None, preview: Path | None = None, *,
    visual_dt: float = 0.1, fps: int = 20, trail_seconds: float = 10.0,
    elev: float = 27.0, azim: float = -55.0, show_heading: bool = True,
    overwrite: bool = False, preview_fraction: float = 0.6, mp4: bool = True,
) -> dict[str, str | None]:
    trace, metadata = load_trace(input_dir)
    visual = interpolate_trace_for_visualization(trace, float(metadata["decision_dt"]), visual_dt)
    directory = Path(input_dir).expanduser().resolve()
    output = (output or directory / "episode.mp4").expanduser().resolve()
    preview = (preview or directory / "preview.png").expanduser().resolve()
    for path in ([preview, output] if mp4 else [preview]):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing render: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    xyz = visual["kinematics"][:, :, :3] / 1000.0
    alive = visual["alive"]
    ranges = episode_cube_ranges(trace["kinematics"], trace["alive"])
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#f7f8fa")
    ax = fig.add_subplot(111, projection="3d", facecolor="#fbfcfe")
    ax.set_xlim(*ranges["x"]); ax.set_ylim(*ranges["y"]); ax.set_zlim(*ranges["z"])
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X / km", ylabel="Y / km", zlabel="Altitude / km")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=.25)
    ground_x, ground_y = np.meshgrid(ranges["x"], ranges["y"])
    ax.plot_surface(ground_x, ground_y, np.zeros_like(ground_x), color="#8d99a6",
                    alpha=.07, shade=False, linewidth=0)
    title = ax.set_title("")
    lines = []; points = []; labels = []; headings = []
    for i, aid in enumerate(ENTITY_IDS):
        style = STYLES[aid]
        line, = ax.plot([], [], [], color=style["color"], linewidth=style["width"],
                        linestyle="--" if style["dash"] == "dash" else "-", label=aid)
        point, = ax.plot([], [], [], marker=style["mpl_marker"], color=style["color"],
                         markersize=9 if aid == "MAV" else 7, linestyle="None")
        heading, = ax.plot([], [], [], color=style["color"], linewidth=1.4, alpha=.75)
        lines.append(line); points.append(point); headings.append(heading)
        labels.append(ax.text(0, 0, 0, "", color=style["color"], fontsize=8, weight="bold" if aid == "MAV" else "normal"))
    attack_line, = ax.plot([], [], [], color="#e83e3e", linewidth=2.2, alpha=.8)
    deaths = death_records(trace, metadata)
    death_artists = []
    for death in deaths:
        p = np.asarray(death["position"]) / 1000
        death_artists.append((death, ax.scatter([p[0]], [p[1]], [p[2]], marker="x", s=90,
                                                 color=STYLES[death["entity"]]["color"], visible=False)))
    ax.legend(loc="upper left", framealpha=.85)
    trail_frames = max(1, int(round(trail_seconds / visual_dt)))

    def update(frame: int):
        t = float(visual["time_s"][frame]); start = max(0, frame - trail_frames) if trail_seconds >= 0 else 0
        for i, aid in enumerate(ENTITY_IDS):
            valid = alive[start:frame + 1, i]
            pts = xyz[start:frame + 1, i][valid]
            lines[i].set_data_3d(pts[:, 0] if len(pts) else [], pts[:, 1] if len(pts) else [], pts[:, 2] if len(pts) else [])
            if alive[frame, i]:
                p = xyz[frame, i]; points[i].set_data_3d([p[0]], [p[1]], [p[2]])
                labels[i].set_position((p[0], p[1])); labels[i].set_3d_properties(p[2]); labels[i].set_text(" " + aid)
                if show_heading:
                    theta, psi = visual["kinematics"][frame, i, 4:6]; length = .5
                    q = p + length * np.array([np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), np.sin(theta)])
                    headings[i].set_data_3d([p[0], q[0]], [p[1], q[1]], [p[2], q[2]])
            else:
                points[i].set_data_3d([], [], []); labels[i].set_text(""); headings[i].set_data_3d([], [], [])
        segments = []
        for event in metadata.get("events", []):
            if event.get("type") == "attack" and 0 <= t - float(event["time_s"]) <= .8:
                raw_frame = int(event["trace_frame"])
                a = trace["kinematics"][raw_frame, ENTITY_IDS.index(event["attacker"]), :3] / 1000
                b = trace["kinematics"][raw_frame, ENTITY_IDS.index(event["target"]), :3] / 1000
                segments.extend([a, b, np.full(3, np.nan)])
        if segments:
            seg = np.asarray(segments); attack_line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        else: attack_line.set_data_3d([], [], [])
        for death, artist in death_artists: artist.set_visible(t >= float(death["time_s"]))
        title.set_text(f"{metadata['algorithm']} | Blue: {metadata['blue_target_mode']} | "
                       f"t={t:.1f}s, decision={visual['raw_step'][frame]} | {metadata['evaluation_profile']}")
        return [*lines, *points, *labels, *headings, attack_line, title, *(x[1] for x in death_artists)]

    preview_frame = min(len(visual["time_s"]) - 1, int(round((len(visual["time_s"]) - 1) * preview_fraction)))
    update(preview_frame); fig.tight_layout(); fig.savefig(preview, dpi=180, bbox_inches="tight")
    mp4_result: str | None = None
    if mp4:
        if shutil.which("ffmpeg") is None:
            plt.close(fig)
            raise RuntimeError(f"preview generated at {preview}; ffmpeg is required for MP4 rendering")
        animation = FuncAnimation(fig, update, frames=len(visual["time_s"]), interval=1000/fps, blit=False)
        animation.save(output, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=120)
        mp4_result = str(output)
    plt.close(fig)
    return {"preview": str(preview), "mp4": mp4_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True); parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", type=Path); parser.add_argument("--visual-dt", type=float, default=.1)
    parser.add_argument("--fps", type=int, default=20); parser.add_argument("--trail-seconds", type=float, default=10.)
    parser.add_argument("--elev", type=float, default=27.); parser.add_argument("--azim", type=float, default=-55.)
    parser.add_argument("--no-heading", action="store_true"); parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(render_episode(args.input_dir, args.output, args.preview, visual_dt=args.visual_dt, fps=args.fps,
                         trail_seconds=args.trail_seconds, elev=args.elev, azim=args.azim,
                         show_heading=not args.no_heading, overwrite=args.overwrite, mp4=not args.preview_only))


if __name__ == "__main__": main()
