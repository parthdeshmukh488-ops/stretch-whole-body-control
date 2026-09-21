"""Render the robot reaching a target, with the safety filter in the loop.

    python examples/04_demo_animation.py

Writes docs/figures/05_demo.gif. The controller is IK run closed-loop: solve
from the current configuration, take the difference as a velocity, push it
through the filter, integrate one step.

The second half of the run is the interesting part. The controller is given a
target it cannot reach without violating the tipping constraint, so the filter
starts intervening and the trace shows what it does instead of what was asked.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from stretchwbc.ik import solve
from stretchwbc.model import ARM, BASE_THETA, BASE_X, BASE_Y, LIFT, StretchModel
from stretchwbc.safety import SafetyFilter, integrate, tipping_barrier

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"

DT = 0.05
INK = "#1b1f24"
MUTED = "#8c959f"
ACCENT = "#0969da"
WARM = "#bc4c00"
GOOD = "#1a7f37"
BAD = "#cf222e"
RAW_GREY = "#dfe3e8"


def link_points(model: StretchModel, q: np.ndarray) -> dict:
    """The robot's skeleton in world coordinates, for drawing."""
    theta = q[BASE_THETA]
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    base = np.array([q[BASE_X], q[BASE_Y], 0.0])

    mast_foot = base + rotation @ np.array(
        [model.mast_offset_xy[0], model.mast_offset_xy[1], 0.0]
    )
    mast_top = mast_foot + np.array([0.0, 0.0, q[LIFT]])

    carriage = mast_top + rotation @ np.array([0.0, model.arm_retracted_y, 0.0])
    arm_end = carriage + rotation @ np.array([0.0, q[ARM], 0.0])

    tool = model.position(q)

    # Base footprint, roughly Stretch's wheelbase.
    half_w, half_l = 0.17, 0.16
    corners = np.array(
        [
            [-half_l, -half_w, 0.0],
            [half_l, -half_w, 0.0],
            [half_l, half_w, 0.0],
            [-half_l, half_w, 0.0],
            [-half_l, -half_w, 0.0],
        ]
    )
    footprint = np.array([base + rotation @ c for c in corners])

    return {
        "base": base,
        "footprint": footprint,
        "mast_foot": mast_foot,
        "mast_top": mast_top,
        "carriage": carriage,
        "arm_end": arm_end,
        "tool": tool,
    }


def run_episode(model: StretchModel, filt: SafetyFilter, target, q0, steps: int):
    """Closed-loop IK through the filter. Returns per-step state."""
    q = np.asarray(q0, dtype=float).copy()
    frames = []

    for _ in range(steps):
        result = solve("jacobian_transpose", model, target, q, max_iterations=25)
        desired = (result.q - q) / DT
        desired = np.clip(desired, -model.limits.velocity, model.limits.velocity)

        solution = filt.filter(q, desired)
        frames.append(
            {
                "q": q.copy(),
                "error": float(np.linalg.norm(target - model.position(q))),
                "intervened": solution.modified,
                "relaxed": solution.relaxed,
            }
        )
        q = integrate(model, q, solution.dq, DT)

    return frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    model = StretchModel()
    filt = SafetyFilter(model)
    barrier = tipping_barrier()

    start = (model.limits.lower + model.limits.upper) / 2.0
    start[BASE_X], start[BASE_Y], start[BASE_THETA] = 0.0, 0.0, 0.0
    start[LIFT], start[ARM] = 0.35, 0.05

    # Reachable. Then one that needs height and reach at once, which the
    # tipping constraint will not allow.
    easy = model.position(np.concatenate([[0.45, 0.35, 0.2], [0.55, 0.22], start[5:]]))
    hard = model.position(np.concatenate([[0.5, 0.9, 0.3], [1.05, 0.50], start[5:]]))

    frames = run_episode(model, filt, easy, start, 70)
    frames += run_episode(model, filt, hard, frames[-1]["q"], 90)

    # Every second step is drawn. The motion is smooth at 0.1 s per frame and
    # halving the frame count halves the encode time, which dominates.
    frames = frames[::2]

    fig = plt.figure(figsize=(10, 4.4), dpi=84)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_state = fig.add_subplot(2, 2, 2)
    ax_margin = fig.add_subplot(2, 2, 4)

    times = np.arange(len(frames)) * DT
    errors = np.array([f["error"] for f in frames])
    margins = np.array([barrier.margin(f["q"]) for f in frames])
    intervened = np.array([f["intervened"] for f in frames])

    def draw(index: int) -> None:
        frame = frames[index]
        q = frame["q"]
        pts = link_points(model, q)
        target = easy if index < 35 else hard

        ax3d.clear()
        ax3d.plot(*pts["footprint"].T, color=INK, lw=2.0)
        ax3d.plot(*np.array([pts["mast_foot"], pts["mast_top"]]).T, color=MUTED, lw=4.0)
        ax3d.plot(
            *np.array([pts["mast_top"], pts["carriage"], pts["arm_end"]]).T,
            color=ACCENT,
            lw=4.0,
        )
        ax3d.plot(
            *np.array([pts["arm_end"], pts["tool"]]).T,
            color=WARM if frame["intervened"] else GOOD,
            lw=3.0,
        )
        ax3d.scatter(*pts["tool"], s=45, color=INK, zorder=5)
        ax3d.scatter(*target, s=90, marker="*", color=BAD, zorder=5)

        ax3d.set_xlim(-0.6, 1.4)
        ax3d.set_ylim(-0.4, 1.6)
        ax3d.set_zlim(0.0, 1.4)
        ax3d.set_xlabel("x (m)", fontsize=8)
        ax3d.set_ylabel("y (m)", fontsize=8)
        ax3d.set_zlabel("z (m)", fontsize=8)
        ax3d.tick_params(labelsize=7)
        ax3d.view_init(elev=22, azim=-58)

        phase = "reaching a feasible target" if index < 35 else "target needs height AND reach"
        status = "  [filter intervening]" if frame["intervened"] else ""
        ax3d.set_title(f"{phase}{status}", fontsize=9.5, loc="left")

        for ax, series, label, colour in (
            (ax_state, errors, "distance to target (m)", ACCENT),
            (ax_margin, margins, "tipping margin", GOOD),
        ):
            ax.clear()
            ax.plot(times, series, color=RAW_GREY, lw=1.0)
            ax.plot(times[: index + 1], series[: index + 1], color=colour, lw=2.0)
            ax.axvline(times[index], color=MUTED, lw=0.8)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(times[0], times[-1])

        ax_margin.axhline(0.0, color=BAD, lw=1.0, ls=(0, (4, 3)))
        ax_margin.fill_between(times, -1, 0, color=BAD, alpha=0.07, lw=0)
        ax_margin.set_ylim(min(margins.min(), -0.05) - 0.02, margins.max() + 0.05)
        ax_margin.set_xlabel("time (s)", fontsize=8)

        if intervened[: index + 1].any():
            first = int(np.flatnonzero(intervened)[0])
            ax_state.axvspan(times[first], times[index], color=WARM, alpha=0.08, lw=0)

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=60)
    fig.tight_layout()

    path = OUT / "05_demo.gif"
    animation.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)

    print(f"Wrote {path}")
    print(f"  {len(frames)} frames, filter intervened on {int(intervened.sum())} of them")
    print(f"  final distance to target: {errors[-1] * 1000:.0f} mm")
    print(f"  tipping margin stayed above {margins.min():+.3f}")


if __name__ == "__main__":
    main()
