"""Regenerate the figures in docs/figures/.

    python examples/make_figures.py

Same seeds as the benchmarks, so the pictures and the numbers describe the
same runs.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stretchwbc.ik import SOLVERS, solve
from stretchwbc.model import ARM, LIFT, N_JOINTS, StretchModel
from stretchwbc.safety import (
    SafetyFilter,
    integrate,
    solve_box_soft_qp,
    tipping_barrier,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"
SEED = 20260920

INK = "#1b1f24"
MUTED = "#8c959f"
RAW = "#bcc4cc"
ACCENT = "#0969da"
WARM = "#bc4c00"
GOOD = "#1a7f37"
BAD = "#cf222e"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    }
)


# --------------------------------------------------------------- figure 1


def figure_barrier_ramp():
    """A barrier decelerates into a limit; a clamp arrives at full speed."""
    model = StretchModel()
    filt = SafetyFilter(model)

    dt = 0.05
    steps = 90
    home = (model.limits.lower + model.limits.upper) / 2.0
    ceiling = model.limits.upper[LIFT]

    desired = np.zeros(N_JOINTS)
    desired[LIFT] = model.limits.velocity[LIFT]

    # Barrier-filtered
    q = home.copy()
    filtered_pos, filtered_vel = [], []
    for _ in range(steps):
        dq = filt.filter(q, desired).dq
        filtered_pos.append(q[LIFT])
        filtered_vel.append(dq[LIFT])
        q = integrate(model, q, dq, dt)

    # Naive: run at full speed, then clamp the position on arrival
    q = home.copy()
    clamped_pos, clamped_vel = [], []
    for _ in range(steps):
        dq = desired.copy()
        clamped_pos.append(q[LIFT])
        clamped_vel.append(dq[LIFT] if q[LIFT] < ceiling else 0.0)
        q = model.limits.clamp(integrate(model, q, dq, dt))

    t = np.arange(steps) * dt

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(8.6, 4.8), sharex=True)

    ax_pos.axhline(ceiling, color=BAD, lw=1.2, ls=(0, (4, 3)))
    ax_pos.text(t[-1], ceiling + 0.005, "joint limit", ha="right", fontsize=8, color=BAD)
    ax_pos.plot(t, clamped_pos, color=WARM, lw=1.8, label="clamp on arrival")
    ax_pos.plot(t, filtered_pos, color=ACCENT, lw=2.0, label="barrier")
    ax_pos.set_ylabel("lift position (m)")
    ax_pos.legend(ncol=2, loc="lower right")

    ax_vel.plot(t, clamped_vel, color=WARM, lw=1.8)
    ax_vel.plot(t, filtered_vel, color=ACCENT, lw=2.0)
    ax_vel.set_ylabel("commanded (m/s)")
    ax_vel.set_xlabel("time (s)")

    ax_vel.annotate(
        "full speed until the\ninstant it stops",
        xy=(3.12, 0.075),
        xytext=(1.25, 0.095),
        fontsize=8.5,
        color=WARM,
        arrowprops={"arrowstyle": "->", "color": WARM, "lw": 1.0},
    )
    ax_vel.annotate(
        "decelerates as the\nmargin shrinks",
        xy=(3.45, 0.035),
        xytext=(3.55, 0.105),
        fontsize=8.5,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )

    fig.suptitle(
        "Barrier versus clamp, approaching the top of the lift",
        y=0.99,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "01_barrier_ramp.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2


def figure_tipping_coupling():
    """The constraint no per-joint bound can express."""
    model = StretchModel()
    filt = SafetyFilter(model)
    barrier = tipping_barrier()

    lift = np.linspace(model.limits.lower[LIFT], model.limits.upper[LIFT], 240)
    arm = np.linspace(model.limits.lower[ARM], model.limits.upper[ARM], 240)
    grid_lift, grid_arm = np.meshgrid(lift, arm)

    margin = np.zeros_like(grid_lift)
    probe = (model.limits.lower + model.limits.upper) / 2.0
    for i in range(grid_lift.shape[0]):
        for j in range(grid_lift.shape[1]):
            q = probe.copy()
            q[LIFT], q[ARM] = grid_lift[i, j], grid_arm[i, j]
            margin[i, j] = barrier.margin(q)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    ax.contourf(grid_lift, grid_arm, margin >= 0, levels=[0.5, 1.5], colors=[GOOD], alpha=0.12)
    ax.contour(grid_lift, grid_arm, margin, levels=[0.0], colors=[GOOD], linewidths=1.6)

    # A controller asking to raise and extend at once, from inside the set.
    q = probe.copy()
    q[LIFT], q[ARM] = 0.30, 0.20
    desired = np.zeros(N_JOINTS)
    desired[LIFT], desired[ARM] = 0.12, 0.12

    path = []
    for _ in range(120):
        path.append((q[LIFT], q[ARM]))
        q = integrate(model, q, filt.filter(q, desired).dq, 0.05)
    path = np.array(path)

    ax.plot(path[:, 0], path[:, 1], color=ACCENT, lw=2.0, label="filtered trajectory")
    ax.scatter(*path[0], s=50, color=ACCENT, zorder=4)
    ax.annotate(
        "asked to raise and extend together;\nallowed to, until the boundary",
        xy=(path[-1, 0], path[-1, 1]),
        xytext=(0.52, 0.16),
        fontsize=8.5,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )

    ax.text(0.22, 0.47, "unsafe", fontsize=10, color=BAD, alpha=0.8)
    ax.text(0.85, 0.09, "safe", fontsize=10, color=GOOD, alpha=0.9)

    ax.set_xlabel("lift height (m)")
    ax.set_ylabel("arm extension (m)")
    ax.set_title("Height costs reach: the one constraint that couples two joints", loc="left")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "02_tipping_coupling.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3


def figure_soft_versus_box():
    """The bug: joint limits in the soft set can be pushed through."""
    model = StretchModel()
    filt = SafetyFilter(model)

    def all_soft_filter(q, desired):
        """The original design: every barrier in the penalty, box = motors only."""
        rows, bounds = [], []
        for constraint in filt.barriers:
            row, bound = constraint.as_row(q)
            rows.append(row)
            bounds.append(bound)

        return solve_box_soft_qp(
            desired,
            -model.limits.velocity,
            model.limits.velocity,
            np.array(rows),
            np.array(bounds),
        ).dq

    dt = 0.05
    episodes, steps = 40, 60

    series = {"joint limits in the penalty": [], "joint limits in the box": []}

    for label, step_fn in (
        ("joint limits in the penalty", all_soft_filter),
        ("joint limits in the box", lambda q, d: filt.filter(q, d).dq),
    ):
        local = np.random.default_rng(SEED)
        cumulative = np.zeros(steps)

        for _ in range(episodes):
            q = model.limits.clamp(model.random_configuration(local))
            push = local.uniform(-3.0, 3.0, size=N_JOINTS)  # persistent direction

            for k in range(steps):
                q = integrate(model, q, step_fn(q, push), dt)
                if not model.limits.within(q, tolerance=1e-6):
                    cumulative[k] += 1

        series[label] = np.cumsum(cumulative)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))

    t = np.arange(steps) * dt
    ax.plot(
        t,
        series["joint limits in the penalty"],
        color=BAD,
        lw=2.0,
        label="joint limits in the penalty (first design)",
    )
    ax.plot(
        t,
        series["joint limits in the box"],
        color=ACCENT,
        lw=2.4,
        label="joint limits in the box (current)",
    )

    final_bad = series["joint limits in the penalty"][-1]
    ax.annotate(
        f"{final_bad:.0f} violations",
        xy=(t[-1], final_bad),
        xytext=(t[-1] - 0.9, final_bad * 0.72),
        fontsize=9,
        color=BAD,
        arrowprops={"arrowstyle": "->", "color": BAD, "lw": 1.0},
    )
    ax.annotate(
        "0 violations",
        xy=(t[-1], 0),
        xytext=(t[-1] - 0.9, final_bad * 0.16),
        fontsize=9,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
    )

    ax.set_xlabel("time within each episode (s)")
    ax.set_ylabel("cumulative joint-limit violations")
    ax.set_title(f"{episodes} episodes of persistent one-directional pressure", loc="left")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "03_soft_versus_box.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4


def figure_solver_convergence():
    """Success rate against iteration budget.

    Plotting median *error* against budget was the first attempt and it was a
    bad figure: with a fixed step cap every solver sits at roughly the starting
    distance until it has had enough iterations to travel, then drops several
    orders of magnitude at once, and on a log axis the exact zeros fall off the
    bottom. Success rate is bounded, monotone, and actually separates them.
    """
    model = StretchModel()
    rng = np.random.default_rng(SEED)

    targets = [model.random_reachable_target(rng) for _ in range(120)]
    seed_q = (model.limits.lower + model.limits.upper) / 2.0
    caps = [8, 16, 24, 32, 48, 64, 96, 128, 192]
    tolerance = 1e-3

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colours = {
        "jacobian_transpose": ACCENT,
        "damped_least_squares": WARM,
        "selectively_damped": GOOD,
    }
    styles = {
        "jacobian_transpose": "-",
        "damped_least_squares": "--",
        "selectively_damped": ":",
    }

    for name in SOLVERS:
        rates = []
        for cap in caps:
            solved = sum(
                solve(name, model, t, seed_q, max_iterations=cap, tolerance=tolerance).success
                for t in targets
            )
            rates.append(100.0 * solved / len(targets))

        ax.plot(
            caps,
            rates,
            marker="o",
            ms=4,
            lw=2.0,
            ls=styles[name],
            color=colours[name],
            label=name.replace("_", " "),
        )

    ax.set_xlabel("iteration budget")
    ax.set_ylabel(f"targets solved to {tolerance * 1000:.0f} mm (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(
        f"{len(targets)} reachable targets, same neutral seed for all three", loc="left"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.15, lw=0.6)

    fig.tight_layout()
    fig.savefig(OUT / "04_solver_convergence.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_barrier_ramp()
    figure_tipping_coupling()
    figure_soft_versus_box()
    figure_solver_convergence()
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
