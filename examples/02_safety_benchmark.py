"""What the safety filter is worth, measured against running without it.

A hostile controller — large random velocities, changing every step — is
integrated forward from random starts, with and without the filter in the
command path. Nothing clamps the configuration afterwards, so a joint that
leaves its limits stays out and is counted.

The hostile controller is the point. A well-behaved controller rarely asks for
anything unsafe, so testing against one measures almost nothing. What a safety
filter is for is the controller that is broken, mistuned, or learned from data
and has no idea the limits exist.
"""

from __future__ import annotations

import os

import numpy as np

from stretchwbc.model import N_JOINTS, StretchModel
from stretchwbc.safety import SafetyFilter, integrate, tipping_barrier

N_EPISODES = int(os.environ.get("WBC_N_EPISODES", "200"))
STEPS = 60
DT = 0.05
SEED = 20260920


def run(model: StretchModel, use_filter: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    filt = SafetyFilter(model) if use_filter else None
    barrier = tipping_barrier()

    joint_violations = 0
    tipping_violations = 0
    worst_excursion = 0.0
    commanded_over_speed = 0
    steps = 0

    for _ in range(N_EPISODES):
        q = model.limits.clamp(model.random_configuration(rng))

        for _ in range(STEPS):
            desired = rng.uniform(-3.0, 3.0, size=N_JOINTS)
            dq = filt.filter(q, desired).dq if filt else desired

            if np.any(np.abs(dq) > model.limits.velocity + 1e-9):
                commanded_over_speed += 1

            q = integrate(model, q, dq, DT)
            steps += 1

            excursion = float(np.max(model.limits.violations(q)))
            if excursion > 1e-6:
                joint_violations += 1
                worst_excursion = max(worst_excursion, excursion)

            if barrier.margin(q) < -1e-6:
                tipping_violations += 1

    return {
        "steps": steps,
        "joint_violations": joint_violations,
        "tipping_violations": tipping_violations,
        "worst_excursion_m": worst_excursion,
        "commanded_over_speed": commanded_over_speed,
    }


def main() -> None:
    model = StretchModel()

    unfiltered = run(model, use_filter=False, seed=SEED)
    filtered = run(model, use_filter=True, seed=SEED)

    print(f"{N_EPISODES} episodes x {STEPS} steps at {DT * 1000:.0f} ms, hostile controller")
    print(f"{unfiltered['steps']} commands per condition\n")

    header = f"{'':<34}{'unfiltered':>13}{'filtered':>12}"
    print(header)
    print("-" * len(header))

    rows = [
        ("joint-limit violations", "joint_violations", "{:d}"),
        ("tipping violations", "tipping_violations", "{:d}"),
        ("commands over velocity limit", "commanded_over_speed", "{:d}"),
        ("worst excursion past a limit (m)", "worst_excursion_m", "{:.3f}"),
    ]

    for label, key, fmt in rows:
        print(f"{label:<34}{fmt.format(unfiltered[key]):>13}{fmt.format(filtered[key]):>12}")

    print()
    if filtered["joint_violations"] == 0 and filtered["commanded_over_speed"] == 0:
        print("Joint limits and velocity limits: zero violations, from a controller")
        print("that was actively trying to break them.")
    else:
        print("Filtered violations are NOT zero. That is a bug, not a tuning problem.")

    print()
    print("The tipping count is the honest part of this table. It does not reach")
    print("zero and it cannot: roughly a fifth of uniformly sampled starting")
    print("configurations are already outside the tipping-safe set, and no")
    print("velocity inside the motors' limits gets them back within one step.")
    print("The filter relaxes that constraint by the smallest amount that admits")
    print("an executable command, drives the margin back up, and reports how much")
    print("it had to give up. A filter claiming zero here would be one that had")
    print("quietly redefined the constraint.")


if __name__ == "__main__":
    main()
