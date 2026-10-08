"""The real Stretch 3 geometry: agreement with MuJoCo, and the IK benchmark on it.

Examples 01-04 use StretchModel, a simplified stand-in. This one uses
Stretch3Chain, built from MuJoCo Menagerie's Stretch 3 model, and:

1. measures how closely its forward kinematics agree with MuJoCo's (skipped if
   the `mujoco` package or the Menagerie model is missing), and
2. reruns the three IK solvers on 1,000 reachable targets, with 95 %
   Clopper-Pearson intervals on each success rate.

Set WBC_N_TARGETS lower for a quick run.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

from stretchwbc.chain import Stretch3Chain
from stretchwbc.ik import SOLVERS, solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
N_TARGETS = int(os.environ.get("WBC_N_TARGETS", "1000"))
SEED = 20261007
TOLERANCE = 1e-3


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval: conservative, and honest at success rates near 100 %."""
    alpha = 1 - level
    low = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    high = (
        1.0
        if successes == trials
        else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return float(low), float(high)


def mujoco_agreement(chain: Stretch3Chain) -> None:
    try:
        import mujoco

        from test_chain_mujoco import XML, mujoco_tool_pose
    except ImportError:
        print("MuJoCo check skipped: `pip install mujoco` first.\n")
        return
    if not XML.exists():
        print(f"MuJoCo check skipped: no Menagerie model at {XML}.\n")
        return
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(SEED)
    errors = []
    for _ in range(1000):
        q = chain.random_configuration(rng)
        expected, _ = mujoco_tool_pose(mujoco, model, data, q)
        errors.append(np.linalg.norm(chain.position(q) - expected))
    errors = np.array(errors)
    print("Forward kinematics against MuJoCo, 1,000 random configurations")
    print(
        f"  grasp centre position error: median {np.median(errors):.1e} m, max {errors.max():.1e} m\n"
    )


def benchmark(chain: Stretch3Chain) -> None:
    rng = np.random.default_rng(SEED)
    targets = np.array([chain.random_reachable_target(rng) for _ in range(N_TARGETS)])
    seed_configuration = (chain.limits.lower + chain.limits.upper) / 2.0
    print(
        f"IK on the real geometry: {N_TARGETS} reachable targets, neutral seed, 1 mm tolerance\n"
    )
    header = f"{'solver':<22}{'solved':>10}{'95% interval':>18}{'median mm':>11}{'p95 mm':>9}{'ms':>7}"
    print(header)
    print("-" * len(header))
    for name in SOLVERS:
        start = time.perf_counter()
        outcomes = [
            solve(name, chain, t, seed_configuration, tolerance=TOLERANCE) for t in targets
        ]
        ms = (time.perf_counter() - start) / N_TARGETS * 1000
        solved = sum(o.success for o in outcomes)
        errors = np.array([o.error_m for o in outcomes]) * 1000
        low, high = clopper_pearson(solved, N_TARGETS)
        interval = f"{100 * low:.1f}-{100 * high:.1f} %"
        print(
            f"{name:<22}{solved:>6}/{N_TARGETS:<4}{interval:>17}"
            f"{np.median(errors):>11.3f}{np.percentile(errors, 95):>9.2f}{ms:>7.1f}"
        )


def main() -> None:
    chain = Stretch3Chain()
    mujoco_agreement(chain)
    benchmark(chain)


if __name__ == "__main__":
    main()
