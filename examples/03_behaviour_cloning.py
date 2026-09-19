"""Cloning the IK controller, and the two things that go wrong.

Trains a small MLP on rollouts from the damped-least-squares controller, then
evaluates it three ways:

1. **Open loop** — action MSE on held-out expert states. The number that looks
   best and predicts least.
2. **Closed loop, unfiltered** — the policy driving the robot on its own,
   counting how often it leaves the joint limits.
3. **Closed loop, filtered** — the same policy through the QP safety filter.

The gap between 1 and 2 is compounding error. The gap between 2 and 3 is the
entire argument for keeping a filter in the command path of a learned
controller.

Requires the learning extra:  pip install "stretchwbc[learning]"
"""

from __future__ import annotations

import os

import numpy as np

from stretchwbc.model import StretchModel
from stretchwbc.policy import (
    BCPolicy,
    collect_demonstrations,
    flatten,
    rollout,
)
from stretchwbc.safety import SafetyFilter

N_TRAIN_EPISODES = int(os.environ.get("WBC_BC_EPISODES", "120"))
N_EVAL_EPISODES = int(os.environ.get("WBC_BC_EVAL", "50"))
EPOCHS = int(os.environ.get("WBC_BC_EPOCHS", "60"))
SEED = 20260920


def main() -> None:
    model = StretchModel()

    print(f"Collecting {N_TRAIN_EPISODES} expert rollouts...")
    train = collect_demonstrations(model, N_TRAIN_EPISODES, seed=SEED)
    held_out = collect_demonstrations(model, 30, seed=SEED + 1)

    observations, actions = flatten(train)
    test_observations, test_actions = flatten(held_out)

    reached = sum(d.reached for d in train)
    print(f"  {len(observations)} transitions, expert reached {reached}/{len(train)}\n")

    print(f"Training for {EPOCHS} epochs...")
    policy = BCPolicy(seed=SEED)
    history = policy.fit(observations, actions, epochs=EPOCHS, verbose=True)
    print(f"  final training loss {history[-1]:.5f}\n")

    # ---- 1. open loop -----------------------------------------------------
    predicted = policy.act(test_observations)
    per_joint = np.mean((predicted - test_actions) ** 2, axis=0)
    open_loop_mse = float(np.mean(per_joint))

    print("1. Open loop, on held-out EXPERT states")
    print(f"   action MSE            {open_loop_mse:.5f}")
    print("   This is the number a training script prints and it is the least")
    print("   informative of the three: it is measured on states the policy")
    print("   will never actually be in once it is driving.\n")

    # ---- 2 and 3. closed loop --------------------------------------------
    rng = np.random.default_rng(SEED + 2)
    episodes = []
    for _ in range(N_EVAL_EPISODES):
        q0 = model.limits.clamp(model.random_configuration(rng))
        episodes.append((model.nearby_target(q0, rng), q0))

    filt = SafetyFilter(model)
    summary = {}

    for label, safety in (("unfiltered", None), ("filtered", filt)):
        reached_count = 0
        violations = 0
        interventions = 0
        errors = []

        for target, q0 in episodes:
            result = rollout(model, policy, target, q0, safety_filter=safety, max_steps=40)
            reached_count += int(result.reached)
            violations += result.limit_violations
            interventions += result.filter_interventions
            errors.append(result.final_error_m)

        errors_m = np.array(errors)
        summary[label] = {
            "reached": reached_count,
            "within_50mm": int(np.sum(errors_m < 0.05)),
            "violations": violations,
            "interventions": interventions,
            "median_error_mm": float(np.median(errors)) * 1000.0,
        }

    header = f"{'':<30}{'unfiltered':>13}{'filtered':>12}"
    print("2 and 3. Closed loop, policy driving from its own states")
    print(header)
    print("-" * len(header))
    for label, key, fmt in (
        ("reached (5 mm tolerance)", "reached", "{:d}"),
        ("within 50 mm", "within_50mm", "{:d}"),
        ("joint-limit violations", "violations", "{:d}"),
        ("filter interventions", "interventions", "{:d}"),
        ("median final error (mm)", "median_error_mm", "{:.1f}"),
    ):
        print(
            f"{label:<30}{fmt.format(summary['unfiltered'][key]):>13}"
            f"{fmt.format(summary['filtered'][key]):>12}"
        )

    print()
    print(f"The expert finishes {reached}/{len(train)} of its own episodes. The clone")
    print(
        f"finishes {summary['unfiltered']['reached']}/{N_EVAL_EPISODES} at the same 5 mm "
        f"tolerance, while getting its median"
    )
    print(f"error down to {summary['unfiltered']['median_error_mm']:.0f} mm.")
    print()
    print("Read that as the clone learning the gross motion and not the endgame,")
    print("which is what compounding error looks like: fine near states the")
    print("expert visited, drifting once it is not. More epochs do not fix it --")
    print("it is a property of training on one state distribution and running on")
    print("another. DAgger is the standard fix and is not implemented here.")
    print()
    print("The violation row is the one that matters for deployment. The network")
    print("was never told joint limits exist, and nothing in its loss made")
    print("exceeding one different from any other error. The filter is what")
    print("makes a learned controller safe to run, not better training.")


if __name__ == "__main__":
    main()
