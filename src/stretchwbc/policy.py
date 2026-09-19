"""Behaviour cloning: learning the reaching controller from demonstrations.

An IK solver already reaches targets, so cloning it is not useful in itself.
What it is useful for is the thing that is hard to get from a solver — a policy
that runs in constant time per step, with no iteration and no convergence
question — and as an honest test bed for the two failure modes every
imitation-learned controller has.

**Compounding error.** The policy is trained on states the expert visited. At
run time it visits its own states, and a small action error moves it slightly
off the expert's distribution, where its next action is slightly worse. The
error compounds along the rollout. This is why the open-loop action MSE reported
during training and the closed-loop success rate are different questions, and
why this module reports both rather than the flattering one.

**No notion of constraints.** A network trained on velocity targets has learned
a correlation, not a rule. Nothing stops it emitting a velocity that drives a
joint past its limit or tips the robot, and nothing in its loss ever told it
those were different from any other mistake. That is not a defect to train
away — it is the reason the QP filter in :mod:`stretchwbc.safety` sits between
this policy and the robot.

Requires the ``learning`` extra: ``pip install "stretchwbc[learning]"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ik import solve
from .model import N_JOINTS, StretchModel

OBSERVATION_DIM = N_JOINTS + 3 + 3  # configuration, target, target minus current position
ACTION_DIM = N_JOINTS


def observation(model: StretchModel, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build the policy's input.

    The error vector ``target - position(q)`` is included even though it is a
    deterministic function of the other two. It is the quantity the controller
    actually acts on, and making the network rediscover it from a configuration
    and a target wastes capacity on re-deriving forward kinematics. Handing it
    over is a modelling decision, not a shortcut — the network keeps the raw
    inputs too, so nothing is hidden from it.
    """
    q = np.asarray(q, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.concatenate([q, target, target - model.position(q)])


@dataclass
class Demonstration:
    """One expert rollout."""

    observations: np.ndarray  # (T, OBSERVATION_DIM)
    actions: np.ndarray  # (T, ACTION_DIM), joint velocities
    target: np.ndarray
    reached: bool


def collect_demonstrations(
    model: StretchModel,
    n_episodes: int,
    *,
    method: str = "damped_least_squares",
    seed: int = 0,
    dt: float = 0.05,
    max_steps: int = 40,
    tolerance: float = 5e-3,
    expert_iterations: int = 12,
) -> list[Demonstration]:
    """Generate expert rollouts by running an IK solver as a controller.

    Each step solves IK from the current configuration, takes the difference as
    a velocity, and moves one ``dt`` along it. That makes the expert a genuine
    closed-loop controller rather than a one-shot plan, so the demonstrations
    contain recovery behaviour: states slightly off the ideal path, with the
    action that corrects them. A policy cloned from one-shot plans never sees
    those and falls apart the first time it drifts.
    """
    rng = np.random.default_rng(seed)
    demonstrations: list[Demonstration] = []

    for _ in range(n_episodes):
        q = model.random_configuration(rng)
        target = model.nearby_target(q, rng)

        observations, actions = [], []
        reached = False

        for _ in range(max_steps):
            error = float(np.linalg.norm(target - model.position(q)))
            if error < tolerance:
                reached = True
                break

            # A handful of iterations is enough. The expert only needs a good
            # velocity *direction* for one dt step, not a converged pose, and
            # solving to convergence at every step makes collection an order of
            # magnitude slower for a direction that barely changes.
            result = solve(method, model, target, q, max_iterations=expert_iterations)
            dq = (result.q - q) / dt

            speed_cap = model.limits.velocity
            dq = np.clip(dq, -speed_cap, speed_cap)

            observations.append(observation(model, q, target))
            actions.append(dq)

            q = model.limits.clamp(q + dq * dt)

        if observations:
            demonstrations.append(
                Demonstration(
                    observations=np.array(observations),
                    actions=np.array(actions),
                    target=target,
                    reached=reached,
                )
            )

    return demonstrations


def flatten(demonstrations: list[Demonstration]) -> tuple[np.ndarray, np.ndarray]:
    """Stack demonstrations into one (observations, actions) pair."""
    if not demonstrations:
        raise ValueError("no demonstrations to flatten")
    return (
        np.vstack([d.observations for d in demonstrations]),
        np.vstack([d.actions for d in demonstrations]),
    )


@dataclass
class BCPolicy:
    """An MLP mapping observations to joint velocities.

    Args:
        hidden: widths of the hidden layers.
        seed: torch seed, so a training run is reproducible.
    """

    hidden: tuple[int, ...] = (256, 256)
    seed: int = 0

    _net: object | None = field(default=None, repr=False)
    _obs_mean: np.ndarray | None = field(default=None, repr=False)
    _obs_std: np.ndarray | None = field(default=None, repr=False)
    _act_scale: np.ndarray | None = field(default=None, repr=False)

    @property
    def trained(self) -> bool:
        return self._net is not None

    def _build(self):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)

        layers: list[nn.Module] = []
        width = OBSERVATION_DIM
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, ACTION_DIM))
        return nn.Sequential(*layers)

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = False,
    ) -> list[float]:
        """Train on (observation, action) pairs. Returns the loss per epoch."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                'BCPolicy needs the learning extra: pip install "stretchwbc[learning]"'
            ) from exc

        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if observations.shape[1] != OBSERVATION_DIM:
            raise ValueError(f"observations must have {OBSERVATION_DIM} columns")
        if actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must have {ACTION_DIM} columns")

        self._obs_mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        self._obs_std = np.where(std < 1e-6, 1.0, std)

        # Actions are normalised by each joint's own velocity limit rather than
        # by the data's spread. The base and the wrist differ by an order of
        # magnitude in rad/s, and an unnormalised MSE would spend its whole
        # budget on the wrist while ignoring the base entirely.
        self._act_scale = np.maximum(np.abs(actions).max(axis=0), 1e-6)

        x = torch.from_numpy((observations - self._obs_mean) / self._obs_std)
        y = torch.from_numpy(actions / self._act_scale)

        self._net = self._build()
        optimiser = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        generator = torch.Generator().manual_seed(self.seed)
        history = []

        for epoch in range(epochs):
            order = torch.randperm(len(x), generator=generator)
            total, seen = 0.0, 0

            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self._net(x[batch]), y[batch])
                loss.backward()
                optimiser.step()

                total += float(loss.detach()) * len(batch)
                seen += len(batch)

            history.append(total / seen)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1:3d}  loss {history[-1]:.5f}")

        return history

    def act(self, observations: np.ndarray) -> np.ndarray:
        """Predict joint velocities. Accepts one observation or a batch."""
        if not self.trained:
            raise RuntimeError("policy is not trained; call fit() first")

        import torch

        single = np.asarray(observations).ndim == 1
        batch = np.atleast_2d(np.asarray(observations, dtype=np.float32))

        with torch.no_grad():
            normalised = torch.from_numpy((batch - self._obs_mean) / self._obs_std)
            predicted = self._net(normalised).numpy() * self._act_scale

        return predicted[0] if single else predicted


@dataclass(frozen=True)
class RolloutResult:
    """Closed-loop outcome for one episode."""

    reached: bool
    final_error_m: float
    steps: int
    limit_violations: int
    filter_interventions: int


def rollout(
    model: StretchModel,
    policy: BCPolicy,
    target: np.ndarray,
    q0: np.ndarray,
    *,
    safety_filter=None,
    dt: float = 0.05,
    max_steps: int = 60,
    tolerance: float = 5e-3,
) -> RolloutResult:
    """Run the policy closed-loop, optionally through the safety filter.

    Counts limit violations along the way. Without the filter that count is the
    honest measure of what a learned controller does unsupervised; with it, it
    should be zero, and the benchmark exists to check that rather than assume
    it.
    """
    q = np.asarray(q0, dtype=float).copy()
    target = np.asarray(target, dtype=float)

    violations = 0
    interventions = 0
    reached = False
    step = 0

    for step in range(1, max_steps + 1):  # noqa: B007 - reported below
        if float(np.linalg.norm(target - model.position(q))) < tolerance:
            reached = True
            break

        dq = policy.act(observation(model, q, target))

        if safety_filter is not None:
            solution = safety_filter.filter(q, dq)
            if solution.modified:
                interventions += 1
            dq = solution.dq

        q = q + dq * dt

        if not model.limits.within(q, tolerance=1e-6):
            violations += 1

    return RolloutResult(
        reached=reached,
        final_error_m=float(np.linalg.norm(target - model.position(q))),
        steps=step,
        limit_violations=violations,
        filter_interventions=interventions,
    )
