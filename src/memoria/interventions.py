"""Interventions: controlled, seeded perturbations of an experience stream.

An intervention maps a :class:`~memoria.core.Dataset` to a new dataset. It never touches
a memory log: stored history is immutable, and what an experiment perturbs is the input
a system under study receives. Probes, the measuring instrument, are never changed.

Every random choice is ``unit_interval(seed, name, step.digest)``: it depends on the
step's content and the seed, not on its position or on call order, so an intervention
selects the same steps wherever they appear in the stream.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import ClassVar, Protocol

from memoria.core import Dataset, Experience, Scalar, Step, unit_interval
from memoria.formation import parse_statement


class Intervention(Protocol):
    name: ClassVar[str]

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        """Every parameter that determines the intervention, including its seed."""
        ...

    def apply(self, dataset: Dataset) -> Dataset: ...


def _derive(dataset: Dataset, name: str, steps: Iterable[Step]) -> Dataset:
    """A derived dataset: steps stably re-sorted into ingestion order, probes unchanged."""
    return Dataset(
        name=dataset.name,
        version=f"{dataset.version}+{name}",
        steps=tuple(sorted(steps, key=lambda s: s.recorded_at)),
        probes=dataset.probes,
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_rate_seed(rate: float, seed: int) -> None:
    _check(0 <= rate <= 1, "rate must be within [0, 1]")
    _check(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")


@dataclass(frozen=True)
class Drop:
    """Delete: remove each step with probability ``rate``. Models lost observations."""

    rate: float
    seed: int
    name: ClassVar[str] = "drop"

    def __post_init__(self) -> None:
        _check_rate_seed(self.rate, self.seed)

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("rate", self.rate), ("seed", self.seed))

    def apply(self, dataset: Dataset) -> Dataset:
        kept = (
            s for s in dataset.steps if unit_interval(self.seed, self.name, s.digest) >= self.rate
        )
        return _derive(dataset, self.name, kept)


@dataclass(frozen=True)
class Delay:
    """Delay: ingest each step with probability ``rate`` ``days`` later than planned.

    The experience (and when it occurred) is unchanged; only its record time moves.
    """

    rate: float
    seed: int
    days: float
    name: ClassVar[str] = "delay"

    def __post_init__(self) -> None:
        _check_rate_seed(self.rate, self.seed)
        _check(self.days > 0, "days must be positive")

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("days", self.days), ("rate", self.rate), ("seed", self.seed))

    def apply(self, dataset: Dataset) -> Dataset:
        shift = timedelta(days=self.days)
        steps = (
            Step(experience=s.experience, recorded_at=s.recorded_at + shift)
            if unit_interval(self.seed, self.name, s.digest) < self.rate
            else s
            for s in dataset.steps
        )
        return _derive(dataset, self.name, steps)


@dataclass(frozen=True)
class Reorder:
    """Reorder: jitter ingestion order within ``window_days``.

    Steps are ordered by ``recorded_at + U * window`` (U seeded per step), then each
    step's record time is raised to at least its predecessor's. No step is ever recorded
    earlier than originally planned, so no experience is ingested before it occurred.
    """

    seed: int
    window_days: float
    name: ClassVar[str] = "reorder"

    def __post_init__(self) -> None:
        _check_rate_seed(0, self.seed)
        _check(self.window_days > 0, "window_days must be positive")

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("seed", self.seed), ("window_days", self.window_days))

    def apply(self, dataset: Dataset) -> Dataset:
        def jittered(indexed: tuple[int, Step]) -> tuple[datetime, int]:
            i, s = indexed
            u = unit_interval(self.seed, self.name, s.digest)
            return s.recorded_at + timedelta(days=u * self.window_days), i

        steps: list[Step] = []
        for _, s in sorted(enumerate(dataset.steps), key=jittered):
            t = max(s.recorded_at, steps[-1].recorded_at) if steps else s.recorded_at
            steps.append(Step(experience=s.experience, recorded_at=t))
        return _derive(dataset, self.name, steps)


@dataclass(frozen=True)
class Contaminate:
    """Contaminate: for each selected ``set``/``correct`` statement, inject a false copy.

    The copy asserts ``<value> (contaminated)`` for the same key, occurring and recorded
    ``delay_days`` after the original, from source ``<source>:<original experience digest>``
    so every contaminant is traceable to what it imitates. With ``delay_days == 0`` it
    contradicts the original at the same instant; with a positive delay it supersedes it.
    Non-statement steps are never selected.
    """

    rate: float
    seed: int
    delay_days: float = 0
    source: str = "contaminant"
    name: ClassVar[str] = "contaminate"

    def __post_init__(self) -> None:
        _check_rate_seed(self.rate, self.seed)
        _check(self.delay_days >= 0, "delay_days must be non-negative")
        _check(bool(self.source) and ":" not in self.source, "source must be non-empty, no ':'")

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (
            ("delay_days", self.delay_days),
            ("rate", self.rate),
            ("seed", self.seed),
            ("source", self.source),
        )

    def apply(self, dataset: Dataset) -> Dataset:
        shift = timedelta(days=self.delay_days)
        added = []
        for s in dataset.steps:
            statement = parse_statement(s.experience.content)
            if statement is None or statement.value is None:
                continue
            if unit_interval(self.seed, self.name, s.digest) >= self.rate:
                continue
            fake = Experience(
                source=f"{self.source}:{s.experience.digest}",
                content=f"{statement.verb} {statement.key} = {statement.value} (contaminated)",
                occurred_at=s.experience.occurred_at + shift,
            )
            added.append(Step(experience=fake, recorded_at=s.recorded_at + shift))
        return _derive(dataset, self.name, [*dataset.steps, *added])


@dataclass(frozen=True)
class Inject:
    """Inject: merge another dataset's steps into the stream (its probes are ignored).

    For hand-built contradictions or contamination with exactly specified content and
    timing. On equal record times, original steps are ingested first.
    """

    source: Dataset = field(repr=False)
    name: ClassVar[str] = "inject"

    @property
    def params(self) -> tuple[tuple[str, Scalar], ...]:
        return (("dataset", self.source.digest),)

    def apply(self, dataset: Dataset) -> Dataset:
        return _derive(dataset, self.name, [*dataset.steps, *self.source.steps])
