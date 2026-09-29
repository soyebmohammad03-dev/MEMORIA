"""Source trust and evidence independence (Phases 11-12).

Source reliability is an explicit, declared variable, never ground truth:

- a :class:`SourceModel` declares, per source class, a Beta(alpha, beta) prior on the
  probability that one of its reports is right, and which sources *copy* which. Classes it
  does not declare get the uninformative Beta(1, 1) (cold start): no source is silently
  authoritative;
- a :class:`TrustLedger` adds counts learned from how each source's reports fare against the
  beliefs the system itself formed. That is circular by construction (the system grades
  sources with its own conclusions), so every learned snapshot is labelled ``circular``.
  No external feedback exists before Phase 13.

Independence is a property of provenance, not of counts: reports from sources that share a
root (a source and everything derived from it through declared copy links) are one piece of
evidence, however often they are repeated. :func:`independence` reports the naive count
next to the independent count.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Self

from pydantic import Field, model_validator

from memoria.core import Digest, Record, quantize


def source_class(source: str) -> str:
    """The class of ``<class>:<id>``; a source without a structured id has class ``""``."""
    cls, sep, ident = source.partition(":")
    return cls if sep and cls and ident else ""


class SourceClass(Record):
    """A declared Beta prior on the reliability of a source class."""

    name: str = Field(min_length=1)
    alpha: float = Field(gt=0, allow_inf_nan=False)
    beta: float = Field(gt=0, allow_inf_nan=False)


class SourceModel(Record):
    """What is declared about sources: class priors and copy links (derivation ancestry).

    ``copies`` lists (copy, origin) pairs: the copy reports what the origin reported. The
    chain length is the provenance depth of a report; the chain's end is its root.
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    classes: tuple[SourceClass, ...] = ()
    copies: tuple[tuple[str, str], ...] = ()
    default_alpha: float = Field(default=1.0, gt=0)  # uninformative unless declared otherwise
    default_beta: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        names = [c.name for c in self.classes]
        if names != sorted(set(names)):
            raise ValueError("source classes must be unique and sorted")
        if list(self.copies) != sorted(set(self.copies)):
            raise ValueError("copies must be unique and sorted")
        origin = dict(self.copies)
        if len(origin) != len(self.copies):
            raise ValueError("a copy has exactly one origin")
        for start in origin:
            seen, node = {start}, start
            while node in origin:
                node = origin[node]
                if node in seen:
                    raise ValueError(f"copy links form a cycle at {node}")
                seen.add(node)
        return self

    def prior(self, source: str) -> tuple[float, float]:
        """The Beta prior of the source's class (uninformative when undeclared)."""
        cls = source_class(source)
        for c in self.classes:
            if c.name == cls:
                return c.alpha, c.beta
        return self.default_alpha, self.default_beta

    def root(self, source: str) -> str:
        """The source at the end of the copy chain: the identity of the evidence."""
        origin = dict(self.copies)
        while source in origin:
            source = origin[source]
        return source

    def depth(self, source: str) -> int:
        """Provenance depth: 1 for an original report, +1 per copy link."""
        origin = dict(self.copies)
        n = 1
        while source in origin:
            source = origin[source]
            n += 1
        return n


def beta_mean(alpha: float, beta: float) -> float:
    return quantize(alpha / (alpha + beta))


def beta_variance(alpha: float, beta: float) -> float:
    n = alpha + beta
    return quantize(alpha * beta / (n * n * (n + 1)))


class TrustRecord(Record):
    """One source's trust at one moment, with everything it was computed from."""

    source: str
    klass: str
    reports: int = Field(ge=0)
    support: float = Field(ge=0)  # reports found consistent with the system's beliefs
    contradict: float = Field(ge=0)  # reports found inconsistent with them
    alpha: float
    beta: float
    mean: float
    variance: float
    circular: bool  # learned from the system's own beliefs (always true if counts are used)


class TrustLedger:
    """Learned source trust: counts credited per report and revisable.

    A report is credited +1 (consistent with the belief it lands in), -1 (inconsistent:
    it sits in a rejected or corrected belief) or 0 (undecided). Credits are set per report,
    so a later resolution moves the credit of the reports it decides; counts are sums of the
    current credits. Deterministic in the order credits are set.
    """

    def __init__(self, model: SourceModel) -> None:
        self.model = model
        self._credit: dict[str, tuple[str, int]] = {}  # report -> (source, credit)
        self._support: dict[str, int] = {}
        self._contra: dict[str, int] = {}
        self._reports: dict[str, int] = {}

    def register(self, report: str, source: str) -> None:
        if report not in self._credit:
            self._credit[report] = (source, 0)
            self._reports[source] = self._reports.get(source, 0) + 1

    def credit(self, report: str, value: int) -> None:
        source, old = self._credit[report]
        if old == value:
            return
        if old > 0:
            self._support[source] -= 1
        elif old < 0:
            self._contra[source] -= 1
        if value > 0:
            self._support[source] = self._support.get(source, 0) + 1
        elif value < 0:
            self._contra[source] = self._contra.get(source, 0) + 1
        self._credit[report] = (source, value)

    def forget(self, report: str) -> None:
        """Withdrawn evidence stops counting for or against its source."""
        if report in self._credit:
            self.credit(report, 0)
            source, _ = self._credit.pop(report)
            self._reports[source] -= 1

    def record(self, source: str, learned: bool = True) -> TrustRecord:
        a0, b0 = self.model.prior(source)
        s = self._support.get(source, 0) if learned else 0
        c = self._contra.get(source, 0) if learned else 0
        a, b = a0 + s, b0 + c
        return TrustRecord(
            source=source,
            klass=source_class(source),
            reports=self._reports.get(source, 0),
            support=s,
            contradict=c,
            alpha=a,
            beta=b,
            mean=beta_mean(a, b),
            variance=beta_variance(a, b),
            circular=learned and (s + c) > 0,
        )

    def mean(self, source: str, learned: bool = True) -> float:
        return self.record(source, learned).mean

    def snapshot(self) -> tuple[TrustRecord, ...]:
        return tuple(self.record(s) for s in sorted(self._reports))

    def state(self) -> tuple[tuple[str, int, int], ...]:
        """Counts only, for state fingerprints."""
        return tuple(
            (s, self._support.get(s, 0), self._contra.get(s, 0)) for s in sorted(self._reports)
        )


class Independence(Record):
    """Naive evidence count next to the independent count and what collapsed."""

    naive: int = Field(ge=0)
    independent: int = Field(ge=0)
    duplicates: int = Field(ge=0)  # naive - independent
    roots: tuple[str, ...]
    max_depth: int = Field(ge=0)


def independence(sources: Iterable[str], model: SourceModel) -> Independence:
    """Count reports and count distinct roots. Repeats from one root, and copies of an
    origin, are one independent piece of evidence."""
    listed = list(sources)
    roots = sorted({model.root(s) for s in listed})
    return Independence(
        naive=len(listed),
        independent=len(roots),
        duplicates=len(listed) - len(roots),
        roots=tuple(roots),
        max_depth=max((model.depth(s) for s in listed), default=0),
    )


def trust_error(
    learned: Mapping[str, float], truth: Mapping[str, float]
) -> tuple[float | None, float | None]:
    """(mean absolute error, Spearman rank correlation) between learned trust and the true
    reliability the world hid from the system. Analysis only: never an input."""
    common = sorted(set(learned) & set(truth))
    if not common:
        return None, None
    mae = quantize(sum(abs(learned[s] - truth[s]) for s in common) / len(common))
    return mae, spearman([learned[s] for s in common], [truth[s] for s in common])


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks; ``None`` when undefined."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return quantize(
        sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sxx * syy) ** 0.5
    )


class SourceReport(Record):
    """Descriptive statistics of a source model over one evidence set: analysis, not input."""

    model: Digest
    sources: int
    classes: tuple[tuple[str, int], ...]
    copy_links: int
    max_depth: int
