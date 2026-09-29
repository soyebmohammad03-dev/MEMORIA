"""Generated worlds for adaptive-memory experiments, with hidden truth and gold labels.

A world has, per key (``<entity>.<attribute>``), a hidden sequence of true values over days.
Sources (declared classes with hidden reliabilities) report on it as *text*: a fraction is the
statement language, the rest is free text with names, aliases, typos, hedges, pronouns and
retractions ("Ana no longer lives in Paris."). Every report carries gold labels (entity,
attribute, value, kind, and whether a careful reader could extract it) used only to
*measure* extraction and identity, never by a system. Users query keys over time; how many
queries a key gets (skew), how fast the world changes, how noisy the reports are and whether an
adversary floods stale values and pumps access are world parameters, not tuned to any policy.

Every choice is ``unit_interval(seed, ...)`` (I21). Values are lower-case; times are days.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from memoria.belief_worlds import CITIES, EMPLOYERS
from memoria.core import Digest, Record, quantize, unit_interval
from memoria.identity import Registry, make_registry

WORLD_NAMES = ("stable", "changing", "repetitive", "noisy", "adversarial")
HORIZON = 360.0
AUDIT_EVERY = 10.0
CHECKPOINTS = (30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 360.0)
CLASSES = ("clinic", "email", "chat", "forum")
DECLARED = (("chat", 0.7), ("clinic", 0.95), ("email", 0.85), ("forum", 0.4))
BASE_ENTITIES = (
    ("ana", "Ana"),
    ("ben", "Ben"),
    ("chen", "Chen"),
    ("dara", "Dara"),
    ("eli", "Eli"),
    ("farid", "Farid"),
)
COLLIDING = (("anna", "Anna"), ("benn", "Benn"))
ALIASES = {"ana": "Annie", "anna": "Annabel", "ben": "Benji", "chen": "Chenny"}
AMBIGUOUS_ALIAS = "Ann"  # registered for both ana and anna

Form = Literal[
    "structured",
    "free",
    "hedged",
    "pronoun",
    "alias",
    "ambiguous",
    "typo",
    "negation",
    "correct",
    "flood",
]

HOME = ("{N} lives in {V}.", "{N} moved to {V}.", "{N}'s home is {V}.", "{N} is based in {V}.")
EMPLOYER = ("{N} works at {V}.", "{N} joined {V}.", "{N} works for {V}.")
HEDGED = {"home": "I think {N} lives in {V}.", "employer": "Maybe {N} works at {V}?"}
PRONOUN = {"home": "She moved to {V}.", "employer": "He works at {V}."}
NEGATION = {"home": "{N} no longer lives in {V}.", "employer": "{N} no longer works at {V}."}


class WorldSpec(Record):
    name: str
    seed: int
    entities: int = Field(ge=1, le=8)  # base entities first; collisions add Anna and Benn
    collisions: bool = False
    change_every: float = Field(ge=0)  # mean days between true changes; 0 = never
    reports_per_period: float = Field(gt=0)
    repeat: int = Field(default=0, ge=0)  # duplicate reports of the first report's value
    stray: float = Field(default=0.0, ge=0, le=1)  # wrong report at a random time in the period
    reliability: tuple[tuple[str, float], ...]  # hidden, by class
    p_free: float = Field(default=0.6, ge=0, le=1)
    p_hedge: float = Field(default=0.0, ge=0, le=1)
    p_pronoun: float = Field(default=0.0, ge=0, le=1)
    p_alias: float = Field(default=0.0, ge=0, le=1)
    p_ambiguous: float = Field(default=0.0, ge=0, le=1)
    p_typo: float = Field(default=0.0, ge=0, le=1)
    p_announce: float = Field(default=0.0, ge=0, le=1)  # a change is announced by a retraction
    p_correct: float = Field(default=0.0, ge=0, le=1)  # ... or by a structured correction
    delay_days: float = Field(default=3.0, ge=0)
    query_rate: float = Field(gt=0)  # mean user queries per key per day
    skew: float = Field(default=0.0, ge=0)  # zipf exponent of query popularity over keys
    targets: int = Field(default=0, ge=0)  # adversary: keys flooded and pumped
    flood: int = Field(default=0, ge=0)  # stale-value reports per target after its change
    pump: int = Field(default=0, ge=0)  # extra queries per target before its change
    notice_wrong: float = Field(default=0.5, ge=0, le=1)  # simulated user
    false_alarm: float = Field(default=0.02, ge=0, le=1)
    confirm: float = Field(default=0.4, ge=0, le=1)  # relevance confirmations, independent of truth


class Report(Record):
    id: str
    key: str
    entity: str  # gold
    attribute: str  # gold
    value: str  # gold: the value the report asserts (may be wrong)
    occurred: float
    recorded: float
    source: str
    text: str
    form: Form
    kind: Literal["assert", "correct", "negate"]  # gold
    certain: bool  # a careful reader could extract it (False: hedged, no subject, ambiguous)
    true: bool  # analysis only: the asserted value is true when the report occurred


class UserQuery(Record):
    at: float
    key: str
    pumped: bool = False


class World(Record):
    spec: Digest
    seed: int
    name: str
    registry: Registry
    keys: tuple[str, ...]
    truth: tuple[tuple[str, tuple[tuple[float, str], ...]], ...]  # key -> (start, value) periods
    reports: tuple[Report, ...]  # by (recorded, id)
    queries: tuple[UserQuery, ...]  # by time
    targets: tuple[str, ...]  # adversary's keys
    declared: tuple[tuple[str, float], ...]
    user: tuple[tuple[str, float], ...]


def _u(spec: WorldSpec, *labels: str | int | float) -> float:
    return unit_interval(spec.seed, spec.name, *labels)


def registry_for(collisions: bool) -> Registry:
    ents = list(BASE_ENTITIES) + (list(COLLIDING) if collisions else [])
    table: dict[str, tuple[str, list[str]]] = {
        i: (n, [ALIASES[i]] if i in ALIASES else []) for i, n in ents
    }
    if collisions:
        table["ana"][1].append(AMBIGUOUS_ALIAS)
        table["anna"][1].append(AMBIGUOUS_ALIAS)
    return make_registry(table, distinct=[("ana", "anna"), ("ben", "benn")] if collisions else [])


def _typo(name: str, taken: set[str]) -> str | None:
    for i, c in enumerate(name):
        if c in "aeiou":
            t = name[: i + 1] + c + name[i + 1 :]
            return t if t.casefold() not in taken else None
    return None


def build_world(spec: WorldSpec) -> World:
    reg = registry_for(spec.collisions)
    canon = {e.id: e.name for e in reg.entries}
    taken = {n.casefold() for n in canon.values()} | {a for e in reg.entries for a in e.aliases}
    ids = [i for i, _ in BASE_ENTITIES][: spec.entities] + (
        [i for i, _ in COLLIDING] if spec.collisions else []
    )
    keys = tuple(sorted(f"{e}.{a}" for e in ids for a in ("employer", "home")))
    rel = dict(spec.reliability)
    pools = {"home": [c.casefold() for c in CITIES], "employer": [c.casefold() for c in EMPLOYERS]}
    order = sorted(keys, key=lambda k: _u(spec, "target", k))
    targets = tuple(sorted(order[: spec.targets]))
    truth: dict[str, list[tuple[float, str]]] = {}
    reports: list[Report] = []

    def pick_source(*lab: str | int) -> str:
        x = _u(spec, "class", *lab)
        return f"{CLASSES[int(x * len(CLASSES))]}:{int(_u(spec, 'sid', *lab) * 3)}"

    def render(
        rid: str,
        key: str,
        value: str,
        occurred: float,
        source: str,
        true: bool,
        kind: Literal["assert", "correct", "negate"],
        form: Form | None = None,
    ) -> Report:
        ent, _, attr = key.partition(".")
        name, certain = canon[ent], True
        f: Form = form or "free"
        x = _u(spec, "form", rid)
        if kind == "correct":
            text, f = f"correct {key} = {value}", "correct"
        elif kind == "negate":
            text, f = NEGATION[attr].format(N=name, V=value.capitalize()), "negation"
        elif f == "flood":
            text = (
                HOME[0].format(N=name, V=value.capitalize())
                if attr == "home"
                else EMPLOYER[0].format(N=name, V=value.capitalize())
            )
        elif x >= spec.p_free:
            text, f = f"set {key} = {value}", "structured"
        else:
            y = _u(spec, "wording", rid)
            cut = spec.p_hedge, spec.p_hedge + spec.p_pronoun
            z = _u(spec, "variant", rid)
            surface = name
            if y < cut[0]:
                text, f, certain = (
                    HEDGED[attr].format(N=name, V=value.capitalize()),
                    "hedged",
                    False,
                )
            elif y < cut[1]:
                text, f, certain = PRONOUN[attr].format(V=value.capitalize()), "pronoun", False
            else:
                f = "free"
                if z < spec.p_ambiguous and ent in ("ana", "anna"):
                    surface, f, certain = AMBIGUOUS_ALIAS, "ambiguous", False
                elif z < spec.p_ambiguous + spec.p_alias and ent in ALIASES:
                    surface, f = ALIASES[ent], "alias"
                elif z < spec.p_ambiguous + spec.p_alias + spec.p_typo:
                    t = _typo(name, taken)
                    if t:
                        surface, f = t, "typo"
                tpl = HOME if attr == "home" else EMPLOYER
                text = tpl[int(_u(spec, "tpl", rid) * len(tpl))].format(
                    N=surface, V=value.capitalize()
                )
        recorded = quantize(occurred + spec.delay_days * (0.3 + 1.4 * _u(spec, "delay", rid)))
        return Report(
            id=rid,
            key=key,
            entity=ent,
            attribute=attr,
            value=value,
            occurred=quantize(occurred),
            recorded=recorded,
            source=source,
            text=text,
            form=f,
            kind=kind,
            certain=certain,
            true=true,
        )

    for key in keys:
        attr = key.partition(".")[2]
        pool = sorted(pools[attr], key=lambda v: _u(spec, "perm", key, v))
        starts = [0.0]
        if key in targets:
            starts.append(150.0 + 20 * _u(spec, "tchange", key))
        elif spec.change_every > 0:
            t, k = 0.0, 0
            while True:
                t += spec.change_every * (0.5 + _u(spec, "gap", key, k))
                if t > HORIZON - 30:
                    break
                starts.append(t)
                k += 1
        periods = [(s, pool[i % len(pool)]) for i, s in enumerate(starts)]
        truth[key] = periods
        ends = [*starts[1:], HORIZON]
        for p, ((a, val), b) in enumerate(zip(periods, ends, strict=True)):
            n = int(spec.reports_per_period) + (_u(spec, "n", key, p) < spec.reports_per_period % 1)
            n = max(1, n)
            first: tuple[str, float, str] | None = None
            for j in range(n):
                lab = (key, p, j)
                src = pick_source(*lab)
                stray = j > 0 and _u(spec, "stray", *lab) < spec.stray
                early = j == 0 and p == 0
                span = b - a if stray else min(b - a, 10.0 if early else 40.0)
                occ = a + span * _u(spec, "occ", *lab)
                ok = (not stray) and _u(spec, "ok", *lab) < rel[src.partition(":")[0]]
                wrong = [w for w in pool if w != val]
                v = val if ok else wrong[int(_u(spec, "wrongv", *lab) * len(wrong))]
                if first is None:
                    first = (v, occ, src)
                reports.append(render(f"r:{key}:{p}:{j}", key, v, occ, src, v == val, "assert"))
            if first and spec.repeat:
                v0, o0, s0 = first
                for j in range(spec.repeat):
                    occ = min(b - 1e-3, o0 + (b - o0) * (j + 1) / (spec.repeat + 1))
                    reports.append(
                        render(f"r:{key}:{p}:rep{j}", key, v0, occ, s0, v0 == val, "assert")
                    )
            if p >= 1:
                old = periods[p - 1][1]
                if _u(spec, "ann", key, p) < spec.p_announce:
                    reports.append(
                        render(
                            f"r:{key}:{p}:neg",
                            key,
                            old,
                            a + 2 * _u(spec, "annt", key, p),
                            pick_source(key, p, "ann"),
                            True,
                            "negate",
                        )
                    )
                elif _u(spec, "cor", key, p) < spec.p_correct:
                    reports.append(
                        render(
                            f"r:{key}:{p}:cor",
                            key,
                            val,
                            a + 3 * _u(spec, "cort", key, p),
                            "email:0",
                            True,
                            "correct",
                        )
                    )
        if key in targets:  # adversary: the old value again, from forum sources, after the change
            c1, old = periods[1][0], periods[0][1]
            for j in range(spec.flood):
                occ = c1 + 5 + 60 * _u(spec, "fl", key, j)
                reports.append(
                    render(
                        f"r:{key}:flood{j}",
                        key,
                        old,
                        occ,
                        f"forum:{j % 3}",
                        False,
                        "assert",
                        "flood",
                    )
                )
    reports = [r for r in reports if r.recorded <= HORIZON]
    reports.sort(key=lambda r: (r.recorded, r.id))

    weights = {
        k: 1 / (1 + rank) ** spec.skew
        for rank, k in enumerate(sorted(keys, key=lambda k: _u(spec, "pop", k)))
    }
    mean_w = sum(weights.values()) / len(weights)
    queries: list[UserQuery] = []
    for qk in keys:
        n = round(spec.query_rate * HORIZON * weights[qk] / mean_w)
        for j in range(n):
            queries.append(UserQuery(at=quantize(5 + (HORIZON - 5) * _u(spec, "q", qk, j)), key=qk))
    for tk in targets:
        for j in range(spec.pump):
            queries.append(
                UserQuery(at=quantize(10 + 135 * _u(spec, "pump", tk, j)), key=tk, pumped=True)
            )
    queries.sort(key=lambda q: (q.at, q.key, q.pumped))
    return World(
        spec=spec.digest,
        seed=spec.seed,
        name=spec.name,
        registry=reg,
        keys=keys,
        truth=tuple((k, tuple(v)) for k, v in sorted(truth.items())),
        reports=tuple(reports),
        queries=tuple(queries),
        targets=targets,
        declared=DECLARED,
        user=(
            ("confirm", spec.confirm),
            ("false_alarm", spec.false_alarm),
            ("notice_wrong", spec.notice_wrong),
        ),
    )


def truth_index(world: World) -> dict[str, tuple[list[float], list[str]]]:
    return {k: ([s for s, _ in p], [v for _, v in p]) for k, p in world.truth}


def truth_at(
    index: dict[str, tuple[list[float], list[str]]], key: str, t: float
) -> tuple[str, int]:
    """(value, period number) true for ``key`` at day ``t``."""
    starts, values = index[key]
    i = max(0, bisect.bisect_right(starts, t) - 1)
    return values[i], i


RELIABILITY = (("chat", 0.75), ("clinic", 0.97), ("email", 0.85), ("forum", 0.45))
NOISY_RELIABILITY = (("chat", 0.6), ("clinic", 0.9), ("email", 0.75), ("forum", 0.35))
ADVERSARIAL_RELIABILITY = (("chat", 0.75), ("clinic", 0.97), ("email", 0.85), ("forum", 0.2))


def world_specs(base_seed: int, replicate: int) -> tuple[WorldSpec, ...]:
    """The five worlds of one replicate. ``base_seed`` separates test (1) and calibration (2)."""
    seed = base_seed * 1000 + replicate

    def w(name: str, **kw: object) -> WorldSpec:
        return WorldSpec(name=name, seed=seed, entities=6, **kw)  # type: ignore[arg-type]

    return (
        w(
            "stable",
            change_every=0,
            reports_per_period=6,
            reliability=RELIABILITY,
            query_rate=0.15,
            p_free=0.5,
            p_alias=0.05,
            p_announce=0.0,
        ),
        w(
            "changing",
            change_every=45,
            reports_per_period=3,
            reliability=RELIABILITY,
            query_rate=0.15,
            p_free=0.6,
            p_alias=0.05,
            p_announce=0.4,
            p_correct=0.2,
        ),
        w(
            "repetitive",
            change_every=70,
            reports_per_period=2,
            repeat=6,
            reliability=RELIABILITY,
            query_rate=0.25,
            skew=1.6,
            p_free=0.6,
            p_alias=0.05,
            p_announce=0.3,
        ),
        w(
            "noisy",
            change_every=60,
            reports_per_period=3,
            stray=0.25,
            reliability=NOISY_RELIABILITY,
            query_rate=0.15,
            p_free=0.8,
            p_hedge=0.12,
            p_pronoun=0.08,
            p_alias=0.1,
            p_ambiguous=0.0,
            p_typo=0.06,
            p_announce=0.3,
        ),
        w(
            "adversarial",
            change_every=0,
            reports_per_period=3,
            stray=0.1,
            reliability=ADVERSARIAL_RELIABILITY,
            query_rate=0.1,
            collisions=True,
            p_free=0.8,
            p_alias=0.05,
            p_ambiguous=0.3,
            p_typo=0.03,
            p_announce=0.2,
            targets=3,
            flood=12,
            pump=40,
        ),
    )


def query_keys(queries: Sequence[UserQuery]) -> set[str]:
    return {q.key for q in queries}
