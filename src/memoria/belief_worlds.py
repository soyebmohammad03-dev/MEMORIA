"""Generated worlds for belief-revision experiments, with exact ground truth.

A world has, per key (``<entity>.<attribute>``), a hidden sequence of true values over valid
time. Sources report on it: each source has a hidden reliability (the probability that one
of its reports is right), reports arrive after a delay, and the world can add the
phenomena belief revision must cope with: same-time rival reports, stray wrong reports that
read as changes, copies of a wrong report by other sources, retroactive corrections,
reports filed under a near-collision entity, floods of weak sources, a source that earns
trust and then lies, and alternating reports.

The reliabilities and the truth are for analysis only: policies see the evidence and the
*declared* source model, never the hidden values. Every choice is seeded (I21).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from memoria.beliefs import EvidenceItem, evidence_from_statement
from memoria.comparison import normalize
from memoria.core import (
    Dataset,
    Experience,
    Probe,
    Record,
    Step,
    unit_interval,
)
from memoria.scenarios import day, known
from memoria.sources import SourceClass, SourceModel

ENTITY_NAMES = ("Ana", "Ben", "Chen", "Dara", "Eli", "Farid")
COLLIDING_NAMES = ("Ana", "Anna", "Ben", "Benn", "Chen", "Dara")
PARTNER = {"ana": "anna", "anna": "ana", "ben": "benn", "benn": "ben"}
CITIES = ("Paris", "Berlin", "Munich", "Rome", "Oslo", "Lisbon", "Vienna", "Prague", "Madrid")
EMPLOYERS = ("Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark", "Wayne")
DOSES = tuple(f"{n} mg" for n in range(5, 65, 5))


class SourceSpec(Record):
    klass: str
    pool: int = Field(ge=1)
    reliability: float = Field(ge=0, le=1)
    jitter: float = Field(default=0.05, ge=0, le=0.3)
    reports: float = Field(ge=0)  # expected reports per key-period


DEFAULT_SOURCES = (
    SourceSpec(klass="clinic", pool=2, reliability=0.97, jitter=0.02, reports=0.6),
    SourceSpec(klass="email", pool=3, reliability=0.85, jitter=0.05, reports=1.6),
    SourceSpec(klass="chat", pool=3, reliability=0.75, jitter=0.08, reports=1.6),
    SourceSpec(klass="forum", pool=4, reliability=0.45, jitter=0.1, reports=1.5),
)


class BeliefWorldSpec(Record):
    """Parameters of a generated belief world. Its digest identifies it."""

    name: str = Field(min_length=1)
    seed: int
    entities: int = Field(ge=1, le=len(ENTITY_NAMES))
    near_collisions: bool = False  # Ana/Anna and Ben/Benn are distinct entities
    attributes: tuple[str, ...] = ("dose", "employer", "home")
    horizon_days: int = Field(ge=30)
    change_every_days: float = Field(gt=10)
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCES
    declared: tuple[tuple[str, float], ...] = ()  # class -> declared mean (default: the truth)
    rival_rate: float = Field(default=0.0, ge=0, le=1)  # per key-period: a concurrent rival
    stray: float = Field(default=0.15, ge=0, le=1)  # share of wrong reports at a random time
    delay_days: float = Field(default=3.0, ge=0)
    correction_rate: float = Field(default=0.0, ge=0, le=1)
    copy_rate: float = Field(default=0.0, ge=0, le=1)  # share of wrong reports that are copied
    copies: int = Field(default=0, ge=0, le=6)
    mislabel_rate: float = Field(default=0.0, ge=0, le=1)
    flood: int = Field(default=0, ge=0, le=12)  # weak sources agreeing on one wrong value
    flood_rate: float = Field(default=0.0, ge=0, le=1)
    poison: bool = False  # a trusted source that turns unreliable halfway
    alternate: int = Field(default=0, ge=0, le=8)  # rounds of alternating reports
    probes_per_key: int = Field(default=4, ge=1)


class BeliefProbe(Record):
    id: str
    key: str
    valid_at_day: float
    known_at_day: float
    truth: str  # normalised true value at valid_at


@dataclass
class BeliefWorld:
    spec: BeliefWorldSpec
    evidence: list[EvidenceItem]
    dataset: Dataset  # the same experiences as steps, for the memory pipeline
    sources: SourceModel
    reliability: dict[str, float]  # hidden, analysis only
    truth: dict[str, list[tuple[float, str]]]
    probes: list[BeliefProbe]
    wrong_reports: set[str] = field(default_factory=set)  # experience ids of erroneous reports
    origin_of: dict[str, str] = field(default_factory=dict)  # copy source -> origin source

    def truth_at(self, key: str, t: float) -> str | None:
        current = None
        for start, value in self.truth.get(key, []):
            if start <= t:
                current = value
        return current


def _value(attribute: str, index: int) -> str:
    pool = {"home": CITIES, "employer": EMPLOYERS, "dose": DOSES}.get(attribute, CITIES)
    return pool[index % len(pool)]


def belief_world(spec: BeliefWorldSpec) -> BeliefWorld:
    """Generate the evidence stream, the hidden truth and the probes of one world."""

    def u(*labels: str | int | float) -> float:
        return unit_interval(spec.seed, spec.name, *labels)

    names = list((COLLIDING_NAMES if spec.near_collisions else ENTITY_NAMES)[: spec.entities])
    keys = [f"{n.lower()}.{a}" for n in names for a in spec.attributes]
    reliability: dict[str, float] = {}
    pools: dict[str, list[str]] = {}
    for sp in spec.sources:
        pools[sp.klass] = []
        for j in range(sp.pool):
            sid = f"{sp.klass}:{sp.klass[0]}{j + 1}"
            jitter = (u("rel", sid) - 0.5) * 2 * sp.jitter
            reliability[sid] = round(min(0.995, max(0.02, sp.reliability + jitter)), 4)
            pools[sp.klass].append(sid)
    if spec.poison:
        reliability["email:trusted"] = 0.97  # true until halfway, then wrong
    truth: dict[str, list[tuple[float, str]]] = {}
    for key in keys:
        attribute = key.split(".", 1)[1]
        starts, t = [0.0], 0.0
        while True:
            t += max(12.0, spec.change_every_days * (0.7 + 0.6 * u(key, "gap", len(starts))))
            if t >= spec.horizon_days - 10:
                break
            starts.append(round(t, 2))
        offset = int(u(key, "offset") * 50)
        vals: list[str] = []
        for i in range(len(starts)):
            v = _value(attribute, offset + i * 3 + int(u(key, "step", i) * 3))
            if vals and v == vals[-1]:
                v = _value(attribute, offset + i * 3 + 1)
            vals.append(v)
        truth[key] = list(zip(starts, vals, strict=True))

    def wrong_value(key: str, true: str, *labels: str | int | float) -> str:
        attribute = key.split(".", 1)[1]
        if attribute == "dose":
            n = int(true.split()[0]) + (1 + int(u(*labels, "d") * 3)) * (
                1 if u(*labels, "sign") < 0.5 else -1
            )
            return f"{max(1, n)} mg"
        pool = {"home": CITIES, "employer": EMPLOYERS}.get(attribute, CITIES)
        options = [x for x in pool if x != true]
        return options[int(u(*labels, "w") * len(options))]

    reports: list[tuple[str, str, str, float, float, str, bool]] = []
    copy_of: dict[str, str] = {}
    counter = [0]

    def add(key: str, verb: str, value: str, occurred: float, source: str, wrong: bool) -> None:
        counter[0] += 1
        delay = u("delay", counter[0]) * spec.delay_days
        reports.append((key, verb, value, occurred, occurred + delay, source, wrong))

    def maybe_mislabel(key: str, *labels: str | int | float) -> str:
        ent, attr = key.split(".", 1)
        if spec.mislabel_rate > 0 and ent in PARTNER and u(*labels, "mis") < spec.mislabel_rate:
            return f"{PARTNER[ent]}.{attr}"
        return key

    for key in keys:
        periods = truth[key]
        for i, (start, true) in enumerate(periods):
            end = periods[i + 1][0] if i + 1 < len(periods) else float(spec.horizon_days)
            length = end - start
            made_wrong: list[tuple[str, float, str]] = []
            for sp in spec.sources:
                n = int(sp.reports) + (1 if u(key, i, sp.klass, "n") < sp.reports % 1 else 0)
                for r in range(n):
                    src = pools[sp.klass][int(u(key, i, sp.klass, r, "src") * sp.pool)]
                    right = u(key, i, sp.klass, r, "ok") < reliability[src]
                    near = u(key, i, sp.klass, r, "near") < 0.6
                    if right:
                        if near:
                            at = start + u(key, i, sp.klass, r, "t") * 1.4
                        else:
                            at = start + 2 + u(key, i, sp.klass, r, "t2") * max(1.0, length - 3)
                        value = true
                    else:
                        stray = u(key, i, sp.klass, r, "stray") < spec.stray
                        if stray:
                            span = max(1.0, min(length - 3, 25.0))
                            at = start + 2 + u(key, i, sp.klass, r, "t3") * span
                        else:
                            at = start + u(key, i, sp.klass, r, "t4") * 1.4
                        value = wrong_value(key, true, key, i, sp.klass, r)
                        made_wrong.append((src, at, value))
                    add(maybe_mislabel(key, key, i, sp.klass, r), "set", value, at, src, not right)
            if u(key, i, "rival") < spec.rival_rate:
                forum = pools.get("forum") or [next(iter(reliability))]
                src = forum[int(u(key, i, "rsrc") * len(forum))]
                value = wrong_value(key, true, key, i, "rival")
                at = start + u(key, i, "rt") * 1.4
                add(key, "set", value, at, src, True)
                made_wrong.append((src, at, value))
            for src, at, value in made_wrong:
                if spec.copies and u(key, src, at, "copy") < spec.copy_rate:
                    for c in range(spec.copies):
                        copy = f"bot:c{len(copy_of) + 1}"
                        copy_of[copy] = src
                        reliability[copy] = 0.0
                        add(key, "set", value, at + 0.01 * (c + 1), copy, True)
            if spec.flood and u(key, i, "flood") < spec.flood_rate:
                value = wrong_value(key, true, key, i, "flood")
                for f in range(spec.flood):
                    src = f"forum:f{int(u(key, i, f, 'fsrc') * 40) + 1}"
                    reliability.setdefault(src, 0.4)
                    add(key, "set", value, start + u(key, i, f, "ft") * 1.0, src, True)
            if spec.correction_rate and u(key, i, "corr") < spec.correction_rate and length > 12:
                src = pools["forum"][0] if "forum" in pools else next(iter(reliability))
                value = wrong_value(key, true, key, i, "corrwrong")
                add(key, "set", value, start + 0.2, src, True)
                fix = start + 3 + u(key, i, "fix") * 6
                cl = pools.get("clinic", [next(iter(reliability))])[0]
                counter[0] += 1
                delay = u("delay", counter[0]) * spec.delay_days * 2
                reports.append((key, "correct", true, fix, fix + delay, cl, False))
            if spec.alternate and i == 0 and u(key, "alt") < 0.7:
                other = wrong_value(key, true, key, "alt")
                for rd in range(spec.alternate):
                    add(key, "set", true, start + 3 + 4 * rd, "chat:c1", False)
                    add(key, "set", other, start + 5 + 4 * rd, "chat:c2", True)
        if spec.poison:
            for i, (start, true) in enumerate(periods):
                end = periods[i + 1][0] if i + 1 < len(periods) else float(spec.horizon_days)
                late = start > spec.horizon_days / 2
                value = wrong_value(key, true, key, i, "poison") if late else true
                for extra in range(2 if late else 1):
                    add(key, "set", value, start + 0.3 + 0.2 * extra, "email:trusted", late)
    # Build experiences, evidence and the dataset.
    steps: list[Step] = []
    evidence: list[EvidenceItem] = []
    wrong_ids: set[str] = set()
    for key, verb, value, occurred, recorded, source, wrong in sorted(
        reports, key=lambda r: (r[4], r[5], r[0], r[2])
    ):
        content = f"{verb} {key} = {value}"
        e = Experience(source=source, content=content, occurred_at=day(occurred))
        if any(s.experience.digest == e.digest for s in steps[-8:]):
            continue
        steps.append(Step(experience=e, recorded_at=day(recorded)))
        item = evidence_from_statement(e.digest, content, day(occurred), day(recorded), source)
        assert item is not None
        evidence.append(item)
        if wrong:
            wrong_ids.add(e.digest)
    steps.sort(key=lambda s: (s.recorded_at, s.experience.source))
    probes: list[BeliefProbe] = []
    dprobes: list[Probe] = []
    for key in keys:
        for p in range(spec.probes_per_key):
            valid = 12 + u(key, "pv", p) * (spec.horizon_days - 14)
            known_at = min(spec.horizon_days + spec.delay_days * 2, valid + u(key, "pk", p) * 25)
            tv = max((s, v) for s, v in truth[key] if s <= valid)[1]
            norm = " ".join(normalize(tv))
            pid = f"{key}-{p}"
            probes.append(BeliefProbe(id=pid, key=key, valid_at_day=round(valid, 3),
                                      known_at_day=round(known_at, 3), truth=norm))  # fmt: skip
            entity, attribute = key.split(".", 1)
            dprobes.append(Probe(id=pid, text=f"What is {attribute} of {entity}?",
                                 valid_at=day(valid), known_at=day(known_at), expected=known(tv),
                                 key=key))  # fmt: skip
    classes = []
    declared = dict(spec.declared)
    for sp in spec.sources:
        mean = declared.get(sp.klass, sp.reliability)
        classes.append(
            SourceClass(name=sp.klass, alpha=round(mean * 10, 2), beta=round((1 - mean) * 10, 2))
        )
    model = SourceModel(
        name=f"declared:{spec.name}", version="1",
        classes=tuple(sorted(classes, key=lambda c: c.name)),
        copies=tuple(sorted(copy_of.items())),
    )  # fmt: skip
    dataset = Dataset(name=f"belief:{spec.name}", version=spec.digest, steps=tuple(steps),
                      probes=tuple(dprobes))  # fmt: skip
    return BeliefWorld(
        spec=spec, evidence=evidence, dataset=dataset, sources=model, reliability=reliability,
        truth=truth, probes=probes, wrong_reports=wrong_ids, origin_of=dict(copy_of),
    )  # fmt: skip


def worlds(seed: int, replicate: int = 0) -> tuple[BeliefWorldSpec, ...]:
    """The ten worlds of the belief matrix for one seed (replicates differ by seed only)."""
    s = seed * 100 + replicate
    base = {"seed": s, "horizon_days": 360}

    def w(name: str, **kw: object) -> BeliefWorldSpec:
        return BeliefWorldSpec.model_validate({"name": name, **base, "entities": 3,
                                               "change_every_days": 90} | kw)  # fmt: skip

    dispute = (
        SourceSpec(klass="email", pool=3, reliability=0.8, jitter=0.05, reports=2.5),
        SourceSpec(klass="chat", pool=3, reliability=0.55, jitter=0.08, reports=2.5),
    )
    return (
        w("stable", change_every_days=150, delay_days=2.0, stray=0.1),
        w("temporal-change", change_every_days=25, delay_days=6.0, rival_rate=0.1),
        w("contradiction", rival_rate=0.6, stray=0.2),
        w("delayed-correction", correction_rate=0.5, delay_days=12.0),
        w("source-disagreement", sources=dispute, stray=0.3),
        w("duplicated-evidence", rival_rate=0.5, copy_rate=0.8, copies=3),
        w(
            "adversarial",
            change_every_days=60,
            rival_rate=0.3,
            copy_rate=0.4,
            copies=2,
            flood=4,
            flood_rate=0.5,
            poison=True,
            alternate=3,
            declared=(("email", 0.9),),
        ),
        w("ambiguous-entities", entities=6, near_collisions=True, mislabel_rate=0.25),
        w("numeric", attributes=("dose",), entities=4, rival_rate=0.5),
        w("long-horizon", horizon_days=1080, change_every_days=40, delay_days=5.0),
    )
