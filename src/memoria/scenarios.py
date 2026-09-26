"""Deterministic research datasets: experience streams with probes and ground truth.

Contents follow the statement language of :class:`memoria.formation.StatementPolicy`,
so the same stream exercises every formation outcome under that policy and serves as
unstructured episodes for others.

Probe expectations are *epistemic*: what an ideal system could believe about
``valid_at`` given every experience recorded by ``known_at``. They are independent of
any policy, which is what makes a policy's behaviour measurable against them. Under
that standard the most recent report *by occurrence* holds; later ingestion does not
make a report more true.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from memoria.core import (
    Dataset,
    Expectation,
    ExpectationStatus,
    Experience,
    InterventionSpec,
    Probe,
    RunManifest,
    Scalar,
    Step,
    unit_interval,
)
from memoria.retrieval import EXTRACTIVE, lexical_recency
from memoria.semantic_eval import Judgement, Relevance, SemanticBenchmark, SemanticQuery

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def day(n: float) -> datetime:
    return EPOCH + timedelta(days=n)


def known(value: str) -> Expectation:
    return Expectation(status=ExpectationStatus.KNOWN, values=(value,))


UNKNOWN = Expectation(status=ExpectationStatus.UNKNOWN)


def contested(*values: str) -> Expectation:
    return Expectation(status=ExpectationStatus.CONTESTED, values=values)


def _step(occurred: float, content: str, source: str, recorded: float | None = None) -> Step:
    return Step(
        experience=Experience(source=source, content=content, occurred_at=day(occurred)),
        recorded_at=day(occurred if recorded is None else recorded),
    )


def _probe(pid: str, text: str, valid: float, known_at: float, expected: Expectation) -> Probe:
    return Probe(id=pid, text=text, valid_at=day(valid), known_at=day(known_at), expected=expected)


def relocation_year() -> Dataset:
    """One user over a year: moves, a job misremembered then corrected, a forgotten and
    relearned employer, late and conflicting reports, duplicates and malformed input.

    The comment on each step is the outcome ``statement-v1`` records for it.
    """
    steps = (
        _step(0, "set home = Paris", "chat:0"),  # NEW_KEY
        _step(3, "set employer = Acme", "chat:3"),  # NEW_KEY
        _step(5, "We had pasta for lunch near the office.", "chat:5"),  # UNPARSED
        _step(40, "set employer = Acme", "chat:40"),  # REDUNDANT
        _step(55, "set home = Berlin", "chat:55", recorded=60),  # VALUE_CHANGED (late)
        _step(61, "correct employer = Globex", "chat:61"),  # EXPLICIT_CORRECTION
        _step(50, "set home = Munich", "email:50", recorded=90),  # STALE
        _step(0, "set home = Paris", "chat:0", recorded=120),  # DUPLICATE_EXPERIENCE
        _step(150, "forget employer", "chat:150"),  # EXPLICIT_FORGET
        _step(200, "set employer = Initech", "chat:200"),  # RELEARNED -> employer#2
        _step(250, "correct pet = cat", "chat:250"),  # UNKNOWN_KEY
        _step(298, "set home = Hamburg", "chat:298a", recorded=300),  # VALUE_CHANGED
        _step(298, "set home = Bremen", "chat:298b", recorded=300),  # CONFLICTING
        _step(310, "forget home = Hamburg", "chat:310"),  # UNPARSED (forget takes no value)
        _step(330, "set pet = dog", "chat:330"),  # NEW_KEY
        _step(340, "Remind me where I live these days?", "chat:340"),  # UNPARSED
    )
    probes = (
        _probe("home-57-before-report", "where is home", 57, 59, known("Paris")),
        _probe("home-57-after-report", "where is home", 57, 61, known("Berlin")),
        # The late Munich report (occurred day 50) is older than the Berlin move (55).
        _probe("home-100", "where is home", 100, 100, known("Berlin")),
        # Two reports for the same instant: the evidence does not decide.
        _probe("home-345", "where is home", 345, 345, contested("Bremen", "Hamburg")),
        _probe("employer-10-before-correction", "employer", 10, 60, known("Acme")),
        _probe("employer-10-after-correction", "employer", 10, 70, known("Globex")),
        # An ideal system honours "forget": nothing is held until the employer is relearned.
        _probe("employer-175-forgotten", "employer", 175, 175, UNKNOWN),
        _probe("employer-345-relearned", "employer", 345, 345, known("Initech")),
        _probe("pet-345", "pet", 345, 345, known("dog")),
        # "correct pet = cat" still asserts cat, though statement-v1 had nothing to correct.
        _probe("pet-260-correct-without-memory", "pet", 260, 260, known("cat")),
        _probe("unrelated", "zebra", 345, 345, UNKNOWN),
        _probe("before-anything", "home", 0, -1, UNKNOWN),
    )
    return Dataset(name="relocation-year", version="1", steps=steps, probes=probes)


def _minutes(days: float) -> timedelta:
    """Whole minutes, so generated times never depend on float formatting."""
    return timedelta(minutes=round(days * 1440))


def drifting_facts(
    seed: int,
    *,
    keys: int = 4,
    changes: int = 5,
    horizon_days: float = 365,
    max_delay_days: float = 10,
    probes_per_key: int = 6,
) -> Dataset:
    """A seeded long-horizon stream: each key's value changes over time; every change is
    reported once, as ``set <key> = <value>``, after a random ingestion delay.

    Delays let reports arrive out of occurrence order, so a system that ignores late
    reports (``statement-v1`` records them as STALE) diverges from the ground truth.
    Probe times are drawn uniformly; ``known_at`` lies up to twice the maximum delay
    after ``valid_at``, so some probes precede the report they would need.
    """
    if min(keys, changes, probes_per_key) < 1 or horizon_days <= 0 or max_delay_days < 0:
        raise ValueError("keys, changes and probes_per_key >= 1; horizon > 0; delay >= 0")

    def u(*labels: str | int) -> float:
        return unit_interval(seed, "drifting-facts", *labels)

    steps: list[Step] = []
    reports: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    for k in range(keys):
        key = f"k{k}"
        times = sorted(u(key, "change", c) * horizon_days for c in range(changes))
        for c, t in enumerate(times):
            value = f"{key}v{c}"
            occurred = EPOCH + _minutes(t)
            recorded = occurred + _minutes(u(key, "delay", c) * max_delay_days)
            steps.append(
                Step(
                    experience=Experience(
                        source=f"world:{key}:{c}",
                        content=f"set {key} = {value}",
                        occurred_at=occurred,
                    ),
                    recorded_at=recorded,
                )
            )
            reports[key].append((occurred, recorded, value))

    probes = []
    for k in range(keys):
        key = f"k{k}"
        for p in range(probes_per_key):
            valid = EPOCH + _minutes(u(key, "probe-valid", p) * horizon_days)
            known_at = valid + _minutes(u(key, "probe-known", p) * 2 * max_delay_days)
            probes.append(
                Probe(
                    id=f"{key}-{p}",
                    text=key,
                    valid_at=valid,
                    known_at=known_at,
                    expected=epistemic_truth(reports[key], valid_at=valid, known_at=known_at),
                )
            )
    steps.sort(key=lambda s: (s.recorded_at, s.digest))
    return Dataset(
        name="drifting-facts",
        version=f"1:seed={seed}:keys={keys}:changes={changes}:horizon={horizon_days}"
        f":delay={max_delay_days}:probes={probes_per_key}",
        steps=tuple(steps),
        probes=tuple(probes),
    )


def epistemic_truth(
    reports: list[tuple[datetime, datetime, str]], *, valid_at: datetime, known_at: datetime
) -> Expectation:
    """Among reports (occurred, recorded, value) recorded by ``known_at`` that occurred by
    ``valid_at``, the latest by occurrence holds; a tie between values is contested."""
    visible = [(o, v) for o, r, v in reports if r <= known_at and o <= valid_at]
    if not visible:
        return UNKNOWN
    latest = max(o for o, _ in visible)
    values = {v for o, v in visible if o == latest}
    return known(values.pop()) if len(values) == 1 else contested(*values)


def neighbours() -> Dataset:
    """Interference: keys that share vocabulary (``home`` / ``home.office``) and a value
    that changes. Probes ask about one key using words that also match the other."""
    steps = (
        _step(0, "set home = Paris", "chat:0"),
        _step(1, "set home.office = Lyon", "chat:1"),
        _step(10, "set home.office = Nice", "chat:10"),
        _step(12, "Working from the home office today.", "chat:12"),
    )
    probes = (
        _probe("home", "home", 20, 20, known("Paris")),
        _probe("office-before-move", "home office", 5, 20, known("Lyon")),
        _probe("office-after-move", "home office", 15, 20, known("Nice")),
        _probe("office-alone", "office", 15, 20, known("Nice")),
    )
    return Dataset(name="neighbours", version="1", steps=steps, probes=probes)


def conditions(dataset: str, policy: str, *, seed: int = 1) -> dict[str, RunManifest]:
    """The standard intervention grid over one dataset and policy, one manifest per
    condition. Each differs from ``baseline`` in exactly one variable (its intervention),
    so every condition can be compared with the baseline under Principle 4."""

    def run(name: str, *interventions: InterventionSpec) -> RunManifest:
        return RunManifest(
            name=name,
            dataset=dataset,
            interventions=interventions,
            policy=policy,
            retriever=lexical_recency().spec,
            responder=EXTRACTIVE,
        )

    def spec(name: str, **params: Scalar) -> InterventionSpec:
        return InterventionSpec(name=name, params=tuple(params.items()))

    return {
        "baseline": run("baseline"),
        "drop": run("drop", spec("drop", rate=0.3, seed=seed)),
        "delay": run("delay", spec("delay", rate=0.5, seed=seed, days=20)),
        "reorder": run("reorder", spec("reorder", seed=seed, window_days=15)),
        "contradict": run(
            "contradict",
            spec("contaminate", rate=0.5, seed=seed, delay_days=0, source="contaminant"),
        ),
        "contaminate": run(
            "contaminate",
            spec("contaminate", rate=0.5, seed=seed, delay_days=1, source="contaminant"),
        ),
    }


# --- Phase 5 semantic diagnostic -------------------------------------------------------------

# (candidate id, text, relevance to its subject's queries or None if unjudged)
_SEMANTIC_SUBJECTS: dict[str, tuple[tuple[str, str], list[tuple[str, str, str]]]] = {
    "residence": (
        ("Where does Ana live?", "What city is Ana's home?"),
        [
            ("res-base", "Ana lives in Berlin.", "paraphrase"),
            ("res-para", "Ana's home is in Berlin.", "paraphrase"),
            ("res-equiv", "Home for Ana is the capital of Germany.", "equivalent"),
            ("res-source-same", "Ana lives in Berlin.", "other_source"),
            ("res-source-text", "Ana's mother says Ana lives in Berlin.", "other_source"),
            ("res-temporal", "Ana lived in Munich until 2024.", "temporal_variant"),
            ("res-contra", "Ana lives in Hamburg.", "contradiction"),
            ("res-neg", "Ana does not live in Berlin.", "negation"),
            ("res-entity", "Ben lives in Berlin.", "entity_substitution"),
            ("res-lexical", "Ana visits Berlin every summer.", "lexical_distractor"),
            ("res-vocab", "Berlin has many lakes.", "shared_vocabulary"),
        ],
    ),
    "meeting": (
        ("When is the project meeting?", "What time do we sync about the project each week?"),
        [
            ("mtg-base", "The project meeting starts at 3 pm on Thursday.", "paraphrase"),
            ("mtg-para", "Thursday's project meeting begins at 3 pm.", "paraphrase"),
            (
                "mtg-equiv",
                "The weekly project sync is Thursday afternoon at fifteen hundred.",
                "equivalent",
            ),
            ("mtg-number", "The project meeting starts at 5 pm on Thursday.", "numeric_change"),
            (
                "mtg-temporal",
                "Last month the project meeting started at 10 am on Mondays.",
                "temporal_variant",
            ),
            ("mtg-neg", "The project meeting does not start at 3 pm on Thursday.", "negation"),
            ("mtg-lexical", "The project starts on Thursday.", "lexical_distractor"),
            ("mtg-vocab", "Thursday is garbage collection day.", "shared_vocabulary"),
        ],
    ),
    "medication": (
        ("How much lisinopril does Leo take?", "What dose is Leo's blood pressure medication?"),
        [
            ("med-base", "Leo takes 20 mg of lisinopril each morning.", "paraphrase"),
            ("med-para", "Every morning Leo has a 20 mg lisinopril dose.", "paraphrase"),
            (
                "med-equiv",
                "Leo's blood-pressure pill is twenty milligrams, taken at breakfast.",
                "equivalent",
            ),
            ("med-number", "Leo takes 40 mg of lisinopril each morning.", "numeric_change"),
            ("med-contra", "Leo stopped taking lisinopril.", "contradiction"),
            ("med-entity", "Mia takes 20 mg of lisinopril each morning.", "entity_substitution"),
            ("med-vocab", "Lisinopril can cause a dry cough.", "shared_vocabulary"),
        ],
    ),
    "employer": (
        ("Where does Priya work?", "Who employs Priya?"),
        [
            ("job-base", "Priya works at Acme Corp as a data engineer.", "paraphrase"),
            ("job-para", "Priya is a data engineer employed by Acme Corp.", "paraphrase"),
            ("job-equiv", "Acme pays Priya to build its data pipelines.", "equivalent"),
            ("job-temporal", "Priya worked at Globex before joining Acme.", "temporal_variant"),
            ("job-contra", "Priya works at Initech as a data engineer.", "contradiction"),
            ("job-entity", "Omar works at Acme Corp as a data engineer.", "entity_substitution"),
            ("job-lexical", "Acme Corp hired a new data engineer.", "lexical_distractor"),
        ],
    ),
    "pet": (
        ("What is the name of Sam's dog?", "What did Sam call the puppy?"),
        [
            ("pet-base", "Sam's dog is called Biscuit.", "paraphrase"),
            ("pet-para", "Sam has a dog named Biscuit.", "paraphrase"),
            ("pet-equiv", "Biscuit is the puppy that belongs to Sam.", "equivalent"),
            ("pet-neg", "Sam's dog is not called Biscuit.", "negation"),
            ("pet-entity", "Sam's cat is called Biscuit.", "entity_substitution"),
            ("pet-vocab", "Biscuit recipes need butter and flour.", "shared_vocabulary"),
        ],
    ),
    "allergy": (
        ("What is Noah allergic to?", "Which food gives Noah a reaction?"),
        [
            ("alg-base", "Noah is allergic to peanuts.", "paraphrase"),
            ("alg-para", "Noah has a peanut allergy.", "paraphrase"),
            ("alg-equiv", "Peanuts make Noah's throat swell up.", "equivalent"),
            ("alg-neg", "Noah is not allergic to peanuts.", "negation"),
            ("alg-contra", "Noah is allergic to shellfish, not peanuts.", "contradiction"),
            ("alg-lexical", "Noah sells peanuts at the market.", "lexical_distractor"),
        ],
    ),
    "flight": (
        ("When does Kim fly to Tokyo?", "What date is Kim's trip to Japan?"),
        [
            ("flt-base", "Kim's flight to Tokyo departs on 14 March.", "paraphrase"),
            ("flt-para", "Kim flies to Tokyo on 14 March.", "paraphrase"),
            ("flt-number", "Kim's flight to Tokyo departs on 21 March.", "numeric_change"),
            ("flt-temporal", "Kim flew to Tokyo last year in October.", "temporal_variant"),
            ("flt-entity", "Kim's flight to Seoul departs on 14 March.", "entity_substitution"),
            ("flt-vocab", "Tokyo is famous for cherry blossoms in March.", "shared_vocabulary"),
        ],
    ),
}


def semantic_diagnostic() -> tuple[Dataset, SemanticBenchmark]:
    """The Phase 5 controlled diagnostic for representations (not the Phase 15 benchmark).

    Each subject has a base fact and variants labelled by their relationship to the
    subject's two queries — one phrased with the facts' own words, one without them.
    Candidates are experiences (one per day, ingested as they occur); an experience's
    ``source`` is its candidate id. Unjudged candidates (other subjects) are unrelated.
    Labels are the design, stated before any measurement: which variants a
    representation *should* place close to a query is what the experiment tests.
    """
    steps = []
    queries = []
    n = 0
    for subject, (texts, candidates) in _SEMANTIC_SUBJECTS.items():
        for cid, text, _ in candidates:
            steps.append(_step(n, text, cid))
            n += 1
        judgements = tuple(
            Judgement(candidate=cid, relevance=Relevance(r)) for cid, _, r in candidates
        )
        for i, text in enumerate(texts):
            style = "worded" if i == 0 else "reworded"
            queries.append(SemanticQuery(id=f"{subject}-{style}", text=text, judgements=judgements))
    dataset = Dataset(name="semantic-diagnostic", version="1", steps=tuple(steps))
    return dataset, SemanticBenchmark(
        name="semantic-diagnostic", version="1", dataset=dataset.digest, queries=tuple(queries)
    )
