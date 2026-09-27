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

from pydantic import Field

from memoria.comparison import normalize
from memoria.consolidation_eval import ConsolidationBenchmark, FactLabel
from memoria.core import (
    Dataset,
    Expectation,
    ExpectationStatus,
    Experience,
    InterventionSpec,
    Probe,
    Record,
    RunManifest,
    Scalar,
    Step,
    unit_interval,
)
from memoria.hybrid_eval import (
    BenchmarkQuery,
    CandidateFact,
    HybridBenchmark,
    Perturbation,
    Role,
)
from memoria.hybrid_eval import Judgement as HybridJudgement
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


# --- Phase 6 hybrid-retrieval benchmark ------------------------------------------------------

# (candidate id = experience source, day it occurred, text, fact it expresses or None).
# Ids are "<class>:<id>" except "import-a81", whose source carries no structured class.
_HYBRID_CANDIDATES: tuple[tuple[str, int, str, str | None], ...] = (
    # Ana: a move, a later move, duplicates, a wrong attribute, a wrong entity.
    ("chat:a0", 0, "set ana.home = Paris", "ana.home=paris"),
    ("chat:a2", 2, "Ana lives in Paris, near the Canal Saint-Martin.", "ana.home=paris"),
    ("chat:a55", 55, "set ana.home = Berlin", "ana.home=berlin"),
    ("email:a56", 56, "Ana lives in Berlin now.", "ana.home=berlin"),
    ("email:a57", 57, "Home for Ana is the German capital these days.", "ana.home=berlin"),
    ("chat:b58", 58, "set ben.home = Berlin", "ben.home=berlin"),
    ("chat:b59", 59, "Ben lives in Berlin now.", "ben.home=berlin"),
    ("chat:a60", 60, "set ana.work_city = Paris", "ana.work_city=paris"),
    ("chat:a61", 61, "Ana works in Paris three days a week.", "ana.work_city=paris"),
    ("chat:a80", 80, "Ana lives in Berlin now.", "ana.home=berlin"),
    ("import-a81", 81, "Ana lives in Berlin now.", "ana.home=berlin"),
    ("chat:a120", 120, "Paris has excellent restaurants, Ana says.", None),
    ("chat:a200", 200, "set ana.home = Munich", "ana.home=munich"),
    ("chat:a201", 201, "Ana lives in Munich now.", "ana.home=munich"),
    ("chat:a299", 299, "Ana bought a new bicycle yesterday.", None),
    # Leo: two sources disagree about the same instant.
    ("clinic:m10", 10, "set leo.dose = 20 mg", "leo.dose=20 mg"),
    ("forum:m10", 10, "set leo.dose = 40 mg", "leo.dose=40 mg"),
    ("clinic:m11", 10, "Leo takes 20 mg of lisinopril each morning.", "leo.dose=20 mg"),
    ("clinic:m12", 12, "set mia.dose = 20 mg", "mia.dose=20 mg"),
    ("forum:m40", 40, "Lisinopril can cause a dry cough.", None),
    # Kim: a same-instant contradiction between near-identical statements.
    ("chat:f20", 20, "set kim.flight = 14 March", "kim.flight=14 march"),
    ("email:f20", 20, "set kim.flight = 21 March", "kim.flight=21 march"),
    ("chat:f21", 20, "Kim's flight to Tokyo departs on 14 March.", "kim.flight=14 march"),
    ("chat:f22", 22, "Kim's flight to Seoul departs on 14 March.", None),
    # Team meeting: a retroactive correction.
    ("chat:t20", 20, "set team.meeting = Thursday 3 pm", "team.meeting=thursday 3 pm"),
    ("chat:t21", 21, "The team meeting starts at 3 pm on Thursday.", "team.meeting=thursday 3 pm"),
    ("email:t30", 30, "correct team.meeting = Thursday 4 pm", "team.meeting=thursday 4 pm"),
    (
        "email:t31",
        31,
        "Correction: the team meeting is at 4 pm on Thursday.",
        "team.meeting=thursday 4 pm",
    ),
    # Sam: a fact, a same-value fact about another attribute, then a forget.
    ("chat:d5", 5, "set sam.dog = Biscuit", "sam.dog=biscuit"),
    ("chat:d6", 6, "Sam's dog is called Biscuit.", "sam.dog=biscuit"),
    ("chat:d7", 7, "set sam.cat = Biscuit", "sam.cat=biscuit"),
    ("chat:d90", 90, "forget sam.dog", None),
    # Noah: one fact stated five ways, and distractors.
    ("clinic:n15", 15, "set noah.allergy = peanuts", "noah.allergy=peanuts"),
    ("chat:n16", 16, "Noah is allergic to peanuts.", "noah.allergy=peanuts"),
    ("email:n17", 17, "Noah is allergic to peanuts.", "noah.allergy=peanuts"),
    ("chat:n18", 18, "Noah has a peanut allergy.", "noah.allergy=peanuts"),
    ("chat:n19", 19, "Peanuts make Noah's throat swell up.", "noah.allergy=peanuts"),
    ("chat:n20", 20, "Noah's sister is allergic to shellfish.", None),
    ("chat:n25", 25, "Noah sells peanuts at the market.", None),
    # Priya: a job change, a colleague, a recent unrelated note.
    ("email:p30", 30, "set priya.employer = Globex", "priya.employer=globex"),
    ("chat:p31", 31, "Priya works at Globex as a data engineer.", "priya.employer=globex"),
    ("email:p150", 150, "set priya.employer = Acme", "priya.employer=acme"),
    ("chat:p151", 151, "Priya works at Acme as a data engineer.", "priya.employer=acme"),
    ("chat:p152", 152, "Omar works at Acme as a data engineer.", "omar.employer=acme"),
    ("chat:p298", 298, "Priya had lunch with the Acme team today.", None),
)

HYBRID_CASES = (
    "temporally_wrong",  # lexically (near-)identical to the target but not valid yet
    "contradictory_similar",  # semantically similar, contradictory value
    "recent_irrelevant",  # recent, shares the entity, answers nothing
    "old_but_exact",  # old memory that exactly answers a historical query
    "redundant_cluster",  # the answer stated several times
    "wrong_entity",  # high similarity, another entity
    "wrong_attribute",  # high lexical overlap, another attribute of the entity
    "paraphrase_low_overlap",  # the answer with little shared wording
    "superseded",  # an older value a later claim replaced
    "source_conflict",  # sources of different declared reliability disagree
    "retroactive_correction",  # a correction that applies before it occurred
    "retracted",  # a fact that was later forgotten
)

# (role, candidates, cases the candidates are the focus of)
_Roles = tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]


def _judge(roles: _Roles) -> tuple[HybridJudgement, ...]:
    return tuple(
        HybridJudgement(candidate=c, role=Role(role), cases=cases)
        for role, candidates, cases in roles
        for c in candidates
    )


_ANA_NOW: _Roles = (
    ("target", ("chat:a55", "email:a56", "chat:a80", "import-a81"), ()),
    ("target", ("email:a57",), ("paraphrase_low_overlap",)),
    ("related", ("chat:a0", "chat:a2"), ("superseded",)),
    ("trap", ("chat:a200", "chat:a201"), ("temporally_wrong",)),
    ("trap", ("chat:b58", "chat:b59"), ("wrong_entity",)),
    ("trap", ("chat:a60", "chat:a61"), ("wrong_attribute",)),
    ("trap", ("chat:a299",), ("recent_irrelevant",)),
    ("trap", ("chat:a120",), ()),
)
_NOAH: _Roles = (
    ("target", ("clinic:n15", "chat:n16", "email:n17", "chat:n18"), ()),
    ("target", ("chat:n19",), ("paraphrase_low_overlap",)),
    ("trap", ("chat:n20", "chat:n25"), ()),
)
_LEO: _Roles = (
    ("target", ("clinic:m10", "forum:m10"), ("source_conflict", "contradictory_similar")),
    ("target", ("clinic:m11",), ()),
    ("trap", ("clinic:m12",), ("wrong_entity",)),
    ("trap", ("forum:m40",), ()),
)
_KIM: _Roles = (
    ("target", ("chat:f20", "email:f20"), ("contradictory_similar",)),
    ("target", ("chat:f21",), ()),
    ("trap", ("chat:f22",), ("wrong_entity",)),
)
_PRIYA_NOW: _Roles = (
    ("target", ("email:p150", "chat:p151"), ()),
    ("related", ("email:p30", "chat:p31"), ("superseded",)),
    ("trap", ("chat:p152",), ("wrong_entity",)),
    ("trap", ("chat:p298",), ("recent_irrelevant",)),
)

# (id, text, valid day, key, roles, query-level cases)
_HYBRID_QUERIES: tuple[tuple[str, str, int, str, _Roles, tuple[str, ...]], ...] = (
    ("ana-home-now", "Where does Ana live?", 100, "ana.home", _ANA_NOW, ("redundant_cluster",)),
    (
        "ana-home-history",
        "Where did Ana live at the start of the year?",
        30,
        "ana.home",
        (
            ("target", ("chat:a0", "chat:a2"), ("old_but_exact",)),
            ("trap", ("email:a56", "chat:a80"), ("temporally_wrong",)),
            ("trap", ("chat:a299",), ("recent_irrelevant",)),
        ),
        (),
    ),
    (
        "ana-home-later",
        "Where does Ana live?",
        250,
        "ana.home",
        (
            ("target", ("chat:a200", "chat:a201"), ()),
            (
                "related",
                ("chat:a55", "email:a56", "email:a57", "chat:a80", "import-a81"),
                ("superseded",),
            ),
            ("related", ("chat:a0", "chat:a2"), ()),
            ("trap", ("chat:a299",), ("recent_irrelevant",)),
            ("trap", ("chat:b59",), ("wrong_entity",)),
        ),
        (),
    ),
    ("leo-dose", "How much lisinopril does Leo take?", 100, "leo.dose", _LEO, ()),
    ("kim-flight", "When does Kim fly to Tokyo?", 100, "kim.flight", _KIM, ()),
    (
        "team-meeting",
        "When is the team meeting?",
        100,
        "team.meeting",
        (
            ("target", ("email:t30", "email:t31"), ()),
            ("related", ("chat:t20", "chat:t21"), ("superseded",)),
        ),
        (),
    ),
    (
        "team-meeting-history",
        "When was the team meeting in January?",
        25,
        "team.meeting",
        (
            ("target", ("email:t30", "email:t31"), ("retroactive_correction",)),
            ("related", ("chat:t20", "chat:t21"), ()),
        ),
        (),
    ),
    (
        "sam-dog-before",
        "What is Sam's dog called?",
        50,
        "sam.dog",
        (
            ("target", ("chat:d5", "chat:d6"), ()),
            ("trap", ("chat:d7",), ("wrong_attribute",)),
        ),
        (),
    ),
    (
        "sam-dog-after",
        "What is Sam's dog called?",
        150,
        "sam.dog",
        (
            ("related", ("chat:d5", "chat:d6"), ("retracted",)),
            ("related", ("chat:d90",), ()),
            ("trap", ("chat:d7",), ("wrong_attribute",)),
        ),
        (),
    ),
    ("noah-allergy", "What is Noah allergic to?", 100, "noah.allergy", _NOAH,
     ("redundant_cluster",)),
    ("noah-reworded", "Which food gives Noah a reaction?", 100, "noah.allergy", _NOAH,
     ("paraphrase_low_overlap",)),
    ("priya-now", "Where does Priya work?", 299, "priya.employer", _PRIYA_NOW, ()),
    (
        "priya-history",
        "Where did Priya work in the spring?",
        100,
        "priya.employer",
        (
            ("target", ("email:p30", "chat:p31"), ("old_but_exact",)),
            ("trap", ("email:p150", "chat:p151"), ("temporally_wrong",)),
            ("trap", ("chat:p298",), ("recent_irrelevant",)),
        ),
        (),
    ),
)  # fmt: skip

# (base, perturbation, text, valid day or None (base's), key or None (base's),
#  roles or None (base's))
_HYBRID_VARIANTS: tuple[tuple[str, str, str, int | None, str | None, _Roles | None], ...] = (
    ("ana-home-now", "paraphrase", "What city is Ana's home?", None, None, None),
    ("ana-home-now", "reorder", "Ana lives where?", None, None, None),
    ("ana-home-now", "irrelevant_wording",
     "Quick question before lunch: where does Ana live?", None, None, None),
    ("ana-home-now", "entity", "Where does Ben live?", None, "ben.home",
     (("target", ("chat:b58", "chat:b59"), ()),)),
    ("ana-home-now", "attribute", "Which city does Ana work in?", None, "ana.work_city",
     (("target", ("chat:a60", "chat:a61"), ()),)),
    ("ana-home-now", "time", "Where does Ana live?", 30, None,
     (("target", ("chat:a0", "chat:a2"), ()),)),
    ("ana-home-now", "negation", "Where does Ana not live?", None, None, ()),
    ("noah-allergy", "reorder", "Noah is allergic to what?", None, None, None),
    ("noah-allergy", "irrelevant_wording",
     "Sorry to bother you, but what is Noah allergic to?", None, None, None),
    ("noah-allergy", "negation", "What is Noah not allergic to?", None, None, ()),
    ("noah-allergy", "entity", "What is Ben allergic to?", None, "ben.allergy", ()),
    ("priya-now", "paraphrase", "Who employs Priya?", None, None, None),
    ("priya-now", "irrelevant_wording", "By the way, where does Priya work these days?",
     None, None, None),
    ("priya-now", "entity", "Where does Omar work?", None, "omar.employer",
     (("target", ("chat:p152",), ()),)),
    ("priya-now", "time", "Where does Priya work?", 100, None,
     (("target", ("email:p30", "chat:p31"), ()),)),
    ("leo-dose", "numeric", "Does Leo take 40 mg of lisinopril?", None, None, None),
    ("leo-dose", "reorder", "Leo takes how much lisinopril?", None, None, None),
    ("kim-flight", "numeric", "Does Kim fly to Tokyo on 21 March?", None, None, None),
    ("kim-flight", "paraphrase", "What date is Kim's trip to Japan?", None, None, None),
    ("team-meeting", "reorder", "The team meeting is when?", None, None, None),
)  # fmt: skip

HYBRID_KNOWN_AT = 300


def hybrid_benchmark() -> tuple[Dataset, HybridBenchmark]:
    """The Phase 6 controlled benchmark for hybrid retrieval (not the Phase 15 benchmark).

    45 experiences about seven subjects, ingested as they occur and stored verbatim
    (episodic), mixing statement-language facts (which carry structured claims) with
    free-text notes (which do not). 13 primary queries pin (valid_at, known_at = day 300)
    and a claim key; each judged candidate has a designed role — target (answers at
    valid_at by the Phase 3 epistemic standard: latest occurrence holds, same-instant
    disagreement is contested so both values are targets, corrections are retroactive,
    a forget leaves no target), related (same subject and attribute, not the answer) or
    trap — and may be the focus of an adversarial case. 20 variants perturb primary
    queries (paraphrase, reordering, irrelevant wording; entity, attribute, time,
    negation, number). Roles are the design, fixed before any measurement.
    """
    steps = sorted(
        (_step(d, text, cid) for cid, d, text, _ in _HYBRID_CANDIDATES),
        key=lambda s: (s.recorded_at, s.experience.source),
    )
    dataset = Dataset(name="hybrid-benchmark", version="1", steps=tuple(steps))
    known = day(HYBRID_KNOWN_AT)
    queries = [
        BenchmarkQuery(
            id=qid,
            text=text,
            valid_at=day(valid),
            known_at=known,
            key=key,
            judgements=_judge(roles),
            cases=cases,
        )
        for qid, text, valid, key, roles, cases in _HYBRID_QUERIES
    ]
    primary = {q.id: q for q in queries}
    for base, perturbation, text, valid, key, roles in _HYBRID_VARIANTS:
        b = primary[base]
        queries.append(
            BenchmarkQuery(
                id=f"{base}~{perturbation}",
                text=text,
                valid_at=b.valid_at if valid is None else day(valid),
                known_at=known,
                key=b.key if key is None else key,
                judgements=tuple(j.model_copy(update={"cases": ()}) for j in b.judgements)
                if roles is None
                else _judge(roles),
                base=base,
                perturbation=Perturbation(perturbation),
            )
        )
    return dataset, HybridBenchmark(
        name="hybrid-benchmark",
        version="1",
        dataset=dataset.digest,
        queries=tuple(queries),
        facts=tuple(
            CandidateFact(candidate=cid, fact=fact)
            for cid, _, _, fact in _HYBRID_CANDIDATES
            if fact is not None
        ),
        cases=HYBRID_CASES,
    )


# --- Phase 7 consolidation worlds ------------------------------------------------------------

_ENTITIES = ("Ana", "Ben", "Chen", "Dara", "Eli")
_VALUES = {
    "home": ("Paris", "Berlin", "Munich", "Rome", "Oslo", "Lisbon", "Vienna", "Prague"),
    "employer": ("Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay"),
    "dose": ("5 mg", "10 mg", "20 mg", "40 mg"),
}
_NOTES = {
    "home": ("{e} lives in {v}.", "{e}'s home is {v}."),
    "employer": ("{e} works at {v}.", "{e} is employed by {v}."),
    "dose": ("{e} takes {v} each morning.", "{e} takes {v}."),  # the second drops a qualifier
}
_QUESTIONS = {
    "home": "Where does {e} live?",
    "employer": "Where does {e} work?",
    "dose": "How much does {e} take?",
}
_NEGATED = {
    "home": "{e} does not live in {v}.",
    "employer": "{e} does not work at {v}.",
    "dose": "{e} does not take {v}.",
}


class WorldSpec(Record):
    """Parameters of a generated consolidation world. Every choice is seeded (I21)."""

    name: str = Field(min_length=1)
    seed: int
    entities: int = Field(ge=1, le=len(_ENTITIES))
    attributes: tuple[str, ...] = ("dose", "employer", "home")
    horizon_days: int = Field(ge=10)
    change_every_days: float = Field(gt=0)
    repetition: int = Field(default=2, ge=0)  # extra reports per period
    note_share: float = Field(default=0.5, ge=0, le=1)  # repetitions stated as free text
    contradiction_rate: float = Field(default=0.0, ge=0, le=1)  # same-instant rival reports
    poison: int = Field(default=0, ge=0)  # rival copies per contradiction (unreliable)
    correction_rate: float = Field(default=0.0, ge=0, le=1)  # wrong first, then corrected
    delay_days: float = Field(default=0.0, ge=0)  # ingestion delay (reports arrive late)
    temporary_rate: float = Field(default=0.0, ge=0, le=1)  # brief values that revert
    noise: int = Field(default=0, ge=0)  # irrelevant notes
    negations: bool = False  # negated notes about the previous value after a change
    grid_days: float | None = Field(default=None, gt=0)  # snap change times (coincidences)
    coupled: bool = False  # the first entity's home and employer change together
    probes_per_key: int = Field(default=4, ge=1)


def consolidation_world(spec: WorldSpec) -> tuple[Dataset, ConsolidationBenchmark]:
    """A seeded long-horizon world of changing facts and the reports about them.

    Each key (``<entity>.<attribute>``) has true periods; each period is reported by a
    statement and ``repetition`` restatements (statements or free-text notes), possibly
    delayed, contradicted at the same instant by ``poison`` unreliable reports, reported
    wrongly first and then corrected, or briefly interrupted by a temporary value. Every
    report is labelled with its (key, value, true period); probes carry the key and an
    epistemic expectation computed from the reports (Phase 3 standard).
    """

    def u(*labels: str | int) -> float:
        return unit_interval(spec.seed, spec.name, *labels)

    def pick(options: tuple[str, ...], avoid: str | None, *labels: str | int) -> str:
        choices = [o for o in options if o != avoid]
        return choices[int(u(*labels) * len(choices))]

    def snap(t: float) -> float:
        g = spec.grid_days
        return round(t / g) * g if g else t

    steps: list[Step] = []
    labels: list[FactLabel] = []
    reports: dict[str, list[tuple[datetime, datetime, str, str | None]]] = defaultdict(list)
    counter = [0]

    def report(key: str, occurred: float, recorded: float, content: str, cls: str,
               verb: str, value: str | None, period: int) -> None:  # fmt: skip
        counter[0] += 1
        source = f"{cls}:{spec.name}-{counter[0]}"
        steps.append(
            Step(
                experience=Experience(source=source, content=content, occurred_at=day(occurred)),
                recorded_at=day(recorded),
            )
        )
        if value is not None:
            labels.append(FactLabel(candidate=source, key=key, value=" ".join(normalize(value)),
                                    period=period))  # fmt: skip
        if verb in ("set", "correct", "forget"):
            reports[key].append((day(occurred), day(recorded), verb, value))

    change_times: dict[str, list[float]] = {}
    keys = [(e, a) for e in _ENTITIES[: spec.entities] for a in spec.attributes]
    for e, a in keys:
        key = f"{e.lower()}.{a}"
        partner = f"{_ENTITIES[0].lower()}.home"
        if spec.coupled and e == _ENTITIES[0] and a == "employer" and partner in change_times:
            times = list(change_times[partner])
        else:
            times, t = [0.0], 0.0
            while True:
                t += spec.change_every_days * (0.5 + u(key, "gap", len(times)))
                if t >= spec.horizon_days:
                    break
                times.append(snap(t))
            times = sorted(set(times))
        change_times[key] = times
        value: str | None = None
        for i, start in enumerate(times):
            end = times[i + 1] if i + 1 < len(times) else float(spec.horizon_days)
            previous, value = value, pick(_VALUES[a], value, key, "value", i)
            delay = u(key, "delay", i) * spec.delay_days
            first = value
            if u(key, "wrong", i) < spec.correction_rate:
                first = pick(_VALUES[a], value, key, "wrong-value", i)
            report(key, start, start + delay, f"set {key} = {first}", "chat", "set", first, i)
            if first != value:  # reported wrongly, corrected a little later (retroactively)
                fix = min(end - 0.01, start + 1 + 2 * u(key, "fix", i))
                report(key, fix, fix + delay, f"correct {key} = {value}", "clinic", "correct",
                       value, i)  # fmt: skip
            if u(key, "rival", i) < spec.contradiction_rate:
                rival = pick(_VALUES[a], value, key, "rival-value", i)
                for _ in range(max(1, spec.poison)):
                    report(key, start, start + delay, f"set {key} = {rival}", "forum", "set",
                           rival, i)  # fmt: skip
            for r in range(spec.repetition):
                at = start + (end - start) * (r + 1) / (spec.repetition + 1)
                if u(key, "note", i, r) < spec.note_share:
                    template = _NOTES[a][int(u(key, "template", i, r) * 2)]
                    report(key, at, at + delay, template.format(e=e, v=value), "email", "note",
                           value, i)  # fmt: skip
                else:
                    report(key, at, at + delay, f"set {key} = {value}", "email", "set", value, i)
            if spec.negations and previous is not None:
                at = start + 0.5
                report(key, at, at + delay, _NEGATED[a].format(e=e, v=previous), "chat", "note",
                       None, i)  # fmt: skip
            if u(key, "temporary", i) < spec.temporary_rate and end - start > 10:
                brief = pick(_VALUES[a], value, key, "temporary-value", i)
                at = start + (end - start) / 2
                report(key, at, at + delay, f"set {key} = {brief}", "chat", "set", brief, i)
                back = at + 3
                report(key, back, back + delay, f"set {key} = {value}", "chat", "set", value, i)
    for n in range(spec.noise):
        e = _ENTITIES[int(u("noise-entity", n) * spec.entities)]
        at = u("noise-time", n) * spec.horizon_days
        text = ("{e} enjoyed the weather today.", "{e} called about the weekend plans.")[n % 2]
        report("-", at, at, text.format(e=e), "chat", "note", None, 0)

    probes = []
    for e, a in keys:
        key = f"{e.lower()}.{a}"
        for p in range(spec.probes_per_key):
            valid = u(key, "probe-valid", p) * spec.horizon_days
            known = min(spec.horizon_days + spec.delay_days, valid + u(key, "probe-known", p) * 30)
            expected = epistemic(reports[key], valid_at=day(valid), known_at=day(known))
            probes.append(
                Probe(
                    id=f"{key}-{p}",
                    text=_QUESTIONS[a].format(e=e),
                    valid_at=day(valid),
                    known_at=day(known),
                    expected=expected,
                    key=key,
                )
            )
    steps.sort(key=lambda s: (s.recorded_at, s.experience.source))
    dataset = Dataset(name=f"world:{spec.name}", version=spec.digest, steps=tuple(steps),
                      probes=tuple(probes))  # fmt: skip
    coupled = ((f"{_ENTITIES[0].lower()}.employer", f"{_ENTITIES[0].lower()}.home"),)
    return dataset, ConsolidationBenchmark(
        name=spec.name,
        version=spec.digest,
        dataset=dataset.digest,
        labels=tuple(labels),
        coupled=coupled if spec.coupled else (),
    )


def epistemic(
    reports: list[tuple[datetime, datetime, str, str | None]],
    *,
    valid_at: datetime,
    known_at: datetime,
) -> Expectation:
    """Epistemic truth over (occurred, recorded, verb, value) reports: among reports
    recorded by ``known_at``, the latest occurring by ``valid_at`` holds; a later
    ``correct`` replaces it retroactively (up to the next report); equal-time
    disagreement is contested; ``forget`` leaves nothing (the Phase 3 standard)."""
    visible = sorted((o, v, x) for o, r, v, x in reports if r <= known_at)
    held = [x for x in visible if x[0] <= valid_at]
    if not held:
        return UNKNOWN
    latest = max(o for o, _, _ in held)
    current = [x for x in held if x[0] == latest]
    later = [o for o, v, _ in visible if o > latest and v != "correct"]
    horizon = min(later, default=None)
    fixes = [x for x in visible if x[1] == "correct" and x[0] > latest
             and (horizon is None or x[0] < horizon)]  # fmt: skip
    if fixes:
        last = max(o for o, _, _ in fixes)
        current = [x for x in fixes if x[0] == last]
    if any(v == "forget" for _, v, _ in current):
        return UNKNOWN
    values = {x for _, _, x in current if x is not None}
    return known(values.pop()) if len(values) == 1 else contested(*sorted(values))


PHASE7_WORLDS = (
    WorldSpec(name="baseline", seed=7, entities=3, horizon_days=360, change_every_days=90),
    WorldSpec(name="short", seed=7, entities=3, horizon_days=90, change_every_days=45),
    WorldSpec(name="long", seed=7, entities=3, horizon_days=1080, change_every_days=120),
    WorldSpec(name="contradiction", seed=7, entities=3, horizon_days=360, change_every_days=90,
              contradiction_rate=0.5, correction_rate=0.3),
    WorldSpec(name="drift", seed=7, entities=3, horizon_days=360, change_every_days=25,
              delay_days=15, temporary_rate=0.3),
    WorldSpec(name="repetitive", seed=7, entities=3, horizon_days=360, change_every_days=90,
              repetition=6),
    WorldSpec(name="noisy", seed=7, entities=3, horizon_days=360, change_every_days=90,
              noise=60, delay_days=10),
    WorldSpec(name="adversarial", seed=7, entities=3, horizon_days=360, change_every_days=60,
              repetition=3, contradiction_rate=0.4, poison=3, negations=True, grid_days=30,
              coupled=True),
)  # fmt: skip
