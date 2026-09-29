import pytest
from pydantic import ValidationError

from memoria.sources import (
    SourceClass,
    SourceModel,
    TrustLedger,
    beta_mean,
    beta_variance,
    independence,
    source_class,
    spearman,
    trust_error,
)

MODEL = SourceModel(
    name="t",
    version="1",
    classes=(
        SourceClass(name="clinic", alpha=9, beta=1),
        SourceClass(name="forum", alpha=2, beta=8),
    ),
    copies=(("bot:c1", "forum:o"), ("bot:c2", "bot:c1")),
)


def test_source_class_needs_a_structured_id() -> None:
    assert source_class("clinic:a1") == "clinic"
    assert source_class("clinic") == ""
    assert source_class(":x") == ""


def test_copy_chain_gives_root_and_depth() -> None:
    assert MODEL.root("bot:c2") == "forum:o"
    assert MODEL.root("forum:o") == "forum:o"
    assert [MODEL.depth(s) for s in ("forum:o", "bot:c1", "bot:c2")] == [1, 2, 3]


def test_declared_model_is_closed_and_acyclic() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        SourceModel(name="t", version="1", copies=(("a:1", "b:1"), ("b:1", "a:1")))
    with pytest.raises(ValidationError, match="one origin"):
        SourceModel(name="t", version="1", copies=(("a:1", "b:1"), ("a:1", "c:1")))
    with pytest.raises(ValidationError, match="unique and sorted"):
        SourceModel(
            name="t", version="1",
            classes=(
                SourceClass(name="b", alpha=1, beta=1),
                SourceClass(name="a", alpha=1, beta=1),
            ),
        )  # fmt: skip


def test_undeclared_classes_are_uninformative_never_authoritative() -> None:
    assert MODEL.prior("chat:x") == (1.0, 1.0)
    assert beta_mean(*MODEL.prior("chat:x")) == 0.5
    assert beta_mean(*MODEL.prior("clinic:a")) == 0.9
    assert beta_variance(1, 1) > beta_variance(90, 10)  # more evidence, less uncertainty


def test_copies_are_not_independent_evidence() -> None:
    naive = ["forum:o", "bot:c1", "bot:c2", "forum:o"]
    r = independence(naive, MODEL)
    assert (r.naive, r.independent, r.duplicates) == (4, 1, 3)
    assert r.max_depth == 3
    mixed = independence([*naive, "clinic:a1", "clinic:a2"], MODEL)
    assert (mixed.naive, mixed.independent) == (6, 3)  # two clinic sources are two roots
    assert independence([], MODEL).independent == 0


def test_trust_counts_are_credits_that_can_move_and_be_withdrawn() -> None:
    t = TrustLedger(MODEL)
    for i in range(4):
        t.register(f"r{i}", "forum:f1")
    cold = t.record("forum:f1")
    assert (cold.support, cold.contradict, cold.circular) == (0, 0, False)
    assert cold.mean == 0.2  # the declared prior alone
    for i in range(3):
        t.credit(f"r{i}", 1)
    t.credit("r3", -1)
    warm = t.record("forum:f1")
    assert (warm.support, warm.contradict, warm.reports) == (3, 1, 4)
    assert warm.circular  # learned from the system's own beliefs: always labelled
    assert warm.mean == pytest.approx((2 + 3) / (10 + 4))
    t.credit("r0", -1)  # a later resolution moves the credit
    assert t.record("forum:f1").support == 2
    t.forget("r0")  # withdrawn evidence stops counting either way
    assert t.record("forum:f1").reports == 3
    assert t.record("forum:f1", learned=False).mean == 0.2  # declared-only view
    assert len(t.snapshot()) == 1


def test_spearman_and_trust_error() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2], [1, 2]) is None
    mae, rho = trust_error({"a": 0.9, "b": 0.5, "c": 0.3}, {"a": 0.95, "b": 0.6, "c": 0.4})
    assert mae == pytest.approx(0.0833333, abs=1e-6)
    assert rho == 1.0
    assert trust_error({}, {"a": 1.0}) == (None, None)
