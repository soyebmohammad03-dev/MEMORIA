"""Answer comparison: what an output asserts, and whether that agrees with an expectation.

Values are compared as **token sequences**: Unicode NFKC, case-folded, split into word
tokens (the retrieval tokenizer). Punctuation, spacing and case are not significant;
nothing else is normalised — no stemming, synonyms, or fuzzy similarity. "Berlin" and
"berlin." are equal; "Berlin (contaminated)" and "Berlin" are not.

An output is *read* by the first applicable reader, in the order an evaluation
specifies:

- ``statement``: the output is a statement (``set|correct <key> = <value>``).
- ``assignment``: the output is memory content ``<key> = <value>`` (``key#n`` allowed).
- ``mention``: the output mentions known values (the dataset's vocabulary) as whole
  token sequences, e.g. "My home is Paris". Longer matches absorb shorter ones they
  contain; two or more distinct values make the reading ambiguous.

The reading keeps the original text, the reader used, and the normalised tokens, so
every comparison can be audited.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Self

from pydantic import model_validator

from memoria.core import Expectation, Record
from memoria.formation import parse_statement
from memoria.retrieval import tokenize

READERS = ("statement", "assignment", "mention")
_ASSIGNMENT = re.compile(r"([a-z0-9_.-]+)(?:#[0-9]+)? = (\S(?:.*\S)?)")

Tokens = tuple[str, ...]


def normalize(text: str) -> Tokens:
    return tuple(tokenize(unicodedata.normalize("NFKC", text)))


class ReadingKind(StrEnum):
    ABSTAINED = "abstained"  # no output
    VALUE = "value"  # exactly one value asserted
    AMBIGUOUS = "ambiguous"  # several distinct values mentioned
    UNREADABLE = "unreadable"  # no value can be read (including empty output)


class Reading(Record):
    """What an output asserts, with the evidence for that reading."""

    output: str | None
    kind: ReadingKind
    reader: str | None = None  # the reader that produced the reading
    key: str | None = None  # the key, when the reader identifies one
    values: tuple[str, ...] = ()  # asserted values, original text, in order found
    tokens: tuple[Tokens, ...] = ()  # their normalised forms

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        expected = {
            ReadingKind.ABSTAINED: len(self.values) == 0 and self.output is None,
            ReadingKind.VALUE: len(self.values) == 1,
            ReadingKind.AMBIGUOUS: len(self.values) >= 2,
            ReadingKind.UNREADABLE: len(self.values) == 0 and self.output is not None,
        }
        if not expected[self.kind]:
            raise ValueError(f"{self.kind.value} reading has the wrong number of values")
        if self.tokens != tuple(normalize(v) for v in self.values):
            raise ValueError("tokens must be the normalised values")
        return self


def _contains(haystack: Tokens, needle: Tokens) -> bool:
    k = len(needle)
    return k > 0 and any(haystack[i : i + k] == needle for i in range(len(haystack) - k + 1))


def read(output: str | None, readers: Sequence[str], vocabulary: Iterable[str]) -> Reading:
    """Read ``output`` with the first applicable reader; ``vocabulary`` feeds ``mention``."""
    unknown = set(readers) - set(READERS)
    if unknown:
        raise ValueError(f"unknown readers: {sorted(unknown)}")
    if output is None:
        return Reading(output=None, kind=ReadingKind.ABSTAINED)

    def value(reader: str, key: str, text: str) -> Reading:
        return Reading(
            output=output,
            kind=ReadingKind.VALUE,
            reader=reader,
            key=key,
            values=(text,),
            tokens=(normalize(text),),
        )

    for reader in readers:
        if reader == "statement":
            s = parse_statement(output)
            if s is not None and s.value is not None:
                return value(reader, s.key, s.value)
        elif reader == "assignment":
            m = _ASSIGNMENT.fullmatch(output.strip())
            if m is not None:
                return value(reader, m[1], m[2])
        else:  # mention
            said = normalize(output)
            found = {
                v: normalize(v) for v in sorted(set(vocabulary)) if _contains(said, normalize(v))
            }
            maximal = sorted(
                v
                for v, t in found.items()
                if not any(t != u and _contains(u, t) for u in found.values())
            )
            distinct: dict[Tokens, str] = {}
            for v in maximal:
                distinct.setdefault(found[v], v)
            texts = tuple(distinct.values())
            kind = (
                ReadingKind.UNREADABLE
                if not texts
                else ReadingKind.VALUE
                if len(texts) == 1
                else ReadingKind.AMBIGUOUS
            )
            return Reading(
                output=output,
                kind=kind,
                reader=reader,
                values=texts,
                tokens=tuple(normalize(t) for t in texts),
            )
    return Reading(output=output, kind=ReadingKind.UNREADABLE)


class Agreement(StrEnum):
    MATCH = "match"  # the asserted value is (one of) the expected value(s)
    MISMATCH = "mismatch"  # a value was asserted that is not expected
    ABSTAINED = "abstained"
    AMBIGUOUS = "ambiguous"
    UNREADABLE = "unreadable"


def agree(reading: Reading, expected: Expectation) -> Agreement:
    """Compare a reading with an expectation. ``unknown`` has no values, so any value
    asserted against it is a mismatch; the classifier decides what kind."""
    if reading.kind is not ReadingKind.VALUE:
        return Agreement(reading.kind.value)
    wanted = {normalize(v) for v in expected.values}
    return Agreement.MATCH if reading.tokens[0] in wanted else Agreement.MISMATCH
