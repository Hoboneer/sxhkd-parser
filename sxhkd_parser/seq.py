"""Classes and functions for expanding sequences of the form {s1,s2,...,sn}.

The utility functions expand_range and expand_sequences expand ranges such as
a-f or 1-6, and sequences, respectively.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, NoReturn, Optional, Tuple, Union, cast

from .errors import SequenceParseError

__all__ = [
    "SequenceSpan",
    "Span",
    "SpanPermutation",
    "SpanTree",
    "TextSpan",
    "expand_range",
    "expand_sequences",
]


@dataclass
class Span:
    """Span of text that separates sequences and non-sequence text.

    This also contains line and column information to allow for exact positions
    if necessary for error messages.

    Instance variables:
        line: the line number.
        col: the column number.
    """

    line: int = -1
    col: int = -1

    @property
    def pos(self) -> Tuple[int, int]:
        """Return line and column as a tuple."""
        return (self.line, self.col)

    @pos.setter
    def pos(self, position: Tuple[int, int]) -> None:
        self.line, self.col = position


@dataclass
class TextSpan(Span):
    """A string of non-sequence text.

    Instance variables:
        text: the string of text.
        is_expansion: whether this span was the result of range expansion, meaning it has no line and col--their values are undefined.
    """

    text: str = ""
    is_expansion: bool = False


@dataclass
class SequenceSpan(Span):
    """A sequence of text that itself contains further `TextSpan` objects.

    Instance variables:
        choices: the list of `TextSpan` objects in order of their appearance.
    """

    choices: List[TextSpan] = field(default_factory=list)


@dataclass
class SpanPermutation:
    """A permutation of some text after sequence expansion.

    Instance variables:
        spans: the list of `TextSpan` objects comprising the permutation--sequences are irrelevant here.
        sequence_choices: the list of choices made at each sequence span, taking `None` at non-sequence spans.
    """

    spans: List[TextSpan]
    # Each element corresponds to the element of the same index in `spans`.
    # `sequence_choices[i] is None` if `spans[i]` is *not* part of a sequence.
    sequence_choices: List[Optional[int]]

    def __str__(self) -> str:
        # Ensure empty sequence elements don't appear at all.
        return "".join(
            span.text if choice is None or span.text != "_" else ""
            for choice, span in zip(self.sequence_choices, self.spans)
        )


@dataclass
class SpanTree:
    """Decision tree representing the spans of text that result from expanding sequences of the form {s1,s2,...,sn}.

    An instance can be parsed from some text by passing it to
    `expand_sequences`.

    Instance variables:
        levels: the levels of the decision tree, each of which is a `Span` object.
    """

    levels: List[Span]

    def __init__(self, levels: List[Span]):
        """Create an instance from a list representing a decision tree.

        Each element of the list represents the levels of the decision tree and
        are stored as `Span` objects, which are, in addition to line and column
        information, essentially:

            - single strings for parts of the original input that weren't
              contained in sequences; or
            - lists of strings for those parts that were.

        Each node on a given level of the decision tree has the same children
        as every other node sharing its level.
        """
        self.levels = levels

    @staticmethod
    def _generate_permutations_rec(
        curr_levels: List[Span],
    ) -> List[SpanPermutation]:
        level, *rest = curr_levels
        # Whether it's the last level or not.
        if not rest:
            if isinstance(level, TextSpan):
                return [SpanPermutation([level], [None])]
            else:
                assert isinstance(level, SequenceSpan)
                return [
                    SpanPermutation([choice], [i])
                    for i, choice in enumerate(level.choices)
                ]
        perms: List[SpanPermutation] = []
        subperms = SpanTree._generate_permutations_rec(rest)
        if isinstance(level, TextSpan):
            for subperm in subperms:
                perms.append(
                    SpanPermutation(
                        [level] + subperm.spans,
                        cast("List[Optional[int]]", [None])
                        + subperm.sequence_choices,
                    )
                )
        else:
            assert isinstance(level, SequenceSpan)
            for i, choice in enumerate(level.choices):
                for subperm in subperms:
                    perms.append(
                        SpanPermutation(
                            [choice] + subperm.spans,
                            cast("List[Optional[int]]", [i])
                            + subperm.sequence_choices,
                        )
                    )
        return perms

    def generate_permutations(self) -> List[SpanPermutation]:
        """Return all the permutations of the text in order.

        Each permutation is the path from the root to a leaf.
        The permutations are ordered such that the leftmost paths come first.
        """
        return SpanTree._generate_permutations_rec(self.levels)

    @staticmethod
    def _print_tree_rec(curr_levels: List[Span], level: int) -> None:
        if not curr_levels:
            return
        curr_level, *rest = curr_levels
        if isinstance(curr_level, TextSpan):
            print(f"{' ' * (level)}└{'─' * (level)} {curr_level.text!r}")
            SpanTree._print_tree_rec(rest, level + 1)
        else:
            assert isinstance(curr_level, SequenceSpan)
            for choice in curr_level.choices:
                print(f"{' ' * (level)}└{'─' * (level)} {choice.text!r}")
                SpanTree._print_tree_rec(rest, level + 1)

    def print_tree(self) -> None:
        """Print the tree."""
        print(None)
        self._print_tree_rec(self.levels, 1)


def expand_range(maybe_range: str) -> Union[List[str], str]:
    """Expand ranges in the form A-Z or 0-9 found in {s1,s2,...,sn} sequences.

    Returns the input unchanged if there was no range to expand, otherwise
    returning the list of all the values in the given range.

    For now, only ranges of the same type work.  e.g., `9-A' doesn't work.
    Also, the alphabetic ranges only work on ascii letters for now.
    """
    m = re.match(r"^([A-Za-z0-9])-([A-Za-z0-9])$", maybe_range)
    if m:
        expanded: List[str] = []
        start, end = m.group(1), m.group(2)
        if start > end:
            raise ValueError(
                f"invalid range (start is past end): {start}-{end}"
            )
        if start in string.ascii_letters and end in string.ascii_letters:
            start_index = string.ascii_letters.index(start)
            end_index = string.ascii_letters.index(end) + 1
            expanded.extend(string.ascii_letters[start_index:end_index])
        elif start in string.digits and end in string.digits:
            expanded.extend(map(str, range(int(start), int(end) + 1)))
        else:
            raise ValueError(f"type mismatch in range: {start}-{end}")
        return expanded
    else:
        return maybe_range


class _SequenceParseMode(Enum):
    """Mode when parsing out the spans formed by expanding {s1,s2,...,sn} sequences."""

    NORMAL = auto()
    NORMAL_ESCAPE_NEXT = auto()
    SEQUENCE = auto()
    SEQUENCE_ESCAPE_NEXT = auto()


def expand_sequences(
    text: Union[str, Iterable[str]],
    start_line: int = 1,
    column_shift: int = 0,
) -> SpanTree:
    """Expand sequences of the form {s1,s2,...,sn} and return its decision tree.

    Allows `text` to be an iterable for each line of the text.
    Note that both types of `text` must not contain newline characters.

    `start_line` may be set so that line numbers match up with the text being
    read from a file.  It is an integer that must be at least 1.

    `column_shift` is how many characters the first column of the *first* line
    of `text` must be shifted so that error messages are accurate.  It must be
    a non-negative integer.  It may be useful for passing in cleaned-up
    `Command` text.
    """
    assert start_line >= 1
    assert column_shift >= 0
    if isinstance(text, str):
        assert "\n" not in text, repr(text)
        lines = [text]
    else:
        lines = list(text)
        assert lines
        assert "\n" not in lines[0], repr(lines[0])
    # Spans of normal text and sequence text (i.e., choices of text).
    spans: List[Optional[Span]] = [None]
    currseqspans: List[Optional[TextSpan]] = [None]
    mode = _SequenceParseMode.NORMAL

    # Only shift for the first row.
    def _get_shifted_column() -> int:
        if row == 0:
            return col + column_shift
        else:
            return col

    def _curr_pos() -> Tuple[int, int]:
        return (start_line + row, _get_shifted_column())

    def _err(msg: str) -> NoReturn:
        pos = _curr_pos()
        err = SequenceParseError(
            msg,
            text=" ".join(lines),
            line=pos[0],
            column=pos[1],
        )
        raise err

    # Operates on the most recent span.
    for row, line in enumerate(lines):
        for col, c in enumerate(line, start=1):
            if mode == _SequenceParseMode.NORMAL:
                if c == "}":
                    _err("Unmatched closing brace")
                elif c == "{":
                    mode = _SequenceParseMode.SEQUENCE
                    # Reuse the slot if unused.
                    if spans[-1] is None:
                        del spans[-1]
                    spans.append(
                        SequenceSpan(
                            *_curr_pos(),
                        )
                    )
                    continue
                elif c == "\\":
                    mode = _SequenceParseMode.NORMAL_ESCAPE_NEXT
                if spans[-1] is None:
                    spans[-1] = TextSpan(*_curr_pos(), text=c)
                else:
                    assert isinstance(spans[-1], TextSpan)
                    spans[-1].text += c
            elif mode == _SequenceParseMode.NORMAL_ESCAPE_NEXT:
                assert isinstance(spans[-1], TextSpan)
                # Sequences can be escaped in normal mode, so the backslash shouldn't remain.
                if c in ("{", "}"):
                    # The last character was a backslash, so replace it.
                    spans[-1].text = spans[-1].text[:-1] + c
                else:
                    spans[-1].text += c
                mode = _SequenceParseMode.NORMAL
            elif mode == _SequenceParseMode.SEQUENCE:
                if c == "{":
                    _err(
                        "No nested sequences allowed (see https://github.com/baskerville/sxhkd/issues/67)"
                    )
                assert isinstance(spans[-1], SequenceSpan)
                if c in ",}":
                    if currseqspans[-1] is None:
                        # TODO: decide whether this undefined behaviour should be allowed
                        currseqspans[-1] = TextSpan(*_curr_pos())
                        continue
                        # _err("Empty sequence elements must use '_'")
                    expanded_seq = expand_range(currseqspans[-1].text)
                    if isinstance(expanded_seq, list):
                        first, *rest = expanded_seq
                        currseqspans[-1] = TextSpan(
                            *currseqspans[-1].pos, first
                        )
                        currseqspans.extend(
                            TextSpan(text=text, is_expansion=True)
                            for text in rest
                        )
                    # If not a list, then no expansion was done.
                    if c == ",":
                        currseqspans.append(None)
                    else:
                        mode = _SequenceParseMode.NORMAL
                        spans[-1].choices = cast(
                            "List[TextSpan]", currseqspans
                        )
                        currseqspans = [None]
                        spans.append(None)
                    continue
                elif c == "\\":
                    mode = _SequenceParseMode.SEQUENCE_ESCAPE_NEXT
                if currseqspans[-1] is None:
                    currseqspans[-1] = TextSpan(*_curr_pos(), text=c)
                else:
                    currseqspans[-1].text += c
            elif mode == _SequenceParseMode.SEQUENCE_ESCAPE_NEXT:
                assert currseqspans[-1] is not None
                # Allow escaping special sequence characters while within a sequence.
                if c in "{},":
                    # The last character was a backslash, so replace it.
                    currseqspans[-1].text = currseqspans[-1].text[:-1] + c
                else:
                    currseqspans[-1].text += c
                mode = _SequenceParseMode.SEQUENCE
    if mode != _SequenceParseMode.NORMAL:
        _err("Input ended while parsing a sequence or escaping a character")
    # Remove unused normal span at the end.
    if spans[-1] is None:
        spans.pop()

    return SpanTree(cast("List[Span]", spans))
