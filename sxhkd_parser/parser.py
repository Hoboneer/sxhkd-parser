"""Classes and functions for hotkeys, commands, and keybinds.

The utility functions expand_range and expand_sequences expand ranges such as
a-f or 1-6, and sequences of the form {s1,s2,...,sn}, respectively.

Terminology:
    Hotkey: The sequence of chords needed to activate a command.
    Command: The command passed to the shell after the hotkey is completed.
    Keybind: The entity that encompasses the above.

Hotkey and Command objects may be created directly, but are already done as
part of creating Keybind objects.  The decision tree produced by sequence
expansion can also be accessed with their respective get_tree() methods.
"""
from __future__ import annotations

import itertools as it
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from weakref import ProxyType, proxy

from .errors import (
    ConflictingChainPrefixError,
    DuplicateChordPermutationError,
    DuplicateModifierError,
    HotkeyParseError,
    HotkeyTokenizeError,
    InconsistentKeybindCasesError,
    InconsistentNoabortError,
    NonTerminalStateExitError,
    PossiblyInvalidKeysyms,
    SequenceParseError,
    UnexpectedTokenError,
)
from .keysyms import KEYSYMS

__all__ = [
    "Chord",
    "ChordRunEvent",
    "Command",
    "Hotkey",
    "HotkeyToken",
    "HotkeyTree",
    "Keybind",
    "KeypressTreeChordRunEventNode",
    "KeypressTreeInternalNode",
    "KeypressTreeKeysymNode",
    "KeypressTreeModifierSetNode",
    "KeypressTreeNode",
    "KeypressTreeReplayNode",
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


class ChordRunEvent(Enum):
    """Enum representing the event on which a keybind will be run.

    This is `@' in the syntax.
    """

    KEYPRESS = auto()
    KEYRELEASE = auto()


@dataclass(unsafe_hash=True)
class Chord:
    """Data object representing a key-chord, which is part of a `Hotkey`.

    Instance variables:
        modifiers: the set of modifiers that must precede a chord.
        keysym: the keysym name, given by the output of `xev -event keyboard`.
        run_event: whether the chord (or whole command? TODO) runs on key-press or key-release.
        replay: whether the captured event will be replayed for the other clients.
        noabort: whether the ':' was used to indicate non-abort at chain-tail.
    """

    modifiers: FrozenSet[str]
    keysym: str
    run_event: ChordRunEvent
    replay: bool
    noabort: bool

    # XXX: are `@' and `~' significant enough to cause differing keybinds?
    # answer: yes (https://github.com/baskerville/sxhkd/issues/198)
    def __init__(
        self,
        modifiers: Iterable[str],
        keysym: str,
        run_event: Optional[ChordRunEvent] = None,
        replay: bool = False,
        noabort: bool = False,
    ):
        self.modifiers = frozenset(modifiers)
        self.keysym = keysym
        if run_event is None:
            run_event = ChordRunEvent.KEYPRESS
        self.run_event = run_event
        self.replay = replay
        self.noabort = noabort


# Not really a "node", but a value of a node.
@dataclass
class KeypressTreeModifierSetNode:
    """Value of a `Chord`'s `modifiers` attribute for the `value` of `KeypressTreeNode`."""

    value: FrozenSet[str]


@dataclass
class KeypressTreeKeysymNode:
    """Value of a `Chord`'s `keysym` attribute for the `value` of `KeypressTreeNode`."""

    value: str


@dataclass
class KeypressTreeChordRunEventNode:
    """Value of a `Chord`'s `run_event` attribute for the `value` of `KeypressTreeNode`."""

    value: ChordRunEvent


@dataclass
class KeypressTreeReplayNode:
    """Value of a `Chord`'s `replay` attribute for the `value` of `KeypressTreeNode`."""

    value: bool


KeypressTreeInternalNode = Union[
    KeypressTreeModifierSetNode,
    KeypressTreeKeysymNode,
    KeypressTreeChordRunEventNode,
    KeypressTreeReplayNode,
]


@dataclass
class KeypressTreeNode:
    """Node for a decision tree representing the keypresses needed to complete a hotkey.

    Hotkeys are each represented by a path from root to leaf, with each `Chord`
    object in that path being the key-chords needed to complete it.

    It has internal nodes with names following the pattern `KeypressTree*Node` to group chords
    that share a common feature:
        - KeypressTreeModifierSetNode: *sets* of modifiers, so order doesn't matter (matches behaviour of sxhkd)
        - KeypressTreeKeysymNode: keysym
        - KeypressTreeChordRunEventNode: run on key-press or key-release
        - KeypressTreeReplayNode: whether to replay event to other clients

    Invariants (assuming all node additions and removals are done through the methods):
        - a (chord) node ends a permutation IFF the node has no children.
        - each level has no duplicate nodes EXCEPT when they differ only by
          `noabort` or `ends_permutation`.
    """

    value: Union[Chord, KeypressTreeInternalNode]
    children: List[KeypressTreeNode]
    # Need to be kept in sync with `children`.
    # Maybe use properties?
    keysym_children: Dict[str, KeypressTreeNode]
    modifierset_children: Dict[FrozenSet[str], KeypressTreeNode]
    runevent_children: Dict[ChordRunEvent, KeypressTreeNode]
    replay_children: Dict[bool, KeypressTreeNode]

    # Both are non-None when this (chord) node ends a permutation.
    permutation_index: Optional[int]
    # Reference to the hotkey that this permutation comes from.
    hotkey: Optional[ProxyType[Hotkey]]

    def __init__(self, value: Union[Chord, KeypressTreeInternalNode]):
        self.value = value
        self.children = []
        self.keysym_children = {}
        self.modifierset_children = {}
        self.runevent_children = {}
        self.replay_children = {}

        self.permutation_index = None
        self.hotkey = None

    @property
    def ends_permutation(self) -> bool:
        """Return whether this node ends a hotkey permutation."""
        return self.permutation_index is not None

    def merge_node(self, node: KeypressTreeNode) -> None:
        """Add `node` under this node, ensuring no duplicates.

        NOTE: permutation-ending nodes are left unmerged to maintain the invariant that
              a (chord) node ends a permutation IFF the node has no children.
        """
        if isinstance(node.value, KeypressTreeKeysymNode):
            if node.value.value in self.keysym_children:
                for child in node.children:
                    self.keysym_children[node.value.value].merge_node(child)
            else:
                self.add_child(node)
        elif isinstance(node.value, KeypressTreeModifierSetNode):
            if node.value.value in self.modifierset_children:
                for child in node.children:
                    self.modifierset_children[node.value.value].merge_node(
                        child
                    )
            else:
                self.add_child(node)
        elif isinstance(node.value, KeypressTreeChordRunEventNode):
            if node.value.value in self.runevent_children:
                for child in node.children:
                    self.runevent_children[node.value.value].merge_node(child)
            else:
                self.add_child(node)
        elif isinstance(node.value, KeypressTreeReplayNode):
            if node.value.value in self.replay_children:
                for child in node.children:
                    self.replay_children[node.value.value].merge_node(child)
            else:
                self.add_child(node)
        else:
            assert isinstance(node.value, Chord)
            for rootchild in self.children:
                if not isinstance(rootchild.value, Chord):
                    continue
                # Don't merge permutation-ending chords.
                if rootchild.value == node.value and not (
                    rootchild.ends_permutation or node.ends_permutation
                ):
                    for child in node.children:
                        rootchild.merge_node(child)
                    break
            else:
                self.add_child(node)

    def add_child(
        self,
        value: Union[Chord, KeypressTreeInternalNode, KeypressTreeNode],
        merge: bool = False,
    ) -> KeypressTreeNode:
        """Add an existing node as a child, or create and add a new one with the given value.

        Synchronizes with the relevant `*_children` dicts.  For non-`Chord`
        node inputs, if `merge` is `False`, raises `ValueError` if the child
        already exists in the relevant `*_children` dicts.  if `merge` is
        `True`, calls the `merge_node` method on the new child.
        """
        if isinstance(value, KeypressTreeNode):
            child = value
        else:
            child = KeypressTreeNode(value)
        if isinstance(child.value, Chord):
            self.children.append(child)
            return child

        curr_children: Dict[Any, KeypressTreeNode]
        if isinstance(child.value, KeypressTreeKeysymNode):
            curr_children = self.keysym_children
        elif isinstance(child.value, KeypressTreeModifierSetNode):
            curr_children = self.modifierset_children
        elif isinstance(child.value, KeypressTreeChordRunEventNode):
            curr_children = self.runevent_children
        elif isinstance(child.value, KeypressTreeReplayNode):
            curr_children = self.replay_children
        else:
            raise RuntimeError(
                f"Unhandled internal node type {type(child.value)!r}"
            )

        assert not isinstance(child.value, Chord)
        if merge:
            self.merge_node(child)
        elif child.value.value in curr_children:
            typename = re.match(
                r"^KeypressTree(.+)Node$", type(child.value).__name__
            )
            assert typename is not None
            raise ValueError(
                f"{typename[1]} child '{child.value.value}' already exists at this node."
            )
        else:
            curr_children[child.value.value] = child
            self.children.append(child)
        return child

    def remove_child(self, index: int) -> KeypressTreeNode:
        """Remove the child at `index` in `children`.

        Synchronizes with the relevant `*_children` dicts.
        """
        child = self.children.pop(index)
        if isinstance(child.value, KeypressTreeKeysymNode):
            del self.keysym_children[child.value.value]
        elif isinstance(child.value, KeypressTreeModifierSetNode):
            del self.modifierset_children[child.value.value]
        elif isinstance(child.value, KeypressTreeChordRunEventNode):
            del self.runevent_children[child.value.value]
        elif isinstance(child.value, KeypressTreeReplayNode):
            del self.replay_children[child.value.value]
        return child

    def _find_permutation_ends_rec(
        self: KeypressTreeNode,
    ) -> List[KeypressTreeNode]:
        perm_ends = []
        if self.ends_permutation:
            perm_ends.append(self)
        if not self.children:
            return perm_ends
        for child in self.children:
            perm_ends.extend(child._find_permutation_ends_rec())
        return perm_ends

    def find_permutation_ends(self) -> List[KeypressTreeNode]:
        """Return all the permutation-ending nodes from this node, excluding it."""
        perm_ends = []
        for child in self.children:
            perm_ends.extend(child._find_permutation_ends_rec())
        return perm_ends

    def _print_tree_rec(self, level: int) -> None:
        assert level >= 0
        if level == 0:
            print(repr(self.value))
        else:
            if self.ends_permutation:
                assert self.hotkey is not None
                perm = self.hotkey.permutations[self.permutation_index]
                print(
                    f"{' ' * (level-1)}└{'─' * (level-1)} {self.value!r} i={self.permutation_index}, hotkey={Hotkey.static_hotkey_str(perm, self.hotkey.noabort_index)!r}"
                )
            else:
                print(f"{' ' * (level-1)}└{'─' * (level-1)} {self.value!r}")

        for child in self.children:
            child._print_tree_rec(level + 1)

    def print_tree(self) -> None:
        """Print the tree rooted at this node."""
        self._print_tree_rec(0)


@dataclass
class HotkeyTree:
    """The decision tree for a single hotkey or multiple hotkeys.

    Instance variables:
        root: the root `KeypressTreeNode`.
        internal_nodes: the list of internal node types included in the tree.

    Internal node types:
        keysym: KeypressTreeKeysymNode
        modifierset: KeypressTreeModifierSetNode
        runevent: KeypressTreeChordRunEventNode
        replay: KeypressTreeReplayNode
    """

    root: KeypressTreeNode
    internal_nodes: List[str]
    INTERNAL_NODE_TYPES: ClassVar[Set[str]] = {
        "keysym",
        "modifierset",
        "runevent",
        "replay",
    }

    def __init__(self, internal_nodes: Optional[List[str]] = None):
        """Create a hotkey tree with the given list of internal node types.

        The order in which internal node types are included determines the
        grouping.  For example, the order (1) modifierset, (2) keysym results in
        modifierset nodes being closest to the root node, and vice versa with the
        reverse order.
        """
        self.root = KeypressTreeNode(None)  # type: ignore
        if internal_nodes is None:
            internal_nodes = []
        for nodetype in internal_nodes:
            if nodetype not in HotkeyTree.INTERNAL_NODE_TYPES:
                raise ValueError(f"Invalid nodetype '{nodetype}'")
        self.internal_nodes = internal_nodes

    def print_tree(self) -> None:
        """Print the tree starting from its root."""
        self.root.print_tree()

    def merge_hotkey(self, hotkey: Hotkey) -> None:
        """Merge the tree of chord permutations for `hotkey` into this tree."""
        for i, perm in enumerate(hotkey.permutations):
            subtree = self._create_subtree_from_chord_chain(perm, i, hotkey)
            self.root.add_child(subtree, merge=True)

    def merge_tree(self, tree: HotkeyTree) -> None:
        """Merge `tree` into this tree."""
        if self.internal_nodes != tree.internal_nodes:
            raise ValueError(
                f"Trees had different internal node types ({self.internal_nodes} vs {tree.internal_nodes}). Not merging."
            )
        for child in tree.root.children:
            self.root.merge_node(child)

    def _create_subtree_from_chord_chain(
        self, perm: List[Chord], index: int, hotkey: Hotkey
    ) -> KeypressTreeNode:
        assert perm, "got non-empty permutation"
        node_values: List[Union[Chord, KeypressTreeInternalNode]] = []
        for chord in perm:
            for nodetype in self.internal_nodes:
                if nodetype == "keysym":
                    node_values.append(KeypressTreeKeysymNode(chord.keysym))
                elif nodetype == "modifierset":
                    # No null sets.
                    if chord.modifiers:
                        node_values.append(
                            KeypressTreeModifierSetNode(chord.modifiers)
                        )
                elif nodetype == "runevent":
                    node_values.append(
                        KeypressTreeChordRunEventNode(chord.run_event)
                    )
                elif nodetype == "replay":
                    node_values.append(KeypressTreeReplayNode(chord.replay))
                else:
                    raise RuntimeError(f"invalid nodetype '{nodetype}'")
            node_values.append(chord)

        root = new_node = KeypressTreeNode(node_values[0])
        for value in node_values[1:]:
            # No need to merge as this is just one permutation.
            new_node = new_node.add_child(value)
        new_node.permutation_index = index
        new_node.hotkey = proxy(hotkey)
        assert isinstance(new_node.value, Chord)
        return root

    @staticmethod
    def _find_duplicate_chords_rec(
        node: KeypressTreeNode,
    ) -> List[List[KeypressTreeNode]]:
        if not node.children:
            return []

        # Take only chord nodes and group them by their value.
        groups: DefaultDict[Chord, List[KeypressTreeNode]] = defaultdict(list)
        for child in node.children:
            if isinstance(child.value, Chord) and child.ends_permutation:
                groups[child.value].append(child)
        dups = list(
            dupnodes for dupnodes in groups.values() if len(dupnodes) > 1
        )

        for child in node.children:
            dups.extend(HotkeyTree._find_duplicate_chords_rec(child))
        return dups

    def find_duplicate_chord_nodes(self) -> List[List[KeypressTreeNode]]:
        """Return duplicate chord nodes, with each sublist representing a set of duplicates.

        Each KeypressTreeNode instance returned ends a permutation.
        """
        return HotkeyTree._find_duplicate_chords_rec(self.root)

    @staticmethod
    def _find_conflicting_chain_prefixes_rec(
        node: KeypressTreeNode,
    ) -> List[Tuple[KeypressTreeNode, List[KeypressTreeNode]]]:
        if not node.children:
            return []

        conflicts = []
        # Check for conflicts between chord nodes with the same value of `noabort`.
        groups: DefaultDict[Chord, List[KeypressTreeNode]] = defaultdict(list)
        for child in node.children:
            if isinstance(child.value, Chord):
                groups[child.value].append(child)
        for matches in groups.values():
            assert matches
            prefix_perm_ends: List[KeypressTreeNode] = []
            longer_chains: List[KeypressTreeNode] = []
            for child in matches:
                assert isinstance(child.value, Chord)
                if child.ends_permutation:
                    assert (
                        not child.children
                    ), "got a permutation-ending node with children"
                    prefix_perm_ends.append(child)
                else:
                    longer_chains.extend(child.find_permutation_ends())
            if longer_chains:
                for prefix in prefix_perm_ends:
                    conflicts.append((prefix, longer_chains))

        # Now check for conflicts between noabort and non-noabort chords.
        noabort_groups: DefaultDict[
            Chord, List[KeypressTreeNode]
        ] = defaultdict(list)
        for child in node.children:
            if isinstance(child.value, Chord):
                noabort_groups[dc_replace(child.value, noabort=False)].append(
                    child
                )
        for noabort_matches in noabort_groups.values():
            assert noabort_matches
            noaborts: List[KeypressTreeNode] = []
            normals: List[KeypressTreeNode] = []
            for child in noabort_matches:
                assert isinstance(child.value, Chord)
                if child.value.noabort:
                    assert (
                        child.children
                    ), "got a noabort chord node without children"
                    noaborts.extend(child.find_permutation_ends())
                elif child.ends_permutation:
                    normals.append(child)
                else:
                    normals.extend(child.find_permutation_ends())
            # It's more natural to think of non-noabort chords conflicting with noabort ones.
            if noaborts:
                for normal in normals:
                    conflicts.append((normal, noaborts))

        for child in node.children:
            conflicts.extend(
                HotkeyTree._find_conflicting_chain_prefixes_rec(child)
            )

        return conflicts

    def find_conflicting_chain_prefixes(
        self,
    ) -> List[Tuple[KeypressTreeNode, List[KeypressTreeNode]]]:
        """Return pairs of conflicting chord chain prefixes and the permutation-ending nodes under them.

        Each KeypressTreeNode instance returned ends a permutation, including
        the first item of each pair (as they would not conflict otherwise).
        """
        return HotkeyTree._find_conflicting_chain_prefixes_rec(self.root)


class _HotkeyParseMode(Enum):
    MODIFIER_NAME = auto()  # initial state
    MODIFIER_CONNECTIVE = auto()
    KEYSYM_GOT_ATSIGN = auto()
    KEYSYM_GOT_TILDE = auto()
    KEYSYM_NAME = auto()
    CHORD = auto()  # terminal state


# TODO: add line and col number (how though? they're pre-expanded?)
@dataclass
class HotkeyToken:
    """Token used for parsing hotkeys.

    Generated by the static method Hotkey.tokenize_static_hotkey.

    See Hotkey.TOKEN_SPEC for the token types.
    """

    type: str
    value: str


@dataclass
class Hotkey:
    """The hotkey for a keybind, containing the sequence of chords needed to execute the keybind.

    Call the get_tree() method to get the decision tree produced by sequence
    expansion, with fully parsed chord sequences.

    Instance variables:
        raw: the unexpanded hotkey text.
        line: the starting line number.
        span_tree: the `SpanTree` instance for the decision tree of the hotkey text, resulting from sequence expansion.
        permutations: all possible choices of chord chains, resulting from sequence expansion.
        noabort_index: the index of the chord which had ":" used after it to indicate noabort.
    """

    MODIFIERS: ClassVar[Set[str]] = {
        "super",
        "hyper",
        "meta",
        "alt",
        "control",
        "ctrl",
        "shift",
        "mode_switch",
        "lock",
        "mod1",
        "mod2",
        "mod3",
        "mod4",
        "mod5",
        "any",
    }
    TOKEN_SPEC: ClassVar[List[Tuple[str, str]]] = [
        ("SEMICOLON", r";"),
        ("COLON", r":"),
        ("PLUS", r"\+"),
        ("MODIFIER", "|".join(mod for mod in MODIFIERS)),
        ("ATSIGN", r"@"),
        ("TILDE", r"~"),
        # This seems good enough.
        ("KEYSYM", r"[A-Za-z0-9_]+"),
        ("WHITESPACE", r"[ \t]+"),
        ("MISMATCH", r"."),
    ]
    TOKENIZER_RE: ClassVar[str] = "|".join(
        "(?P<%s>%s)" % pair for pair in TOKEN_SPEC
    )

    raw: Union[str, List[str]]
    line: Optional[int]
    span_tree: SpanTree = field(repr=False)
    permutations: List[List[Chord]] = field(repr=False)
    noabort_index: Optional[int]
    keybind: Optional[ProxyType[Keybind]]

    def __init__(
        self,
        hotkey: Union[str, List[str]],
        line: Optional[int] = None,
        check_duplicate_permutations: bool = True,
        check_conflicting_permutations: bool = True,
        check_maybe_invalid_keysyms: bool = False,
    ):
        """Create an instance with the hotkey text and the starting line number.

        `hotkey` is passed straight to `expand_sequences` along with `line`,
        and the output used to tokenize and then parse into chord permutations
        due to sequence expansion.

        If `line` is `None`, assume it starts at line 1: user code can
        interpret it as they see fit.
        """
        self.raw = hotkey
        self.line = line
        # To be set by `Keybind` constructor.
        self.keybind = None

        # It's okay if the error messages say it's at line 1:
        # since it's at line 1 of the input anyway.
        self.span_tree = expand_sequences(hotkey, start_line=self.line or 1)

        self.permutations = []
        seen_chords: Dict[Tuple[Chord, ...], Tuple[int, Optional[int]]] = {}
        prev_perm: List[Chord]

        for i, flat_perm in enumerate(self.span_tree.generate_permutations()):
            tokens = Hotkey.tokenize_static_hotkey(str(flat_perm), self.line)
            try:
                noabort_index, chords = Hotkey.parse_static_hotkey(tokens)
            except HotkeyParseError as e:
                if isinstance(e, DuplicateModifierError):
                    e.message = f"{e.message} in '{flat_perm}'"
                e.line = self.line
                raise
            if i == 0:
                prev_perm = chords
                self.noabort_index = noabort_index
                seen_chords[tuple(chords)] = (i, noabort_index)
                self.permutations.append(chords)
                continue

            if noabort_index != self.noabort_index:
                if isinstance(self.raw, str):
                    raw_hotkey = self.raw
                else:
                    raw_hotkey = " ".join(self.raw)
                hotkey_str1 = Hotkey.static_hotkey_str(
                    prev_perm, self.noabort_index
                )
                hotkey_str2 = Hotkey.static_hotkey_str(chords, noabort_index)
                raise InconsistentNoabortError(
                    f"Noabort indicated in different places among permutations for '{raw_hotkey}': '{hotkey_str1}' vs '{hotkey_str2}'",
                    perm1=prev_perm,
                    index1=self.noabort_index,
                    perm2=chords,
                    index2=noabort_index,
                    line=line,
                )

            if check_duplicate_permutations and tuple(chords) in seen_chords:
                assert noabort_index == seen_chords[tuple(chords)][1]
                raise DuplicateChordPermutationError(
                    f"Duplicate permutation '{Hotkey.static_hotkey_str(chords, noabort_index)}'",
                    dup_perm=tuple(chords),
                    perm1=(i, noabort_index),
                    perm2=seen_chords[tuple(chords)],
                    line=line,
                )
            seen_chords[tuple(chords)] = (i, noabort_index)
            self.permutations.append(chords)
            prev_perm = chords

        if check_conflicting_permutations:
            tree = self.get_tree(["modifierset"])
            for (
                prefix,
                conflicts,
            ) in tree.find_conflicting_chain_prefixes():
                conflicts_str = []
                for conflict in conflicts:
                    assert conflict.hotkey is not None
                    assert conflict.permutation_index is not None
                    chords = self.permutations[conflict.permutation_index]
                    hk_str = Hotkey.static_hotkey_str(
                        chords, self.noabort_index
                    )
                    conflicts_str.append(f"'{hk_str}'")

                assert prefix.permutation_index is not None
                chords = self.permutations[prefix.permutation_index]
                chain_hk_str = Hotkey.static_hotkey_str(
                    chords, self.noabort_index
                )
                # Fail on the first conflict.
                raise ConflictingChainPrefixError(
                    f"'{chain_hk_str}' conflicts with {', '.join(conflicts_str)}",
                    chain_prefix=prefix,
                    conflicts=conflicts,
                    line=line,
                )

        if check_maybe_invalid_keysyms:
            maybe_invalid_keysyms = set()
            for perm in self.permutations:
                for chord in perm:
                    if chord.keysym not in KEYSYMS:
                        maybe_invalid_keysyms.add(chord.keysym)
            if maybe_invalid_keysyms:
                keysym_str = " ,".join(f"'{k}'" for k in maybe_invalid_keysyms)
                raise PossiblyInvalidKeysyms(
                    f"Possibly invalid keysyms: {keysym_str}",
                    keysyms=maybe_invalid_keysyms,
                    line=line,
                )

    @property
    def noabort(self) -> bool:
        """Return whether the chain won't be aborted when the chain tail is reached (`:').

        Synchronized with `noabort_index`.
        """
        return self.noabort_index is not None

    def get_tree(
        self, internal_nodes: Optional[List[str]] = None
    ) -> HotkeyTree:
        """Return the decision tree resulting from the permutations."""
        if internal_nodes is None:
            internal_nodes = []
        tree = HotkeyTree(internal_nodes)
        tree.merge_hotkey(self)
        return tree

    @staticmethod
    def static_hotkey_str(
        chain: List[Chord], noabort_index: Optional[int] = None
    ) -> str:
        """Return the string representation of the chord chain.

        `noabort_index` is the index of the chord where ':' is used, if present.
        """
        hotkey = ""
        for i, chord in enumerate(chain):
            keysym_prefix = ""
            if chord.replay:
                keysym_prefix += "~"
            if chord.run_event == ChordRunEvent.KEYRELEASE:
                keysym_prefix += "@"
            hotkey += " + ".join(
                it.chain(
                    sorted(chord.modifiers), [keysym_prefix + chord.keysym]
                )
            )
            if noabort_index is not None and i == noabort_index:
                hotkey += ": "
            elif i == len(chain) - 1:
                # last chord: no ';'
                continue
            else:
                hotkey += "; "
        return hotkey

    @staticmethod
    def tokenize_static_hotkey(
        hotkey: str, line: Optional[int] = None
    ) -> List[HotkeyToken]:
        """Tokenize a hotkey with pre-expanded {s1,s2,...,sn} sequences.

        Setting `line` to an int will include it in any error messages.
        It should be the starting line number of the hotkey.
        """
        tokens = []
        for m in re.finditer(Hotkey.TOKENIZER_RE, hotkey):
            type_ = m.lastgroup
            value = m.group()
            assert type_ is not None
            assert value is not None
            if type_ == "WHITESPACE":
                continue
            elif type_ == "MISMATCH":
                msg = f"Encountered unexpected value {value!r} for hotkey '{hotkey}'"
                raise HotkeyTokenizeError(
                    msg, hotkey=hotkey, value=value, line=line
                )
            tokens.append(HotkeyToken(type_, value))
        return tokens

    @staticmethod
    def parse_static_hotkey(
        tokens: List[HotkeyToken],
    ) -> Tuple[Optional[int], List[Chord]]:
        """Parse a hotkey with pre-expanded {s1,s2,...,sn} sequences.

        Returns noabort_index and the sequence of chords.

        Based on the informal grammar from the sxhkd 0.6.2 manual on Debian Bullseye.
        """
        noabort_index: Optional[int] = None
        chords: List[Chord] = []

        # Temporary data for the current chord.
        curr_modifiers: List[str] = []
        seen_modifiers: Set[str] = set()
        curr_keysym: Optional[str] = None
        curr_run_event: Optional[ChordRunEvent] = None
        curr_replay: bool = False

        def reset_temp_state() -> None:
            nonlocal curr_modifiers, curr_keysym, curr_run_event, curr_replay
            curr_modifiers = []
            seen_modifiers.clear()
            curr_keysym = None
            curr_run_event = None
            curr_replay = False

        def MODIFIER_NAME_on_MODIFIER(tok: HotkeyToken) -> None:
            if tok.value in seen_modifiers:
                raise DuplicateModifierError(
                    f"Duplicate modifier '{tok.value}'", modifier=tok.value
                )
            seen_modifiers.add(tok.value)
            curr_modifiers.append(tok.value)

        # We know that getting a keysym means completing a chord.
        def on_KEYSYM(tok: HotkeyToken) -> None:
            nonlocal curr_keysym
            curr_keysym = tok.value
            chords.append(
                Chord(
                    curr_modifiers,
                    curr_keysym,
                    curr_run_event,
                    curr_replay,
                )
            )
            reset_temp_state()

        def on_ATSIGN(tok: HotkeyToken) -> None:
            nonlocal curr_run_event
            curr_run_event = ChordRunEvent.KEYRELEASE

        def on_TILDE(tok: HotkeyToken) -> None:
            nonlocal curr_replay
            curr_replay = True

        def CHORD_on_COLON(tok: HotkeyToken) -> None:
            nonlocal noabort_index
            if noabort_index is None:
                # this runs after receiving a keysym, which creates a chord
                noabort_index = len(chords) - 1
                chords[-1].noabort = True
            else:
                raise UnexpectedTokenError(
                    "Got a second COLON token",
                    token=tok,
                    mode=mode,
                    transitions=transition_table,
                    tokens=tokens,
                )

        # Transitions from state-to-state based on received token,
        # with their callback functions upon transition.
        STATE_TABLE: Dict[
            _HotkeyParseMode,
            Dict[str, Tuple[_HotkeyParseMode, Callable[[HotkeyToken], None]]],
        ]
        STATE_TABLE = {
            _HotkeyParseMode.MODIFIER_NAME: {
                "MODIFIER": (
                    _HotkeyParseMode.MODIFIER_CONNECTIVE,
                    MODIFIER_NAME_on_MODIFIER,
                ),
                "KEYSYM": (_HotkeyParseMode.CHORD, on_KEYSYM),
                "ATSIGN": (_HotkeyParseMode.KEYSYM_GOT_ATSIGN, on_ATSIGN),
                "TILDE": (_HotkeyParseMode.KEYSYM_GOT_TILDE, on_TILDE),
            },
            _HotkeyParseMode.MODIFIER_CONNECTIVE: {
                "PLUS": (_HotkeyParseMode.MODIFIER_NAME, lambda tok: None),
            },
            _HotkeyParseMode.KEYSYM_GOT_ATSIGN: {
                "TILDE": (_HotkeyParseMode.KEYSYM_NAME, on_TILDE),
                "KEYSYM": (_HotkeyParseMode.CHORD, on_KEYSYM),
            },
            _HotkeyParseMode.KEYSYM_GOT_TILDE: {
                "ATSIGN": (_HotkeyParseMode.KEYSYM_NAME, on_ATSIGN),
                "KEYSYM": (_HotkeyParseMode.CHORD, on_KEYSYM),
            },
            _HotkeyParseMode.KEYSYM_NAME: {
                "KEYSYM": (_HotkeyParseMode.CHORD, on_KEYSYM),
            },
            _HotkeyParseMode.CHORD: {
                "SEMICOLON": (
                    _HotkeyParseMode.MODIFIER_NAME,
                    lambda tok: None,
                ),
                "COLON": (_HotkeyParseMode.MODIFIER_NAME, CHORD_on_COLON),
            },
        }

        mode = _HotkeyParseMode.MODIFIER_NAME
        for token in tokens:
            try:
                transition_table = STATE_TABLE[mode]
            except KeyError as e:
                raise RuntimeError(
                    f"Unhandled mode '{mode.name}'! this shouldn't happen!"
                ) from e
            try:
                next_mode, callback = transition_table[token.type]
            except KeyError as e:
                raise UnexpectedTokenError(
                    f"{mode.name} parser state expected token out of {list(transition_table.keys())} but got {token!r} from {tokens!r}",
                    token=token,
                    mode=mode,
                    transitions=transition_table,
                    tokens=tokens,
                ) from e
            else:
                callback(token)
                mode = next_mode
        if mode != _HotkeyParseMode.CHORD:
            raise NonTerminalStateExitError(
                f"Input ended on parser state {mode.name}: the only terminal state is CHORD",
                mode=mode,
            )

        return (noabort_index, chords)


@dataclass
class Command:
    """The command for a keybind, containing the text to be executed in the shell.

    Call the get_tree() method to get the decision tree produced by sequence expansion.

    Instance variables:
        raw: the unexpanded and unprocessed command text.
        line: the starting line number.
        span_tree: the `SpanTree` instance for the decision tree of the command text, resulting from sequence expansion.
        permutations: all possible choices of command text, resulting from sequence expansion.
        synchronous: whether the command should be executed synchronously or asynchronously.
    """

    raw: Union[str, List[str]]
    line: Optional[int]
    span_tree: SpanTree = field(repr=False)
    permutations: List[str] = field(repr=False)
    synchronous: bool

    def __init__(
        self, command: Union[str, List[str]], line: Optional[int] = None
    ):
        """Create an instance with the command text and the starting line number.

        After some processing to strip any leading whitespace from the first
        line of `command` and to determine whether the command is synchronous
        or not, `command` is passed to `expand_sequences` along with `line`.

        If `line` is `None`, assume it starts at line 1: user code can
        interpret it as they see fit.
        """
        self.line = line
        if isinstance(command, str):
            self.raw = command
            command = command.lstrip()
            col_shift = len(self.raw) - len(command)
            if command[0] == ";":
                self.synchronous = True
                command = command[1:]
                col_shift += 1
            else:
                self.synchronous = False
        else:
            self.raw = command.copy()
            command[0] = command[0].lstrip()
            col_shift = len(self.raw[0]) - len(command[0])
            if command[0][0] == ";":
                self.synchronous = True
                command[0] = command[0][1:]
                col_shift += 1
            else:
                self.synchronous = False

        self.span_tree = root = expand_sequences(
            command, start_line=line or 1, column_shift=col_shift
        )

        self.permutations = [
            str(perm) for perm in root.generate_permutations()
        ]

    def get_tree(self) -> SpanTree:
        """Return the decision tree resulting from sequence expansion."""
        return self.span_tree


@dataclass
class Keybind:
    """A hotkey and its associated command, along with any metadata.

    Instance variables:
        hotkey: the `Hotkey` object representing the keypresses to activate this keybind.
        command: the `Command` object storing the text to be executed when the hotkey is completed.
        metadata: the dictionary of metadata parsed from comments immediately above the keybind.
    """

    hotkey: Hotkey
    command: Command
    metadata: Dict[str, Any]

    def __init__(
        self,
        hotkey: Union[str, List[str]],
        command: Union[str, List[str]],
        hotkey_start_line: Optional[int] = None,
        command_start_line: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hotkey_errors: Optional[Mapping[str, bool]] = None,
    ):
        """Create an instance with the the hotkey and command text.

        `hotkey` and `hotkey_start_line` are directly passed to the constructor
        for `Hotkey`, and `command` and `command_start_line` are directly
        passed to that of `Command`.

        If the hotkey and command differ in the number of cases/permutations
        after sequence expansion occurs, InconsistentKeybindCasesError is raised.
        """
        if metadata is None:
            metadata = {}
        if hotkey_errors is None:
            hotkey_errors = {}
        self.metadata = metadata

        self.hotkey: Hotkey = Hotkey(
            hotkey,
            line=hotkey_start_line,
            **{f"check_{k}": v for k, v in hotkey_errors.items()},
        )
        self.hotkey.keybind = proxy(self)
        self.command: Command = Command(command, line=command_start_line)

        hotkey_cases = len(self.hotkey.permutations)
        command_cases = len(self.command.permutations)
        if hotkey_cases != command_cases:
            raise InconsistentKeybindCasesError(
                f"Inconsistent number of cases: hotkey-cases={hotkey_cases}, command-cases={command_cases}",
                hotkey_cases=hotkey_cases,
                command_cases=command_cases,
                line=hotkey_start_line,
            )

    @property
    def line(self) -> Optional[int]:
        """Return the starting line of the keybind, which is that of its hotkey."""
        return self.hotkey.line
