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
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from .errors import (
    HotkeyTokenizeError,
    InconsistentNoabortError,
    NonTerminalStateExitError,
    SequenceParseError,
    UnexpectedTokenError,
)

__all__ = [
    "Chord",
    "ChordRunEvent",
    "Command",
    "Hotkey",
    "HotkeyToken",
    "Keybind",
    "KeypressTreeChordRunEventNode",
    "KeypressTreeInternalNode",
    "KeypressTreeKeysymNode",
    "KeypressTreeModifierNode",
    "KeypressTreeModifierSetNode",
    "KeypressTreeNode",
    "KeypressTreeReplayNode",
    "SpanTree",
    "expand_range",
    "expand_sequences",
]


@dataclass
class SpanTree:
    """Decision tree representing the spans of text that result from expanding sequences of the form {s1,s2,...,sn}.

    An instance can be parsed from some text by passing it to
    `expand_sequences`.

    Instance variables:
        levels: the levels of the decision tree, each of which is either a string or a list of strings.
    """

    levels: List[Union[str, List[str]]]

    def __init__(self, levels: List[Union[str, List[str]]]):
        """Create an instance from a list representing a decision tree.

        Each element of the list represents the levels of the decision tree,
        which may be single strings for parts of the original input that
        weren't contained in sequences, or lists of strings for those parts
        that were.

        For sxhkdrc sequences, each node on a given level of the decision tree
        has the same children as every other node sharing its level.
        """
        self.levels = levels

    @staticmethod
    def _generate_permutations_rec(
        curr_levels: List[Union[str, List[str]]]
    ) -> List[str]:
        level, *rest = curr_levels
        # Whether it's the last level or not.
        if not rest:
            if isinstance(level, str):
                return [level]
            else:
                out = []
                for choice in level:
                    # The empty sequence element needs to still be in the list
                    # so that the branches aren't messed up.
                    if choice == "_":
                        out.append("")
                    else:
                        out.append(choice)
                return out
        perms: List[str] = []
        if isinstance(level, str):
            perms.extend(
                level + subperm
                for subperm in SpanTree._generate_permutations_rec(rest)
            )
        else:
            assert isinstance(level, list)
            for choice in level:
                # The empty sequence element needs to still be in the list
                # so that the branches aren't messed up.
                if choice == "_":
                    choice = ""
                perms.extend(
                    choice + subperm
                    for subperm in SpanTree._generate_permutations_rec(rest)
                )
        return perms

    def generate_permutations(self) -> List[str]:
        """Return all the permutations of the text in order.

        Each permutation is the path from the root to a leaf.
        The permutations are ordered such that the leftmost paths come first.
        """
        return SpanTree._generate_permutations_rec(self.levels)

    @staticmethod
    def _print_tree_rec(
        curr_levels: List[Union[str, List[str]]], level: int
    ) -> None:
        if not curr_levels:
            return
        curr_level, *rest = curr_levels
        if isinstance(curr_level, str):
            print(f"{' ' * (level)}└{'─' * (level)} {curr_level!r}")
            SpanTree._print_tree_rec(rest, level + 1)
        else:
            assert isinstance(curr_level, list)
            for choice in curr_level:
                print(f"{' ' * (level)}└{'─' * (level)} {choice!r}")
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
    spans: List[Union[str, List[str]]] = [""]
    mode = _SequenceParseMode.NORMAL

    # Only shift for the first row.
    def _get_shifted_column() -> int:
        if row == 0:
            return col + column_shift
        else:
            return col

    def _err(msg: str) -> NoReturn:
        err = SequenceParseError(
            msg,
            text=" ".join(lines),
            line=start_line + row,
            column=_get_shifted_column(),
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
                    if spans[-1] == "":
                        spans[-1] = [""]
                    else:
                        spans.append([""])
                    continue
                elif c == "\\":
                    mode = _SequenceParseMode.NORMAL_ESCAPE_NEXT
                spans[-1] += c  # type: ignore
            elif mode == _SequenceParseMode.NORMAL_ESCAPE_NEXT:
                # Sequences can be escaped in normal mode, so the backslash shouldn't remain.
                # There are no nested sequences, so nothing to be done for sequence mode.
                if c in ("{", "}"):
                    # The last character was a backslash, so replace it.
                    spans[-1] = spans[-1][:-1] + c  # type: ignore
                else:
                    spans[-1] += c  # type: ignore
                mode = _SequenceParseMode.NORMAL
            elif mode == _SequenceParseMode.SEQUENCE:
                if c == "{":
                    _err(
                        "No nested sequences allowed (see https://github.com/baskerville/sxhkd/issues/67)"
                    )
                seq = cast("List[str]", spans[-1])
                if c == ",":
                    expanded_seq = expand_range(seq[-1])
                    if isinstance(expanded_seq, list):
                        del seq[-1]
                        seq.extend(expanded_seq)
                    # If not a list, then no expansion was done.
                    seq.append("")
                    continue
                elif c == "}":
                    mode = _SequenceParseMode.NORMAL
                    expanded_seq = expand_range(seq[-1])
                    if isinstance(expanded_seq, list):
                        del seq[-1]
                        seq.extend(expanded_seq)
                    # If not a list, then no expansion was done.
                    spans.append("")
                    continue
                elif c == "\\":
                    mode = _SequenceParseMode.SEQUENCE_ESCAPE_NEXT
                seq[-1] += c
            elif mode == _SequenceParseMode.SEQUENCE_ESCAPE_NEXT:
                seq = cast("List[str]", spans[-1])
                seq[-1] += c
                mode = _SequenceParseMode.SEQUENCE
    if mode != _SequenceParseMode.NORMAL:
        _err("Input ended while parsing a sequence or escaping a character")
    # Remove unused normal span at the end.
    if spans[-1] == "":
        spans.pop()

    return SpanTree(spans)


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
        modifiers: the sorted tuple of modifiers that must precede a chord.
        keysym: the keysym name, given by the output of `xev -event keyboard`.
        run_event: whether the chord (or whole command? TODO) runs on key-press or key-release.
        replay: whether the captured event will be replayed for the other clients.
    """

    modifiers: Tuple[str, ...]
    keysym: str
    run_event: ChordRunEvent
    replay: bool

    # XXX: are `@' and `~' significant enough to cause differing keybinds?
    # answer: yes (https://github.com/baskerville/sxhkd/issues/198)
    def __init__(
        self,
        modifiers: Iterable[str],
        keysym: str,
        run_event: Optional[ChordRunEvent] = None,
        replay: bool = False,
    ):
        self.modifiers = tuple(modifiers)
        self.keysym = keysym
        if run_event is None:
            run_event = ChordRunEvent.KEYPRESS
        self.run_event = run_event
        self.replay = replay


# not really a "node", but a value of a node
@dataclass
class KeypressTreeModifierNode:
    value: str


@dataclass
class KeypressTreeModifierSetNode:
    value: FrozenSet[str]


@dataclass
class KeypressTreeKeysymNode:
    value: str


@dataclass
class KeypressTreeChordRunEventNode:
    value: ChordRunEvent


@dataclass
class KeypressTreeReplayNode:
    value: bool


KeypressTreeInternalNode = Union[
    KeypressTreeModifierNode,
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
        - KeypressTreeModifierNode: modifiers (but order matters)
        - KeypressTreeModifierSetNode: *sets* of modifiers, so order doesn't matter (matches behaviour of sxhkd)
        - KeypressTreeKeysymNode: keysym
        - KeypressTreeChordRunEventNode: run on key-press or key-release
        - KeypressTreeReplayNode: whether to replay event to other clients

    Internal nodes must be manually included with their respective `include_*_nodes` method:
        - include_modifier_nodes
        - include_modifierset_nodes
        - include_keysym_nodes
        - include_runevent_nodes
        - include_replay_nodes

    They can be deduplicated by calling their respective `dedupe_*_nodes` method:
        - dedupe_modifier_nodes
        - dedupe_modifierset_nodes
        - dedupe_keysym_nodes
        - dedupe_runevent_nodes
        - dedupe_replay_nodes
    There is also `dedupe_chord_nodes`, which may be useful but may remove some
    permutations.

    The order in which internal node types are included determines the
    grouping.  For example, the order (1) modifierset, (2) keysym results in
    modifierset nodes being closest to the root node, and vice versa with the
    reverse order.
    """

    value: Union[Chord, KeypressTreeInternalNode]
    children: List[KeypressTreeNode]

    def __init__(self, value: Union[Chord, KeypressTreeInternalNode]):
        self.value = value
        self.children = []

    def include_modifier_nodes(self) -> None:
        """Include internal nodes whose type is `KeypressTreeModifierNode`."""
        if not self.children:
            return
        for child in self.children:
            child.include_modifier_nodes()
        i = len(self.children) - 1
        while i >= 0:
            child = self.children[i]
            if not isinstance(child.value, Chord):
                i -= 1
                continue
            new_node = self
            # old path: self -> child
            # Add modifiers such that this path is formed: self -> m1 -> m2 -> ... -> mn -> child(chord)
            for modifier in child.value.modifiers:
                new_node = new_node.add_child(
                    KeypressTreeModifierNode(modifier)
                )
            new_node = new_node.add_child(child)
            del self.children[i]
            i -= 1

    def include_keysym_nodes(self) -> None:
        """Include internal nodes whose type is `KeypressTreeKeysymNode`."""
        if not self.children:
            return
        for child in self.children:
            child.include_keysym_nodes()
        i = len(self.children) - 1
        while i >= 0:
            child = self.children[i]
            if not isinstance(child.value, Chord):
                i -= 1
                continue
            new_node = self
            # old path: self -> child
            # Add keysym in between between to get: self -> keysym -> child
            new_node = new_node.add_child(
                KeypressTreeKeysymNode(child.value.keysym)
            )
            new_node = new_node.add_child(child)
            del self.children[i]
            i -= 1

    def include_runevent_nodes(self) -> None:
        """Include internal nodes whose type is `KeypressTreeChordRunEventNode`."""
        if not self.children:
            return
        for child in self.children:
            child.include_runevent_nodes()
        i = len(self.children) - 1
        while i >= 0:
            child = self.children[i]
            if not isinstance(child.value, Chord):
                i -= 1
                continue
            new_node = self
            # old path: self -> child
            # Add chord run event in between between to get: self -> chord run event -> child
            new_node = new_node.add_child(
                KeypressTreeChordRunEventNode(child.value.run_event)
            )
            new_node = new_node.add_child(child)
            del self.children[i]
            i -= 1

    def include_replay_nodes(self) -> None:
        """Include internal nodes whose type is `KeypressTreeReplayNode`."""
        if not self.children:
            return
        for child in self.children:
            child.include_replay_nodes()
        i = len(self.children) - 1
        while i >= 0:
            child = self.children[i]
            if not isinstance(child.value, Chord):
                i -= 1
                continue
            new_node = self
            # old path: self -> child
            # Add replay node in between between to get: self -> replay node -> child
            new_node = new_node.add_child(
                KeypressTreeReplayNode(child.value.replay)
            )
            new_node = new_node.add_child(child)
            del self.children[i]
            i -= 1

    def _dedupe(self, kind: Type[KeypressTreeInternalNode]) -> None:
        """Deduplicate internal nodes whose value has the type `kind`."""
        # Take only nodes of the kind, and group them by their value.
        # KEY type of defaultdict should be same types as its `value` field
        groups: DefaultDict[
            Any, List[Tuple[int, KeypressTreeNode]]
        ] = defaultdict(list)
        for i, child in enumerate(self.children):
            if isinstance(child.value, kind):
                assert not isinstance(child.value, Chord)
                groups[child.value.value].append((i, child))

        # Take the first match per group and move the children of the other matches
        # into the first match.
        removed_children: List[int] = []
        for _, matches in groups.items():
            assert matches
            tokeep, *tomerge = matches
            i, keep_child = tokeep
            for j, merge_child in tomerge:
                for subchild in merge_child.children:
                    keep_child.add_child(subchild)
                removed_children.append(j)

        # Remove from right-to-left in children list.
        removed_children.sort(reverse=True)
        for i in removed_children:
            del self.children[i]

    def _dedupe_nodes(self, kind: Type[KeypressTreeInternalNode]) -> None:
        """Coordinate deduplication of internal nodes to maximise deduplication."""
        # No children to deduplicate.
        if not self.children:
            return
        # Dedupe what can be deduped already.
        self._dedupe(kind)

        # Dedupe the children so that all subtrees below this node are
        # deduplicated already.
        for child in self.children:
            child._dedupe_nodes(kind)

        # Dedupe anything that was made available by deduping the children.
        self._dedupe(kind)

    def dedupe_modifier_nodes(self) -> None:
        """Deduplicate internal nodes whose value has the type `KeypressTreeModifierNode`.

        This merges sibling nodes with equal `value` attributes, if they are
        both of type `KeypressTreeModifierNode`.
        """
        self._dedupe_nodes(KeypressTreeModifierNode)

    def dedupe_keysym_nodes(self) -> None:
        """Deduplicate internal nodes whose value has the type `KeypressTreeKeysymNode`.

        This merges sibling nodes with equal `value` attributes, if they are
        both of type `KeypressTreeKeysymNode`.
        """
        self._dedupe_nodes(KeypressTreeKeysymNode)

    def dedupe_runevent_nodes(self) -> None:
        """Deduplicate internal nodes whose value has the type `KeypressTreeChordRunEventNode`.

        This merges sibling nodes with equal `value` attributes, if they are
        both of type `KeypressTreeChordRunEventNode`.
        """
        self._dedupe_nodes(KeypressTreeChordRunEventNode)

    def dedupe_replay_nodes(self) -> None:
        """Deduplicate internal nodes whose value has the type `KeypressTreeReplayNode`.

        This merges sibling nodes with equal `value` attributes, if they are
        both of type `KeypressTreeReplayNode`.
        """
        self._dedupe_nodes(KeypressTreeReplayNode)

    # Lots of duplication with `_dedupe(kind)`, but it's pretty much the same.
    def _dedupe_chords(self) -> None:
        # Take only nodes of the kind, and group them by their value.
        # KEY type of defaultdict should be same types as its `value` field
        groups: DefaultDict[
            Any, List[Tuple[int, KeypressTreeNode]]
        ] = defaultdict(list)
        for i, child in enumerate(self.children):
            if isinstance(child.value, Chord):
                groups[child.value].append((i, child))

        # Take the first match per group and move the children of the other matches
        # into the first match.
        removed_children: List[int] = []
        for _, matches in groups.items():
            assert matches
            tokeep, *tomerge = matches
            i, keep_child = tokeep
            for j, merge_child in tomerge:
                for subchild in merge_child.children:
                    keep_child.add_child(subchild)
                removed_children.append(j)

        # Remove from right-to-left in children list.
        removed_children.sort(reverse=True)
        for i in removed_children:
            del self.children[i]

    def dedupe_chord_nodes(self) -> None:
        """Deduplicate nodes that contain `Chord` in its `value` attribute.

        This merges sibling nodes with equal `value` attributes, if they are
        both of type `Chord`.

        Note that deduplicating chords may delete permutations, thereby
        limiting its use as a decision tree.
        """
        # No children to deduplicate.
        if not self.children:
            return
        # Dedupe what can be deduped already.
        self._dedupe_chords()

        # Dedupe the children so that all subtrees below this node are
        # deduplicated already.
        for child in self.children:
            child._dedupe_chords()

        # Dedupe anything that was made available by deduping the children.
        self._dedupe_chords()

    def include_modifierset_nodes(self, nullsets: bool = False) -> None:
        """Include internal nodes whose type is `KeypressTreeModifierSetNode`.

        If `nullsets` is False, empty modifiers will not create modifierset
        nodes.
        """
        if not self.children:
            return
        for child in self.children:
            child.include_modifierset_nodes(nullsets=nullsets)
        i = len(self.children) - 1
        while i >= 0:
            child = self.children[i]
            if not isinstance(child.value, Chord):
                i -= 1
                continue
            # old path: self -> child
            # Add modifiers such that this path is formed: self -> {m1,m2,...,mn} -> child(chord)
            if (not nullsets and child.value.modifiers) or nullsets:
                new_node = self.add_child(
                    KeypressTreeModifierSetNode(
                        frozenset(child.value.modifiers)
                    )
                )
                new_node.add_child(child)
                del self.children[i]
            i -= 1

    def _dedupe_modifierset_nodes_rec(self) -> None:
        modset_nodes: List[Tuple[int, KeypressTreeNode]] = []
        for i, child in enumerate(self.children):
            if isinstance(child.value, KeypressTreeModifierSetNode):
                modset_nodes.append((i, child))
        modset_nodes.sort(
            key=lambda x: len(
                cast(KeypressTreeModifierSetNode, x[1].value).value
            )
        )

        removed_children: List[int] = []
        i = 0
        while i < len(modset_nodes):
            n1_index, n1_node = modset_nodes[i]
            assert isinstance(n1_node.value, KeypressTreeModifierSetNode)
            modset1 = n1_node.value.value

            j = len(modset_nodes) - 1
            while j >= 0:
                n2_index, n2_node = modset_nodes[j]
                assert isinstance(n2_node.value, KeypressTreeModifierSetNode)
                modset2 = n2_node.value.value
                if n1_node is n2_node:
                    j -= 1
                    continue

                if modset1 == modset2:
                    # Necessary since moving subsets might make those with
                    # equal value siblings?
                    # Move all children of n2 under n1 and remove n2.
                    for subchild in n2_node.children:
                        n1_node.add_child(subchild)
                    removed_children.append(n2_index)
                    del modset_nodes[j]
                elif modset1 < modset2:
                    # Keep n2, but make it a child of n1 since modset1 being a
                    # strict subset of modset2 means that all of its modifiers
                    # must be pressed.
                    # XXX: no good solution to this
                    #   - can't just take their union
                    n1_node.add_child(n2_node)
                    removed_children.append(n2_index)
                    del modset_nodes[j]
                j -= 1
            i += 1

        # Remove from right-to-left in children list.
        removed_children.sort(reverse=True)
        for i in removed_children:
            del self.children[i]

    def dedupe_modifierset_nodes(self, subsets: bool = True) -> None:
        """Deduplicate modifierset nodes.

        Equal sets are merged into single branches and, if `subsets` is True,
        subsets become parents of the nodes they are (strict) subsets of.
        """
        # No children to deduplicate.
        if not self.children:
            return
        # Merge equal modifierset nodes first.
        self._dedupe_nodes(KeypressTreeModifierSetNode)

        if not subsets:
            return

        # Now merge those with subset relationships:

        # Dedupe what can be deduped already.
        self._dedupe_modifierset_nodes_rec()

        # Dedupe the children so that all subtrees below this node are
        # deduplicated already.
        for child in self.children:
            child._dedupe_modifierset_nodes_rec()

        # Dedupe anything that was made available by deduping the children.
        self._dedupe_modifierset_nodes_rec()

    @classmethod
    def build_tree(cls, chord_perms: List[List[Chord]]) -> KeypressTreeNode:
        """Create an instance from a list of chord permutations."""
        root = cls(None)  # type: ignore
        # Add each permutation as a path from the root
        for perm in chord_perms:
            new_node = root
            for chord in perm:
                new_node = new_node.add_child(chord)
        return root

    def add_child(
        self, value: Union[Chord, KeypressTreeInternalNode, KeypressTreeNode]
    ) -> KeypressTreeNode:
        """Add an existing node as a child, or create and add a new one with the given value."""
        if isinstance(value, KeypressTreeNode):
            child = value
        else:
            child = KeypressTreeNode(value)
        self.children.append(child)
        return child

    def _print_tree_rec(self, level: int) -> None:
        assert level >= 0
        if level == 0:
            print(repr(self.value))
        else:
            print(f"{' ' * (level-1)}└{'─' * (level-1)} {self.value!r}")
        for child in self.children:
            child._print_tree_rec(level + 1)

    def print_tree(self) -> None:
        """Print the tree, rooted at this node."""
        self._print_tree_rec(0)


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
    permutations: List[List[Chord]] = field(repr=False)
    noabort_index: Optional[int]

    def __init__(
        self, hotkey: Union[str, List[str]], line: Optional[int] = None
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

        # It's okay if the error messages say it's at line 1:
        # since it's at line 1 of the input anyway.
        root = expand_sequences(hotkey, start_line=self.line or 1)

        self.permutations = []
        noabort_indices: List[Optional[int]] = []

        for flat_perm in root.generate_permutations():
            tokens = Hotkey.tokenize_static_hotkey(flat_perm, self.line)
            noabort_index, chords = Hotkey.parse_static_hotkey(tokens)
            self.permutations.append(chords)
            noabort_indices.append(noabort_index)

        unique_indices = set(noabort_indices)
        if len(unique_indices) > 1:
            index_counts = dict(
                sorted(
                    ((i, noabort_indices.count(i)) for i in unique_indices),
                    key=lambda x: x[1],
                )
            )
            raise InconsistentNoabortError(
                f"Noabort indicated in different places among permutations of '{self.raw if isinstance(self.raw, str) else ' '.join(self.raw)}' with index count: {index_counts}",
                perms=self.permutations,
                indices=noabort_indices,
                index_counts=index_counts,
            )
        assert len(unique_indices) == 1
        self.noabort_index = unique_indices.pop()

    @property
    def noabort(self) -> bool:
        """Return whether the chain won't be aborted when the chain tail is reached (`:').

        Synchronized with `noabort_index`.
        """
        return self.noabort_index is not None

    def get_tree(self) -> KeypressTreeNode:
        """Return the decision tree resulting from the permutations."""
        return KeypressTreeNode.build_tree(self.permutations)

    @staticmethod
    def static_hotkey_str(
        chain: List[Chord], noabort_index: Optional[int] = None
    ) -> str:
        """Return the string representation of the chord chain.

        `noabort_index` is the index of the chord where ':' is used, if present.
        """
        hotkey = ""
        for i, chord in enumerate(chain):
            hotkey += " + ".join(it.chain(chord.modifiers, [chord.keysym]))
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
            # Empty sequence element.
            elif type_ == "KEYSYM" and value == "_":
                continue
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
        curr_keysym: Optional[str] = None
        curr_run_event: Optional[ChordRunEvent] = None
        curr_replay: bool = False

        def reset_temp_state() -> None:
            nonlocal curr_modifiers, curr_keysym, curr_run_event, curr_replay
            curr_modifiers = []
            curr_keysym = None
            curr_run_event = None
            curr_replay = False

        def MODIFIER_NAME_on_MODIFIER(tok: HotkeyToken) -> None:
            curr_modifiers.append(tok.value)

        # We know that getting a keysym means completing a chord.
        def on_KEYSYM(tok: HotkeyToken) -> None:
            nonlocal curr_keysym
            curr_keysym = tok.value
            # normalise them
            curr_modifiers.sort()
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
        permutations: all possible choices of command text, resulting from sequence expansion.
        synchronous: whether the command should be executed synchronously or asynchronously.
    """

    raw: Union[str, List[str]]
    line: Optional[int]
    _span_tree: SpanTree = field(repr=False)
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

        self._span_tree = root = expand_sequences(
            command, start_line=line or 1, column_shift=col_shift
        )

        self.permutations = root.generate_permutations()

    def get_tree(self) -> SpanTree:
        """Return the decision tree resulting from sequence expansion."""
        return self._span_tree


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
    ):
        """Create an instance with the the hotkey and command text.

        `hotkey` and `hotkey_start_line` are directly passed to the constructor
        for `Hotkey`, and `command` and `command_start_line` are directly
        passed to that of `Command`.

        If the hotkey and command differ in the number of cases/permutations
        after sequence expansion occurs, ValueError is raised.
        """
        if metadata is None:
            metadata = {}
        self.metadata = metadata

        self.hotkey: Hotkey = Hotkey(hotkey, line=hotkey_start_line)
        self.command: Command = Command(command, line=command_start_line)

        if len(self.hotkey.permutations) != len(self.command.permutations):
            raise ValueError(
                f"inconsistent number of cases: hotkey-cases={len(self.hotkey.permutations)}, command-cases={len(self.command.permutations)}"
            )

    @property
    def line(self) -> Optional[int]:
        """Return the starting line of the keybind, which is that of its hotkey."""
        return self.hotkey.line
