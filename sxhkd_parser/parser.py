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
from abc import ABC
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
    Generic,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
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
    "HotkeyPermutation",
    "HotkeyToken",
    "HotkeyTree",
    "HotkeyTreeChordData",
    "HotkeyTreeChordRunEventData",
    "HotkeyTreeInternalNode",
    "HotkeyTreeKeysymData",
    "HotkeyTreeLeafNode",
    "HotkeyTreeModifierSetData",
    "HotkeyTreeNode",
    "HotkeyTreeNodeData",
    "HotkeyTreeReplayData",
    "HotkeyTreeRootData",
    "Keybind",
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
    """

    modifiers: FrozenSet[str]
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
        noabort: bool = False,
    ):
        self.modifiers = frozenset(modifiers)
        self.keysym = keysym
        if run_event is None:
            run_event = ChordRunEvent.KEYPRESS
        self.run_event = run_event
        self.replay = replay

    def __str__(self) -> str:
        keysym_prefix = ""
        if self.replay:
            keysym_prefix += "~"
        if self.run_event == ChordRunEvent.KEYRELEASE:
            keysym_prefix += "@"
        return " + ".join(
            it.chain(sorted(self.modifiers), [keysym_prefix + self.keysym])
        )


T = TypeVar("T")


@dataclass(frozen=True)
class HotkeyTreeNodeData(Generic[T]):
    """Value for a `HotkeyTreeNode`."""

    value: T
    SUBCLASSES: ClassVar[Set[Type[HotkeyTreeNodeData[Any]]]] = set()

    def __init_subclass__(cls) -> None:
        HotkeyTreeNodeData.SUBCLASSES.add(
            cast("Type[HotkeyTreeNodeData[Any]]", cls)
        )


@dataclass(frozen=True)
class HotkeyTreeRootData(HotkeyTreeNodeData[None]):
    """Value for the root node of a `HotkeyTree`."""

    value: None = None


@dataclass(frozen=True)
class HotkeyTreeModifierSetData(HotkeyTreeNodeData[FrozenSet[str]]):
    """Value of a `Chord`'s `modifiers` attribute."""

    value: FrozenSet[str]


@dataclass(frozen=True)
class HotkeyTreeKeysymData(HotkeyTreeNodeData[str]):
    """Value of a `Chord`'s `keysym` attribute."""

    value: str


@dataclass(frozen=True)
class HotkeyTreeChordRunEventData(HotkeyTreeNodeData[ChordRunEvent]):
    """Value of a `Chord`'s `run_event` attribute."""

    value: ChordRunEvent


@dataclass(frozen=True)
class HotkeyTreeReplayData(HotkeyTreeNodeData[bool]):
    """Value of a `Chord`'s `replay`."""

    value: bool


@dataclass(frozen=True)
class HotkeyTreeChordData(HotkeyTreeNodeData[Chord]):
    """Value for a chord `HotkeyTreeNode`.

    NOTE: Only internal chord nodes can have noabort--it makes no sense for
    leaf nodes to have a noabort value.
    """

    value: Chord
    noabort: bool


@dataclass
class HotkeyTreeNode(ABC):
    """Node for a decision tree representing the keypresses needed to complete a hotkey.

    NOTE: This class should never be instantiated--enforced by inheriting from `ABC`.

    Instance variables:
        data: the data stored in the node.

    Class variables:
        ends_permutation: whether this type of node ends a permutation.

    Invariants:
        - each level has no duplicate nodes EXCEPT among leaf nodes.
    """

    data: HotkeyTreeNodeData[Any]
    ends_permutation: ClassVar[bool]


@dataclass
class HotkeyTreeInternalNode(HotkeyTreeNode):
    """Hotkey tree node with children.

    Instance variables:
        children:
            The list of children of this node, including internal nodes.
        internal_children:
            The dict that keeps dicts of references to the children that are
            internal nodes.  This is kept in sync with changes in `children`.

    By definition, these cannot end permutations, so the class variable `ends_permutation` is False.
    """

    ends_permutation = False

    children: List[HotkeyTreeNode]
    internal_children: Dict[
        Type[HotkeyTreeNodeData[Any]],
        Dict[HotkeyTreeNodeData[Any], HotkeyTreeInternalNode],
    ] = field(repr=False)

    def __init__(self, data: HotkeyTreeNodeData[Any]):
        super().__init__(data)
        self.children = []
        self.internal_children = {}
        for cls in HotkeyTreeNodeData.SUBCLASSES:
            self.internal_children[cls] = {}

    @overload
    def add_child(
        self, node: HotkeyTreeInternalNode
    ) -> HotkeyTreeInternalNode:
        ...

    @overload
    def add_child(self, node: HotkeyTreeLeafNode) -> HotkeyTreeLeafNode:
        ...

    @overload
    def add_child(self, node: HotkeyTreeNode) -> HotkeyTreeNode:
        ...

    def add_child(
        self,
        node: HotkeyTreeNode,
    ) -> HotkeyTreeNode:
        """Add the subtree rooted at `node` as a child, ensuring no duplicate internal nodes.

        Cases:
            - `node` is a leaf node:
                return `node`
            - `node` is an internal node which doesn't already exist at this node:
                return `node`
            - `node` is a duplicate internal node:
                recursively merge the subtrees of `node` into the one already there, ensuring no duplicate internal nodes
                return the existing internal node

        NOTE:
            Permutation-ending nodes (i.e., leaf nodes) are left unmerged in
            order to preserve the one-to-one correspondence between a leaf node
            and a hotkey permutation.
        """
        if isinstance(node, HotkeyTreeLeafNode):
            self.children.append(node)
            return node
        assert isinstance(node, HotkeyTreeInternalNode)

        curr_children = self.internal_children[type(node.data)]
        existing_node = curr_children.get(node.data)
        if existing_node is None:
            curr_children[node.data] = node
            self.children.append(node)
            # `node` is unique at this path.
            return node
        for child in node.children:
            # Ignore return value as we only care about the highest-level internal node.
            existing_node.add_child(child)
        return existing_node

    def remove_child(self, index: int) -> HotkeyTreeNode:
        """Remove the child at `index` in `children` and return it."""
        child = self.children.pop(index)
        if isinstance(child, HotkeyTreeLeafNode):
            return child
        curr_children = self.internal_children[type(child.data)]
        del curr_children[child.data]
        return child

    def find_permutation_ends(self) -> List[HotkeyTreeLeafNode]:
        """Return all the permutation-ending nodes from this node.

        NOTE: only leaf nodes can end permutations.
        """
        perm_ends = []
        for child in self.children:
            if isinstance(child, HotkeyTreeLeafNode):
                perm_ends.append(child)
                continue
            assert isinstance(child, HotkeyTreeInternalNode)
            perm_ends.extend(child.find_permutation_ends())
        return perm_ends


@dataclass
class HotkeyTreeLeafNode(HotkeyTreeNode):
    """Leaf node of a Hotkey tree.

    NOTE: only `HotkeyTreeChordData` objects are stored in the `data` attribute.

    Instance variables:
        permutation_index: the index to the permutation contained in the list of permutations held by `hotkey`.
        hotkey: the reference to the hotkey that contains the permutation represented by this path from root to leaf.

    By definition, these end permutations, so the class variable `ends_permutation` is True.
    """

    ends_permutation = True

    hotkey: Hotkey
    permutation_index: int

    def __init__(
        self,
        data: HotkeyTreeNodeData[Any],
        hotkey: Hotkey,
        permutation_index: int,
    ):
        super().__init__(data)
        assert isinstance(data, HotkeyTreeChordData)
        assert not data.noabort, f"got leaf node with noabort (data: {data})"
        self.hotkey = hotkey
        self.permutation_index = permutation_index

    @property
    def permutation(self) -> HotkeyPermutation:
        """Return the associated `HotkeyPermutation`."""
        return self.hotkey.permutations[self.permutation_index]


@dataclass
class HotkeyPermutation:
    """A permutation of a hotkey.

    Instance variables:
        chords: the list of `Chord` objects comprising the permutation.
        noabort_index: see the `noabort_index` attribute of `Hotkey` for description.
    """

    chords: List[Chord]
    noabort_index: Optional[int]

    def __str__(self) -> str:
        """Return the string representation of the chord chain."""
        hotkey = ""
        for i, chord in enumerate(self.chords):
            hotkey += str(chord)
            if self.noabort_index is not None and i == self.noabort_index:
                hotkey += ": "
            elif i == len(self.chords) - 1:
                # last chord: no ';'
                continue
            else:
                hotkey += "; "
        return hotkey


@dataclass
class HotkeyTree:
    """The decision tree for a single hotkey or multiple hotkeys.

    Hotkeys are each represented by a path from root to leaf composed of
    `HotkeyTreeNode` objects, with each `HotkeyTreeChordData` in that path
    being the key-chords needed to complete it.

    Instance variables:
        root: the root `HotkeyTreeNode` with data of type `HotkeyTreeRootData`.
        internal_nodes: the list of internal node types included in the tree.

    Internal node types:
        keysym: HotkeyTreeKeysymData
        modifierset: HotkeyTreeModifierSetData
        runevent: HotkeyTreeChordRunEventData
        replay: HotkeyTreeReplayData

    NOTE: `HotkeyTreeInternalNode` objects with data of type
    `HotkeyTreeChordData` are automatically included for non-permutation ending
    chords.
    """

    root: HotkeyTreeInternalNode
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
        self.root = HotkeyTreeInternalNode(HotkeyTreeRootData())
        if internal_nodes is None:
            internal_nodes = []
        for nodetype in internal_nodes:
            if nodetype not in HotkeyTree.INTERNAL_NODE_TYPES:
                raise ValueError(f"Invalid nodetype '{nodetype}'")
        self.internal_nodes = internal_nodes

    @staticmethod
    def _print_tree_rec(node: HotkeyTreeNode, level: int) -> None:
        prefix = f"{' ' * (level-1)}└{'─' * (level-1)}"
        if isinstance(node, HotkeyTreeLeafNode):
            print(
                f"{prefix} {node.data!r} i={node.permutation_index}, hotkey={node.permutation}"
            )
            return
        assert isinstance(node, HotkeyTreeInternalNode)
        print(f"{prefix} {node.data!r}")
        for child in node.children:
            HotkeyTree._print_tree_rec(child, level + 1)

    def print_tree(self) -> None:
        """Print the tree starting from its root."""
        print(repr(self.root.data.value))
        for child in self.root.children:
            HotkeyTree._print_tree_rec(child, 1)

    def merge_hotkey(self, hotkey: Hotkey) -> None:
        """Merge the tree of chord permutations for `hotkey` into this tree."""
        for i, perm in enumerate(hotkey.permutations):
            subtree = self._create_subtree_from_chord_chain(perm, i, hotkey)
            self.root.add_child(subtree)

    def merge_tree(self, tree: HotkeyTree) -> None:
        """Merge `tree` into this tree."""
        if self.internal_nodes != tree.internal_nodes:
            raise ValueError(
                f"Trees had different internal node types ({self.internal_nodes} vs {tree.internal_nodes}). Not merging."
            )
        for child in tree.root.children:
            self.root.add_child(child)

    def _create_subtree_from_chord_chain(
        self, perm: HotkeyPermutation, index: int, hotkey: Hotkey
    ) -> HotkeyTreeNode:
        assert perm.chords, "got empty permutation"
        node_values: List[HotkeyTreeNodeData[Any]] = []
        for i, chord in enumerate(perm.chords):
            for nodetype in self.internal_nodes:
                if nodetype == "keysym":
                    node_values.append(HotkeyTreeKeysymData(chord.keysym))
                elif nodetype == "modifierset":
                    node_values.append(
                        HotkeyTreeModifierSetData(chord.modifiers)
                    )
                elif nodetype == "runevent":
                    node_values.append(
                        HotkeyTreeChordRunEventData(chord.run_event)
                    )
                elif nodetype == "replay":
                    node_values.append(HotkeyTreeReplayData(chord.replay))
                else:
                    raise RuntimeError(f"invalid nodetype '{nodetype}'")
            is_noabort: bool = (
                perm.noabort_index is not None and i == perm.noabort_index
            )
            node_values.append(
                HotkeyTreeChordData(
                    chord,
                    noabort=is_noabort,
                )
            )

        if len(node_values) == 1:
            assert isinstance(node_values[0], HotkeyTreeChordData)
            return HotkeyTreeLeafNode(
                node_values[0], hotkey=hotkey, permutation_index=index
            )

        root = new_node = HotkeyTreeInternalNode(node_values[0])
        leaf = HotkeyTreeLeafNode(
            node_values[-1], hotkey=hotkey, permutation_index=index
        )
        assert isinstance(leaf.data, HotkeyTreeChordData)
        # No need to merge as this is just one permutation.
        if len(node_values) == 2:
            root.add_child(leaf)
            return root
        for value in node_values[1:-1]:
            new_node = new_node.add_child(HotkeyTreeInternalNode(value))
        new_node.add_child(leaf)
        return root

    @staticmethod
    def _find_duplicate_chords_rec(
        node: HotkeyTreeNode,
    ) -> List[List[HotkeyTreeLeafNode]]:
        if isinstance(node, HotkeyTreeLeafNode):
            return []
        assert isinstance(node, HotkeyTreeInternalNode)

        # Take only chord nodes and group them by their value.
        groups: DefaultDict[Chord, List[HotkeyTreeLeafNode]] = defaultdict(
            list
        )
        for child in node.children:
            if isinstance(child, HotkeyTreeLeafNode) and isinstance(
                child.data, HotkeyTreeChordData
            ):
                assert (
                    not child.data.noabort
                ), "got perm-ending noabort chord node"
                groups[child.data.value].append(child)
        dups = list(
            dupnodes for dupnodes in groups.values() if len(dupnodes) > 1
        )

        for child in node.children:
            dups.extend(HotkeyTree._find_duplicate_chords_rec(child))
        return dups

    def find_duplicate_chord_nodes(self) -> List[List[HotkeyTreeLeafNode]]:
        """Return duplicate chord nodes, with each sublist representing a set of duplicates."""
        return HotkeyTree._find_duplicate_chords_rec(self.root)

    @staticmethod
    def _find_conflicting_chain_prefixes_rec(
        node: HotkeyTreeNode,
    ) -> List[Tuple[HotkeyTreeLeafNode, List[HotkeyTreeLeafNode]]]:
        if isinstance(node, HotkeyTreeLeafNode):
            return []
        assert isinstance(node, HotkeyTreeInternalNode)

        conflicts = []
        # Check for conflicts between chord nodes with the same value of `noabort`.
        groups: DefaultDict[
            HotkeyTreeChordData, List[HotkeyTreeNode]
        ] = defaultdict(list)
        for child in node.children:
            if isinstance(child.data, HotkeyTreeChordData):
                groups[child.data].append(child)
        for matches in groups.values():
            assert matches
            prefix_perm_ends: List[HotkeyTreeLeafNode] = []
            longer_chains: List[HotkeyTreeLeafNode] = []
            for child in matches:
                if isinstance(child, HotkeyTreeLeafNode):
                    # Leaves on this level are the entire conflicting prefix.
                    prefix_perm_ends.append(child)
                else:
                    assert isinstance(child, HotkeyTreeInternalNode)
                    # All descendant leaves of internal nodes on this level
                    # conflict with the leaves on this level.
                    longer_chains.extend(child.find_permutation_ends())
            if longer_chains:
                for prefix in prefix_perm_ends:
                    conflicts.append((prefix, longer_chains))

        # Now check for conflicts between noabort and non-noabort chords.
        noabort_groups: DefaultDict[Chord, List[HotkeyTreeNode]] = defaultdict(
            list
        )
        for child in node.children:
            if isinstance(child.data, HotkeyTreeChordData):
                noabort_groups[child.data.value].append(child)
        for noabort_matches in noabort_groups.values():
            assert noabort_matches
            noaborts: List[HotkeyTreeLeafNode] = []
            normals: List[HotkeyTreeLeafNode] = []
            for child in noabort_matches:
                assert isinstance(child.data, HotkeyTreeChordData)
                if child.data.noabort:
                    assert (
                        isinstance(child, HotkeyTreeInternalNode)
                        and child.children
                    ), "got a noabort chord without children"
                    noaborts.extend(child.find_permutation_ends())
                elif isinstance(child, HotkeyTreeLeafNode):
                    normals.append(child)
                else:
                    assert isinstance(child, HotkeyTreeInternalNode)
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
    ) -> List[Tuple[HotkeyTreeLeafNode, List[HotkeyTreeLeafNode]]]:
        """Return pairs of conflicting chord chain prefixes and the permutation-ending nodes under them."""
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
    permutations: List[HotkeyPermutation] = field(repr=False)
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
        # Map perms to indices in `self.permutations`.
        seen_perms: Dict[Tuple[Tuple[Chord, ...], Optional[int]], int] = {}

        for i, flat_perm in enumerate(self.span_tree.generate_permutations()):
            tokens = Hotkey.tokenize(str(flat_perm), self.line)
            try:
                curr_perm = Hotkey.parse_hotkey_permutation(tokens)
            except HotkeyParseError as e:
                if isinstance(e, DuplicateModifierError):
                    e.message = f"{e.message} in '{flat_perm}'"
                e.line = self.line
                raise

            perm_tuple = (tuple(curr_perm.chords), curr_perm.noabort_index)
            if i == 0:
                self.noabort_index = curr_perm.noabort_index
                seen_perms[perm_tuple] = i
                self.permutations.append(curr_perm)
                continue

            if curr_perm.noabort_index != self.noabort_index:
                if isinstance(self.raw, str):
                    raw_hotkey = self.raw
                else:
                    raw_hotkey = " ".join(self.raw)
                hotkey_str1 = str(self.permutations[-1])
                hotkey_str2 = str(curr_perm)
                raise InconsistentNoabortError(
                    f"Noabort indicated in different places among permutations for '{raw_hotkey}': '{hotkey_str1}' vs '{hotkey_str2}'",
                    perm1=self.permutations[-1],
                    perm1_index=len(self.permutations) - 1,
                    perm2=curr_perm,
                    perm2_index=i,
                    line=line,
                )

            if check_duplicate_permutations and perm_tuple in seen_perms:
                raise DuplicateChordPermutationError(
                    f"Duplicate permutation '{curr_perm}'",
                    dup_perm=curr_perm,
                    perm1_index=i,
                    perm2_index=seen_perms[perm_tuple],
                    line=line,
                )
            seen_perms[perm_tuple] = i
            self.permutations.append(curr_perm)

        if check_conflicting_permutations:
            tree = self.get_tree(["modifierset"])
            for (
                prefix,
                conflicts,
            ) in tree.find_conflicting_chain_prefixes():
                conflicts_str = []
                for conflict in conflicts:
                    assert conflict.permutation is not None
                    conflicts_str.append(f"'{conflict.permutation}'")

                assert prefix.permutation is not None
                # Fail on the first conflict.
                raise ConflictingChainPrefixError(
                    f"'{prefix.permutation}' conflicts with {', '.join(conflicts_str)}",
                    chain_prefix=prefix,
                    conflicts=conflicts,
                    line=line,
                )

        if check_maybe_invalid_keysyms:
            maybe_invalid_keysyms = set()
            for perm in self.permutations:
                for chord in perm.chords:
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
    def tokenize(hotkey: str, line: Optional[int] = None) -> List[HotkeyToken]:
        """Tokenize a hotkey without sequences of the form {s1,s2,...,sn}.

        `hotkey` can be obtained from input that contains sequences by passing
        it to `expand_sequences`, and then stringifying and passing one
        permutation from the `generate_permutations` method of the `SpanTree`
        output to each call of `tokenize`.

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
    def parse_hotkey_permutation(
        tokens: List[HotkeyToken],
    ) -> HotkeyPermutation:
        """Parse a hotkey with pre-expanded {s1,s2,...,sn} sequences.

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

        return HotkeyPermutation(chords, noabort_index)


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
