from __future__ import annotations

import itertools as it
import re
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Tuple,
)

from .errors import MetadataParserError, SectionEOFError, SectionPushError
from .parser import Chord, Hotkey, Keybind, SpanTreeNode, expand_sequences


# TODO: should sections have metadata too?
# Invariants:
#   - each section completely encloses all subsections recursively
@dataclass
class SectionTreeNode:
    """Node representing a section in the sxhkdrc.

    Note that only the root section will have `name` be `None`.
    """

    name: Optional[str]
    start: int
    end: Optional[int]
    children: List[SectionTreeNode]
    keybind_children: List[Keybind]

    def __init__(self, name: Optional[str], start: int, end: Optional[int]):
        self.name = name
        self.start = start
        # initially None when created
        self.end = end
        self.children = []
        self.keybind_children = []

    def add_child(
        self, name: str, start: int, end: Optional[int]
    ) -> SectionTreeNode:
        child = SectionTreeNode(name, start, end)
        self.children.append(child)
        return child

    def add_keybind(self, keybind: Keybind) -> None:
        self.keybind_children.append(keybind)

    @staticmethod
    def _get_level_prefix(level: int) -> str:
        return f"{' ' * (level-1)}└{'─' * (level-1)}"

    @staticmethod
    def _default_keybind_child_callback(keybind: Keybind, level: int) -> None:
        msg = f"{SectionTreeNode._get_level_prefix(level)}"
        if "description" in keybind.metadata:
            desc = keybind.metadata["description"]
            expanded = expand_sequences(desc)
            # print(f"{expanded=}")
            if isinstance(expanded, str):
                assert desc == expanded
                # No permutations, so just print it plainly.
                msg = f"{msg} {keybind.hotkey.raw!r} (line {keybind.line})"
                if "mode" in keybind.metadata:
                    msg = f"{msg} (mode: {keybind.metadata['mode']})"
                msg = f"{msg}: {desc}"
                print(msg)
                return
            else:
                assert isinstance(expanded, SpanTreeNode)
                # Try to print each permutation separately.
                permutations = list(
                    it.chain.from_iterable(
                        child.generate_permutations(empty_elem_strat="delete")
                        for child in expanded.children
                    )
                )
                if len(permutations) == len(keybind.hotkey.permutations):
                    prefix = msg
                    for hotkey_perm, desc_perm in zip(
                        keybind.hotkey.permutations, permutations
                    ):
                        assert isinstance(hotkey_perm[0], Chord)
                        msg = f"{prefix} {Hotkey.static_hotkey_str(hotkey_perm, keybind.hotkey.noabort_index)!r} (line {keybind.line})"
                        if "mode" in keybind.metadata:
                            msg = f"{msg} (mode: {keybind.metadata['mode']})"
                        msg = f"{msg}: {desc_perm}"
                        print(msg)
                else:
                    if isinstance(keybind.hotkey.raw, str):
                        msg = f"{msg} {keybind.hotkey.raw!r} (line {keybind.line})"
                    else:
                        msg = f"{msg} {' '.join(keybind.hotkey.raw)!r} (line {keybind.line})"
                    if "mode" in keybind.metadata:
                        msg = f"{msg} (mode: {keybind.metadata['mode']})"
                    msg = f"{msg}: {desc}"
                    print(msg)
                return

        else:
            if isinstance(keybind.hotkey.raw, str):
                msg = f"{msg} {keybind.hotkey.raw!r} (line {keybind.line})"
            else:
                msg = f"{msg} {' '.join(keybind.hotkey.raw)!r} (line {keybind.line})"
            if "mode" in keybind.metadata:
                msg = f"{msg} (mode: {keybind.metadata['mode']})"
            print(msg)

    def _print_tree_rec(
        self,
        level: int,
        keybind_child_callback: Callable[[Keybind, int], None],
    ) -> None:
        assert level >= 0
        pos = (self.start, self.end)
        if level == 0:
            print(f"{self.name} {pos}")
        else:
            print(f"{' ' * (level-1)}└{'─' * (level-1)} {self.name} {pos}")
        # Print this section's keybinds under this node.
        for keybind in self.keybind_children:
            keybind_child_callback(keybind, level + 1)
        # And now the descendants.
        for child in self.children:
            child._print_tree_rec(level + 1, keybind_child_callback)

    def print_tree(
        self,
        keybind_child_callback: Optional[
            Callable[[Keybind, int], None]
        ] = None,
    ) -> None:
        """Print the section tree rooted at this node, with keybinds."""
        if not keybind_child_callback:
            keybind_child_callback = (
                SectionTreeNode._default_keybind_child_callback
            )
        self._print_tree_rec(0, keybind_child_callback)

    @classmethod
    def build_root(cls) -> SectionTreeNode:
        """Return a new root node."""
        return cls(None, 1, None)


# recursive and works because:
#   - base case: empty list
#   - few cases above base: child_gaps is a flat list of (node,gap) pairs
def _find_enclosing_section_rec(
    node: SectionTreeNode, keybind: Keybind
) -> List[Tuple[SectionTreeNode, int]]:
    assert node.start is not None, node
    assert node.end is not None, node
    assert keybind.line is not None, keybind
    if not (node.start <= keybind.line <= node.end):
        return []
    gap = abs(keybind.line - node.start) + abs(node.end - keybind.line)
    child_gaps = []
    for child in node.children:
        child_gaps.extend(_find_enclosing_section_rec(child, keybind))
    return [(node, gap)] + child_gaps


def find_enclosing_section(
    node: SectionTreeNode, keybind: Keybind
) -> Optional[SectionTreeNode]:
    """Find smallest enclosing section that is a descendant of `node` for `keybind`.

    Assumes that `keybind` has a non-None line number.
    """
    assert keybind.line is not None, keybind
    gaps = []
    for child in node.children:
        gaps.extend(_find_enclosing_section_rec(child, keybind))
    # sort by gap size
    gaps.sort(key=lambda x: x[1])
    if gaps:
        # smallest gap -> smallest section that contains the keybind
        section, gap = gaps[0]
        return section
    else:
        return None


class SectionHandler(ABC):
    @abstractmethod
    def push(self, text: str, line: int) -> bool:
        raise NotImplementedError

    def push_eof(self, last_line: int) -> None:
        self.root.end = last_line

    # should be the tree that the handler operates on -> it should update!
    @abstractproperty
    def root(self) -> SectionTreeNode:
        raise NotImplementedError

    @abstractmethod
    def current_section(self) -> SectionTreeNode:
        raise NotImplementedError


@dataclass
class NullSectionHandler(SectionHandler):
    _section: SectionTreeNode

    def __init__(self) -> None:
        self._section = SectionTreeNode.build_root()

    def push(self, text: str, line: int) -> bool:
        return False

    @property
    def root(self) -> SectionTreeNode:
        return self._section

    def current_section(self) -> SectionTreeNode:
        return self._section


@dataclass
class SimpleSectionHandler(SectionHandler):
    section_header_re: Pattern[str]
    sections: List[SectionTreeNode]
    _root: SectionTreeNode = field(repr=False)

    def __init__(self, section_header_re: str):
        self.section_header_re = re.compile(section_header_re)
        if "name" not in self.section_header_re.groupindex:
            raise ValueError(
                "section header regex must have the named group 'name'"
            )
        self._root = SectionTreeNode.build_root()
        self.sections = [self._root]

    def push(self, text: str, line: int) -> bool:
        m = self.section_header_re.search(text)
        if m:
            # starting a new section ends the previous one
            if self.sections[-1] is not self._root:
                self.sections[-1].end = line - 1
            # self.sections[self._curr_section_name] = (self._curr_section_start, end)
            self.sections.append(
                self._root.add_child(
                    m.group("name"),
                    line,
                    None,
                )
            )
            return True
        else:
            return False

    @property
    def root(self) -> SectionTreeNode:
        return self._root

    def current_section(self) -> SectionTreeNode:
        return self.sections[-1]


@dataclass
class StackSectionHandler(SectionHandler):
    section_header_re: Pattern[str]
    section_footer_re: Pattern[str]
    _section_tree: SectionTreeNode = field(repr=False)
    _section_stack: List[SectionTreeNode] = field(repr=False)

    def __init__(self, section_header_re: str, section_footer_re: str):
        self.section_header_re = re.compile(section_header_re)
        self.section_footer_re = re.compile(section_footer_re)
        if "name" not in self.section_header_re.groupindex:
            raise ValueError(
                "section header regex must have the named group 'name'"
            )
        self._section_tree = SectionTreeNode.build_root()
        self._section_stack = [self._section_tree]

    def push(self, text: str, line: int) -> bool:
        m = self.section_header_re.search(text)
        if m:
            name = m.group("name")
            node = self._section_stack[-1].add_child(name, line, None)
            self._section_stack.append(node)
            return True
        else:
            m = self.section_footer_re.search(text)
            if m:
                # first *must* always be root
                if len(self._section_stack) == 1:
                    raise SectionPushError(
                        "Ended a section without opening one first",
                        line=line,
                    )
                node = self._section_stack.pop()
                node.end = line
                return True
            else:
                return False

    def push_eof(self, last_line: int) -> None:
        if len(self._section_stack) > 1:
            raise SectionEOFError(
                f"Got EOF while reading section '{self._section_stack[-1].name}'",
                last_line=last_line,
                sections=self._section_stack[1:],
            )
        super().push_eof(last_line)

    @property
    def root(self) -> SectionTreeNode:
        return self._section_tree

    def current_section(self) -> SectionTreeNode:
        return self._section_stack[-1]


class MetadataParser(ABC):
    @abstractmethod
    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class NullMetadataParser(MetadataParser):
    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        return {}


@dataclass
class SimpleDescriptionParser(MetadataParser):
    description_re: Pattern[str]

    def __init__(self, description_re: str):
        self.description_re = re.compile(description_re)
        if "value" not in self.description_re.groupindex:
            raise ValueError(
                "description regex must have the named group 'value'"
            )

    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        comments = list(lines)
        assert len(comments) > 0
        maybe_description = comments[-1]
        m = self.description_re.search(maybe_description)
        if m:
            return {"description": m.group("value")}
        else:
            return {}


@dataclass
class KeyValueMetadataParser(MetadataParser):
    pair_re: Pattern[str]
    empty_re: Pattern[str]  # part of description but no pair

    def __init__(self, pair_re: str, empty_re: str):
        self.pair_re = re.compile(pair_re)
        if (
            "key" not in self.pair_re.groupindex
            or "value" not in self.pair_re.groupindex
        ):
            raise ValueError(
                "pair regex must have named groups 'key' and 'value'"
            )
        self.empty_re = re.compile(empty_re)

    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        comments = list(lines)
        metadata = {}
        # Reads them bottom-up, so "duplicates" are actually earlier in the file.
        i = len(comments) - 1
        while i >= 0:
            comment = comments[i]
            m = self.pair_re.search(comment)
            if m:
                key = m.group("key")
                value = m.group("value")
                if key in metadata:
                    raise MetadataParserError(
                        f"Duplicate key '{key}'",
                        key=key,
                        value=value,
                        line=start_line + i,
                    )
                metadata[key] = value
                i -= 1
            elif self.empty_re.search(comment):
                # metadata continues
                i -= 1
                continue
            else:
                break
        return metadata
