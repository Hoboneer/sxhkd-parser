"""Classes for managing sections and metadata in config files.

These classes parse sections and metadata found in the comments of sxhkdrc
files, in which different styles of formatting are handled specially.

As for sections, they may be "simple" in which there are no subsections, or
represent a "stack" in which there *are* subsections, which require that they
are completely enclosed by their parent sections.  There may also be *no*
sections below the root, in which no comments create new sections.

With regard to metadata, there may only be descriptions on single lines
("simple") or they may be key-value pairs across multiple lines ("key-value").

Many of them will also take regular expressions to configure the recognition of
sections and metadata.

Section classes:
    SectionHandler: abstract base class.
    RootSectionHandler: no new sections and places all keybinds in the root section.
    SimpleSectionHandler: flat sections.
    StackSectionHandler: subsections.

Metadata classes:
    MetadataParser: abstract base class.
    NullMetadataParser: no-op for all operations.
    SimpleDescriptionParser: description line immediately above the keybind.
    KeyValueMetadataParser: key-value pairs above the keybind.
"""
from __future__ import annotations

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
    Union,
)

from .errors import MetadataParserError, SectionEOFError, SectionPushError
from .parser import Keybind, SpanTree, expand_sequences

__all__ = [
    # General.
    "SectionTreeNode",
    "find_enclosing_section",
    # Section handlers.
    "SectionHandler",
    "RootSectionHandler",
    "SimpleSectionHandler",
    "StackSectionHandler",
    # Metadata parsers.
    "MetadataParser",
    "NullMetadataParser",
    "SimpleDescriptionParser",
    "KeyValueMetadataParser",
]


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
        """Add a subsection with the given name and line range."""
        child = SectionTreeNode(name, start, end)
        self.children.append(child)
        return child

    def add_keybind(self, keybind: Keybind) -> None:
        """Add `keybind` to this section as a direct child."""
        if keybind.line is not None:
            assert keybind.line >= self.start
            if self.end is not None:
                assert keybind.line <= self.end
        self.keybind_children.append(keybind)

    @staticmethod
    def _get_level_prefix(level: int) -> str:
        return f"{' ' * (level-1)}└{'─' * (level-1)}"

    @staticmethod
    def _default_keybind_child_callback(keybind: Keybind, level: int) -> None:
        msg = f"{SectionTreeNode._get_level_prefix(level)}"
        if "description" in keybind.metadata:
            desc = keybind.metadata["description"]
            # TODO: handle exceptions
            expanded = expand_sequences(desc)
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
                assert isinstance(expanded, SpanTree)
                # Try to print each permutation separately.
                permutations = expanded.generate_permutations()
                if len(permutations) == len(keybind.hotkey.permutations):
                    prefix = msg
                    for hotkey_perm, desc_perm in zip(
                        keybind.hotkey.permutations, permutations
                    ):
                        msg = f"{prefix} {str(hotkey_perm)!r} (line {keybind.line})"
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
    """Abstract base class for managing sxhkdrc sections.

    Abstract methods/properties:
        - reset
        - clone_config
        - push
        - root
        - current_section
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the section handler."""
        raise NotImplementedError

    @abstractmethod
    def clone_config(self) -> SectionHandler:
        """Create an empty instance with the same configuration."""
        raise NotImplementedError

    @abstractmethod
    def push(self, text: str, line: int) -> bool:
        """Return whether `text` can be parsed as a section header or footer.

        This should be used for delimiting sections and metadata comments.
        """
        raise NotImplementedError

    def push_eof(self, last_line: int) -> None:
        """Do clean-up actions after the input ends.

        This should be called by overriding subclasses so that the root section
        gets its `end` attribute defined.
        """
        self.root.end = last_line

    @abstractproperty
    def root(self) -> SectionTreeNode:
        """Return the root section."""
        raise NotImplementedError

    @abstractproperty
    def current_section(self) -> SectionTreeNode:
        """Return the current section."""
        raise NotImplementedError


@dataclass
class RootSectionHandler(SectionHandler):
    """Handler for a single-section sxhkdrc, where all keybinds are children of the root."""

    _section: SectionTreeNode

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the section handler."""
        self._section = SectionTreeNode.build_root()

    def clone_config(self) -> RootSectionHandler:
        """Create an empty instance with the same configuration."""
        return RootSectionHandler()

    def push(self, text: str, line: int) -> bool:
        """Reject the input.

        No new sections will be defined.
        """
        return False

    @property
    def root(self) -> SectionTreeNode:
        """Return the root section (which is the only node)."""
        return self._section

    @property
    def current_section(self) -> SectionTreeNode:
        """Return the current section (which is always the root)."""
        return self._section


@dataclass
class SimpleSectionHandler(SectionHandler):
    """Handler for sections one level under the root section.

    Each new section is a direct child of the root and each such section ends
    when a new section begins or EOF.

    Instance variables:
        section_header_re: the pattern to match and parse out section headers.
        sections: the sections of the config in the order they were defined.
    """

    section_header_re: Pattern[str]
    sections: List[SectionTreeNode]
    _root: SectionTreeNode = field(repr=False)

    def __init__(self, section_header_re: Union[str, re.Pattern[str]]):
        """Create an instance with a regex for section headers.

        `section_header_re` must have a named group 'name'.
        """
        if isinstance(section_header_re, re.Pattern):
            self.section_header_re = section_header_re
        else:
            self.section_header_re = re.compile(section_header_re)
        if "name" not in self.section_header_re.groupindex:
            raise ValueError(
                "section header regex must have the named group 'name'"
            )
        self.reset()

    def reset(self) -> None:
        """Reset the section handler."""
        self._root = SectionTreeNode.build_root()
        self.sections = [self._root]

    def clone_config(self) -> SimpleSectionHandler:
        """Create an empty instance with the same configuration."""
        return SimpleSectionHandler(self.section_header_re)

    def push(self, text: str, line: int) -> bool:
        """Return whether `text` can be parsed as a section header.

        If `section_header_re` matches `text`, it is a section header: a new
        section is added and the previous section ended.
        """
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
        """Return the root section."""
        return self._root

    @property
    def current_section(self) -> SectionTreeNode:
        """Return the latest section."""
        return self.sections[-1]


@dataclass
class StackSectionHandler(SectionHandler):
    """Handler for recursive sections and subsections.

    This uses a stack to keep track of sections, and requires that inner
    sections be merged before their parent sections.

    Instance variables:
        section_header_re: the pattern to match and parse out section headers.
        section_footer_re: the pattern to match section footers.
    """

    section_header_re: Pattern[str]
    section_footer_re: Pattern[str]
    _section_tree: SectionTreeNode = field(repr=False)
    _section_stack: List[SectionTreeNode] = field(repr=False)

    def __init__(
        self,
        section_header_re: Union[str, re.Pattern[str]],
        section_footer_re: Union[str, re.Pattern[str]],
    ):
        """Create an instance with regexes for section headers and footers.

        `section_header_re` must have a named group 'name'.
        `section_footer_re` doesn't need any named groups.
        """
        if isinstance(section_header_re, re.Pattern):
            self.section_header_re = section_header_re
        else:
            self.section_header_re = re.compile(section_header_re)
        if isinstance(section_footer_re, re.Pattern):
            self.section_footer_re = section_footer_re
        else:
            self.section_footer_re = re.compile(section_footer_re)
        if "name" not in self.section_header_re.groupindex:
            raise ValueError(
                "section header regex must have the named group 'name'"
            )
        self.reset()

    def reset(self) -> None:
        """Reset the section handler."""
        self._section_tree = SectionTreeNode.build_root()
        self._section_stack = [self._section_tree]

    def clone_config(self) -> StackSectionHandler:
        """Create an empty instance with the same configuration."""
        return StackSectionHandler(
            self.section_header_re, self.section_footer_re
        )

    def push(self, text: str, line: int) -> bool:
        """Return whether `text` can be parsed as a section header or footer.

        If `section_header_re` matches `text`, it is a section header: a new
        section is created and added to the stack.

        If `section_footer_re` matches `text`, it is a section footer: the
        current section's ending is defined and it is popped from the stack.

        NOTE: ending a section before any have been defined raises SectionPushError.
        """
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
        """Ensure no sections have been left unclosed.

        If any *have* been left open, SectionEOFError is raised.
        Otherwise, defines the ending line number for the root section.
        """
        if len(self._section_stack) > 1:
            raise SectionEOFError(
                f"Got EOF while reading section '{self._section_stack[-1].name}'",
                last_line=last_line,
                sections=self._section_stack[1:],
            )
        super().push_eof(last_line)

    @property
    def root(self) -> SectionTreeNode:
        """Return the root section."""
        return self._section_tree

    @property
    def current_section(self) -> SectionTreeNode:
        """Return the section at the top of the stack."""
        return self._section_stack[-1]


class MetadataParser(ABC):
    """Abstract base class for parsing metadata comments for a keybind.

    Abstract methods:
        - parse
    """

    @abstractmethod
    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        """Parse metadata from the comments immediately preceding a keybind."""
        raise NotImplementedError


@dataclass
class NullMetadataParser(MetadataParser):
    """Parser that always returns an empty mapping when parsing."""

    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        """Return an empty dict."""
        return {}


@dataclass
class SimpleDescriptionParser(MetadataParser):
    """Parser for the description line immediately above a keybind.

    Instance variables:
        description_re: the pattern to parse out the description from the comment.
    """

    description_re: Pattern[str]

    def __init__(self, description_re: str):
        """Create an instance with a regex for description lines.

        `description_re` must have a named group 'description'.
        """
        self.description_re = re.compile(description_re)
        if "description" not in self.description_re.groupindex:
            raise ValueError(
                "description regex must have the named group 'description'"
            )

    def parse(self, lines: Iterable[str], start_line: int) -> Dict[str, Any]:
        """Parse a description from the last element of `lines`.

        If the regex match with `description_re` succeeds, a dict with the
        key-value pair "description" and the 'description' match group is
        returned.  Otherwise, an empty dict is returned instead.
        """
        comments = list(lines)
        assert len(comments) > 0
        maybe_description = comments[-1]
        m = self.description_re.search(maybe_description)
        if m:
            return {"description": m.group("description")}
        else:
            return {}


@dataclass
class KeyValueMetadataParser(MetadataParser):
    """Parser for key-value pairs in metadata.

    Instance variables:
        pair_re: the pattern to parse out key-value pairs.
        empty_re: the pattern to match empty metadata lines.
    """

    pair_re: Pattern[str]
    empty_re: Pattern[str]  # part of description but no pair

    def __init__(self, pair_re: str, empty_re: str):
        """Create an instance with regexes for pairs and empty metadata lines.

        `pair_re` must have the named groups 'key' and 'value'.
        `empty_re` doesn't need any named groups.
        """
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
        """Parse metadata comments for key-value pairs.

        They are parsed in the reverse order of `lines` so that the contiguous
        block of metadata comments immediately preceding a keybind isn't
        interrupted by non-metadata comments.

        `empty_re` is used to match comments that are part of that contiguous
        block, but don't define key-value pairs.  Useful for "blank" lines
        between key-value pairs or for comments about them in a separate line.

        NOTE: duplicate keys cause MetadataParserError to be raised.
        """
        comments = list(lines)
        # Each dict value stores the line number of the pair that is latest in
        # the file, so that errors on duplicates show the actual duplicate
        # rather than the first occurrence.
        metadata: Dict[str, Tuple[int, Any]] = {}
        i = len(comments) - 1
        while i >= 0:
            comment = comments[i]
            m = self.pair_re.search(comment)
            if m:
                key = m.group("key")
                value = m.group("value")
                if key in metadata:
                    dup_line, _ = metadata[key]
                    raise MetadataParserError(
                        f"Duplicate key '{key}'",
                        key=key,
                        value=value,
                        line=dup_line,
                    )
                metadata[key] = (start_line + i, value)
                i -= 1
            elif self.empty_re.search(comment):
                # metadata continues
                i -= 1
                continue
            else:
                break
        return {key: val for key, (_, val) in metadata.items()}
