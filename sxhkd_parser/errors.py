"""Exceptions for the library.

SXHKDParserError is the ancestor for almost all of the exceptions in the
library.  Some functions and methods raise ValueError, but they are few and
small in scope.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .metadata import SectionTreeNode
    from .parser import Chord, HotkeyToken, _HotkeyParseMode

    TransitionTable = Dict[
        str, Tuple[_HotkeyParseMode, Callable[[HotkeyToken], None]]
    ]

__all__ = [
    "SXHKDParserError",
    # ---
    "SequenceParseError",
    # ---
    "HotkeyError",
    "HotkeyTokenizeError",
    "HotkeyParseError",
    "UnexpectedTokenError",
    "NonTerminalStateExitError",
    "InconsistentNoabortError",
    # ---
    "SectionHandlerError",
    "SectionPushError",
    "SectionEOFError",
    # ---
    "MetadataParserError",
]


class SXHKDParserError(Exception):
    """Ancestor for almost all of the exceptions in the library."""

    pass


class SequenceParseError(SXHKDParserError):
    """A sequence of the form {s1,s2,...,sn} failed to parse."""

    def __init__(
        self,
        message: str,
        text: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ):
        self.message = message
        self.text = text
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line is not None:
            if self.column is None:
                return f"{self.line}: {self.message}"
            else:
                return f"{self.line}:{self.column}: {self.message}"
        elif self.column is not None:
            return f"{self.message} at column {self.column}"
        else:
            return self.message


class HotkeyError(SXHKDParserError):
    """Base class for errors related to `Hotkey` objects."""

    pass


class HotkeyTokenizeError(HotkeyError):
    """An invalid character was passed to the tokenizer."""

    def __init__(
        self, message: str, hotkey: str, value: str, line: Optional[int] = None
    ):
        self.message = message
        self.hotkey = hotkey
        self.value = value
        self.line = line

    def __str__(self) -> str:
        if self.line is not None:
            return f"{self.line}: {self.message}"
        else:
            return self.message


class HotkeyParseError(HotkeyError):
    """Base class for issues related to parsing hotkeys into Hotkey objects."""

    pass


class UnexpectedTokenError(HotkeyParseError):
    """A token was in the wrong position."""

    def __init__(
        self,
        message: str,
        token: HotkeyToken,
        mode: _HotkeyParseMode,
        transitions: TransitionTable,
        tokens: List[HotkeyToken],
    ):
        super().__init__(message)
        self.token = token
        self.mode = mode
        self.transitions = transitions
        self.tokens = tokens


class NonTerminalStateExitError(HotkeyParseError):
    """The input ended on a non-terminal parser state."""

    def __init__(self, message: str, mode: _HotkeyParseMode):
        super().__init__(message)
        self.mode = mode


class InconsistentNoabortError(HotkeyParseError):
    """The colon character was used in different places along the permutations of a hotkey."""

    def __init__(
        self,
        message: str,
        perms: List[List[Chord]],
        indices: List[Optional[int]],
        index_counts: Dict[Optional[int], int],
    ):
        super().__init__(message)
        self.perms = perms
        self.indices = indices
        self.index_counts = index_counts


class SectionHandlerError(SXHKDParserError):
    """Base class for issues related to the management of sections."""

    pass


class SectionPushError(SectionHandlerError):
    """Miscellaneous errors while pushing a potential section."""

    def __init__(self, message: str, line: int):
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return f"{self.line}: {self.message}"


class SectionEOFError(SectionHandlerError):
    """Miscellaneous errors after receiving EOF."""

    def __init__(
        self, message: str, last_line: int, sections: List[SectionTreeNode]
    ):
        self.message = message
        self.last_line = last_line
        self.sections = sections


class MetadataParserError(SXHKDParserError):
    """Miscellaneous errors while instance was parsing comments for metadata."""

    def __init__(self, message: str, key: str, value: Any, line: int):
        self.message = message
        self.key = key
        self.value = value
        self.line = line

    def __str__(self) -> str:
        return f"{self.line}: {self.message}"
