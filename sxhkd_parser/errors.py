"""Exceptions for the library.

SXHKDParserError is the ancestor for almost all of the exceptions in the
library.  Some functions and methods raise ValueError, but they are few and
small in scope.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from .metadata import SectionTreeNode
    from .parser import (
        HotkeyPermutation,
        HotkeyToken,
        HotkeyTreeLeafNode,
        _HotkeyParseMode,
    )

    TransitionTable = Dict[
        str, Tuple[_HotkeyParseMode, Callable[[HotkeyToken], None]]
    ]

__all__ = [
    "SXHKDParserError",
    # ---
    "ConfigReadError",
    "MissingHotkeyError",
    # ---
    "KeybindError",
    "InconsistentKeybindCasesError",
    # ---
    "SequenceParseError",
    # ---
    "HotkeyError",
    "HotkeyTokenizeError",
    "HotkeyParseError",
    "UnexpectedTokenError",
    "NonTerminalStateExitError",
    "InconsistentNoabortError",
    "DuplicateModifierError",
    "DuplicateChordPermutationError",
    "ConflictingChainPrefixError",
    "PossiblyInvalidKeysyms",
    # ---
    "SectionHandlerError",
    "SectionPushError",
    "SectionEOFError",
    # ---
    "MetadataParserError",
]


class SXHKDParserError(Exception):
    """Ancestor for almost all of the exceptions in the library."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
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


class ConfigReadError(SXHKDParserError):
    """Base class for errors related to reading sxhkdrc files."""

    pass


class MissingHotkeyError(ConfigReadError):
    """Command text was read without an accompanying hotkey."""

    pass


class KeybindError(SXHKDParserError):
    """Base class for errors related to `Keybind` objects."""

    pass


class InconsistentKeybindCasesError(KeybindError):
    """A keybind had an inconsistent number of cases."""

    def __init__(
        self,
        message: str,
        hotkey_cases: int,
        command_cases: int,
        line: Optional[int] = None,
    ):
        super().__init__(message=message, line=line)
        self.hotkey_cases = hotkey_cases
        self.command_cases = command_cases


class SequenceParseError(SXHKDParserError):
    """A sequence of the form {s1,s2,...,sn} failed to parse."""

    def __init__(
        self,
        message: str,
        text: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ):
        super().__init__(message=message, line=line, column=column)
        self.text = text
        self.line = line


class HotkeyError(SXHKDParserError):
    """Base class for errors related to `Hotkey` objects."""

    pass


class HotkeyTokenizeError(HotkeyError):
    """An invalid character was passed to the tokenizer."""

    def __init__(
        self, message: str, hotkey: str, value: str, line: Optional[int] = None
    ):
        super().__init__(message=message, line=line)
        self.hotkey = hotkey
        self.value = value


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
        super().__init__(message=message)
        self.token = token
        self.mode = mode
        self.transitions = transitions
        self.tokens = tokens


class NonTerminalStateExitError(HotkeyParseError):
    """The input ended on a non-terminal parser state."""

    def __init__(self, message: str, mode: _HotkeyParseMode):
        super().__init__(message=message)
        self.mode = mode


class InconsistentNoabortError(HotkeyParseError):
    """The colon character was used in different places along the permutations of a hotkey."""

    def __init__(
        self,
        message: str,
        perm1: HotkeyPermutation,
        perm1_index: int,
        perm2: HotkeyPermutation,
        perm2_index: int,
        line: Optional[int] = None,
    ):
        super().__init__(message=message, line=line)
        self.perm1 = perm1
        self.perm1_index = perm1_index
        self.perm2 = perm2
        self.perm2_index = perm2_index


class DuplicateModifierError(HotkeyParseError):
    """A modifier was repeated in the same chord."""

    def __init__(self, message: str, modifier: str):
        super().__init__(message=message)
        self.modifier = modifier


class DuplicateChordPermutationError(HotkeyError):
    """A chord permutation was a duplicate of another."""

    def __init__(
        self,
        message: str,
        dup_perm: HotkeyPermutation,
        perm1_index: int,
        perm2_index: int,
        line: Optional[int] = None,
    ):
        super().__init__(message=message, line=line)
        self.dup_perm = dup_perm
        self.perm1_index = perm1_index
        self.perm2_index = perm2_index


class ConflictingChainPrefixError(HotkeyError):
    """An entire permutation of a hotkey was a prefix of another."""

    def __init__(
        self,
        message: str,
        chain_prefix: HotkeyTreeLeafNode,
        conflicts: List[HotkeyTreeLeafNode],
        line: Optional[int] = None,
    ):
        super().__init__(message=message, line=line)
        self.chain_prefix = chain_prefix
        self.conflicts = conflicts


class PossiblyInvalidKeysyms(HotkeyError):
    """Possibly invalid keysyms were found in a hotkey.

    Based on the keysyms found in X11 include files `keysymdef.h` and
    `XF86keysym.h` from Debian Bullseye's `x11proto-dev` package.

    There may be non-standard keysyms that people may want to use, so this
    shouldn't be fatal by default.
    """

    def __init__(
        self,
        message: str,
        keysyms: Set[str],
        line: Optional[int] = None,
    ):
        super().__init__(message=message, line=line)
        self.keysyms = keysyms


class SectionHandlerError(SXHKDParserError):
    """Base class for issues related to the management of sections."""

    pass


class SectionPushError(SectionHandlerError):
    """Miscellaneous errors while pushing a potential section."""

    def __init__(self, message: str, line: int):
        super().__init__(message=message, line=line)


class SectionEOFError(SectionHandlerError):
    """Miscellaneous errors after receiving EOF."""

    def __init__(
        self, message: str, last_line: int, sections: List[SectionTreeNode]
    ):
        super().__init__(message=message, line=last_line)
        self.message = message
        self.last_line = last_line
        self.sections = sections


class MetadataParserError(SXHKDParserError):
    """Miscellaneous errors while instance was parsing comments for metadata."""

    def __init__(self, message: str, key: str, value: Any, line: int):
        super().__init__(message=message, line=line)
        self.key = key
        self.value = value
