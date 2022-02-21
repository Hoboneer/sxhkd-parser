from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .metadata import SectionTreeNode
    from .parser import Chord, HotkeyToken, _HotkeyParseMode

    TransitionTable = Dict[
        str, Tuple[_HotkeyParseMode, Callable[[HotkeyToken], None]]
    ]


class SXHKDParserError(Exception):
    pass


class SequenceParseError(SXHKDParserError):
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
        prefix = ""
        if self.line is not None:
            prefix = f"{prefix}{self.line}:"
        if self.column is not None:
            prefix = f"{prefix}{self.column}: "
        return f"{prefix}{self.message}"


class HotkeyTokenizeError(SXHKDParserError):
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


class HotkeyParseError(SXHKDParserError):
    pass


class UnexpectedTokenError(HotkeyParseError):
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
    def __init__(self, message: str, mode: _HotkeyParseMode):
        super().__init__(message)
        self.mode = mode


class InconsistentNoabortError(HotkeyParseError):
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
    pass


class SectionPushError(SectionHandlerError):
    def __init__(self, message: str, line: int):
        self.message = message
        self.line = line

    def __str__(self) -> str:
        return f"{self.line}: {self.message}"


class SectionEOFError(SectionHandlerError):
    def __init__(
        self, message: str, last_line: int, sections: List[SectionTreeNode]
    ):
        self.message = message
        self.last_line = last_line
        self.sections = sections


class MetadataParserError(SXHKDParserError):
    def __init__(self, message: str, key: str, value: Any, line: int):
        self.message = message
        self.key = key
        self.value = value
        self.line = line

    def __str__(self) -> str:
        return f"{self.line}: {self.message}"
