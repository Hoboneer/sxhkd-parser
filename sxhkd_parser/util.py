"""Convenience functions for using the library."""
import re
from os import PathLike
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from .errors import (
    MissingHotkeyError,
    SXHKDParserError,
    UnterminatedParserConfigDirectiveError,
)
from .metadata import MetadataParser, SectionHandler
from .parser import Keybind

__all__ = ["read_sxhkdrc"]


T = TypeVar("T", SectionHandler, MetadataParser)


def _generate_from_options_func(
    type_: Type[T],
) -> Callable[[Dict[str, str]], T]:
    def cls_from_options(options: Dict[str, str]) -> T:
        typename = options["type"]
        args = []
        cls = type_.TYPES[typename]
        for opt in cls.OPTIONS:
            args.append(options[opt])
        return cls(*args)

    return cls_from_options


make_section_handler_from_options = _generate_from_options_func(SectionHandler)  # type: ignore
make_metadata_parser_from_options = _generate_from_options_func(MetadataParser)  # type: ignore


DIRECTIVE_COMMENT_RE = re.compile(
    r"^#\?\s*(?P<name>[A-Za-z0-9-]+):\s*(?P<value>.+)\s*$"
)


def read_sxhkdrc(
    file: Union[str, PathLike[str], TextIO],
    section_handler: Optional[SectionHandler] = None,
    metadata_parser: Optional[MetadataParser] = None,
    hotkey_errors: Optional[Mapping[str, bool]] = None,
) -> Generator[
    Union[SXHKDParserError, Keybind],
    None,
    Tuple[Optional[SectionHandler], Optional[MetadataParser]],
]:
    """Parse keybinds from a given file or path, yielding a stream.

    `file` may be a filename, `os.PathLike`, or an opened file.  In the case of
    an opened file, it is not closed at the end of the function.

    A block of comments is maintained so that it can be passed to
    `MetadataParser.parse` and given to a `Keybind`.

    Upon encountering an empty line, this isolated block of comments is cleared
    so that it can't be attached to any keybinds.

    Upon encountering a comment line, the section first has its `push` method
    called on the line to define any sections if it matches.
    If a comment *doesn't* create a new section (`push` returns `False`), the
    line is added to the block of comments.
    But if a comment *does* create a new section, the block of comments is
    cleared, cutting off comments above section comments from any keybinds
    below them.

    If `section_handler` is `None`, no sections (i.e., `SectionTreeNode`
    objects) are made: they must be created manually afterwards.  Otherwise,
    the given SectionHandler instance is used to manage sections and add
    keybinds to them.

    If `metadata_parser` is `None`, all metadata comments are ignored and every
    `Keybind` instance is given an empty dict instead.
    """
    if isinstance(file, (str, PathLike)):
        f = open(file)
        close_io = True
    else:
        f = file
        # Passed in a file object.
        close_io = False

    hotkey: Optional[List[str]] = None
    hotkey_start_line: Optional[int] = None
    command: Optional[List[str]] = None
    command_start_line: Optional[int] = None

    comment_block_start_line: Optional[int] = None
    comment_buf: List[str] = []

    lines: List[str] = []
    line_no: int = 0
    line_block_start: int
    metadata: Dict[str, Any] = {}
    num_yields = 0
    try:
        while True:
            line = f.readline()
            # readline() returns the empty string upon EOF.
            if not line:
                break
            line_no += 1
            line_block_start = line_no
            line = line.rstrip("\n")
            # sxhkd ignores empty lines.
            if not line:
                # Cut off any isolated comment blocks.
                comment_block_start_line = None
                comment_buf.clear()
                continue

            # Only accept inline specification of section and/or metadata type
            # before any keybinds.
            if num_yields == 0 and re.match(
                r"^#\?\s*Begin-Parser-Config\s*$", line
            ):
                format_lines = []
                line = f.readline()
                if not line:
                    raise UnterminatedParserConfigDirectiveError(
                        "sxhkdrc ended before End-Parser-Config", line=line_no
                    )
                line_no += 1
                while not re.match(r"^#\?\s*End-Parser-Config\s*$", line):
                    assert line.startswith(
                        "#"
                    ), f"line {line_no} must be a comment ({line!r})"
                    format_lines.append(line.rstrip("\n"))
                    line = f.readline()
                    if not line:
                        raise UnterminatedParserConfigDirectiveError(
                            "sxhkdrc ended before End-Parser-Config",
                            line=line_no,
                        )
                    line_no += 1
                directives = {}
                for fline in format_lines:
                    m = DIRECTIVE_COMMENT_RE.match(fline)
                    if not m:
                        continue
                    name = cast(str, m.group("name")).lower()
                    value = cast(str, m.group("value"))
                    directives[name] = value
                # TODO: handle missing keys
                if section_handler is None and "section-type" in directives:
                    section_handler = make_section_handler_from_options(
                        {
                            re.sub(r"^section-", "", name): value
                            for name, value in directives.items()
                            if name.startswith("section-")
                        }
                    )
                if metadata_parser is None and "metadata-type" in directives:
                    metadata_parser = make_metadata_parser_from_options(
                        {
                            re.sub(r"^metadata-", "", name): value
                            for name, value in directives.items()
                            if name.startswith("metadata-")
                        }
                    )
                continue

            if line.startswith("#"):
                # line number so keybinds can be matched to sections later
                if section_handler is not None and section_handler.push(
                    line, line_no
                ):
                    # section handler ate it up
                    comment_block_start_line = None
                    comment_buf.clear()  # cut off any comments that could attach to a keybind
                # Metadata only comes from comments directly preceding the hotkey.
                elif hotkey is None:
                    if not comment_buf:
                        assert comment_block_start_line is None
                        comment_block_start_line = line_no
                    comment_buf.append(line)
                continue

            lines.append(line)
            while lines[-1][-1] == "\\":
                # Exclude the backslash character escaping the newline.
                lines[-1] = lines[-1][:-1]
                next_line = f.readline()
                # readline() returns the empty string upon EOF.
                if not next_line:
                    break
                lines.append(next_line.rstrip("\n"))
                line_no += 1

            if line.startswith((" ", "\t")):
                # command!
                if hotkey is None:
                    # Fatal error, so don't yield the exception.
                    raise MissingHotkeyError(
                        "Missing hotkey while reading a command: did you forget to escape a newline?",
                        line=line_no,
                    )
                command = lines.copy()
                command_start_line = line_block_start

                try:
                    keybind = Keybind(
                        hotkey,
                        command,
                        hotkey_start_line=hotkey_start_line,
                        command_start_line=command_start_line,
                        metadata=metadata,
                        hotkey_errors=hotkey_errors,
                    )
                except SXHKDParserError as e:
                    yield e
                else:
                    if section_handler is not None:
                        section_handler.current_section.add_keybind(keybind)
                    yield keybind
                num_yields += 1

                hotkey = command = None
                hotkey_start_line = command_start_line = None
                lines.clear()
            else:
                # hotkey!
                assert hotkey is None and hotkey_start_line is None, repr(
                    (hotkey, hotkey_start_line, line)
                )
                hotkey = lines.copy()
                hotkey_start_line = line_block_start
                lines.clear()
                if comment_buf and metadata_parser is not None:
                    metadata = metadata_parser.parse(
                        comment_buf,
                        start_line=cast(int, comment_block_start_line),
                    )
                else:
                    metadata = {}
                comment_buf.clear()
                comment_block_start_line = None

    except Exception:
        raise
    finally:
        try:
            if section_handler is not None:
                section_handler.push_eof(line_no)
        finally:
            if close_io:
                f.close()
            return (section_handler, metadata_parser)
