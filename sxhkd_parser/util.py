from os import PathLike
from typing import Iterable, List, Optional, TextIO, Union, cast

from .metadata import MetadataParser, NullMetadataParser, SectionHandler
from .parser import Keybind


def read_sxhkdrc(
    file: Union[str, PathLike[str], TextIO],
    section_handler: Optional[SectionHandler] = None,
    metadata_parser: Optional[MetadataParser] = None,
) -> Iterable[Keybind]:
    if isinstance(file, (str, PathLike)):
        f = open(file)
        close_io = True
    else:
        f = file
        # Passed in a file object.
        close_io = False
    if metadata_parser is None:
        metadata_parser = NullMetadataParser()

    hotkey: Optional[List[str]] = None
    hotkey_start_line: Optional[int] = None
    command: Optional[List[str]] = None
    command_start_line: Optional[int] = None

    comment_block_start_line: Optional[int] = None
    comment_buf: List[str] = []

    lines: List[str] = []
    line_no: int = 0
    line_block_start: int
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

            # TODO: add descriptions (or should it be for any metadata element?) with sequence expansion
            #   - maybe that should be up to the user? (expand_sequences is public!)
            if line.startswith("#"):
                # line number so keybinds can be matched to sections later
                if section_handler is not None:
                    if section_handler.push(line, line_no):
                        # section handler ate it up
                        comment_block_start_line = None
                        comment_buf.clear()  # cut off any comments that could attach to a keybind
                else:
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
                assert hotkey is not None
                command = lines.copy()
                command_start_line = line_block_start

                if comment_buf:
                    metadata = metadata_parser.parse(
                        comment_buf,
                        start_line=cast(int, comment_block_start_line),
                    )
                    comment_buf.clear()
                    comment_block_start_line = None
                else:
                    metadata = {}
                keybind = Keybind(
                    hotkey,
                    command,
                    hotkey_start_line=hotkey_start_line,
                    command_start_line=command_start_line,
                    metadata=metadata,
                )
                if section_handler is not None:
                    section_handler.current_section.add_keybind(keybind)
                yield keybind

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

    except Exception:
        raise
    finally:
        if section_handler is not None:
            section_handler.push_eof(line_no)
        if close_io:
            f.close()
