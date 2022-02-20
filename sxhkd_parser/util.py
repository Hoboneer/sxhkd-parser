from os import PathLike
from typing import Iterable, List, Optional, TextIO, Union, cast

from .metadata import (
    MetadataParser,
    NullMetadataParser,
    NullSectionHandler,
    SectionHandler,
)
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
    if section_handler is None:
        section_handler = NullSectionHandler(accept=False)
    if metadata_parser is None:
        metadata_parser = NullMetadataParser()

    hotkey: Optional[str] = None
    hotkey_start_line: Optional[int] = None
    command: Optional[str] = None
    command_start_line: Optional[int] = None

    comment_block_start_line: Optional[int] = None
    comment_buf: List[str] = []

    line_no: int = 0
    try:
        while True:
            line = f.readline()
            # readline() returns the empty string upon EOF.
            if not line:
                break
            line_no += 1
            line = line.rstrip("\n")
            # sxhkd ignores empty lines.
            if not line:
                continue

            # TODO: add descriptions (or should it be for any metadata element?) with sequence expansion
            #   - maybe that should be up to the user? (expand_sequences is public!)
            if line.startswith("#"):
                # line number so keybinds can be matched to sections later
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

            while line and line[-1] == "\\":
                # Exclude the backslash character escaping the newline.
                line = line[:-1]
                next_line = f.readline()
                # readline() returns the empty string upon EOF.
                if not next_line:
                    break
                line += next_line.rstrip("\n")
                line_no += 1

            if line.startswith((" ", "\t")):
                # command!
                assert hotkey is not None
                command = line
                command_start_line = line_no

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
                section_handler.current_section().add_keybind(keybind)
                yield keybind

                hotkey = command = None
                hotkey_start_line = command_start_line = None
            else:
                # hotkey!
                assert hotkey is None and hotkey_start_line is None, repr(
                    (hotkey, hotkey_start_line, line)
                )
                hotkey = line
                hotkey_start_line = line_no

    except Exception:
        raise
    finally:
        section_handler.push_eof(line_no)
        if close_io:
            f.close()
