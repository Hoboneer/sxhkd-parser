"""Tool for linting hotkeys."""
from __future__ import annotations

import argparse
from itertools import chain
from typing import List, Optional, Union, cast

from ..errors import SXHKDParserError
from ..parser import HotkeyTree, SequenceSpan
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    Message,
    find_duplicates,
    find_maybe_invalid_keysyms,
    find_prefix_conflicts,
    format_error_msg,
    get_command_name,
    print_exceptions,
    process_args,
)

__all__ = ["main"]


# Current as of sxhkd v0.5.1 - v0.6.2 (inclusive).
SXHKD_MAXLEN = 256
HOTKEY_MAXLEN = COMMAND_MAXLEN = 2 * SXHKD_MAXLEN


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.
    """
    parser = argparse.ArgumentParser(
        get_command_name(__file__),
        description="Check hotkeys for invalid keysyms, duplicates, conflicts, and truncation",
        parents=[BASE_PARSER],
    )
    parser.add_argument(
        "--sxhkd-version",
        "-S",
        default="auto",
        help="minimum sxhkd version: if 'auto', determine from output of `sxhkd -v` (currently a no-op) (default: %(default)s)",
    )

    namespace = parser.parse_args(argv)
    section_handler, metadata_parser = process_args(namespace)

    errors: List[Union[Message, SXHKDParserError]] = []
    # TODO: maybe include other internal nodes?
    INTERNAL_NODES = ["modifierset"]
    hotkey_tree = HotkeyTree(INTERNAL_NODES)
    try:
        for bind_or_err in read_sxhkdrc(
            namespace.sxhkdrc,
            section_handler=section_handler,
            metadata_parser=metadata_parser,
            # Handle them ourselves.
            hotkey_errors=IGNORE_HOTKEY_ERRORS,
        ):
            if isinstance(bind_or_err, SXHKDParserError):
                errors.append(bind_or_err)
                continue

            keybind = bind_or_err
            errors.extend(find_maybe_invalid_keysyms(keybind))

            # Check for truncated keybinds (since sxhkd avoids dynamic memory allocation)
            # See https://github.com/baskerville/sxhkd/issues/139
            if isinstance(keybind.hotkey.raw, list):
                hotkey_txt = "\\\n".join(keybind.hotkey.raw)
            else:
                hotkey_txt = keybind.hotkey.raw
            if len(hotkey_txt.encode()) > HOTKEY_MAXLEN:
                errors.append(
                    Message(
                        keybind.hotkey.line,
                        None,
                        f"Hotkey text exceeds {HOTKEY_MAXLEN} bytes so it may be truncated by sxhkd",
                    )
                )
            if isinstance(keybind.command.raw, list):
                command_txt = "\\\n".join(keybind.command.raw)
            else:
                command_txt = keybind.command.raw
            if len(command_txt.encode()) > COMMAND_MAXLEN:
                errors.append(
                    Message(
                        keybind.command.line,
                        None,
                        f"Command text exceeds {COMMAND_MAXLEN} bytes so it may be truncated by sxhkd",
                    )
                )

            # Check for mistakenly unescaped sequence characters.
            for span in chain(
                keybind.hotkey.span_tree.levels,
                keybind.command.span_tree.levels,
            ):
                if not isinstance(span, SequenceSpan):
                    continue
                if len(span.choices) == 1:
                    errors.append(
                        Message(
                            span.line,
                            span.col,
                            "Sequence with only one element: did you forget to escape the braces?",
                        )
                    )
            hotkey_tree.merge_hotkey(keybind.hotkey)
    except SXHKDParserError as e:
        print_exceptions(e, namespace.sxhkdrc)
        return 1

    errors.extend(find_duplicates(hotkey_tree))
    errors.extend(find_prefix_conflicts(hotkey_tree))

    numbered = []
    rest = []
    for msg in errors:
        if msg.line is not None:
            numbered.append(msg)
        else:
            rest.append(msg)
    numbered.sort(key=lambda x: cast(int, x.line))
    for msg in numbered:
        print(format_error_msg(msg, config_filename=namespace.sxhkdrc))
    for msg in rest:
        print(format_error_msg(msg, config_filename=namespace.sxhkdrc))

    return 0
