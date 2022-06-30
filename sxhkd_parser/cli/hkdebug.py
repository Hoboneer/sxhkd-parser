"""Tool for debugging sxhkd configs."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..errors import SXHKDParserError
from ..metadata import SectionTreeNode
from ..parser import HotkeyTree, Keybind, SequenceSpan, Span, TextSpan
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    format_error_msg,
    get_command_name,
    print_exceptions,
    process_args,
)

__all__ = ["main"]


INTERNAL_NODES = ["modifierset"]


def parse_internal_nodes(s: str) -> Optional[List[str]]:
    if s == "none":
        return None
    types = s.split(",")
    if not (set(types) <= set(HotkeyTree.INTERNAL_NODE_TYPES)):
        raise argparse.ArgumentTypeError(
            f"invalid types {types!r}... must be one of {HotkeyTree.INTERNAL_NODE_TYPES!r}"
        )
    return types


def print_span_tree_level(level: Span) -> None:
    if isinstance(level, TextSpan):
        print(repr(level.text))
    else:
        assert isinstance(level, SequenceSpan)
        print([item.text for item in level.choices])


def print_keybind(keybind: Keybind) -> None:
    print("Hotkey:")
    print(f"\tline: {keybind.hotkey.line}")
    print(f"\traw: {keybind.hotkey.raw}")
    print(f"\tnoabort_index: {keybind.hotkey.noabort_index}")
    print("Command:")
    print(f"\tline: {keybind.command.line}")
    print(f"\traw: {keybind.command.raw}")
    print(f"\tsynchronous?: {keybind.command.synchronous}")
    print("Metadata:")
    for key, val in keybind.metadata.items():
        print(f"\t{key}: {val!r}")


def print_sections(
    dirname: List[Optional[str]], node: SectionTreeNode
) -> None:
    currpath = dirname + [node.name]
    for keybind in node.keybind_children:
        print_keybind(keybind)
        print(f"Section path: {currpath}")
        print()
    for subsection in node.children:
        print_sections(currpath, subsection)


PROGNAME = get_command_name(__file__)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.
    """
    parser = argparse.ArgumentParser(
        PROGNAME,
        description="Debug sxhkd config files",
        parents=[BASE_PARSER],
    )

    subparsers = parser.add_subparsers(
        title="subcommands", metavar="mode", dest="mode", required=True
    )

    parser_keybinds = subparsers.add_parser(
        "keybinds",
        help="print summary of parsed keybinds with their metadata",
    )
    parser_keybinds.add_argument(
        "--include-sections",
        "-S",
        action="store_true",
        help="include section information of each keybind",
    )

    parser_hotkey_tree = subparsers.add_parser(
        "hotkey-tree",
        help="print tree containing the keypresses needed for each hotkey permutation",
    )
    braced_all_choices = "{%s}" % ",".join(HotkeyTree.INTERNAL_NODE_TYPES)
    parser_hotkey_tree.add_argument(
        "--internal-nodes",
        "-I",
        type=parse_internal_nodes,
        default=INTERNAL_NODES,
        help=f"the comma-separated sequence of internal node types to be included in the tree (choices: {braced_all_choices}) (default: {','.join(INTERNAL_NODES)})",
    )

    parser_span_tree = subparsers.add_parser(
        "span-tree",
        help="print tree containing the spans of text resulting from sequence expansion",
    )
    parser_span_tree.add_argument(
        "--type",
        "-t",
        choices=["both", "hotkey", "command"],
        default="both",
        help="whether to include span trees of hotkeys or commands (or both) (default: %(default)s)",
    )
    parser_span_tree.add_argument(
        "--levels",
        "-L",
        action="store_true",
        help="print levels of the span tree rather than its tree representation",
    )

    namespace = parser.parse_args(argv)
    section_handler, metadata_parser = process_args(namespace)

    keybinds = []
    try:
        for bind_or_err in read_sxhkdrc(
            namespace.sxhkdrc,
            section_handler=section_handler,
            metadata_parser=metadata_parser,
            # Ignore them: it is another tool's job to check.
            hotkey_errors=IGNORE_HOTKEY_ERRORS,
        ):
            if isinstance(bind_or_err, SXHKDParserError):
                msg = format_error_msg(bind_or_err, namespace.sxhkdrc)
                print(msg, file=sys.stderr)
                continue
            keybinds.append(bind_or_err)
    except SXHKDParserError as e:
        print_exceptions(e, namespace.sxhkdrc, file=sys.stderr)
        return 1

    # Copied straight from `apt`.
    print(
        f"WARNING: {PROGNAME} does not have a stable CLI interface. Use with caution in scripts.",
        file=sys.stderr,
    )
    if namespace.mode == "keybinds":
        if namespace.include_sections:
            print_sections([], section_handler.root)
        else:
            for keybind in keybinds:
                print_keybind(keybind)
                print()
    elif namespace.mode == "hotkey-tree":
        hotkey_tree = HotkeyTree(namespace.internal_nodes)
        for keybind in keybinds:
            hotkey_tree.merge_hotkey(keybind.hotkey)
        hotkey_tree.print_tree()
    elif namespace.mode == "span-tree":
        for keybind in keybinds:
            if namespace.type in ("hotkey", "both"):
                print(
                    f"hotkey line: {keybind.hotkey.line}; raw hotkey: {keybind.hotkey.raw}"
                )
                if namespace.levels:
                    for level in keybind.hotkey.span_tree.levels:
                        print_span_tree_level(level)
                else:
                    keybind.hotkey.span_tree.print_tree()
            if namespace.type in ("command", "both"):
                print(
                    f"command line: {keybind.command.line}; raw command: {keybind.command.raw}"
                )
                if namespace.levels:
                    for level in keybind.command.span_tree.levels:
                        print_span_tree_level(level)
                else:
                    keybind.command.span_tree.print_tree()
            print()
    else:
        assert 0, "unreachable"
    return 0
