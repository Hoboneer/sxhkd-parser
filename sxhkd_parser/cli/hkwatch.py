"""Program outputting the current sxhkd mode from the status fifo."""
from __future__ import annotations

import argparse
import itertools as it
import re
import subprocess
import sys
from signal import SIGUSR1, SIGUSR2, signal
from typing import Any, FrozenSet, List, Optional, Tuple

from ..errors import SXHKDParserError
from ..metadata import MetadataParser, SectionHandler
from ..parser import Chord, Hotkey, HotkeyTree, KeypressTreeNode
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    find_duplicates,
    find_maybe_invalid_keysyms,
    find_prefix_conflicts,
    format_error_msg,
    get_command_name,
    print_exceptions,
    process_args,
)

__all__ = ["main"]


INTERNAL_NODES = ["modifierset", "keysym"]


def read_config(
    config: str,
    section_handler: SectionHandler,
    metadata_parser: MetadataParser,
) -> Tuple[bool, HotkeyTree]:
    errored = False
    hotkey_tree = HotkeyTree(INTERNAL_NODES)
    for bind_or_err in read_sxhkdrc(
        config,
        section_handler=section_handler,
        metadata_parser=metadata_parser,
        # Handle them ourselves.
        hotkey_errors=IGNORE_HOTKEY_ERRORS,
    ):
        if isinstance(bind_or_err, SXHKDParserError):
            msg = format_error_msg(bind_or_err, config)
            print(msg, file=sys.stderr)
            errored = True
            continue

        keybind = bind_or_err
        for err in find_maybe_invalid_keysyms(keybind):
            print(format_error_msg(err, config), file=sys.stderr)
            errored = True
        hotkey_tree.merge_hotkey(keybind.hotkey)

    for err in it.chain(
        find_duplicates(hotkey_tree), find_prefix_conflicts(hotkey_tree)
    ):
        print(format_error_msg(err, config), file=sys.stderr)
        errored = True
    return (errored, hotkey_tree)


def find_matching_modsets(
    modset: FrozenSet[str], node: KeypressTreeNode
) -> Optional[KeypressTreeNode]:
    if modset in node.modifierset_children:
        return node.modifierset_children[modset]
    for mods, child in node.modifierset_children.items():
        if mods < modset:
            return find_matching_modsets(modset, child)
    return None


def match_hotkey(
    chords: List[Chord], curr_level: KeypressTreeNode
) -> Optional[KeypressTreeNode]:
    if not chords:
        assert isinstance(curr_level.value, Chord)
        return curr_level
    curr, *rest = chords
    assert {"modifierset", "keysym"} == set(INTERNAL_NODES)
    curr_level = find_matching_modsets(curr.modifiers, curr_level)
    if curr_level is None:
        return None

    if curr.keysym in curr_level.keysym_children:
        curr_level = curr_level.keysym_children[curr.keysym]
    else:
        return None

    # Now find next chord
    for child in curr_level.children:
        if not isinstance(child.value, Chord):
            continue
        if curr == child.value:
            return match_hotkey(rest, child)
    return None


PROGNAME = get_command_name(__file__)
NOTIFY_SEND_ERR_PREFIX = [
    "notify-send",
    "-t",
    "10000",
    "-u",
    "critical",
]
NOTIFY_SEND_INFO_PREFIX = [
    "notify-send",
    "-t",
    "5000",
]


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.
    """
    parser = argparse.ArgumentParser(
        PROGNAME,
        description="Tail the status fifo and output the current sxhkd mode until exit.  Needs only the H*, B*, and E* fifo messages.  Send SIGUSR1 to reload config and SIGUSR2 to print the current mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BASE_PARSER],
    )
    parser.add_argument(
        "--status-fifo",
        "-s",
        default="-",
        help="the location of the status fifo ('-' for stdin)",
    )
    parser.add_argument(
        "--mode-field",
        "-m",
        default="mode",
        help="metadata key indicating the mode that a noabort hotkey belongs to",
    )
    parser.add_argument(
        "--notify-send-on-config-read",
        action="store_true",
        help="whether to call `notify-send` when reading configs",
    )
    parser.add_argument(
        "--notify-send-on-config-read-error",
        action="store_true",
        help="whether to call `notify-send` upon config read error",
    )

    namespace = parser.parse_args(argv)
    section_handler, metadata_parser = process_args(namespace)

    msg = f"Loading config file '{namespace.sxhkdrc}'"
    print(msg, file=sys.stderr)
    if namespace.notify_send_on_config_read:
        subprocess.Popen(
            NOTIFY_SEND_INFO_PREFIX
            + [
                f"{PROGNAME}: Loading config file for the first time",
                f"Reading '{namespace.sxhkdrc}'",
            ]
        )
    try:
        errored, tree = read_config(
            namespace.sxhkdrc, section_handler, metadata_parser
        )
    except Exception:
        if namespace.notify_send_on_config_read_error:
            subprocess.Popen(
                NOTIFY_SEND_ERR_PREFIX
                + [
                    f"{PROGNAME}: Fatal error upon initial config read",
                    "Quitting...",
                ]
            )
        raise
    if errored and namespace.notify_send_on_config_read_error:
        subprocess.Popen(
            NOTIFY_SEND_ERR_PREFIX
            + [
                f"{PROGNAME}: Warnings upon initial config read",
                "Read stderr for error messages",
            ]
        )

    # Actually reload the config at a set point in the loop to avoid data
    # becoming stale inside the rest of the loop body.
    # It's delayed, but it's is safe since, for us, the keybinds are
    # essentially unchanged until the first time we get a fifo message after
    # setting the flag.
    reload_config = False

    def handle_sigusr1(*_: Any) -> None:
        nonlocal reload_config
        print(
            f"Arranging to reload config file '{namespace.sxhkdrc}'",
            file=sys.stderr,
        )
        if namespace.notify_send_on_config_read:
            subprocess.Popen(
                NOTIFY_SEND_INFO_PREFIX
                + [
                    f"{PROGNAME}: Preparing to reload config file",
                    f"Will reload '{namespace.sxhkdrc}' just in time",
                ]
            )
        reload_config = True

    signal(SIGUSR1, handle_sigusr1)

    msg_re = re.compile(r"^(?P<type>[A-Z])(?P<msg>.*)$")
    try:
        if namespace.status_fifo == "-":
            status_fifo = sys.stdin
            do_close = False
        else:
            status_fifo = open(namespace.status_fifo)
            do_close = True

        prev_hotkey_str: Optional[str] = None
        curr_mode = "normal"

        def handle_sigusr2(*_: Any) -> None:
            print(curr_mode, flush=True)

        signal(SIGUSR2, handle_sigusr2)

        print(curr_mode, flush=True)
        for line in status_fifo:
            if reload_config:
                reload_config = False
                print(
                    f"Reloading config file '{namespace.sxhkdrc}'",
                    file=sys.stderr,
                )
                if namespace.notify_send_on_config_read:
                    subprocess.Popen(
                        NOTIFY_SEND_INFO_PREFIX
                        + [
                            f"{PROGNAME}: Reloading config file",
                            f"Reading '{namespace.sxhkdrc}'",
                        ]
                    )
                new_section_handler = section_handler.clone_config()
                try:
                    errored, new_tree = read_config(
                        namespace.sxhkdrc, new_section_handler, metadata_parser
                    )
                except Exception as e:
                    print_exceptions(e, namespace.sxhkdrc, file=sys.stderr)
                    print(
                        "Got errors while reloading config: using old keybinds...",
                        file=sys.stderr,
                    )
                    if namespace.notify_send_on_config_read_error:
                        subprocess.Popen(
                            NOTIFY_SEND_ERR_PREFIX
                            + [
                                f"{PROGNAME}: Fatal config reload error",
                                "Rolling back to old keybinds...",
                            ]
                        )
                else:
                    tree = new_tree
                    section_handler = new_section_handler
                    if errored and namespace.notify_send_on_config_read_error:
                        subprocess.Popen(
                            NOTIFY_SEND_ERR_PREFIX
                            + [
                                f"{PROGNAME}: Warnings upon config reload",
                                "Read stderr for error messages",
                            ]
                        )

            m = msg_re.match(line)
            if not m:
                continue
            type_ = m.group("type")
            msg = m.group("msg")
            if type_ == "B" and msg == "Begin chain":
                assert (
                    prev_hotkey_str is not None
                ), "expected to see a chord before 'BBegin chain' but got none"
                tokens = Hotkey.tokenize_static_hotkey(prev_hotkey_str)
                _, chords = Hotkey.parse_static_hotkey(tokens)
                failmsg = "got noabort final chord from sxhkd status"
                assert not chords[-1].noabort, failmsg

                # Try to match a mode first.
                chords[-1].noabort = True
                node = match_hotkey(chords, tree.root)
                if node is None:
                    chords[-1].noabort = False
                    node = match_hotkey(chords, tree.root)

                if node is not None and isinstance(node.value, Chord):
                    if node.value.noabort:
                        perm_ends = node.find_permutation_ends()
                        assert perm_ends
                        hotkey = perm_ends[0].hotkey
                        assert hotkey is not None
                        keybind = hotkey.keybind
                        assert keybind is not None
                        curr_mode = keybind.metadata.get(
                            namespace.mode_field, "__unknown__"
                        )
                        print(curr_mode)
                    else:
                        curr_mode = "in-chain"
                        print(curr_mode)
            elif type_ == "E" and msg == "End chain":
                # Timeout immediately has 'EEnd chain' after it.
                curr_mode = "normal"
                print(curr_mode)
            elif type_ == "H":
                prev_hotkey_str = msg
            sys.stdout.flush()
    finally:
        if do_close:
            status_fifo.close()

    return 0
