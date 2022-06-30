"""Program outputting the current sxhkd mode from the status fifo."""
from __future__ import annotations

import argparse
import itertools as it
import re
import subprocess
import sys
from enum import Enum
from signal import SIGUSR1, SIGUSR2, signal
from typing import Any, List, NamedTuple, Optional, Tuple

from ..errors import SXHKDParserError
from ..metadata import MetadataParser, SectionHandler
from ..parser import (
    Hotkey,
    HotkeyPermutation,
    HotkeyTree,
    HotkeyTreeChordData,
    HotkeyTreeInternalNode,
    HotkeyTreeKeysymData,
    HotkeyTreeLeafNode,
    HotkeyTreeModifierSetData,
    HotkeyTreeNode,
)
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
    modsetdata: HotkeyTreeModifierSetData, node: HotkeyTreeInternalNode
) -> Optional[HotkeyTreeInternalNode]:
    modset_children = node.internal_children[HotkeyTreeModifierSetData]
    modsetnode = modset_children.get(modsetdata)
    if modsetnode is not None:
        return modsetnode
    for mods, child in modset_children.items():
        if mods.value < modsetdata.value:
            return find_matching_modsets(modsetdata, child)
    return None


def match_hotkey(
    perm: HotkeyPermutation, chord_index: int, curr_level: HotkeyTreeNode
) -> Optional[HotkeyTreeNode]:
    assert perm.chords, "got empty hotkey permutation"
    if chord_index >= len(perm.chords):
        assert isinstance(curr_level.data, HotkeyTreeChordData)
        # May or may not be a leaf node: `perm` might not represent a full permutation.
        return curr_level
    if isinstance(curr_level, HotkeyTreeLeafNode):
        # `perm` hasn't been completed yet but already at a leaf.
        return None
    assert isinstance(curr_level, HotkeyTreeInternalNode)

    curr = perm.chords[chord_index]
    curr_level = find_matching_modsets(
        HotkeyTreeModifierSetData(curr.modifiers), curr_level
    )
    if curr_level is None:
        return None

    curr_level = curr_level.internal_children[HotkeyTreeKeysymData].get(
        HotkeyTreeKeysymData(curr.keysym)
    )
    if curr_level is None:
        return None

    # Now find next chord
    for child in curr_level.children:
        if not isinstance(child.data, HotkeyTreeChordData):
            continue
        if curr != child.data.value:
            continue
        # Only match non-noabort nodes if perm has no noabort index.
        if perm.noabort_index is None and child.data.noabort:
            continue
        if perm.noabort_index is not None:
            # Only match noabort nodes when we are not at the index where the
            # noabort node should be.
            if chord_index != perm.noabort_index and child.data.noabort:
                continue
            if chord_index == perm.noabort_index and not child.data.noabort:
                continue
        return match_hotkey(perm, chord_index + 1, child)
    return None


class Mode(Enum):
    normal = "N"
    unknown = "U"
    inchain = "C"
    usermode = "M"


class CurrMode(NamedTuple):
    mode: Mode
    message: str

    def __str__(self) -> str:
        return f"{self.mode.value}{self.message}"


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
        parents=[BASE_PARSER],
    )
    parser.add_argument(
        "--status-fifo",
        "-s",
        default="-",
        help="the location of the status fifo ('-' for stdin) (default: %(default)s)",
    )
    parser.add_argument(
        "--mode-field",
        "-m",
        default="mode",
        help="metadata key indicating the mode that a noabort hotkey belongs to (default: %(default)s)",
    )
    parser.add_argument(
        "--notify-send-on-config-read",
        action="store_true",
        help="call `notify-send` when reading configs",
    )
    parser.add_argument(
        "--notify-send-on-config-read-error",
        action="store_true",
        help="call `notify-send` upon config read error",
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
        curr_mode = CurrMode(Mode.normal, "normal")

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
                tokens = Hotkey.tokenize(prev_hotkey_str)
                perm = Hotkey.parse_hotkey_permutation(tokens)
                # sxhkd never seems to print chains with ':' to its status fifo.
                # Also, it doesn't seem to print all hotkeys in a mode.
                # Therefore this shouldn't be a problem anyway.
                if perm.noabort_index is not None:
                    continue

                # Try to match a mode first.
                perm.noabort_index = len(perm.chords) - 1
                node = match_hotkey(perm, 0, tree.root)
                perm.noabort_index = None
                if node is None:
                    node = match_hotkey(perm, 0, tree.root)

                if node is not None:
                    assert isinstance(
                        node.data, HotkeyTreeChordData
                    ), f"got non-chord output ({node.data!r}) from match"
                    if node.data.noabort:
                        assert isinstance(node, HotkeyTreeInternalNode)
                        perm_ends = node.find_permutation_ends()
                        assert perm_ends
                        hotkey = perm_ends[0].hotkey
                        keybind = hotkey.keybind
                        assert keybind is not None
                        nodemode = keybind.metadata.get(namespace.mode_field)
                        if nodemode is None:
                            curr_mode = CurrMode(Mode.unknown, str(perm))
                        else:
                            curr_mode = CurrMode(Mode.usermode, nodemode)
                    else:
                        curr_mode = CurrMode(Mode.inchain, "in-chain")
                    print(curr_mode)
            elif type_ == "E" and msg == "End chain":
                # Timeout immediately has 'EEnd chain' after it.
                curr_mode = CurrMode(Mode.normal, "normal")
                print(curr_mode)
            elif type_ == "H":
                prev_hotkey_str = msg
            sys.stdout.flush()
    finally:
        if do_close:
            status_fifo.close()

    return 0
