"""Program outputting the current sxhkd mode from the status fifo."""
from __future__ import annotations

import argparse
import re
import sys
from signal import SIGUSR1, SIGUSR2, signal
from typing import Any, FrozenSet, List, Optional, cast

from ..errors import SXHKDParserError
from ..keysyms import KEYSYMS
from ..metadata import MetadataParser, SectionHandler
from ..parser import Chord, Hotkey, HotkeyTree, KeypressTreeNode
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    get_command_name,
    process_args,
)

__all__ = ["main"]


INTERNAL_NODES = ["modifierset", "keysym"]


def read_config(
    config: str,
    section_handler: SectionHandler,
    metadata_parser: MetadataParser,
) -> HotkeyTree:
    hotkey_tree = HotkeyTree(INTERNAL_NODES)
    for bind_or_err in read_sxhkdrc(
        config,
        section_handler=section_handler,
        metadata_parser=metadata_parser,
        # Handle them ourselves.
        hotkey_errors=IGNORE_HOTKEY_ERRORS,
    ):
        if isinstance(bind_or_err, SXHKDParserError):
            print(bind_or_err, file=sys.stderr)
            continue

        keybind = bind_or_err
        # Check for possibly invalid keysyms.
        keysyms = set()
        for perm in keybind.hotkey.permutations:
            for chord in perm:
                if chord.keysym not in KEYSYMS:
                    keysyms.add(chord.keysym)
        if keysyms:
            keysym_str = " ,".join(f"'{k}'" for k in keysyms)
            print(
                f"{keybind.line}: Possibly invalid keysyms: {keysym_str}",
                file=sys.stderr,
            )

        hotkey_tree.merge_hotkey(keybind.hotkey)

    hotkey_tree.dedupe_chord_nodes()
    for dupset in hotkey_tree.find_duplicate_chord_nodes():
        assert dupset
        node = dupset[0]
        assert node.hotkey is not None
        assert node.permutation_index is not None
        chords = node.hotkey.permutations[node.permutation_index]
        noabort_index = node.hotkey.noabort_index
        hotkey_str = Hotkey.static_hotkey_str(chords, noabort_index)
        # XXX: All `line` attributes will be non-`None` since all were read from a file.
        for line in sorted(
            cast(int, cast(Hotkey, node.hotkey).line) for node in dupset
        ):
            print(f"{line}: Duplicate hotkey '{hotkey_str}'", file=sys.stderr)

    for prefix, conflicts in hotkey_tree.find_conflicting_chain_prefixes():
        assert prefix.hotkey is not None
        assert prefix.permutation_index is not None
        chords = prefix.hotkey.permutations[prefix.permutation_index]
        noabort_index = prefix.hotkey.noabort_index
        chain_hk_str = Hotkey.static_hotkey_str(chords, noabort_index)

        lines = set()
        lines.add(prefix.hotkey.line)
        lines.update(
            cast(Hotkey, conflict.hotkey).line for conflict in conflicts
        )

        if len(lines) == 1:
            conflicts_str = []
            for conflict in conflicts:
                assert conflict.hotkey is not None
                assert conflict.permutation_index is not None
                chords = conflict.hotkey.permutations[
                    conflict.permutation_index
                ]
                noabort_index = conflict.hotkey.noabort_index
                hk_str = Hotkey.static_hotkey_str(chords, noabort_index)
                conflicts_str.append(f"'{hk_str}'")
            curr_line = lines.pop()
            print(
                f"{curr_line}: '{chain_hk_str}' conflicts with {', '.join(conflicts_str)}",
                file=sys.stderr,
            )
        else:
            conflicts_str = []
            for conflict in conflicts:
                assert conflict.hotkey is not None
                assert conflict.permutation_index is not None
                chords = conflict.hotkey.permutations[
                    conflict.permutation_index
                ]
                noabort_index = conflict.hotkey.noabort_index
                hk_str = Hotkey.static_hotkey_str(chords, noabort_index)
                conflicts_str.append(
                    f"'{hk_str}' (line {conflict.hotkey.line})"
                )
            print(
                f"{prefix.hotkey.line}: '{chain_hk_str}' conflicts with {', '.join(conflicts_str)}",
                file=sys.stderr,
            )
    return hotkey_tree


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
        if (curr.modifiers, curr.keysym) == (
            child.value.modifiers,
            child.value.keysym,
        ):
            return match_hotkey(rest, child)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.
    """
    parser = argparse.ArgumentParser(
        get_command_name(__file__),
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

    namespace = parser.parse_args(argv)
    section_handler, metadata_parser = process_args(namespace)

    print(f"Loading config file '{namespace.sxhkdrc}'", file=sys.stderr)
    tree = read_config(namespace.sxhkdrc, section_handler, metadata_parser)

    def handle_sigusr1(*_: Any) -> None:
        nonlocal tree
        print(f"Reloading config file '{namespace.sxhkdrc}'", file=sys.stderr)
        section_handler.reset()
        tree = read_config(namespace.sxhkdrc, section_handler, metadata_parser)

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
