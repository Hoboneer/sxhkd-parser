"""Tool for linting hotkeys."""
from __future__ import annotations

import argparse
from typing import List, NamedTuple, Optional, cast

from ..errors import SXHKDParserError
from ..keysyms import KEYSYMS
from ..parser import Hotkey, HotkeyTree
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    get_command_name,
    process_args,
)

__all__ = ["main"]


class Message(NamedTuple):
    line: Optional[int]
    column: Optional[int]
    message: str


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.
    """
    parser = argparse.ArgumentParser(
        get_command_name(__file__),
        description="Check hotkeys for invalid keysyms, duplicates, and conflicts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BASE_PARSER],
    )

    namespace = parser.parse_args(argv)
    section_handler, metadata_parser = process_args(namespace)

    errors: List[Message] = []
    # TODO: maybe include other internal nodes?
    INTERNAL_NODES = ["modifierset"]
    hotkey_tree = HotkeyTree(INTERNAL_NODES)
    for bind_or_err in read_sxhkdrc(
        namespace.sxhkdrc,
        section_handler=section_handler,
        metadata_parser=metadata_parser,
        # Handle them ourselves.
        hotkey_errors=IGNORE_HOTKEY_ERRORS,
    ):
        if isinstance(bind_or_err, SXHKDParserError):
            errors.append(
                Message(bind_or_err.line, bind_or_err.column, str(bind_or_err))
            )
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
            errors.append(
                Message(
                    keybind.line,
                    None,
                    f"{keybind.line}: Possibly invalid keysyms: {keysym_str}",
                )
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
            errors.append(
                Message(line, None, f"{line}: Duplicate hotkey '{hotkey_str}'")
            )

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
            errors.append(
                Message(
                    curr_line,
                    None,
                    f"{curr_line}: '{chain_hk_str}' conflicts with {', '.join(conflicts_str)}",
                )
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
            errors.append(
                Message(
                    prefix.hotkey.line,
                    None,
                    f"{prefix.hotkey.line}: '{chain_hk_str}' conflicts with {', '.join(conflicts_str)}",
                )
            )

    numbered = []
    rest = []
    for msg in errors:
        if msg.line is not None:
            numbered.append(msg)
        else:
            rest.append(msg)
    numbered.sort(key=lambda x: cast(int, x.line))
    for msg in numbered:
        print(f"{namespace.sxhkdrc}:{msg.message}")
    for msg in rest:
        print(f"{namespace.sxhkdrc}: {msg.message}")

    return 0