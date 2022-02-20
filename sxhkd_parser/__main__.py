from typing import Dict

from .metadata import (
    KeyValueMetadataParser,
    SimpleDescriptionParser,
    SimpleSectionHandler,
    StackSectionHandler,
)
from .parser import Keybind
from .util import read_sxhkdrc

# section_handler = SimpleSectionHandler(r'^#\s*(?P<name>[A-Z. /-]+):$')
# metadata_parser = SimpleDescriptionParser(r'^#\s*(?P<value>[A-Z][^.]+\.)$')
section_handler = StackSectionHandler(
    r"^#+\s*(?P<name>[A-Z./-]+)\s*{{{$", r"^#+\s*}}}$"
)
metadata_parser = KeyValueMetadataParser(
    r"^#+\s*@(?P<key>[a-z0-9_]+)\s+(?P<value>.*)$",
    r"^#+\s*$",
)

modes: Dict[str, Keybind] = {}
for keybind in read_sxhkdrc("sxhkdrc", section_handler, metadata_parser):
    if "mode" in keybind.metadata:
        modename = keybind.metadata["mode"]
        if modename in modes:
            # IIRC separate keybinds for the chords after ':' fail.
            raise RuntimeError(
                "warning: multiple keybinds used for the same mode!"
            )
        modes[modename] = keybind
    from pprint import pprint

    print("keybind.hotkey.permutations=", end="")
    pprint(keybind.hotkey.permutations)
    print(keybind.hotkey)

    chord_tree = keybind.hotkey.get_tree()
    print("tree:")
    chord_tree.print_tree()
    print()
    chord_tree.include_modifierset_nodes()
    print("pre-dedup(modifiersets):")
    chord_tree.print_tree()
    chord_tree.dedupe_modifierset_nodes()
    print("post-dedup(modifiersets):")
    chord_tree.print_tree()

    print("pre-dedup(chords):")
    chord_tree.print_tree()
    print("post-dedup(chords):")
    chord_tree.dedupe_chord_nodes()
    chord_tree.print_tree()

    # chord_tree.include_keysym_nodes()
    # print("pre-dedup(keysyms):")
    # chord_tree.print_tree()
    # print("post-dedup(keysyms):")
    # chord_tree.dedupe_keysym_nodes()
    # chord_tree.print_tree()

    # chord_tree.include_runevent_nodes()
    # print("pre-dedup(runevent):")
    # chord_tree.print_tree()
    # print("post-dedup(runevent):")
    # chord_tree.dedupe_runevent_nodes()
    # chord_tree.print_tree()

    # chord_tree.include_replay_nodes()
    # print("pre-dedup(replay):")
    # chord_tree.print_tree()
    # print("post-dedup(replay):")
    # chord_tree.dedupe_replay_nodes()
    # chord_tree.print_tree()
    # print()

    print(f"{keybind.command.permutations=}")
    print(keybind.command)
    print()

    print(keybind)
    print("\n\n")
sections = section_handler.get_tree()
sections.print_tree()

print()
from pprint import pprint

pprint(modes)
