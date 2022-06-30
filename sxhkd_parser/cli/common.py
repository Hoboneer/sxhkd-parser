"""Code and utilities common to all command-line tools using this library."""
from __future__ import annotations

import argparse
import os
import re
import string
import sys
from dataclasses import dataclass
from typing import (
    IO,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

from .._package import __version__
from ..errors import SXHKDParserError
from ..keysyms import KEYSYMS
from ..metadata import (
    KeyValueMetadataParser,
    MetadataParser,
    NullMetadataParser,
    RootSectionHandler,
    SectionHandler,
    SimpleDescriptionParser,
    SimpleSectionHandler,
    StackSectionHandler,
)
from ..parser import HotkeyTree, Keybind


def get_command_name(path: str) -> str:
    """Get command name from __file__."""
    cmd, _ = os.path.splitext(os.path.basename(path))
    return cmd


# XXX: should it raise exceptions instead?
def find_sxhkdrc() -> Optional[str]:
    """Find any existing sxhkdrc in the standard directories.

    Looks in $XDG_CONFIG_HOME (default: $HOME/.config) for subdir
    'sxhkd/'.  Returns `None` if the $HOME variable doesn't exist or
    the sxhkdrc at the standard location doesn't exist.
    """
    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if not xdg_config_home:
        home = os.getenv("HOME")
        if not home:
            return None
        xdg_config_home = os.path.join(home, ".config")
    sxhkdrc = os.path.join(xdg_config_home, "sxhkd", "sxhkdrc")
    if not os.path.exists(sxhkdrc):
        return None
    return sxhkdrc


BASE_PARSER = argparse.ArgumentParser(add_help=False)
BASE_PARSER.add_argument(
    "--version",
    "-V",
    action="version",
    version=f"%(prog)s (sxhkd-parser) {__version__}",
)
_section_group = BASE_PARSER.add_argument_group("section config")
_section_group.add_argument(
    "--section-type",
    choices=["none", "simple", "stack"],
    default="none",
    help="set the type of sections that the config uses (default: %(default)s)",
)
_section_group.add_argument(
    "--header",
    metavar="REGEX",
    help="regex for the header of the 'simple' and 'stack' types: it must have a named group 'name'",
)
_section_group.add_argument(
    "--footer",
    metavar="REGEX",
    help="regex for the footer of the 'stack' type",
)

_metadata_group = BASE_PARSER.add_argument_group("metadata config")
_metadata_group.add_argument(
    "--metadata-type",
    choices=["none", "simple", "key-value"],
    default="none",
    help="set the type of metadata that the config uses (default: %(default)s)",
)
_metadata_group.add_argument(
    "--description",
    metavar="REGEX",
    help="regex for the 'simple' metadata type: it must have a named group 'description'",
)
_metadata_group.add_argument(
    "--pair",
    metavar="REGEX",
    help="regex for key-value pairs of the 'key-value' metadata type: it must have named groups 'key' and 'value'",
)
_metadata_group.add_argument(
    "--empty",
    metavar="REGEX",
    help="regex for empty lines of metadata of type 'key-value'",
)

BASE_PARSER.add_argument(
    "--sxhkdrc",
    "-c",
    default=find_sxhkdrc(),
    help="the location of the config file (default: $XDG_CONFIG_HOME/sxhkd/sxhkdrc)",
)


def process_args(
    namespace: argparse.Namespace,
) -> Tuple[SectionHandler, MetadataParser]:
    """Preprocess the command-line arguments for common options."""
    section_handler: SectionHandler
    if namespace.section_type == "none":
        section_handler = RootSectionHandler()
    elif namespace.section_type in ("simple", "stack"):
        if not namespace.header:
            raise RuntimeError("got no section header regex")
        if namespace.section_type == "simple":
            section_handler = SimpleSectionHandler(namespace.header)
        else:
            if not namespace.footer:
                raise RuntimeError("got no section footer regex")
            section_handler = StackSectionHandler(
                namespace.header, namespace.footer
            )
    else:
        raise RuntimeError(
            f"unreachable! invalid section type {namespace.section_type}"
        )

    metadata_parser: MetadataParser
    if namespace.metadata_type == "none":
        metadata_parser = NullMetadataParser()
    elif namespace.metadata_type == "simple":
        if not namespace.description:
            raise RuntimeError("got no description regex")
        metadata_parser = SimpleDescriptionParser(namespace.description)
    elif namespace.metadata_type == "key-value":
        if not namespace.pair or not namespace.empty:
            raise RuntimeError("got no pair regex or no empty regex")
        metadata_parser = KeyValueMetadataParser(
            namespace.pair, namespace.empty
        )
    else:
        raise RuntimeError(
            f"unreachable! invalid description type {namespace.description_type}"
        )

    return (section_handler, metadata_parser)


IGNORE_HOTKEY_ERRORS: Dict[str, bool] = {
    "duplicate_permutations": False,
    "conflicting_permutations": False,
    "maybe_invalid_keysyms": False,
}


@dataclass
class ReplaceStrEvaluator:
    """Evaluator for command lines containing replacement strings."""

    hotkey: str
    command: str
    re: re.Pattern[str]

    def __init__(self, hotkey: str, command: str):
        self.hotkey = hotkey
        self.command = command
        groups = [
            f"(?P<hk>{re.escape(hotkey)})",
            f"(?P<cmd>{re.escape(command)})",
            "(?P<rest>.)",
        ]
        self.re = re.compile("|".join(groups))

    def eval(
        self,
        args: List[str],
        hotkey: str,
        cmd_file: Union[str, IO[str]],
    ) -> List[str]:
        """Substitute any occurrences of the replacement strings in `args` with `hotkey` and (the name of) `cmd_file`."""
        out_args = []
        for arg in args:
            new_arg = ""
            for m in self.re.finditer(arg):
                if m.group("hk"):
                    new_arg += hotkey
                elif m.group("cmd"):
                    if isinstance(cmd_file, str):
                        new_arg += cmd_file
                    else:
                        new_arg += cmd_file.name
                else:
                    new_arg += m.group("rest")
            out_args.append(new_arg)
        return out_args


def _parse_repl_str(repl: str) -> str:
    if set(string.whitespace) <= set(repl):
        raise argparse.ArgumentTypeError(
            "replacement string cannot contain whitespace"
        )
    return repl


def add_repl_str_options(parser: argparse.ArgumentParser) -> None:
    """Add replacement string options to `parser`.

    These options are added in this separate function rather than added to
    `BASE_PARSER` directly, because not all tools need it.
    """
    parser.add_argument(
        "--hotkey-replace-str",
        "-H",
        default="@",
        action="store",
        type=_parse_repl_str,
        help="set replacement string for hotkey text (default: %(default)s)",
    )
    parser.add_argument(
        "--command-replace-str",
        "-C",
        default="%",
        action="store",
        type=_parse_repl_str,
        help="set replacement string for the filename of the command text (default: %(default)s)",
    )


class Message(NamedTuple):
    """Data object for CLI error messages."""

    line: Optional[int]
    column: Optional[int]
    message: str


def format_error_msg(
    msg: Union[Message, SXHKDParserError], config_filename: str
) -> str:
    """Return formatted error message for an error in the file `config_filename`."""
    parts = []
    parts.append(config_filename)
    if msg.line is None and msg.column is not None:
        raise ValueError(f"missing line but column exists with {msg}")
    if msg.line is not None:
        parts.append(str(msg.line))
    if msg.column is not None:
        parts.append(str(msg.column))
    return f"{':'.join(parts)}: {msg.message}"


def print_exceptions(
    ex: BaseException, config_filename: str, file: Optional[IO[str]] = None
) -> None:
    """Print exceptions for a fatal error in order of their time of raising."""
    if file is None:
        file = sys.stdout
    if ex.__context__ is None:
        if isinstance(ex, SXHKDParserError):
            print(f"{config_filename}:{ex} [FATAL]", file=file)
        else:
            print(f"{config_filename}: {ex} [FATAL]", file=file)
        return
    print_exceptions(ex.__context__, config_filename, file)
    if isinstance(ex, SXHKDParserError):
        print(f"{config_filename}:{ex} [FATAL]", file=file)
    else:
        print(f"{config_filename}: {ex} [FATAL]", file=file)


def find_maybe_invalid_keysyms(keybind: Keybind) -> Iterable[Message]:
    """Yield a `Message` object if `keybind` has possibly invalid keysyms."""
    keysyms = set()
    for perm in keybind.hotkey.permutations:
        for chord in perm.chords:
            if chord.keysym not in KEYSYMS:
                keysyms.add(chord.keysym)
    if keysyms:
        keysym_str = ", ".join(f"'{k}'" for k in keysyms)
        # TODO: either make this function take a HotkeyTree or rewrite type
        # hints such that it isn't an iterable.  As it is now however, this is
        # very convenient in the CLI tools.
        yield Message(
            keybind.line,
            None,
            f"Possibly invalid keysyms: {keysym_str}",
        )


def find_duplicates(tree: HotkeyTree) -> Iterable[Message]:
    """Yield `Message` objects for duplicate hotkeys found in `tree`."""
    for dupset in tree.find_duplicate_chord_nodes():
        assert dupset
        node = dupset[0]
        perm = node.permutation
        assert perm is not None
        hotkey_str = str(perm)
        # XXX: All `line` attributes will be non-`None` since it's assumed that all were read from a file.
        for line in sorted(cast(int, node.hotkey.line) for node in dupset):
            yield Message(line, None, f"Duplicate hotkey '{hotkey_str}'")


def find_prefix_conflicts(tree: HotkeyTree) -> Iterable[Message]:
    """Yield `Message` objects for conflicting chain prefixes in `tree`."""
    for prefix, conflicts in tree.find_conflicting_chain_prefixes():
        assert prefix.hotkey is not None
        perm = prefix.permutation
        assert perm is not None
        chain_hk_str = str(perm)

        conflicts_str = []
        for conflict in conflicts:
            perm = conflict.permutation
            assert perm is not None
            hk_str = str(perm)
            assert conflict.hotkey is not None
            assert conflict.hotkey.line is not None
            if conflict.hotkey.line != prefix.hotkey.line:
                conflicts_str.append(
                    f"{hk_str!r} (line {conflict.hotkey.line})"
                )
            else:
                conflicts_str.append(f"{hk_str!r}")
        yield Message(
            prefix.hotkey.line,
            None,
            f"{chain_hk_str!r} conflicts with {', '.join(conflicts_str)}",
        )
