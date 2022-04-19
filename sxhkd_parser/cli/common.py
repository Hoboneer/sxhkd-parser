"""Code and utilities common to all command-line tools using this library."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

from .._package import __version__
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


def get_command_name(path: str) -> str:
    """Get command name from __file__."""
    cmd, _ = os.path.splitext(os.path.basename(path))
    return cmd


# XXX: should it raise exceptions instead?
def find_sxhkdrc() -> Optional[str]:
    """Find any existing sxhkdrc in the standard directories.

    Looks in $XDG_CONFIG_HOME (default: $HOME/.config) for subdirs
    'sxhkd/sxhkdrc/'.  Returns `None` if the $HOME variable doesn't exist or
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
    "--section-type", choices=["none", "simple", "stack"], default="none"
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
    help="the location of the config file",
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
            raise RuntimeError("got no description regex")
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
