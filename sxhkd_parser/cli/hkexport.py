"""Tool for exporting hotkeys to various formats, including HTML and plaintext."""
from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Set, Type

from ..errors import SXHKDParserError
from ..metadata import SectionTreeNode
from ..parser import expand_sequences
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


class PrintOption(Enum):
    SECTIONS = auto()
    KEYBINDS = auto()


class KeybindEmitter(ABC):
    def __init__(
        self,
        options: Optional[Iterable[PrintOption]] = None,
        expand_sequences: bool = False,
    ):
        self.options: Set[PrintOption]
        if options is None:
            self.options = set()
        else:
            self.options = set(options)
        self.expand_sequences = expand_sequences

    @abstractmethod
    def emit_node_header(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def emit_node_keybinds(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        raise NotImplementedError

    def emit_node_footer(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        return iter([])

    def emit(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        if PrintOption.SECTIONS in self.options:
            yield from self.emit_node_header(node, level=level, fields=fields)
        if PrintOption.KEYBINDS in self.options:
            yield from self.emit_node_keybinds(
                node, level=level, fields=fields
            )
        for subsection in node.children:
            yield from self.emit(subsection, level=level + 1, fields=fields)
        if PrintOption.SECTIONS in self.options:
            yield from self.emit_node_footer(node, level=level, fields=fields)


HTML_BODY_ESCAPES: Dict[int, str] = str.maketrans(
    {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
    }
)
HTML_ATTR_ESCAPES: Dict[int, str] = str.maketrans(
    {
        '"': "&quot;",
        "'": "&apos;",
    }
)


def esc_body(s: str) -> str:
    return s.translate(HTML_BODY_ESCAPES)


def esc_attr(s: str) -> str:
    return s.translate(HTML_ATTR_ESCAPES)


class HTMLEmitter(KeybindEmitter):
    def _get_id_slug(self, node: SectionTreeNode) -> str:
        assert isinstance(node.name, str)
        return node.name.lower().replace(" ", "-").replace("/", "-")

    def emit_node_header(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        header_num = 1 + level
        # clamp it
        if header_num > 6:
            header_num = 6
        assert isinstance(node.name, str)
        id_slug = self._get_id_slug(node)
        yield f'{"  " * level}<div class="section" id="{esc_attr(id_slug)}">'
        yield f'{"  " * (level+1)}<h{header_num}>{esc_body(node.name)}</h{header_num}>'

    def emit_node_keybinds(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        base = "  " * (level + 1)
        yield f'{base}<table class="hotkeys">'
        for i, keybind in enumerate(node.keybind_children):
            curr_base = base + "  "
            if self.expand_sequences:
                if "description" in keybind.metadata:
                    try:
                        desc_perms = expand_sequences(
                            keybind.metadata["description"]
                        ).generate_permutations()
                    except Exception:
                        desc_perms = None
                    else:
                        assert desc_perms is not None
                        if len(desc_perms) != len(keybind.hotkey.permutations):
                            desc_perms = None
                else:
                    desc_perms = None
                for j, perm in enumerate(keybind.hotkey.permutations):
                    yield f'{curr_base}<tr class="hotkey" id="{esc_attr(self._get_id_slug(node))}-{i}-{j}">'
                    field_base = curr_base + "  "
                    for field in fields:
                        if field == "hotkey":
                            hotkey_str = str(perm)
                            yield f'{field_base}<td class="bind">{esc_body(hotkey_str)}</td>'
                        elif field == "mode":
                            yield f'{field_base}<td class="mode">{esc_body(keybind.metadata.get("mode", "normal"))}</td>'
                        elif field == "description" and desc_perms:
                            yield f'{field_base}<td class="field-{esc_attr(field)}">{esc_body(str(desc_perms[j]))}</td>'
                        else:
                            if field in keybind.metadata:
                                body = esc_body(keybind.metadata[field])
                            else:
                                body = (
                                    f"<em>No field '{esc_body(field)}'.</em>"
                                )
                            yield f'{field_base}<td class="field-{esc_attr(field)}">{body}</td>'
                    yield f"{curr_base}</tr>"
            else:
                yield f'{curr_base}<tr class="hotkey" id="{esc_attr(self._get_id_slug(node))}-{i}">'
                field_base = curr_base + "  "
                for field in fields:
                    if field == "hotkey":
                        yield f'{field_base}<td class="bind">{esc_body(keybind.hotkey.raw if isinstance(keybind.hotkey.raw, str) else " ".join(keybind.hotkey.raw))}</td>'
                    elif field == "mode":
                        yield f'{field_base}<td class="mode">{esc_body(keybind.metadata.get("mode", "normal"))}</td>'
                    else:
                        if field in keybind.metadata:
                            body = esc_body(keybind.metadata[field])
                        else:
                            body = f"<em>No field '{esc_body(field)}'.</em>"
                        yield f'{field_base}<td class="field-{esc_attr(field)}">{body}</td>'
                yield f"{curr_base}</tr>"
        yield f"{base}</table>"

    def emit_node_footer(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        yield f"{'  ' * level}</div>"


class PlaintextEmitter(KeybindEmitter):
    def emit_node_header(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        assert isinstance(node.name, str)
        yield f"{'#' * (level+1)}{node.name}"

    def emit_node_keybinds(
        self, node: SectionTreeNode, level: int, fields: List[str]
    ) -> Iterable[str]:
        for keybind in node.keybind_children:
            if self.expand_sequences:
                if "description" in keybind.metadata:
                    try:
                        desc_perms = expand_sequences(
                            keybind.metadata["description"]
                        ).generate_permutations()
                    except Exception:
                        desc_perms = None
                    else:
                        assert desc_perms is not None
                        if len(desc_perms) != len(keybind.hotkey.permutations):
                            desc_perms = None
                else:
                    desc_perms = None
                for j, perm in enumerate(keybind.hotkey.permutations):
                    line = []
                    for field in fields:
                        if field == "hotkey":
                            line.append(str(perm))
                        elif field == "mode":
                            line.append(keybind.metadata.get("mode", "normal"))
                        elif field == "description" and desc_perms:
                            line.append(str(desc_perms[j]))
                        else:
                            line.append(keybind.metadata.get(field, ""))
                    yield "\t".join(line)
            else:
                line = []
                for field in fields:
                    if field == "hotkey":
                        if isinstance(keybind.hotkey.raw, str):
                            line.append(keybind.hotkey.raw)
                        else:
                            line.append(" ".join(keybind.hotkey.raw))
                    elif field == "mode":
                        line.append(keybind.metadata.get("mode", "normal"))
                    else:
                        line.append(keybind.metadata.get(field, ""))
                yield "\t".join(line)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.  SXHKDParserError instances yielded by `read_sxhkdrc`
    are simply printed to stderr.
    """
    parser = argparse.ArgumentParser(
        get_command_name(__file__),
        description="Export keybinds in various formats",
        parents=[BASE_PARSER],
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "txt"],
        default="html",
        help="the format to export to (default: %(default)s)",
    )
    parser.add_argument(
        "--fields",
        "-F",
        default="hotkey,description,mode",
        help="comma-separated list of the metadata fields to include and the order in which to print them ('hotkey' isn't strictly metadata, but oh well) (default: %(default)s)",
    )
    parser.add_argument(
        "--expand",
        "-E",
        action="store_true",
        help="expand embedded sequences",
    )
    records_group = parser.add_mutually_exclusive_group()
    records_group.add_argument(
        "--records",
        "-R",
        choices=["all", "sections", "keybinds"],
        default="all",
        help="what to print (default: %(default)s)",
    )
    records_group.add_argument(
        "--sections-only",
        "-S",
        dest="records",
        action="store_const",
        const="sections",
        help="alias for --records=sections",
    )
    records_group.add_argument(
        "--keybinds-only",
        "-K",
        dest="records",
        action="store_const",
        const="keybinds",
        help="alias for --records=keybinds",
    )

    # TODO: allow configuring whether to specially process 'mode' in the metadata
    #   - maybe allow configuring what key it is?

    namespace = parser.parse_args(argv)
    namespace.fields = namespace.fields.split(",")
    section_handler, metadata_parser = process_args(namespace)

    # Exhaust the generator function to read all keybinds.
    try:
        for bind_or_err in read_sxhkdrc(
            namespace.sxhkdrc,
            section_handler=section_handler,
            metadata_parser=metadata_parser,
            hotkey_errors=IGNORE_HOTKEY_ERRORS,
        ):
            if isinstance(bind_or_err, SXHKDParserError):
                msg = format_error_msg(bind_or_err, namespace.sxhkdrc)
                print(msg, file=sys.stderr)
    except SXHKDParserError as e:
        print_exceptions(e, namespace.sxhkdrc, file=sys.stderr)
        return 1
    section_handler.root.name = "SXHKD keybinds"

    emittercls: Type[KeybindEmitter]
    if namespace.format == "html":
        emittercls = HTMLEmitter
    elif namespace.format == "txt":
        emittercls = PlaintextEmitter
    else:
        raise RuntimeError(
            f"unreachable! invalid export format {namespace.format}"
        )

    print_options: Iterable[PrintOption]
    if namespace.records == "all":
        print_options = PrintOption.__members__.values()
    else:
        print_options = [PrintOption.__members__[namespace.records.upper()]]
    emitter = emittercls(print_options, expand_sequences=namespace.expand)

    for line in emitter.emit(
        section_handler.root, level=0, fields=namespace.fields
    ):
        print(line)
    return 0
