"""Tool for exporting hotkeys to various formats, including HTML and plaintext."""
from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import ClassVar, Dict, Iterable, List, Optional, Set, Type

from ..errors import SXHKDParserError
from ..metadata import SectionTreeNode
from ..parser import Hotkey, expand_sequences
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    get_command_name,
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


class HTMLEmitter(KeybindEmitter):
    BODY_ESCAPES: ClassVar[Dict[str, str]] = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
    }
    ATTR_ESCAPES: ClassVar[Dict[str, str]] = {
        '"': "&quot;",
        "'": "&apos;",
    }

    def __init__(
        self,
        options: Optional[Iterable[PrintOption]] = None,
        expand_sequences: bool = False,
    ):
        super().__init__(options, expand_sequences)
        self._body_escapes = str.maketrans(self.BODY_ESCAPES)
        self._attr_escapes = str.maketrans(self.ATTR_ESCAPES)

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
        yield f'{"  " * level}<div class="section" id="{id_slug.translate(self._attr_escapes)}">'
        yield f'{"  " * (level+1)}<h{header_num}>{node.name.translate(self._body_escapes)}</h{header_num}>'

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
                    yield f'{curr_base}<tr class="hotkey" id="{self._get_id_slug(node).translate(self._attr_escapes)}-{i}-{j}">'
                    field_base = curr_base + "  "
                    for field in fields:
                        if field == "hotkey":
                            hotkey_str = Hotkey.static_hotkey_str(
                                perm, keybind.hotkey.noabort_index
                            )
                            yield f'{field_base}<td class="bind">{hotkey_str.translate(self._body_escapes)}</td>'
                        elif field == "mode":
                            yield f"{field_base}<td class=\"mode\">{keybind.metadata.get('mode', 'normal').translate(self._body_escapes)}</td>"
                        elif field == "description" and desc_perms:
                            yield f'{field_base}<td class="field-{field.translate(self._attr_escapes)}">{desc_perms[j].translate(self._body_escapes)}</td>'
                        else:
                            yield f"{field_base}<td class=\"field-{field.translate(self._attr_escapes)}\">{keybind.metadata.get(field, f'<em>No field {field!r}.</em>').translate(self._body_escapes)}</td>"
                    yield f"{curr_base}</tr>"
            else:
                yield f'{curr_base}<tr class="hotkey" id="{self._get_id_slug(node).translate(self._attr_escapes)}-{i}">'
                field_base = curr_base + "  "
                for field in fields:
                    if field == "hotkey":
                        yield f"{field_base}<td class=\"bind\">{keybind.hotkey.raw.translate(self._body_escapes) if isinstance(keybind.hotkey.raw, str) else ' '.join(keybind.hotkey.raw).translate(self._body_escapes)}</td>"
                    elif field == "mode":
                        yield f"{field_base}<td class=\"mode\">{keybind.metadata.get('mode', 'normal').translate(self._body_escapes)}</td>"
                    else:
                        yield f"{field_base}<td class=\"field-{field.translate(self._attr_escapes)}\">{keybind.metadata.get(field, f'<em>No field {field!r}.</em>').translate(self._body_escapes)}</td>"
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
                            hotkey_str = Hotkey.static_hotkey_str(
                                perm, keybind.hotkey.noabort_index
                            )
                            line.append(hotkey_str)
                        elif field == "mode":
                            line.append(keybind.metadata.get("mode", "normal"))
                        elif field == "description" and desc_perms:
                            line.append(desc_perms[j])
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BASE_PARSER],
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "txt"],
        default="html",
        help="the format to export to",
    )
    parser.add_argument(
        "--fields",
        "-F",
        default=["hotkey", "description", "mode"],
        type=lambda x: x.split(","),
        help="the metadata fields and the order in which to print them ('hotkey' isn't strictly metadata, but oh well)",
    )
    parser.add_argument(
        "--expand",
        "-E",
        action="store_true",
        help="expand embedded sequences (also expands the 'description' field if the permutations match)",
    )
    records_group = parser.add_mutually_exclusive_group()
    records_group.add_argument(
        "--records",
        "-R",
        choices=["all", "sections", "keybinds"],
        default="all",
        help="what to print",
    )
    records_group.add_argument(
        "--sections-only",
        "-S",
        dest="records",
        action="store_const",
        const="sections",
    )
    records_group.add_argument(
        "--keybinds-only",
        "-K",
        dest="records",
        action="store_const",
        const="keybinds",
    )

    # TODO: allow configuring whether to specially process 'mode' in the metadata
    #   - maybe allow configuring what key it is?

    namespace = parser.parse_args(argv)
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
                print(bind_or_err, file=sys.stderr)
    except SXHKDParserError as e:
        # Print errors inside-out.
        def print_errors(ex: BaseException) -> None:
            if ex.__context__ is None:
                print(f"{namespace.sxhkdrc}:{ex} [FATAL]", file=sys.stderr)
                return
            print_errors(ex.__context__)
            print(f"{namespace.sxhkdrc}:{ex} [FATAL]", file=sys.stderr)

        print_errors(e)
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
