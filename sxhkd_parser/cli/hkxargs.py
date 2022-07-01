"""Tool to run commands on the command text of keybinds."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    IO,
    ClassVar,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from ..errors import SXHKDParserError
from ..parser import Hotkey, Keybind
from ..util import read_sxhkdrc
from .common import (
    BASE_PARSER,
    IGNORE_HOTKEY_ERRORS,
    ReplaceStrEvaluator,
    add_repl_str_options,
    format_error_msg,
    get_command_name,
    process_args,
)

__all__ = ["main"]

# Same exit codes as GNU xargs (which is based on those of POSIX).
CODE_1_125 = 123
CODE_255 = 124
CODE_SIGNAL = 125
CODE_CANNOT_RUN = 126
CODE_NOT_FOUND = 127
CODE_OTHER = 1


KeybindDict = Dict[str, Tuple[Keybind, str, bool]]


def get_perms_to_exec(file: IO[str], keybinds: KeybindDict) -> Iterable[str]:
    for line in file:
        if not line.strip():
            continue
        # Normalise
        tokens = Hotkey.tokenize(line)
        norm_str = str(Hotkey.parse_hotkey_permutation(tokens))
        if norm_str in keybinds:
            yield norm_str


class LinterFormatParseMode(Enum):
    NORMAL = auto()
    NEED_LBRACE = auto()
    NEED_CHAR = auto()
    NEED_CHAR_OR_RBRACE = auto()


@dataclass
class LinterFormatField:
    field: str
    DEFINED: ClassVar[Container[str]] = {
        "file",
        "line",
        "column",
        "type",
        "message",
    }

    def __init__(self, field: str):
        self.field = field

    def validate(self) -> None:
        if self.field not in self.DEFINED:
            raise ValueError(f"undefined field {self.field!r}")


def parse_linter_format(format: str) -> List[Union[str, LinterFormatField]]:
    spans: List[Union[str, LinterFormatField]] = [""]
    mode = LinterFormatParseMode.NORMAL
    for c in format:
        if mode == LinterFormatParseMode.NORMAL:
            assert isinstance(
                spans[-1], str
            ), f"non-str latest span in mode {mode}"

            if c == "%":
                mode = LinterFormatParseMode.NEED_LBRACE
            else:
                spans[-1] += c
        elif mode == LinterFormatParseMode.NEED_LBRACE:
            assert isinstance(
                spans[-1], str
            ), f"non-str latest span in mode {mode}"

            # Allow escaping '%' chars.
            if c == "%":
                spans[-1] += c
                mode = LinterFormatParseMode.NORMAL
            elif c == "{":
                if spans[-1] == "":
                    spans.pop()
                spans.append(LinterFormatField(""))
                mode = LinterFormatParseMode.NEED_CHAR
            else:
                raise ValueError(c)
        elif mode == LinterFormatParseMode.NEED_CHAR:
            assert c not in (
                "{",
                "}",
                "%",
            ), f"special char in field name {c!r}"
            assert isinstance(
                spans[-1], LinterFormatField
            ), f"non-field-span latest span in mode {mode}"

            spans[-1].field += c
            mode = LinterFormatParseMode.NEED_CHAR_OR_RBRACE
        elif mode == LinterFormatParseMode.NEED_CHAR_OR_RBRACE:
            assert c not in ("{", "%"), f"special char in field name {c!r}"
            assert isinstance(
                spans[-1], LinterFormatField
            ), f"non-field-span latest span in mode {mode}"
            assert spans[-1].field, f"empty current field name {spans[-1]!r}"

            if c == "}":
                spans[-1].validate()
                spans.append("")
                mode = LinterFormatParseMode.NORMAL
            else:
                spans[-1].field += c
    if isinstance(spans[-1], str) and spans[-1] == "":
        spans.pop()
    return spans


def linter_format_to_regex(
    spans: List[Union[str, LinterFormatField]], filename: str
) -> re.Pattern[str]:
    patt = ""
    for span in spans:
        if isinstance(span, str):
            patt += re.escape(span)
        else:
            if span.field == "file":
                inner_re = re.escape(filename)
            elif span.field in ("line", "column"):
                inner_re = r"\d+"
            else:
                inner_re = ".+"
            patt += f"(?P<{span.field}>{inner_re})"
    return re.compile(patt)


def print_linter_output(
    keybind: Keybind,
    namespace: argparse.Namespace,
    filename: str,
    output: Iterable[str],
) -> None:
    regex = linter_format_to_regex(namespace.linter_format, filename)
    for line in output:
        m = regex.match(line)
        if m:
            msg = m.group("message")
            if "type" in regex.groupindex and m.group("type"):
                print(
                    f"{namespace.sxhkdrc}:{keybind.command.line}: {m.group('type')}: {msg}"
                )
            else:
                print(f"{namespace.sxhkdrc}:{keybind.command.line}:{msg}")


def wait_on_proc(proc: subprocess.Popen[str]) -> Tuple[str, Optional[int]]:
    _, errs = proc.communicate()
    code = None
    if proc.returncode != 0:
        if 1 <= proc.returncode <= 125:
            code = CODE_1_125
        elif proc.returncode < 0:
            code = CODE_SIGNAL
        else:
            code = CODE_CANNOT_RUN
        code -= 64
    return (errs, code)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Exit codes correspond to that of xargs, subtracting 64 for errors during
    commands run due to --exec/-e.

    However, if `Exception` is raised during the intrinsic command invocations
    (from argv), exits early and exit code 1 is returned.  `Exception` during
    --exec/-e commands also exit early and then return the exit code of the
    instrinsic invocation errors + 1 (i.e., exit code 2).

    ENOENT during intrinsic invocations exit early.  Various other software
    errors return exit code 1.
    """
    parser = argparse.ArgumentParser(
        get_command_name(__file__),
        description="Run commands on sxhkdrc keybinds",
        parents=[BASE_PARSER],
    )
    add_repl_str_options(parser)
    parser.add_argument(
        "--mode",
        "-m",
        # "edit [commands]"
        choices=["edit", "filter", "linter"],
        default="edit",
        help="set mode (default: %(default)s)",
    )
    parser.add_argument(
        "--exclude-synchronous-marker",
        "-s",
        action="store_true",
        help="exclude ';' prefix from output for keybind commands that indicate synchronous",
    )
    parser.add_argument(
        "--linter-format",
        "-f",
        default="%{file}:%{line}:%{column}:%{message}",
        type=parse_linter_format,
        help="set linter error message format (default: %(default)s)",
    )
    parser.add_argument(
        "--exec",
        "-e",
        action="store_true",
        help="execute keybind command text after modification by the command from argv",
    )
    parser.add_argument(
        "--delay",
        "-d",
        default=0,
        type=float,
        help="wait DELAY seconds between each --exec command invocation (default: %(default)s)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
    )
    namespace = parser.parse_args(argv)
    if namespace.command and namespace.command[0] == "--":
        namespace.command = namespace.command[1:]
    section_handler, metadata_parser = process_args(namespace)
    if namespace.mode != "edit":
        if namespace.exec:
            print(
                f"cannot use --mode={namespace.mode} with --exec",
                file=sys.stderr,
            )
            return 1

    keybinds: KeybindDict = {}
    try:
        for bind_or_err in read_sxhkdrc(
            namespace.sxhkdrc,
            section_handler=section_handler,
            metadata_parser=metadata_parser,
            # Handle them ourselves.
            hotkey_errors=IGNORE_HOTKEY_ERRORS,
        ):
            if isinstance(bind_or_err, SXHKDParserError):
                msg = format_error_msg(bind_or_err, namespace.sxhkdrc)
                print(msg, file=sys.stderr)
                continue

            for hkperm, cmdperm in zip(
                bind_or_err.hotkey.permutations,
                bind_or_err.command.permutations,
            ):
                norm_str = str(hkperm)
                keybinds[norm_str] = (
                    bind_or_err,
                    cmdperm,
                    bind_or_err.command.synchronous,
                )
    except SXHKDParserError as e:
        msg = format_error_msg(e, namespace.sxhkdrc)
        print(msg, file=sys.stderr)
        return 1

    repl_evaluator = ReplaceStrEvaluator(
        namespace.hotkey_replace_str,
        namespace.command_replace_str,
    )
    sxhkd_shell = os.getenv("SXHKD_SHELL", os.getenv("SHELL"))
    if sxhkd_shell is None:
        print("no $SXHKD_SHELL or $SHELL defined", file=sys.stderr)
        return 1
    if sxhkd_shell == "":
        print("empty $SXHKD_SHELL and $SHELL", file=sys.stderr)
        return 1

    code = 0
    procs: List[subprocess.Popen[str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for hk in get_perms_to_exec(sys.stdin, keybinds):
            bind, cmd, synchronous = keybinds[hk]
            with open(os.path.join(tmpdir, hk), "w") as f:
                f.write(cmd)
            if namespace.mode == "edit":
                print(hk)

            failed = False
            if namespace.command:
                # TODO: fail if hotkey repl str in cmdline but cmdline ends with `+'
                if not any(
                    namespace.hotkey_replace_str in arg
                    or namespace.command_replace_str in arg
                    for arg in namespace.command
                ):
                    cmdline = namespace.command + [f.name]
                else:
                    cmdline = repl_evaluator.eval(
                        namespace.command, hk, f.name
                    )
                try:
                    result = subprocess.run(
                        cmdline,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    failed = True
                    if 1 <= e.returncode <= 125:
                        code = CODE_1_125
                    elif e.returncode == 255:
                        code = CODE_255
                    elif e.returncode < 0:
                        code = CODE_SIGNAL
                    else:
                        code = CODE_CANNOT_RUN
                    do_continue = False
                    if namespace.mode == "filter":
                        print(
                            e.stdout,
                            end="",
                        )
                    elif namespace.mode == "linter":
                        print_linter_output(
                            bind,
                            namespace,
                            filename=f.name,
                            output=e.stdout.split("\n"),
                        )
                        do_continue = True
                    print(
                        e.stderr,
                        end="",
                        file=sys.stderr,
                    )
                    # Immediately exit upon encountering 255.
                    if code == CODE_255:
                        return code
                    if do_continue:
                        continue
                except FileNotFoundError as e:
                    print(e, file=sys.stderr)
                    return CODE_NOT_FOUND
                except Exception as e:
                    print(e, file=sys.stderr)
                    return CODE_OTHER
                else:
                    if namespace.mode == "linter":
                        print_linter_output(
                            bind,
                            namespace,
                            filename=f.name,
                            output=result.stdout.split("\n"),
                        )
                        print(result.stderr, end="", file=sys.stderr)
                        continue
                    else:
                        cmd = result.stdout
                        print(result.stderr, end="", file=sys.stderr)

            assert namespace.mode != "linter"

            if namespace.exec and not failed:
                assert namespace.mode == "edit"
                cmdline = [sxhkd_shell, "-c", cmd]
                # exit codes reduced by 64 (except CODE_OTHER) to distinguish
                # which subprocess out of the intrinsic one and the --exec/-e
                # one errored.
                try:
                    newproc = subprocess.Popen(
                        cmdline,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    )
                except FileNotFoundError as e:
                    print(e, file=sys.stderr)
                    code = CODE_NOT_FOUND - 64
                except Exception as e:
                    print(e, file=sys.stderr)
                    return CODE_OTHER + 1
                else:
                    if synchronous:
                        # Synchronous command may be long-running, so ensure it doesn't hang.
                        errs, newproc_code = wait_on_proc(newproc)
                        print(errs, end="", file=sys.stderr)
                        if newproc_code is not None:
                            code = newproc_code
                    else:
                        procs.append(newproc)
                time.sleep(namespace.delay)

            if cmd and not failed:
                # Escape braces in command.
                # NOTE: At this point, `cmd` is clean and without spans, so any
                # existing braces are literal.
                if namespace.mode == "edit":
                    cmd = cmd.replace("{", "\\{").replace("}", "\\}")

                if not namespace.exclude_synchronous_marker and synchronous:
                    cmd = f";{cmd}"

                if not cmd.endswith("\n"):
                    end = "\n"
                else:
                    end = ""
                if namespace.mode == "filter":
                    prefix = ""
                else:
                    prefix = "\t"
                print(f"{prefix}{cmd}", end=end)

    # Clean up any short-lived processes while ensuring that any long-running
    # subprocesses don't hang.
    with ThreadPoolExecutor() as executor:
        for errs, proc_code in executor.map(wait_on_proc, procs):
            print(errs, end="", file=sys.stderr)
            if proc_code is not None:
                code = proc_code

    # Just return the latest exit code.
    # POSIX xargs doesn't specify it, so this xargs-like tool can do whatever.
    return code
