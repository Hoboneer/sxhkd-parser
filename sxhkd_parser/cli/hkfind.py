"""Tool to search keybinds that match given search criteria."""
import argparse
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    cast,
)

from ..errors import SXHKDParserError
from ..metadata import SectionTreeNode
from ..parser import Keybind
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


@dataclass
class KeybindMatcherContext:
    replace_str: ReplaceStrEvaluator


class Expression:
    @classmethod
    def from_args(cls, args: Deque[str]) -> "Expression":
        return OrExpression.from_args(args)

    # path from root to here (last element is current section)
    # empty path means root
    # RETURN: permutation indices that match
    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        raise NotImplementedError

    def print(self, level: int = 0) -> None:
        raise NotImplementedError


RESERVED_ARGS = set()


@dataclass
class OrExpression(Expression):
    lhs: Expression
    rhs: Expression
    RESERVED_ARGS.add("-o")

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        assert args[0] not in ("-o", "-a"), args
        maybe_lhs = AndExpression.from_args(args)
        # Backtrack in case this is not an OR.
        if not args:
            return maybe_lhs

        out_expr: Expression
        if args[0] == "-o":
            args.popleft()
            rhs = cls.from_args(args)
            out_expr = cls(maybe_lhs, rhs)
            return out_expr
        else:
            return AndExpression.from_args(args, lhs=maybe_lhs)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        # Short-circuit.
        lhs = self.lhs.match(keybind, path, ctx)
        if lhs == set(range(len(keybind.hotkey.permutations))):
            return lhs
        return lhs | self.rhs.match(keybind, path, ctx)

    def print(self, level: int = 0) -> None:
        print(f"{' ' * level}or-lhs:")
        self.lhs.print(level + 1)
        print(f"{' ' * level}or-rhs:")
        self.rhs.print(level + 1)


@dataclass
class AndExpression(Expression):
    lhs: Expression
    rhs: Expression
    RESERVED_ARGS.add("-a")

    @classmethod
    def from_args(
        cls, args: Deque[str], lhs: Optional[Expression] = None
    ) -> Expression:
        assert args[0] != "-o"
        if lhs is None:
            lhs = NotExpression.from_args(args)
            # Backtrack in case this is not an AND.
            if not args:
                return lhs
        # Allow ANDing via juxtaposition
        if args[0] == "-a":
            args.popleft()
        # `lhs` should be LHS of calling function (parse_or).  AND binds tighter
        # than OR, and need to take into account surrounding parens.  Not returning
        # early would cause parse errors in the higher-precedence operators or in
        # the predicates.
        elif args[0] in ("-o", ")"):
            return lhs
        rhs = cls.from_args(args)
        out_expr = cls(lhs, rhs)
        return out_expr

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        # Short-circuit.
        lhs = self.lhs.match(keybind, path, ctx)
        if not lhs:
            return lhs
        return lhs & self.rhs.match(keybind, path, ctx)

    def print(self, level: int = 0) -> None:
        print(f"{' ' * level}and-lhs:")
        self.lhs.print(level + 1)
        print(f"{' ' * level}and-rhs:")
        self.rhs.print(level + 1)


@dataclass
class NotExpression(Expression):
    subexpr: Expression
    RESERVED_ARGS.add("!")

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        if args[0] == "!":
            args.popleft()
            subexpr = cls.from_args(args)
            return cls(subexpr)
        else:
            return _ParenExpression.from_args(args)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        # return not self.subexpr.match(keybind, path, ctx)
        return set(
            range(len(keybind.hotkey.permutations))
        ) - self.subexpr.match(keybind, path, ctx)

    def print(self, level: int = 0) -> None:
        print(f"{' ' * level}not:")
        self.subexpr.print(level + 1)


# RESERVED_ARGS.add("(")
# RESERVED_ARGS.add(")")


@dataclass
class _ParenExpression(Expression):
    subexpr: Expression
    RESERVED_ARGS.add("(")
    RESERVED_ARGS.add(")")

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError("This should never have been instantiated")

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        if args[0] == "(":
            args.popleft()
            subexpr = Expression.from_args(args)
            maybe_rparen = args.popleft()
            if maybe_rparen == ")":
                # return ParenExpression(subexpr)
                return subexpr
            else:
                raise ValueError(f"expected ')' but got {maybe_rparen!r}")
        else:
            return PredicateExpression.from_args(args)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        return self.subexpr.match(keybind, path, ctx)

    def print(self, level: int = 0) -> None:
        print(f"{' ' * level}paren:")
        self.subexpr.print(level + 1)


class PredicateExpression(Expression):
    _PARSERS: ClassVar[Dict[str, Type["PredicateExpression"]]] = {}
    RESERVED: ClassVar[Sequence[str]] = ()

    def __init_subclass__(cls, **kwargs: Any):
        for reserved_arg in getattr(cls, "RESERVED", []):
            cls._PARSERS[reserved_arg] = cls

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        val = args.popleft()
        try:
            pred_expr_cls = cls._PARSERS[val]
        except KeyError as e:
            raise ValueError(f"invalid syntax item: {val!r}") from e
        else:
            return pred_expr_cls.from_args(args)


@dataclass
class CommandPredicate(PredicateExpression):
    args: List[str]
    RESERVED = ["-cmd"]
    RESERVED_ARGS.update(RESERVED)

    def __init__(self, args: List[str]):
        if not args:
            raise ValueError("empty command")
        self.args = args

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        cmdargs = []
        while args and args[0] != ";":
            cmdargs.append(args.popleft())
        if not args:
            raise ValueError(
                f"unterminated command ({'|'.join(cls.RESERVED)}) predicate"
            )
        args.popleft()
        return cls(cmdargs)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        matches = set()
        for i, (hk, cmd) in enumerate(
            zip(keybind.hotkey.permutations, keybind.command.permutations)
        ):
            with tempfile.NamedTemporaryFile("w+") as f:
                f.write(cmd)
                f.flush()
                if not any(
                    ctx.replace_str.hotkey in arg
                    or ctx.replace_str.command in arg
                    for arg in self.args
                ):
                    cmdline = self.args + [f.name]
                else:
                    norm_str = str(hk)
                    cmdline = ctx.replace_str.eval(
                        self.args, hotkey=norm_str, cmd_file=f.name
                    )
                try:
                    result = subprocess.run(
                        cmdline,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(e.stderr, end="", file=sys.stderr)
                    continue
                except FileNotFoundError as e:
                    print(e, file=sys.stderr)
                    continue
                else:
                    print(result.stderr, end="", file=sys.stderr)
                    matches.add(i)
        return matches

    def print(self, level: int = 0) -> None:
        print(f"{' ' * level}command: {self.args}")


@dataclass
class MetadataPredicate(PredicateExpression):
    name: str
    operator: Optional[str]
    value: Optional[str]
    OPERATORS: ClassVar[Sequence[str]] = ("=", "!=")
    RESERVED = ["-has"]
    RESERVED_ARGS.update(RESERVED)

    def __init__(
        self,
        name: str,
        operator: Optional[str] = None,
        value: Optional[str] = None,
    ):
        if value is not None:
            assert operator is not None
        self.name = name
        if value is not None and operator not in MetadataPredicate.OPERATORS:
            raise ValueError(
                f"invalid operator {operator!r}: must be one of {MetadataPredicate.OPERATORS}"
            )
        self.operator = operator
        self.value = value

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        name = args.popleft()
        # -has should only consume further arguments if the next one cannot
        # possibly be interpreted as another predicate or as a logical
        # operator.
        if not args or (args[0].startswith("-") or args[0] in ("!", "(", ")")):
            return cls(name)
        operator = args.popleft()
        value = args.popleft()
        return cls(name, operator, value)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        perms = set(range(len(keybind.hotkey.permutations)))
        # Existence (-has NAME)
        if self.value is None:
            if self.name in keybind.metadata:
                return perms
            else:
                return set()

        # Others (-has NAME(=|!=)VALUE)
        if self.operator == "=":
            if keybind.metadata.get(self.name) == self.value:
                return perms
            else:
                return set()
        elif self.operator == "!=":
            if keybind.metadata.get(self.name) != self.value:
                return perms
            else:
                return set()
        else:
            raise NotImplementedError

    def print(self, level: int = 0) -> None:
        if self.value:
            print(
                f"{' ' * level}metadata-value: {self.name!r} {self.operator} {self.value!r}"
            )
        else:
            print(f"{' ' * level}metadata-exists: {self.name!r}")


def parse_path(raw_path: str) -> Tuple[List[str], bool]:
    assert raw_path
    if raw_path[0] == "/":
        absolute = True
        raw_path = raw_path[1:]
    else:
        absolute = False

    path = []
    component = ""
    do_escape = False
    for c in raw_path:
        if do_escape:
            if c == "/":
                component += c
            else:
                component += "\\" + c
            do_escape = False
        else:
            if c == "\\":
                do_escape = True
            elif c == "/":
                if component:
                    path.append(component)
                component = ""
            else:
                component += c
    if do_escape:
        component += "\\"
    if component:
        path.append(component)
    return (path, absolute)


@dataclass
class SectionChildPredicate(PredicateExpression):
    path: List[str]
    absolute: bool
    RESERVED = ["-in", "-child-of"]
    RESERVED_ARGS.update(RESERVED)

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        path, absolute = parse_path(args.popleft())
        return cls(path, absolute=absolute)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        perms = set(range(len(keybind.hotkey.permutations)))
        if self.absolute:
            if self.path == path:
                return perms
            else:
                return set()
        else:
            assert self.path, f"empty path {self.path}"
            if len(self.path) > len(path):
                return set()
            # Only need direct children, so just match the final path components.
            elif self.path == path[len(path) - len(self.path) :]:
                return perms
            else:
                return set()

    def print(self, level: int = 0) -> None:
        print(
            f"{' ' * level}child-of ({'absolute' if self.absolute else 'relative'}): {self.path}"
        )


@dataclass
class SectionDescendantPredicate(PredicateExpression):
    path: List[str]
    absolute: bool
    RESERVED = ["-under", "-descendant-of"]
    RESERVED_ARGS.update(RESERVED)

    @classmethod
    def from_args(cls, args: Deque[str]) -> Expression:
        path, absolute = parse_path(args.popleft())
        return cls(path, absolute=absolute)

    def match(
        self, keybind: Keybind, path: List[str], ctx: KeybindMatcherContext
    ) -> Set[int]:
        perms = set(range(len(keybind.hotkey.permutations)))
        if len(path) <= len(self.path):
            return set()
        if self.absolute:
            if self.path == path[: len(self.path)]:
                return perms
            else:
                return set()
        else:
            assert self.path, f"empty path {self.path}"
            # WARNING: very bad time-complexity (roughly O(nm) where n=len(path), m=len(self.path)).
            # Might not be so bad in practice since section nesting won't be too deep (2-3 at most), I predict.
            # TODO: maybe use smarter substring finder algorithm (but can't use Python strings---each elem is a path component).
            subpath = path
            # Need non-direct descendants, so can't match on final path components (must use strict >).
            while len(subpath) > len(self.path):
                if self.path == subpath[: len(self.path)]:
                    return perms
                subpath = subpath[1:]
            return set()

    def print(self, level: int = 0) -> None:
        print(
            f"{' ' * level}descendant-of ({'absolute' if self.absolute else 'relative'}): {self.path}"
        )


def match_keybinds(
    node: SectionTreeNode,
    expr: Optional[Expression],
    path: List[str],
    ctx: KeybindMatcherContext,
) -> Iterable[Tuple[Keybind, Set[int]]]:
    for keybind in node.keybind_children:
        if expr is None:
            matches = set(range(len(keybind.hotkey.permutations)))
        else:
            matches = expr.match(keybind, path, ctx)
        yield (keybind, matches)
    for subsection in node.children:
        yield from match_keybinds(
            subsection, expr, path + [cast(str, subsection.name)], ctx
        )


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line tool with the given arguments, without the command name.

    Currently only returns 0 (success) and any non-SXHKDParserError exceptions
    are left unhandled.
    """
    parser = argparse.ArgumentParser(
        get_command_name(__file__),
        usage="%(prog)s [option ...] [expression]",
        description="Find hotkeys that match a search expression",
        epilog="See %(prog)s(1) for the expression syntax.",
        parents=[BASE_PARSER],
    )
    add_repl_str_options(parser)
    parser.add_argument(
        "--print-expression-tree",
        "-T",
        action="store_true",
        help="print parsed tree for the search expression and exit",
    )

    # Separate options from the search expression.
    if argv is None:
        argv = sys.argv[1:]
    argv: Deque[str] = deque(argv)
    options = []
    while argv and argv[0] not in RESERVED_ARGS:
        curr = argv.popleft()
        if curr == "--":
            break
        options.append(curr)
    raw_expr = argv

    namespace = parser.parse_args(options)
    section_handler, metadata_parser = process_args(namespace)
    expr: Optional[Expression]
    if raw_expr:
        expr = Expression.from_args(raw_expr)
    else:
        expr = None
    if namespace.print_expression_tree:
        if expr is not None:
            expr.print()
        return 0

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

    for keybind, matches in match_keybinds(
        section_handler.root,
        expr,
        path=[],
        ctx=KeybindMatcherContext(
            ReplaceStrEvaluator(
                namespace.hotkey_replace_str, namespace.command_replace_str
            )
        ),
    ):
        hk = keybind.hotkey
        for i in matches:
            print(hk.permutations[i])

    return 0
