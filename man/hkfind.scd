hkfind(1)

# NAME

hkfind - find *sxhkd*(1) hotkeys that match a search expression

# SYNOPSIS

*hkfind* [options...] [expression]

# DESCRIPTION

*hkfind* searches a *sxhkdrc* file for keybinds that match a given
expression, similar in spirit to *find*(1).  It writes the hotkeys for
the matching keybinds to standard output.  If _expression_ is omitted,
all keybinds are matched.

# OPTIONS

Run *hkfind --help*.

# EXPRESSION

After the options, the search expression may be entered.  *hkfind*
accepts a similar expression syntax as *find*(1), but is much simpler.
An expression is composed of:

- Predicates: The filters against which the keybinds are matched.
- Operators: Join predicates together as a logical expression.

The expression evaluates against each permutation of a keybind.  Only when
the whole expression evaluates to true for a permutation of a keybind
does its hotkey get written to standard output.

## PREDICATES

\-cmd _command_ ;
	Execute _command_; true if 0 status is returned.  An empty
	_command_ is forbidden.  Replaces all occurrences of the hotkey
	replacement string (*-H*) with the hotkey and all occurrences
	of the command replacement string (*-C*) with the filename of
	a temporary file containing the contents of the command for the
	keybind permutation.

\-has _name_
	True if the key _name_ is in the metadata of the keybind.
\-has _name_ = _value_
	True if the value of _name_ is exactly _value_.  Whitespace is
	significant before and after the "=".
\-has _name_ != _value_
	True if the value of _name_ is *not* _value_.  Whitespace is
	significant before and after the "!=".
*NOTE ON GREEDINESS*: *-has* consumes more than one argument _only_ if the next argument cannot possibly be interpreted as another predicate or as a logical operator.  That is, *-has* consumes three arguments if the argument after _name_:
. does not start with *-* (predicate); and
. is not exactly *!* (negation), *(*, or *)* (grouping).

\-in _path_++
\-child-of _path_
	True if the keybind is a (direct) child of the section referred
	to by _path_.  An empty _path_ is forbidden.
\-under _path_++
\-descendant-of _path_
	True if the keybind is a descendant (but not a direct child) of
	the section referred to by _path_.  An empty _path_ is forbidden.
_path_ is an absolute or relative UNIX path and can have the directory
separators escaped, i.e., */foo\\/bar/baz* has section *baz* underneath
the top-level *foo/bar*, but *foo\\/bar/baz* has section *baz* underneath
a section *foo/bar* which may or may not be at the top-level.

## OPERATORS

Listed in order of decreasing precedence, with the same meaning as those
in *find*(1):

( _expr_ )

! _expr_

_expr1_ -a _expr2_++
_expr1_ _expr2_

_expr1_ -o _expr2_

# EXIT STATUS

Similarly to *find*(1), *hkfind* exits with status 0 if no errors occur,
greater than 0 if any do.

# SEE ALSO

*sxhkd*(1), *find*(1)
