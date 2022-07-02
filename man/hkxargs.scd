hkxargs(1)

# NAME

hkxargs - run commands on *sxhkd*(1) keybinds

# SYNOPSYS

hkxargs [options...] [_command_]

# DESCRIPTION

*hkxargs* parses each line of standard input as a hotkey (which must not
have embedded sequences of the form *{s1,s2,...,sn}*) and then executes
_command_ on each keybind permutation referred to by a hotkey.

Similarly to *xargs*(1), _command_ can have replacement strings for
the hotkey (see *-H* option) and for the filename of a temporary file
containing the contents of the command (excluding any ";" prefix;
see *-C* option).  _command_ may contain any number of each type of
replacement string.  However, if no replacement strings are contained
in _command_, then *hkxargs* will interpret _command_ as if it had the
command replacement string appended to it as a final argument.

Based on the mode specified by the *--mode* option, *hkxargs* uses the
standard output of each invocation of _command_.

# OPTIONS

Run *hkxargs --help*.

# MODE

*hkxargs* has modes which determine how the output of _command_ will
be interpreted.

## edit

The *edit* mode works similarly to *xargs*(1).  The standard output of
each invocation of _command_ is interpreted to be the new body of the
keybind permutation referred to by the hotkey.

If *--exec* is used, the new command body is executed using *SHELL
\-c* _new\_command\_body_.  If the original command specified it was
synchronous (by prefixing ";"), then *hkxargs* waits for the command
to exit before executing further commands.  A delay between each such
invocation can be specified by the *--delay* option.  *--exec* can only
be used with this mode.

Finally, the keybind permutation is printed according to *sxhkdrc*
syntax, with any embedded braces in the command escaped:

*HOTKEY*
	*NEW_COMMAND*

## filter

The *filter* mode works similarly to the *edit* mode, with the difference
being that the final output is only the command without any whitespace
to prefix it:

*COMMAND*

## linter

The *linter* mode interprets each line of the output of the command to be
a compiler-style error message.  The format of the error message can be
configured by the *-f* option, which contains a mix of literal text and
format fields of the form *%{FIELDNAME}*.  The valid fields are *file*,
*line*, *column*, *type*, *message*.

*hkxargs* itself outputs a compiler-style error message:

If *type* is in the format spec: "*SXHKDRC_FILENAME*:*COMMAND_LINENO*:
*EXTRACTED_TYPE*: *EXTRACTED_MESSAGE*"++
Otherwise: "*SXHKDRC_FILENAME*:*COMMAND_LINENO*:*EXTRACTED_MESSAGE*"

# EXIT STATUS

The exit status of *hkxargs* distinguishes between errors during
the execution of _command_ (which happens first and always happens;
"intrinsic invocation") and errors during commands run due to *--exec*
("non-intrinsic invocation").

The exit status for errors during intrinsic invocation are taken straight
from GNU *xargs*(1) v4.8.0.  During non-intrinsic invocation, there is
the same set of error types, which *hkxargs* reports by subtracting 64
from the corresponding exit code for intrinsic invocation errors *(EXCEPTION:
exit status 1 becomes 2)*.

*hkxargs* exits immediately upon encountering exit status 255 or ENOENT
during intrinsic invocation.

# ENVIRONMENT

; Straight from *sxhkd*(1)
*SXHKD_SHELL*, *SHELL*
	Just like *sxhkd*(1), *SHELL* will be the content of the first
	defined environment variable in the following list: *SXHKD_SHELL*,
	*SHELL*.

# EXAMPLES

## Running shellcheck(1) on the commands of every keybind permutation

```
$ hkfind | hkxargs -m linter shellcheck -s sh -f gcc --color=always | uniq
```

# SEE ALSO

*sxhkd*(1), *xargs*(1), *hkfind*(1), *shellcheck*(1)
