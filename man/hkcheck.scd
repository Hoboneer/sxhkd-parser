hkcheck(1)

# NAME

hkcheck - check config file of *sxhkd*(1) for user errors, violations of limitations in *sxhkd*(1), and undefined behaviour

# SYNOPSIS

hkcheck [options...]

# DESCRIPTION

*hkcheck* is a linter tool for *sxhkd*(1) which warns when it sees parts
of the config that contain user errors, violations of limitations in
*sxhkd*(1), and undefined behaviour.  The warnings are formatted like
compiler error messages and contain the line (and possibly column)
in the config file where the error occurs.

# OPTIONS

Run *hkcheck --help*.

# CHECKS

## User Errors

*Use of potentially invalid keysyms*
	Any keysyms that are not in the set of keysyms defined by the
	headers for X11 or for XFree86 vendor specific keysyms are
	unlikely to be real keysyms.  However, the actual system being
	used is the ultimate authority on this, hence only "_potentially_
	invalid".

*Single-argument sequences*
	Since sequences use braces as delimiters, it is easy to forget
	to escape them when using *awk*(1) in a command (for example).
	This results in a sequence with only one element and a command
	with the braces removed.  This is almost certainly not the
	intended result.

; What follows is most of the content of NOTES.md in the repo root.
; Be sure to keep them synchronised.
; Make sure this version has inline references to other manpages where necessary
; and that unnecessary phrases for a manpage are removed.
## Limitations

*Character limit in hotkeys and commands*
	Version 0.5.1 - <current version (0.6.2)> (inclusive) of
	*sxhkd*(1) limits hotkeys and commands to 512 bytes.

## Undefined Behaviour

*Empty sequence elements without using \_*
	The manpage for *sxhkd*(1) states that *\_* is an empty sequence
	element but doesn't indicate that it's just "syntactic sugar"
	or say anything about what happens with an actual empty string
	as a sequence element:

	- *{a,b,\_}* is well-defined
	- *{a,b,}* is undefined

*Duplicate hotkeys and hotkey conflicts*
	The manpage for *sxhkd*(1) is silent about what happens when
	there are duplicate hotkeys--whether the first or last one
	wins--or when hotkeys conflict.

	Here are some example conflicts:

	. *super + a*
	. *super + a; b*
	. *super + a: {c,d}*

	(1) conflicts with both (2) and (3) because, once the chord
	*super + a* has been entered, sxhkd has two choices:

	- execute the command for (1); or
	- continue to (2) or (3).

	(2) and (3) conflict with each other because, once the chord
	*super + a* has been entered, sxhkd has two choices:

	- wait for more input until *b* is received and then execute the
	  command for (2); or
	- enter the mode associated with *super + a* and execute the associated
	  commands every time when *c* or *d* is pressed while the mode is active.

# SEE ALSO

*sxhkd*(1)
