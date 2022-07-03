sxhkd-parser(7)

# NAME

sxhkd-parser - parser library for *sxhkd*(1) configs

# DESCRIPTION

*sxhkd-parser* is a *python*(1) library for parsing *sxhkd*(1) configs.
This is an overview of some important background information needed when
using any of the tools using this library.  The manpage for *hkdebug*(1)
goes into further detail with regard to the internals of the library.

# TERMINOLOGY

. Hotkey: The sequence of chords needed to activate a command.
. Command: The command passed to the shell after the hotkey is completed.
. Keybind: The entity that encompasses the above.

# SECTIONS AND METADATA

The *sxhkd-parser* library allows sections and metadata to be parsed
from comments in the config.  This relies on passing regular expressions
to certain options.  The list of required options is determined by
the section or metadata type (*--section-type* and *--metadata-type*,
respectively).  These regular expressions may be required to contain
certain Python-style named groups (i.e., "*(?P<name>REGEX)*") so that
relevant information can be extracted.

Sections are recognised greedily: whenever a comment matches the given
regex, a new section is defined.

Metadata is parsed from the contiguous block of comments immediately
preceding a keybind (if any).  If a comment in this block defines a
section, that comment and any comments above it are excluded from the
block parsed for metadata.

; The following two sections are part of the module docstring for sxhkd_parser.metadata
; Be sure to keep them synchronised.
## SECTION TYPES

Sections may be *simple* in which there are no subsections, or represent
a *stack* in which there _are_ subsections, which require that they
are completely enclosed by their parent sections.

There may also be _no_ sections below the root ("*none*"), in which no
comments create new sections.  Consequently, all keybinds are direct
children of the root section.

## METADATA TYPES

With regard to metadata, there may only be descriptions on single
lines ("simple") or they may be key-value pairs across multiple lines
("key-value").

There may also be _no_ metadata for keybinds ("*none*").

# BUGS

; What follows is part of the README.
; Be sure to keep them synchronised.
To maintain simplicity in the implementation, some uncommon features of
sxhkd are unsupported:

- Inconsistent location of "*:*" to indicate noabort across the permutations of
  a hotkey: it's best to keep "*:*" outside of any sequences.
- Alphanumeric ranges within sequences of the form *{s1,s2,s3,...,sn}*:
	- These are OK:
		- Alphabetic: *A-Z*, *a-z*, *A-F*, *a-f*, *A-z* (within ASCII)
		- Numeric: *0-9*, *5-8*
	- These are _not_: *0-A*, *A-0*
	- I'm open to changing this if there's a good justification for it.
- Command cycling:
    - The library enforces the invariant that, within each keybind, the number
      of permutations of the hotkey and the command are equal.
    - It's undocumented and thus unlikely to be used that much anyway.

Also, using *@* or *~* at the start of the hotkey instead of just before
the keysym (as documented in the manpage of *sxhkd*(1)) is unsupported.
Use the documented syntax.

# SEE ALSO

*sxhkd*(1), *python*(1), *hkdebug*(1)
