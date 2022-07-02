hkdebug(1)

# NAME

hkdebug - debug *sxhkd*(1) configs

# SYNOPSIS

*hkdebug* [global-options...] _mode_ [mode-options...]

# DESCRIPTION

*hkdebug* is a debugging tool for *sxhkd*(1) configs which shows the
internal state of the *sxhkd-parser* library backing *hkcheck*(1),
*hkfind*(1), *hkxargs*(1), *hkwatch*(1), etc.

## NOTE ON TERMINOLOGY

. Hotkey: The sequence of chords needed to activate a command.
. Command: The command passed to the shell after the hotkey is completed.
. Keybind: The entity that encompasses the above.

# OPTIONS

Run *hkdebug --help* for global options.++
Run *hkdebug* _mode_ *--help* for options specific to _mode_.

# MODE

The behaviour of *hkdebug* depends entirely on the subcommands indicated
by _mode_.

## keybinds

This mode prints relevant information for each hotkey-command unit
("keybind") found in the *sxhkd*(1) config.  It also includes the parsed
metadata and optionally the path of the section in which the keybind
is contained.

## hotkey-tree

A core data structure for the *sxhkd-parser* library is a tree of
key-chord nodes in which each path from root to leaf is a hotkey.
This tree can also contain internal nodes which aren't chord nodes.
They help to group subtrees by common sets of modifiers, keysyms, etc.
Use the *-I* option to define this.

This mode prints the hotkey tree for the parsed *sxhkd*(1) config.

## span-tree

A core data structure for the *sxhkd-parser* library is a tree of spans
of text used to expand sequences of the form *{s1,s2,...,sn}* in hotkeys
and commands.  Since sequences cannot be nested, the span tree can be
implemented as a decision tree where every node on the same level has the
same children.  Consequently, its graphical tree representation contains
redundancies.  Use the *-L* option to see just the levels of the tree.

For example, the span tree of "*super + {shift +,\_} a*" has the levels:
. "*super + *"
. "*shift +*", "*\_*"
. "* a*"
Walking the tree then expands the sequences into:
. "*super + shift + a*"
. "*super +  a*" (recall: "*\_*" is the empty sequence element)
In this case, the expanded text would then be tokenised and parsed into
a hotkey.

Ranges are also expanded while a span tree is created; the span tree of
"*super + {0-9}*" has the levels:
. "*super + *"
. "*0*" , "*1*" , "*2*" , "*3*" , "*4*" , "*5*" , "*6*" , "*7*" , "*8*" , "*9*"

This mode prints the span tree for the parsed *sxhkd*(1) config.

# BUGS

Since the purpose of *hkdebug* is for debugging configs by eye, its
output is not guaranteed to be stable across versions.  Use with caution
in scripts.

# SEE ALSO

*sxhkd*(1), *hkcheck*(1), *hkfind*(1), *hkxargs*(1), *hkwatch*(1)
