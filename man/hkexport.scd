hkexport(1)

# NAME

hkexport - export keybinds for *sxhkd*(1) and their metadata in various formats

# SYNOPSIS

hkexport [options...] [-f _format_] [-F _fields_] [-E] [-R _records_ | -S | -K]

# DESCRIPTION

*hkexport* generates documentation in various formats for *sxhkd*(1)
keybinds which include the hotkey and the associated metadata.
It optionally expands embedded sequences in the hotkey and *description*
field of the metadata (if the number of permutations match those of the
hotkey) with the *-E* option.

# OPTIONS

Run *hkexport --help*.

# SEE ALSO

*sxhkd*(1)
