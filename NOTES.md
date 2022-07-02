# Implementer's Notes

These are some implementer's notes about sxhkd and its behaviour.

## Limitations

### Character limit in hotkeys and commands

Version 0.5.1 - \<current version (0.6.2)> (inclusive) of sxhkd limits
hotkeys and commands to 512 bytes.  `hkcheck` warns when it is exceeded.

### Single chord limit after noabort character (`:`) in hotkeys

It seems that only one chord may be used after the `:` to indicate noabort in a
hotkey---probably to avoid two layers of modes which can be escaped: once to
abort the current chord chain, and once more to abort the noabort chord chain
(and thus the whole hotkey).  I haven't tested this that much though.

`hkcheck` doesn't yet warn about this.

## Undefined Behaviour

Here is some behaviour left undefined by the sxhkd manpage.

### Empty sequence elements without using `_`

The manpage states that `_` is an empty sequence element but doesn't indicate
that it's just "syntactic sugar" or say anything about what happens with an
actual empty string as a sequence element:

- `{a,b,_}` is well-defined
- `{a,b,}` is undefined

### Duplicate hotkeys and hotkey conflicts

The manpage is silent about what happens when there are duplicate
hotkeys--whether the first or last one wins--or when hotkeys conflict.

Here are some example conflicts:

1. `super + a`
2. `super + a; b`
3. `super + a: {c,d}`

\(1) conflicts with both (2) and (3) because, once the chord `super + a` has
been entered, sxhkd has two choices:

- execute the command for (1); or
- continue to (2) or (3).

\(2) and (3) conflict with each other because, once the chord `super + a` has
been entered, sxhkd has two choices:

- wait for more input until `b` is received and then execute the command for (2); or
- enter the mode associated with `super + a` and execute the associated commands every time when `c` or `d` is pressed while the mode is active.

`hkcheck` warns on duplicates and conflicts.
