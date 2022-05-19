# Implementer's Notes

These are some implementer's notes about sxhkd and its behaviour.

## Limitations

### Character limit in hotkeys and commands

Version 0.5.1 - 0.6.2 (inclusive) of sxhkd limits hotkeys and commands to 512
bytes.  `hkcheck` warns when it is exceeded.

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
