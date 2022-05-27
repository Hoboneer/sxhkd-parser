# sxhkd-parser

sxhkd-parser is a mostly complete library written in Python for parsing
[sxhkd](https://github.com/baskerville/sxhkd) configs.

## Bundled tools

- `hkcheck`: Lint your keybinds.
- `hkexport`: Export your keybinds to various formats including HTML and
  plaintext.
- `hkwatch`: Tail the sxhkd status fifo and output the current mode.
- `hkfind`: Get hotkeys that match given search criteria.
- `hkxargs`: Execute commands on a keybind's command for editing them,
  retrieving some information from them, or running linters on them.

For more, see the modules prefixed with `hk` in `sxhkd_parser/cli/`.

## Quickstart

### Print all your keybinds

```python
from sxhkd_parser import *

for bind_or_err in read_sxhkdrc('sxhkdrc'):
    if isinstance(bind_or_err, SXHKDParserError):
        print(bind_or_err)
        continue
    keybind = bind_or_err
    print(keybind)
    keybind.hotkey.get_tree().print_tree()
    keybind.command.get_tree().print_tree()
```

### Include sections and descriptions

```python
from sxhkd_parser import *

handler = SimpleSectionHandler(r'^#\s*(?P<name>[A-Z. /-]+):$')
parser = SimpleDescriptionParser(r'^#\s*(?P<description>[A-Z][^.]+\.)$')
for bind_or_err in read_sxhkdrc('sxhkdrc', section_handler=handler, metadata_parser=parser):
    if isinstance(bind_or_err, SXHKDParserError):
        print(bind_or_err)
        continue
    keybind = bind_or_err
    print(keybind)
    keybind.hotkey.get_tree().print_tree()
    keybind.command.get_tree().print_tree()
handler.get_tree().print_tree()
```

In this example, the description for a keybind is the comment immediately
preceding the start of the keybind that matches the given regex.

## Terminology

1. Hotkey: The sequence of chords needed to activate a command.
2. Command: The command passed to the shell after the hotkey is completed.
3. Keybind: The entity that encompasses the above.

I'm aware that "hotkey" and "keybind" are interchangeable and have the meaning
of (3) above, so any suggestions for renaming (1) are welcome.

## Limitations

To maintain simplicity in the implementation, some uncommon features of sxhkd
are unsupported.  The list follows:

- Inconsistent location of `:` to indicate noabort across the permutations of a
  hotkey:
    - Each `Hotkey` object has a single `noabort_index` attribute.  This will
      not change.
- Alphanumeric ranges within sequences of the form `{s1,s2,s3,...,sn}`:
    - These are OK:
      - Alphabetic: `A-Z`, `a-z`, `A-F`, `a-f`, `A-z` (within ASCII)
      - Numeric: `0-9`, `5-8`
    - These are *not*: `0-A`, `A-0`
    - I'm open to changing this if there's a good justification for it.
- Command cycling:
    - The library enforces the invariant that, within each keybind, the number
      of permutations of the hotkey and the command are equal.
    - It's undocumented and thus unlikely to be used that much anyway.
