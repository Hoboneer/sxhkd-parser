# sxhkd-parser

sxhkd-parser is a mostly complete library written in Python for parsing
[sxhkd](https://github.com/baskerville/sxhkd) configs.

Future uses:
  - Export keybinds to markdown, html, etc.
  - Search for keybinds
  - Programmatically modify or merge keybinds
  - Keep track of mode for use in status bar, etc.

## Bundled tools

- `hkcheck`: Lint your keybinds.
- `hkexport`: Export your keybinds to various formats including HTML and plaintext.
- `hkwatch`: Tail the sxhkd status fifo and output the current mode.

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
parser = SimpleDescriptionParser(r'^#\s*(?P<value>[A-Z][^.]+\.)$')
for bind_or_err in read_sxhkdrc('sxhkdrc', section_handler=handler, metadata_parser=parser):
    if isinstance(bind_or_err, SXHKDParserError):
        print(bind_or_err)
        continue
    print(keybind)
    keybind.hotkey.get_tree().print_tree()
    keybind.command.get_tree().print_tree()
handler.get_tree().print_tree()
```

In this example, the description for a keybind is the comment immediately
preceding the start of the keybind that matches the given regex.

## Terminology

- Hotkey: The sequence of chords needed to activate a command.
- Command: The command passed to the shell after the hotkey is completed.
- Keybind: The entity that encompasses the above.

## Limitations

- Inconsistent location of `:` to indicate noabort across the permutations of a hotkey:
    - Each `Hotkey` object has a single `noabort_index` attribute.  This will not change.
