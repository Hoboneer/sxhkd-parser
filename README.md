# sxhkd-parser

sxhkd-parser is a mostly complete library written in Python 3.7 for
parsing [sxhkd](https://github.com/baskerville/sxhkd) configs.

It has no dependencies and will not have any in the future.

## Bundled tools

- `hkcheck`: Lint your keybinds.
- `hkexport`: Export your keybinds to various formats including HTML and
  plaintext.
- `hkwatch`: Tail the sxhkd status fifo and output the current mode.
- `hkfind`: Get hotkeys that match given search criteria.
- `hkxargs`: Execute commands on a keybind's command for editing them,
  retrieving some information from them, or running linters on them.
- `hkdebug`: Retrieve information useful for debugging the config.

For more, see the modules prefixed with `hk` in `sxhkd_parser/cli/`.

## Goals

- Provide a high-level library for manipulating `sxhkd` configs
- Be a test-bed for compatible extensions to `sxhkdrc` syntax

## Interface Stability

The library API has no guarantees about stability (yet), but the interface of
the CLI tools should be relatively more stable.

This project follows [semantic versioning](https://semver.org/).  On v1.0.0,
the library API will be stable and the CLI tools will be split into a separate
Python package after which they can start to have dependencies.

## Quickstart

### Install

We will use [pipx](https://github.com/pypa/pipx) because it is convenient.
`pipx` is available on distro repositories.

Run `pipx install sxhkd-parser`.

#### Manuals

If your system uses [man-db](https://www.nongnu.org/man-db/), include
`export MANPATH="$HOME/.local/pipx/venvs/sxhkd-parser/share/man:"` in
`.profile`, `.bashrc`, or any startup config file of your choice.

Run `man <TOOL>` for the details of each tool.  `man 7 sxhkd-parser`
also has the background information needed to use them.

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
    - So, it's best to keep `:` outside of any sequences.
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

Also, using `@` or `~` at the start of the hotkey instead of just before
the keysym (as documented in the manpage of `sxhkd`) is unsupported.
Use the documented syntax.
