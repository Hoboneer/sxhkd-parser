- hkfind: output chord chains that match search criteria (one per line)
  - simple searches can be done with hkexport and grepping
- hkxargs: xargs for executing programs on Hotkeys, Commands, or Keybinds
  - allow commands to take multiple command texts as input?
    - syntax: "-cmd" ... "+"
    - disallow hotkey replacement string in command
    - modes:
      - =filter or =linter: output to stdout and stderr
      - =edit: output to each file
- hkrun or hkexec: execute *a* keybind corresponding to a given chord chain from argv

- hkview: a graphical program to find hotkeys incrementally with autocompletion basically
  - upon getting the abort keysym, quit the program
  - allow defining the abort keysym (like sxhkd does with its -a option), so it can match that of sxhkd
    - however, no automatic way afaik, because sxhkd doesn't share any info about itself while running
  - execute upon selection?

- hkdebug: a program for debugging hotkeys
  - output is unstable--not intended to be parsed
  - some functions: print hotkey tree, print span tree for command

- cli: Don't use `ArgumentDefaultsHelpFormatter`
- lib: Use "expanded" or "flat" instead of "static" for stuff like
  `parse_static_hotkey`---it's clearer that the hotkey should have no sequences
- lib: Maybe change interface of section handlers to require calling a method
  to update the line counter upon reading each line?
  - so that no arguments need to passed upon EOF---just look at current state
  - to allow for use as a context manager
  - something like `tick()` or `update(...)` to allow for a more detailed
    "tick" interface in the future?
    - e.g., set step size
    - e.g., allow multiple section handlers per file? (e.g., for merging
      configs)
- lib: Allow easy pretty-printing of configs with configurable output
  formatting with regard to metadata and section comments
- lib: Read config file formatting config (e.g., section handler config) from
  comments at top of file as well
  - to allow for easy use of tools like my `hk*` programs without having to set
    up a way to include the parser args with every call
