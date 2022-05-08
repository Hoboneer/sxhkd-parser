- warn in some linter tool if two hotkeys with identical chains except the last differ by their value of `noabort`?
  - prerequisite: noabort used after *only* the second-to-last chord

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
- lib: While expanding sequences, track starting line and columns for each
  (raw) span?
  - to allow for exact line and column count as given in the raw input when
    running checks on the `Hotkey` and `Command` objects
  - e.g., allows converting error messages such that the errors can be
    pinpointed in the input file, including the sequence number
