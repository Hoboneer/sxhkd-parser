- support the special pointer keysyms `button{1-24}`

- warn in some linter tool if two hotkeys with identical chains except the last differ by their value of `noabort`?
  - prerequisite: noabort used after *only* the second-to-last chord

- hkwatch: Add option to call notify-send(1) to inform user that config read failed?
  - Maybe also call it when preparing to read config?

- hkxargs (?): xargs-like cli tool for executing programs on Hotkeys, Commands, or Keybinds

- hkfind: a graphical program to find hotkeys incrementally with autocompletion basically
  - upon getting the abort keysym, quit the program
  - allow defining the abort keysym (like sxhkd does with its -a option), so it can match that of sxhkd
    - however, no automatic way afaik, because sxhkd doesn't share any info about itself while running
  - execute upon selection?

- hkdebug: a program for debugging hotkeys
  - output is unstable--not intended to be parsed
  - some functions: print hotkey tree, print span tree for command
