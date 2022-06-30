# Using `hkwatch` to track the current mode in status bus

This comes from my system, within my `$XDG_CONFIG_HOME/bspwm` directory which
includes a number of watch scripts under `watch/`.

Execute `launch_watchers` to restart the watch scripts.

Use `pkill -<SIGNAL> -x hkwatch` to send signals to the `hkwatch` process
within the `watch/sxhkd` script process.

To get the output of `watch/sxhkd` in your status bar, it should run `tail -f`
on the output fifo.
