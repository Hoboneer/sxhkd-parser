hkwatch(1)

# NAME

hkwatch - keep track of the current mode of *sxhkd*(1) by tailing its status FIFO

# SYNOPSIS

hkwatch [options...] [-s _status-fifo_] [-m _mode-field_]

# DESCRIPTION

*hkwatch* outputs status information in a similar format to the messages
that *sxhkd*(1) sends to its status FIFO.  Its main purpose is to use
the hotkey tree and the metadata gained from parsing the config so that
the current chord state of *sxhkd*(1) can be tracked for a status bar,
which includes the mode.

*hkwatch* only uses the messages from the status FIFO that are prefixed
by *H*, *B*, or *E*.

At startup, *hkwatch* is initialised to assume that *sxhkd*(1) is not
in a mode or mid-chain (i.e., mode is set to *normal*).

When an *H* message is seen, the hotkey that follows the *H* prefix
is stored.  This hotkey must not contain embedded sequences of the form
*{s1,s2,...,sn}*.

Upon seeing a *BBegin chain* message, the stored hotkey is parsed and
each chord in the parsed hotkey is matched to chord nodes on a path
along the hotkey tree (which came from parsing the config).

First, *hkwatch* attempts to find a chord node that begins a mode with
the last chord in the parsed hotkey.  In the example below, if the parsed
hotkey is *super + a; super + a*, the second *super + a* node would be
matched, and *hkwatch* would print the name of the current mode (taken
from the metadata field specified by the *-m* option).

- *super + a*
	- *super + a* (starts mode)
		- *a*
		- *b*
		- ...

If that fails to find a node, *hkwatch* attempts to find a chord node
that matches the parsed hotkey exactly.  In the example below, if the
parsed hotkey is *super + b; super + c*, the *super + c* node would be
matched, and *hkwatch* would print that *sxhkd*(1) is currently completing
a hotkey.

- *super + b*
	- *super + c*
		- *a*
		- *b*
		- ...

Upon seeing an *EEnd chain* message, the mode is set to *normal* again
(which is printed).

*NOTE: The output is undefined when the parsed hotkey doesn't exist in the
config.*

# OPTIONS

Run *hkwatch --help*.  Some important options for *hkwatch* are *-s*
and *-m*.

# MESSAGE FORMAT

Similarly to *sxhkd*(1), each message of *hkwatch* is prefixed by a
character indicating the message type.

## Normal (N)

*sxhkd*(1) is not currently completing a hotkey, nor is it in a mode.

## Unknown (U)

*sxhkd*(1) is in a mode with an unknown name.  The message content is
the hotkey (e.g., *Usuper + a* when the mode specified by *-m* doesn't
exist in the metadata of the hotkey).

## In-chain (C)

*sxhkd*(1) has completed one or more chords but is not in a mode and it
hasn't yet completed a hotkey (i.e., it's currently completing a hotkey).

## User-mode (M)

*sxhkd*(1) is currently in a mode.  The message content is the mode name
(e.g., *Mresize*).

# EXIT STATUS

*hkwatch* has no special exit status codes.  Simply 0 on success,
non-zero on failure.

# SIGNALS

Upon receiving these signals, *hkwatch* will take certain actions.

*SIGUSR1*
	Reload the config.

*SIGUSR2*
	Print the current mode.

# SEE ALSO

*sxhkd*(1)
