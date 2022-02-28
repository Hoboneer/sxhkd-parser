all: keysyms.py

HASH = \#

keysyms.py: XF86keysym.h keysymdef.h
	awk '$$1 ~ "#define" {print $$2}' $^ | sed -e 's/^XK_//' -e 's/^XF86XK_/XF86/' | sort | \
		{ echo "$(HASH) flake8: noqa"; echo ""; \
		  echo '"""Standardised keysyms."""'; echo ""; \
		  echo "from typing import Set"; echo ""; \
		  echo '__all__ = ["KEYSYMS"]'; echo ""; \
		  echo "KEYSYMS: Set[str] = {"; \
		  sed -e 's/^/    "/' -e 's/$$/",/'; \
		  echo "}"; } > $@

clean:
	rm -f keysyms.py

.PHONY: all clean
