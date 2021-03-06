PACKAGE_DIRS := sxhkd_parser sxhkd_parser/cli
SOURCE_FILES := $(foreach dir, $(PACKAGE_DIRS), $(wildcard $(dir)/*.py)) #$(wildcard tests/*.py)
CHECK_FILES := $(SOURCE_FILES) setup.py

# Do nothing.
.PHONY: all
all:

.PHONY: requirements
requirements:
	make -C requirements

.PHONY: clean-requirements
clean-requirements:
	make -C requirements clean

.PHONY: keysyms
keysyms:
	make -C keysyms

tags: $(SOURCE_FILES)
	ctags $(SOURCE_FILES)

# Ensure that `isort` and `black` are not run unnecessarily.
.formatted: $(CHECK_FILES)
	isort $?
	black $?
	$(MAKE) tags
	touch .formatted

.PHONY: format
format: .formatted

.PHONY: check-dev
check-dev: typecheck lint test

.PHONY: typecheck
typecheck:
	mypy

.PHONY: lint
lint:
	flake8 $(CHECK_FILES)

.PHONY: test
test:
	pytest || true

.PHONY: clean
clean:
	-rm -fr *.egg-info/
	-rm -fr build/
	-rm -fr dist/
	-rm -f *.1 *.7

.PHONY: dist
dist: keysyms format check-dev man
	python setup.py sdist bdist_wheel
	twine check dist/*

.PHONY: upload
upload:
	twine upload dist/*

.PHONY: man
man: man/hkfind.1 man/hkxargs.1 man/hkwatch.1 man/hkexport.1 man/hkcheck.1 man/hkdebug.1 man/sxhkd-parser.7
man/%.1 man/%.7: man/%.scd
	scdoc <$< >$@
