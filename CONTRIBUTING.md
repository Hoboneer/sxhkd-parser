# Contributing

## Set up a development environment

```sh
python3 -m venv env
. env/bin/activate
pip install -r requirements/dev.txt
pip install -e .
```

Also ensure that you have GNU Make, (Universal) Ctags, and scdoc.
On Debian, Ubuntu, and their derivatives:

```sh
sudo apt install make universal-ctags scdoc
```

Run `make keysyms` to generate the set of keysyms recognised by the library.

## Make a new release

1. Rename the `[Unreleased]` section in `CHANGELOG.md` to `[NEW_VERSION] - CURRENT_DATE_ISO`.
2. Create a new empty `[Unreleased]` section above the one renamed in step (1).
3. Change the version number in `sxhkd_parser/_package.py` (`__version__`) to `NEW_VERSION`.
4. Run `make clean` to remove any build artifacts from previous builds.
5. Run `make dist`.
6. Check that the resulting wheel and sdist has all that you need.
7. Run `make upload`.
8. Add a tag with the new version number: `git tag NEW_VERSION`.
9. Push your changes: `git push --tags`.
