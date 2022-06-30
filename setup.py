import glob
import os

from setuptools import setup

console_scripts = []
scriptpath = "sxhkd_parser/cli"
for hkscript in os.scandir(scriptpath):
    scriptname, ext = os.path.splitext(hkscript.name)
    if not scriptname.startswith("hk") or ext != ".py":
        continue
    console_scripts.append(
        f"{scriptname} = {scriptpath.replace('/', '.')}.{scriptname}:main"
    )
setup(
    entry_points={"console_scripts": console_scripts},
    data_files=[("share/man/man1", glob.glob("man/*.1"))],
)
