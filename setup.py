from pathlib import Path
from shutil import rmtree

from setuptools import setup
from setuptools.command.build_py import build_py


class CleanDashboardBuild(build_py):
    def run(self) -> None:
        rmtree(Path(self.build_lib) / "drowse" / "web" / "dist", ignore_errors=True)
        super().run()


setup(cmdclass={"build_py": CleanDashboardBuild})
