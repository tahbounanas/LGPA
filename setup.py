import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

IS_WINDOWS = sys.platform == "win32"

# MSVC uses /O2, GCC/Clang use -O3.
# On Windows we also statically link the C++ runtime (/MT) so the built .pyd
# does not depend on the Microsoft Visual C++ Redistributable being installed
# on the end user's machine.
if IS_WINDOWS:
    opt_flags = ["/O2", "/MT"]
    link_flags = []
else:
    opt_flags = ["-O3"]
    link_flags = []


class BuildExt(build_ext):
    """Ensure /MT replaces the default dynamic /MD runtime flag on MSVC."""

    def build_extensions(self):
        if IS_WINDOWS:
            for ext in self.extensions:
                # setuptools/distutils injects /MD by default; /MD and /MT
                # cannot both be present, so drop /MD before /MT is applied.
                ext.extra_compile_args = [
                    f for f in (ext.extra_compile_args or []) if f.upper() != "/MD"
                ]
            compiler_args = getattr(self.compiler, "compile_options", None)
            if compiler_args:
                self.compiler.compile_options = [
                    f for f in compiler_args if f.upper() != "/MD"
                ]
        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "LGPA._lgpa_core",
        ["LGPA/lgpa_core.cpp"],
        extra_compile_args=opt_flags,
        extra_link_args=link_flags,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
