import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# MSVC uses /O2, GCC/Clang use -O3
opt_flags = ["/O2"] if sys.platform == "win32" else ["-O3"]

ext_modules = [
    Pybind11Extension(
        "LGPA._lgpa_core",
        ["LGPA/lgpa_core.cpp"],
        extra_compile_args=opt_flags,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
