from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler
import numpy as np


class CustomBuildExt(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "lcs2",
        ["lcs2.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        language="c++",
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    name='lcs2',
    ext_modules=cythonize(ext_modules, language_level="3"),
    cmdclass={'build_ext': CustomBuildExt},
    include_dirs=[np.get_include()]
)
