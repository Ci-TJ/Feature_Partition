from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="new_binary_search_perplexity",
    sources=["new_binary_search_perplexity.pyx"],
    include_dirs=[np.get_include()],
)

setup(
    ext_modules=cythonize(
        [ext],
        language_level="3",
        compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True},
    )
)
