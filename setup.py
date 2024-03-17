from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension_bpe_vectorizer = Extension(
    "bpe_vectorizer._vectorize_bpe_output_cy",
    ["bpe_vectorizer/_vectorize_bpe_output_cy.pyx"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++"
)

setup(
    ext_modules=cythonize([extension_bpe_vectorizer])
)
