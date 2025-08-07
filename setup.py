import pathlib
import sys

from setuptools import find_packages, setup

assert sys.version_info >= (3, 9, 0), "pysensors requires Python 3.9+"

NAME = "python-sensors"
DESCRIPTION = "Sparse sensor placement"
URL = "https://github.com/dynamicslab/pysensors"
EMAIL = "bdesilva@uw.edu, kmanohar@uw.edu, emily.e.clark93@gmail.com"
AUTHOR = "Brian de Silva, Krithika Manohar, Emily Clark"
PYTHON = ">=3.6"
LICENSE = "MIT"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Mathematics",
]

here = pathlib.Path(__file__).parent

with open(here / "requirements.txt", "r") as f:
    REQUIRED = f.readlines()

with open(here / "README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "examples"]),
    install_requires=REQUIRED,
    python_requires=PYTHON,
    license=LICENSE,
    classifiers=CLASSIFIERS,
)
