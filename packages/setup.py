#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import find_packages, setup

# Package meta-data.
NAME = 'tid-gradient-boosting-model'
DESCRIPTION = "Gradient boosting regression model from Train In Data."
URL = "https://github.com/dots13/testing-and-monitoring-ml-deployments"
EMAIL = "stasya.dots@gmail.com"
AUTHOR = "MarchenkoAA"
REQUIRES_PYTHON = ">=3.12.0"


# Function to read dependencies from a requirements file
def list_reqs(fname="requirements.txt"):
    with open(fname, "r") as fd:
        return fd.read().splitlines()


# Read long description from README.md or similar file (better approach)
long_description = DESCRIPTION
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    pass  # Fallback to the short DESCRIPTION if README.md is missing

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'gradient_boosting_model'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

# Setup function to package the module
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),  # Exclude test directories
    package_data={
        "gradient_boosting_model": [
            "VERSION",
            # Add other static files you want to include, e.g., config, models, etc.
        ]
    },
    install_requires=list_reqs(),  # Read from requirements.txt
    extras_require={
        # Define any optional dependencies here
        # Example:
        # "dev": ["pytest", "sphinx"],
    },
    include_package_data=True,
    license="BSD-3",  # Open source license
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",  # Added 3.12 support
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
