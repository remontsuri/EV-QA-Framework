"""
DEPRECATED — package config moved to pyproject.toml.

This file is kept for backward compatibility with older pip versions
(< 21.3). Remove once Python 3.10- support is dropped.

All metadata, dependencies, and scripts are now defined in pyproject.toml.
"""
from setuptools import setup, find_packages

setup(
    name='ev-qa-framework',
    version='1.0.0',
    packages=find_packages(),
)
