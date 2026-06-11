# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add project source to path so autodoc can find the package
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EV-QA-Framework"
copyright = "2026, remontsuri"
author = "remontsuri"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Suppress warnings -------------------------------------------------------
suppress_warnings = [
    "ref.python",  # duplicate cross-references from re-exported names
    "ref",  # broader: catches duplicate object descriptions in Sphinx 9.x
    "toc.not_included",  # pages intentionally not in toctree
    "ref.objtype",  # duplicate object descriptions from autosummary
    "misc.highlighting_failure",  # syntax highlighting issues
]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "ignore-module-all": True,
}
autodoc_typehints = "description"
autodoc_type_aliases = {}
autodoc_mock_imports = [
    "can",
    "serial",
    "tensorflow",
    "prometheus_client",
]
autodoc_inherit_docstrings = True

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = None
html_favicon = None

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
