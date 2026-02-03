# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../"))


def skip(app, what, name, obj, would_skip, options):
    if name in ["__init__","main"]:
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

project = 'Machine Learning Hessian: MLH'
copyright = '2026, Gereon Feldmann'
author = 'Gereon Feldmann'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ["_templates"]

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
]

autosummary_generate = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store","test_*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"

latex_elements = {
    "preamble": r"""
\usepackage[titles]{tocloft}
\usepackage{amsmath}
""",
}

latex_show_urls = "footnote"
