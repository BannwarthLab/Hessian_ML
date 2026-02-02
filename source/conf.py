import os 
import sys 

sys.path.insert(0, os.path.abspath('../src/mlhess'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Machine Learning Hessian MLH'
copyright = '2026, Gereon Feldmann'
author = 'Gereon Feldmann'
release = '02.02.26'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # Google / NumPy style docstrings
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

autosummary_generate = True