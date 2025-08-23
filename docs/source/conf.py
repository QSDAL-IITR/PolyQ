# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os, sys
sys.path.insert(0, os.path.abspath('../..'))  # ← point at your project root

project = 'PolyQ'
copyright = '2025, jajapuram shiva sai'
author = 'jajapuram shiva sai'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# default options for all autodoc directives
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
