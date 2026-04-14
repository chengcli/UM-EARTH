"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DOCS_SOURCE = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_SOURCE.parents[1]
sys.path.insert(0, os.path.abspath(str(PROJECT_ROOT)))


project = "UM-EARTH"
copyright = "2026, Cheng Li"
author = "Cheng Li"
release = "0.1.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []
autosummary_generate = True
add_module_names = False
add_function_parentheses = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "UM-EARTH Documentation"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

epub_show_urls = "footnote"
