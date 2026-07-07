# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

# Sphinx configuration for the egg documentation.
# Build: uv run --group docs sphinx-build -b html docs docs/_build/html

import subprocess
from pathlib import Path

from pallets_sphinx_themes import ProjectLink

project = "egg"
copyright = "2026, Shahzeb Imran and Egg contributors"
author = "Shahzeb Imran and Egg contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "breathe",
    "pallets_sphinx_themes",
]

# C++ core docs: Doxygen (XML only, see docs/Doxyfile) -> Breathe. Run
# doxygen up front so a plain sphinx-build always sees fresh XML.
_docs_dir = Path(__file__).resolve().parent
subprocess.run(["doxygen", "Doxyfile"], cwd=_docs_dir, check=True)
breathe_projects = {"egg": str(_docs_dir / "_doxygen" / "xml")}
breathe_default_project = "egg"
breathe_default_members = ("members",)
breathe_show_include = False

# Each doxygenfile directive re-emits the enclosing `egg` namespace, so the
# same namespace target appears once per section — harmless, silence it.
# source_code_parser.cpp: Sphinx's C++ parser can't fully express nested
# classes of partial specializations (EntitySoA<Free<D>>::Host/View); the
# rendering is still correct enough.
suppress_warnings = [
    "duplicate_declaration.cpp",
    "source_code_parser.cpp",
    "docutils",
]

# Numpy-style docstrings only (napoleon's Google-style parsing off).
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Render type hints in the description rather than cluttering signatures.
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "_doxygen"]

# Flask's documentation theme (Pallets), plus a hand-rolled light/dark
# toggle: theme-toggle.js sets data-theme on <html> (localStorage-persisted,
# defaulting to the OS preference) and dark.css re-skins the palette under
# html[data-theme="dark"].
html_theme = "flask"
html_title = "egg"
# Persistent "Source Code" link on every page via the theme's project.html
# sidebar block (the same mechanism Flask's own docs use).
html_context = {
    "project_links": [
        ProjectLink("Source Code", "https://github.com/bezmi/egg"),
    ]
}
html_sidebars = {
    "index": ["project.html", "localtoc.html", "searchbox.html"],
    "**": [
        "home.html",  # link back to the index at the top of every page
        "localtoc.html",
        "relations.html",
        "project.html",
        "searchbox.html",
    ],
}
html_static_path = ["_static"]
html_css_files = ["theme.css"]
html_js_files = ["theme-toggle.js"]
