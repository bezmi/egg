# Sphinx configuration for the egg documentation.
# Build: uv run --group docs sphinx-build -b html docs docs/_build/html

project = "egg"
copyright = "2026, egg developers"
author = "egg developers"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "pallets_sphinx_themes",
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
exclude_patterns = ["_build"]

# Flask's documentation theme (Pallets), plus a hand-rolled light/dark
# toggle: theme-toggle.js sets data-theme on <html> (localStorage-persisted,
# defaulting to the OS preference) and dark.css re-skins the palette under
# html[data-theme="dark"].
html_theme = "flask"
html_title = "egg"
html_static_path = ["_static"]
html_css_files = ["dark.css"]
html_js_files = ["theme-toggle.js"]
