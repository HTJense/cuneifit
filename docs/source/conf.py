# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Cuneifit'
copyright = '2026, Hidde T. Jense'
author = 'Hidde T. Jense'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_favicon",
]

templates_path = ['_templates']
exclude_patterns = []

root_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
}
html_use_index = False
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

# Favicons
favicons = {
    "href": "logo.svg",
    "type": "image/svg+xml"
}
