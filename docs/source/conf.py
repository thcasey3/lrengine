# Configuration file for the Sphinx documentation builder.
# -- Project information -----------------------------------------------------

import sys
import os
import re

sys.path.append(os.path.abspath('.'))

# Import Read-the-Docs (RTD) theme
from sphinx.locale import _
from sphinx_rtd_theme import __version__

# Project details
project = 'lrengine'
author = 'Thomas Casey'
language = 'en'


version = open(os.path.join('..','..','version')).read().splitlines()[0]
rst_epilog = f'.. |version| replace:: {version}'

# Add sphinx extensions
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinxcontrib.httpdomain',
    'sphinxcontrib.ghcontributors',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.imgmath',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosummary',
    'm2r2'
]

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

# Warnings suppression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
exclude_patterns = ['.', './functions']
numpydoc_show_class_members = False
# Render Latex math equations as svg instead of rendering with JavaScript
imgmath_image_format = 'svg'
imgmath_dvisvgm = 'dvisvgm'
imgmath_latex_preamble = r'''
\newcommand{\mr}[1]{\mathrm{#1}}
\newcommand{\mx}[1]{\boldsymbol{#1}}
\newcommand{\vc}[1]{\boldsymbol{#1}}
\DeclareMathOperator*{\argmin}{\arg\!\min}
'''

# Setup template stuff
templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = []
master_doc = 'index'
suppress_warnings = ['image.nonlocal_uri']
pygments_style = 'default'
intersphinx_mapping = {
    'rtd': ('https://docs.readthedocs.io/en/latest/', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
}
html_theme = 'sphinx_rtd_theme'

# Integrate version control system
# -------------------------------------------------------------
html_context = {
    "display_github": False, # Integrate GitHub
    "github_user": "thcasey3", # Username
    "github_repo": "lrengine", # Repo name
    "github_version": "master", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}


# Read-the-Docs options configuration
# -------------------------------------------------------------
html_theme_options = {
    'sticky_navigation': False,
    'titles_only': False,
    'collapse_navigation': False,
    'logo_only': True,
    'navigation_depth':1,
    #'style_nav_header_background': '#2a7bf8'
}
html_copy_source = False
html_theme_path = ["../.."]
html_logo = "_static/web_logo.png"
html_show_sourcelink = True
html_favicon = '_static/web_logo.png'
html_static_path = ['_static']

# Extensions to theme docs
def setup(app):
    from sphinx.domains.python import PyField
    from sphinx.util.docfields import Field
    app.add_css_file('/source/_static/lrtheme.css')
    app.add_stylesheet('/source/_static/lrtheme.css')
    app.add_object_type(
        'confval',
        'confval',
        objname='configuration value',
        indextemplate='pair: %s; configuration value',
        doc_field_types=[
            PyField(
                'type',
                label=_('Type'),
                has_arg=False,
                names=('type',),
                bodyrolename='class'
            ),
            Field(
                'default',
                label=_('Default'),
                has_arg=False,
                names=('default',),
            ),
        ]
    )
    
# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# Add path to custom CSS file to overwrite some of the default CSS settings
html_css_files = [
    'lrtheme.css'
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Default role
default_role = 'math'  # with this, :math:`\psi` can be written simply as `\psi`


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_path = ["_themes", ]
html_static_path = ['_static']
html_title = 'lrengine'
highlight_language = 'python'
primary_domain = 'py'
html_logo = '_static/web_logo.png'

