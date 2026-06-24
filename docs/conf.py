import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))

project = 'zestimatr'
copyright = '2025, CodeAstro Group 2'
author = 'CodeAstro Group 2'

root_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
