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

autodoc_mock_imports = [
    'torch',
    'numpy',
    'matplotlib',
    'tqdm',
    'huggingface_hub',
    'pandas',
    'scipy',
    'astropy',
    'wandb',
]

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
