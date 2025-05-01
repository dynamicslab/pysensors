import importlib
import os
import pathlib
import sys

author = "Brian de Silva, Krithika Manohar, Emily Clark"
project = "pysensors"  # package name


# no need to edit below this line

copyright = f"2020, {author}"

module = importlib.import_module(project)
version = release = getattr(module, "__version__")
sys.path.insert(0, os.path.abspath("."))

# The master toctree document.
master_doc = "index"

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copy_examples",
]

apidoc_module_dir = f"../{project}"
apidoc_excluded_paths = ["tests", "version.py"]
apidoc_toc_file = False

autodoc_default_options = {"members": True}
autodoc_member_order = "bysource"
autoclass_content = "init"

language = "en"

here = pathlib.Path(__file__).parent

if (here / "static/custom.css").exists():

    html_static_path = ["static"]

    def setup(app):
        app.add_css_file("custom.css")


exclude_patterns = ["build", "_build", "Thumbs.db", ".DS_Store"]
# pygments_style = "sphinx"

add_module_names = True
add_function_parentheses = False

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

default_role = "any"
html_sourcelink_suffix = ""

example_gallery_config = dict(
    dont_preprocess=True,
    examples_dirs=["../examples"],
    gallery_dirs=["examples"],
    pattern=".+.ipynb",
)

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pysensors", "PySensors Documentation", [author], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Extensions to the  Napoleon GoogleDocstring class ---------------------
# michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
from sphinx.ext.napoleon.docstring import GoogleDocstring  # noqa: E402


def parse_keys_section(self, section):
    return self._format_fields("Keys", self._consume_fields())


GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields("Attributes", self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields("Class Attributes", self._consume_fields())


GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section


def patched_parse(self):
    """
    we now patch the parse method to guarantee that the the above methods are
    assigned to the _section dict
    """
    self._sections["keys"] = self._parse_keys_section
    self._sections["class attributes"] = self._parse_class_attributes_section
    self._unpatched_parse()


GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse
