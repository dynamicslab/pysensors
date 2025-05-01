import shutil
from pathlib import Path

from sphinx.application import Sphinx


def setup(app: Sphinx):
    """Copy notebooks and supporting files from examples/ to docs/examples/"""
    source = Path(__file__).parent.parent / "examples"
    target = Path(__file__).parent / "examples"

    if target.exists():
        shutil.rmtree(target)
    target.mkdir()

    for item in source.rglob("*"):
        # Accomodates for any helper scripts and data files
        rel_path = item.relative_to(source)
        dest_path = target / rel_path

        if item.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_path)
            print(f"Copied {rel_path}")
