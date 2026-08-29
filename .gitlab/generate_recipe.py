#!/usr/bin/env python3
"""Print the Apptainer recipe matching a container name to stdout.

Called by the GitLab Runner custom executor as
``python3 .gitlab/generate_recipe.py ${CONTAINER_NAME}`` to (re)build a
container image that is missing from the runner's image cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <container_name>")

    # CONTAINER_NAME may be a bare name ("mlhess_image") or a path to the
    # built image ("containers/mlhess_image.sif") depending on the job.
    name = Path(sys.argv[1]).stem
    recipe_path = RECIPES_DIR / f"{name}.def"

    if not recipe_path.is_file():
        sys.exit(f"no recipe found for '{sys.argv[1]}' at {recipe_path}")

    sys.stdout.write(recipe_path.read_text())


if __name__ == "__main__":
    main()
