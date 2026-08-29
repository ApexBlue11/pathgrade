"""Guards that the *distributed* package is complete.

Every other test here imports the few modules it exercises, which means a
module missing from a checkout is only caught indirectly - as a collection
error somewhere else, if anything imports it at all. That is exactly how
src/pathgrade/data/ stayed uncommitted for the life of the repo: `.gitignore`
carried an unanchored `data/` rule, git applied it at every depth, and the
working tree everyone developed against still had the files. Only `git clone`
saw the gap.

So this module asserts two different things:

* **Presence.** REQUIRED lists the subpackages by name. A discovery walk
  cannot do this job - ``pkgutil.walk_packages`` enumerates what is on disk,
  so a package absent from the checkout is silently not walked rather than
  reported missing. Naming them is the only way absence becomes a failure.
* **Import health.** The walk then imports everything it does find, which
  catches a module that ships but cannot be imported.

Both are cheap and need no optional dependency: openslide, timm, torchvision
and torch_xla are all imported inside functions, deliberately, so the package
imports on a bare install.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

# Subpackages that must exist in any checkout. Not discovered - enumerated,
# for the reason in the module docstring.
REQUIRED = [
    "pathgrade.data",
    "pathgrade.data.dataset",
    "pathgrade.data.io",
    "pathgrade.data.splits",
    "pathgrade.models",
    "pathgrade.models.asmil_ord",
    "pathgrade.models.attention",
    "pathgrade.preprocessing",
    "pathgrade.preprocessing.single_slide",
    "pathgrade.preprocessing.stream_extract",
    "pathgrade.preprocessing.tiling",
    "pathgrade.config",
    "pathgrade.evaluate",
    "pathgrade.inference",
    "pathgrade.losses",
    "pathgrade.metrics",
    "pathgrade.train",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_required_module_is_present_and_imports(name):
    """A module named here must ship in the checkout and import cleanly."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"{name} is missing from this checkout ({exc}).\n"
            "If it exists in your working tree but not in `git ls-files`, an "
            "ignore rule is eating it - check `git check-ignore -v <path>`."
        )


def test_every_shipped_module_imports():
    """Whatever else is present must import too - no dead or broken modules."""
    import pathgrade

    failures = []
    for info in pkgutil.walk_packages(pathgrade.__path__, prefix="pathgrade."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:                      # noqa: BLE001 - report all
            failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules present but not importable:\n  " + "\n  ".join(failures)


def test_headline_readme_api_is_importable():
    """The first code block in README.md must actually work.

    It is the thing a new reader runs first, and it was broken on every clone
    of this repo until the missing package was committed.
    """
    from pathgrade.inference import GradePredictor, grade_slide  # noqa: F401
