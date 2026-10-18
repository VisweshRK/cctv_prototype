"""Finds plugins: every folder in ai/plugins/ with an __init__.py that defines
a `Plugin` subclass named `Plugin`."""

from __future__ import annotations

import importlib
import pkgutil

import plugins as plugins_pkg

from .plugin import Plugin


def discover() -> dict[str, type[Plugin]]:
    found: dict[str, type[Plugin]] = {}
    for info in pkgutil.iter_modules(plugins_pkg.__path__):
        if not info.ispkg:
            continue
        module = importlib.import_module(f"plugins.{info.name}")
        cls = getattr(module, "Plugin", None)
        if isinstance(cls, type) and issubclass(cls, Plugin) and cls is not Plugin:
            found[info.name] = cls
    return found
