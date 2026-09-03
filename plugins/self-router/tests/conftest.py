"""Conftest for the self-router plugin tests.

Registers the hyphenated plugin dir (`self-router`) as the importable
package ``self_router`` so tests can ``from self_router import ...``.
Python cannot import a hyphenated directory name directly; this shim maps the
on-disk dir to a synthetic package via spec_from_file_location.
"""

import importlib.util
import os
import sys
import types

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../self-router/tests
_PLUGIN_DIR = os.path.dirname(_TESTS_DIR)  # .../self-router (plugin root)
_PARENT = os.path.dirname(_PLUGIN_DIR)  # .../plugins

# Build the synthetic package `self_router` -> `self-router/` dir.
pkg_name = "self_router"
pkg = types.ModuleType(pkg_name)
pkg.__path__ = [_PLUGIN_DIR]
pkg.__file__ = os.path.join(_PLUGIN_DIR, "__init__.py")
sys.modules[pkg_name] = pkg

# Load each submodule from its file path and attach to the package.
_SUBMODULES = ("__init__", "config", "self_assess", "anchoring", "router", "cascade")


def _load_submodule(name: str):
    fname = "__init__.py" if name == "__init__" else f"{name}.py"
    path = os.path.join(_PLUGIN_DIR, fname)
    if not os.path.exists(path):
        return
    mod_name = pkg_name if name == "__init__" else f"{pkg_name}.{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)


for _sm in _SUBMODULES:
    _load_submodule(_sm)

# Attach config/self_assess/anchoring/router/cascade as attributes of the package so
# `from self_router import config` works post-hoc.
for _sm in ("config", "self_assess", "anchoring", "router", "cascade"):
    _mod = sys.modules.get(f"{pkg_name}.{_sm}")
    if _mod is not None:
        setattr(pkg, _sm, _mod)