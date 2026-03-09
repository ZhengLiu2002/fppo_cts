"""CRL extensions for Isaac Lab.

This package contains constrained reinforcement learning building blocks,
environment extensions, and terrain utilities used by the Galileo tasks.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crl-isaaclab")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
