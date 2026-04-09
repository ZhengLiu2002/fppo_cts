"""Repository-local interpreter tweaks.

This file is imported automatically by Python during startup when the repo root
is on ``sys.path``. We use it to keep external pytest plugins from polluting the
local test environment.
"""

from __future__ import annotations

import os


os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
