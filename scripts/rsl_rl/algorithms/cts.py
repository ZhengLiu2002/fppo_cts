# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .ppo import PPO


class CTS(PPO):
    """Legacy compatibility alias for PPO running inside the CTS framework.

    The Galileo benchmark now treats CTS as the shared teacher-student training
    framework and selects the optimizer with `--algo`. This alias is kept so
    older configs and tests that request `CTS` continue to work.
    """

    pass
