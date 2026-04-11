"""Gym registrations for Galileo CRL tasks."""

import gymnasium as gym

from .config import agents, cts_env_cfg


gym.register(
    id="Isaac-Galileo-CTS-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cts_env_cfg.__name__}:GalileoCTSCRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_cts_cfg:GalileoCTSBenchmarkRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Galileo-CTS-Eval-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cts_env_cfg.__name__}:GalileoCTSCRLEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_cts_cfg:GalileoCTSBenchmarkRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Galileo-CTS-Play-v0",
    entry_point="crl_isaaclab.envs:CRLManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{cts_env_cfg.__name__}:GalileoCTSCRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_cts_cfg:GalileoCTSBenchmarkRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_crl_ppo_cfg.yaml",
    },
)
