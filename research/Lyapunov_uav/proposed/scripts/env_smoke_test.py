from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _add_import_paths() -> None:
    script_path = Path(__file__).resolve()
    proposed_dir = script_path.parents[1]
    lyapunov_dir = proposed_dir.parent
    for path in (lyapunov_dir, proposed_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _zero_fast_action(cfg):
    return {
        "rsu_chunks": np.zeros((cfg.num_rsu, cfg.num_user), dtype=np.int32),
        "rsu_layers": np.zeros((cfg.num_rsu, cfg.num_user), dtype=np.int32),
        "uav_chunks": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.int32),
        "uav_layers": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.int32),
        "uav_power": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.float32),
    }


def _zero_slow_action(cfg):
    return {
        "rsu_scheduling": np.zeros((cfg.num_rsu, cfg.num_user), dtype=np.int32),
        "uav_hiring": np.zeros(cfg.num_uav, dtype=np.int32),
        "uav_scheduling": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.int32),
    }


def main() -> None:
    _add_import_paths()

    from proposed.config import EnvConfig
    from proposed.env import Env, FastEnv, SlowEnv

    cfg = EnvConfig()

    env = Env(cfg)
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    fast_env = FastEnv(cfg)
    fast_obs, fast_info = fast_env.reset()
    assert fast_env.observation_space.contains(fast_obs)
    assert fast_env.action_space.contains({})
    fast_action = _zero_fast_action(cfg)
    assert fast_env.action_space.contains(fast_action)
    assert len(fast_env.step(fast_action)) == 5

    try:
        fast_env.apply_slow_action(_zero_slow_action(cfg))
    except RuntimeError:
        pass
    else:
        raise AssertionError("slow action should be blocked outside round boundary")

    slow_env = SlowEnv(cfg)
    slow_obs, slow_info = slow_env.reset()
    slow_action = _zero_slow_action(cfg)
    assert slow_env.observation_space.contains(slow_obs)
    assert slow_env.action_space.contains(slow_action)
    assert len(slow_env.step(slow_action)) == 5

    env = Env(cfg)
    env.reset()
    env.requested_content[0] = env.uav_cached_content[0]
    slow_action = _zero_slow_action(cfg)
    slow_action["uav_hiring"][0] = 1
    slow_action["uav_scheduling"][0, 0] = 1
    env.apply_slow_action(slow_action)
    env.batteries[0].soc = float(cfg.battery.e_min)
    fast_action = _zero_fast_action(cfg)
    fast_action["uav_chunks"][0, 0] = 1
    fast_action["uav_layers"][0, 0] = 1
    fast_action["uav_power"][0, 0] = 1.0
    _, _, terminated, truncated, step_info = env.step(fast_action)
    assert terminated is False
    assert truncated is False
    assert int(step_info["uav_charge_effective"][0]) == 1
    assert float(step_info["delivered_uav_per_user"][0]) == 0.0
    assert step_info["battery_step_info"][0]["mode"] == "charge"

    print("env smoke test passed")


if __name__ == "__main__":
    main()
