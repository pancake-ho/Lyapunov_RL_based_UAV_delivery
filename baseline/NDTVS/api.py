"""Ladipo-adapted RSU PPO and a proposed-without-UAV ablation.

Use the sibling proposed/ physical implementation without editing it.
No GRU, instantaneous CSI, cloud, cache, or bandwidth/power optimization.
This is an adaptation, not a reproduction of the original NDT system."""
# Compatibility API and unchanged train/eval CLI. Implementations live in
# the responsible modules; there is no duplicate training/reward code here.
import numpy as np
import torch
from baseline.NDTVS.common.paths import HERE, PROPOSED
from baseline.NDTVS.common.config import HPPOConfig, VERSION, BASE_COMMIT, read_config
from baseline.NDTVS.common.io import atomic, write_rows, load_bundle, jsonable
from baseline.NDTVS.common.runtime import rng_state, restore_rng, Budget, require_device
from baseline.NDTVS.environment.rsu import RSUEnv, NoUAVCompletion, P3HierarchicalEnv
from baseline.NDTVS.models.policy import mlp, NDTVSNet, ndt_agent, make_agents, PPOAgent
from baseline.NDTVS.metrics.observer import QoELogger, HistoryLogger
from baseline.NDTVS.training.rollout import ndt_observation, ndt_episode, episode, hrl
from baseline.NDTVS.common.checkpoint import (source_hashes, verify_shared_sources,
    verify_checkpoint_source, policy_state, restore_agents,
    LEGACY_ADAPTER_SHA256, REVIEWED_CONFIG_DEFAULT_HASHES)
from baseline.NDTVS.rewards.qoe import (QoEHistory, PSNR_DB, QOE_WEIGHTS, REWARD_SCALE,
    REWARD_VERSION, reward_spec)
from baseline.NDTVS.training.train import run_train
from baseline.NDTVS.evaluation.single import run_eval
from baseline.NDTVS.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
