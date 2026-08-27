from __future__ import annotations

"""
Compatibility entrypoint for frozen-Fast Slow-timescale evaluation.

The canonical Slow controller now lives in
``agent.PPO.fast.fast_train`` because that path already owns the optimized
shared-memory Fast-policy rollout evaluator used by both ``eval_joint`` and
``joint_dpp``.

Do not maintain a second Slow-DPP implementation here.
"""

import os
import sys
from pathlib import Path


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))


def main() -> None:
    requested = os.environ.get(
        "FAST_PPO_PHASE",
        "eval_joint",
    ).strip().lower()

    if requested != "eval_joint":
        raise RuntimeError(
            "agent.PPO.slow.slow_train is a frozen-Fast evaluation "
            "compatibility entrypoint and requires "
            "FAST_PPO_PHASE=eval_joint. "
            f"got={requested!r}."
        )

    if not os.environ.get(
        "FAST_PPO_CHECKPOINT",
        "",
    ).strip():
        raise RuntimeError(
            "FAST_PPO_CHECKPOINT is required. "
            "Select the Fast checkpoint first."
        )

    os.environ[
        "FAST_PPO_PHASE"
    ] = "eval_joint"

    from agent.PPO.fast.fast_train import (
        main as canonical_main,
    )

    canonical_main()


if __name__ == "__main__":
    main()
