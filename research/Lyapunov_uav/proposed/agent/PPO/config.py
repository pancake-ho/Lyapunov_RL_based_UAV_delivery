from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_text(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    value = _env_text(name)
    return int(default if value is None else value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return bool(default)
    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, "
        f"got {value!r}."
    )


def _env_float(name: str, default: float) -> float:
    value = _env_text(name)
    return float(default if value is None else value)


def _env_optional_float(
    name: str,
    default: Optional[float],
) -> Optional[float]:
    value = _env_text(name)
    if value is None:
        return default

    normalized = value.lower()
    if normalized in {"none", "null", "off"}:
        return None

    return float(value)


def _env_int_list(name: str, default: List[int]) -> List[int]:
    value = _env_text(name)
    if value is None:
        return list(default)
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError(f"{name} must contain at least one integer.")
    return parsed


def _env_mobility_curriculum(
    name: str,
    default: Tuple[Tuple[int, float], ...],
) -> Tuple[Tuple[int, float], ...]:
    """Parse ``episode:move_prob`` pairs separated by commas."""
    value = _env_text(name)
    if value is None:
        return tuple(default)

    parsed: list[tuple[int, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            episode_text, probability_text = item.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"{name} entries must use episode:move_prob, got {item!r}."
            ) from exc
        parsed.append((int(episode_text), float(probability_text)))

    if not parsed:
        raise ValueError(f"{name} must contain at least one entry.")
    return tuple(parsed)


@dataclass(frozen=True)
class FastTrainConfig:
    """
    Fast-timescale PPO 실행/학습 설정.

    원칙:
    - argparse를 사용하지 않는다.
    - 이 파일이 fast_train.py의 단일 실행 설정 source다.
    - proposed/config.py는 env/system/channel/battery 상수만 관리한다.
    - agent/PPO/config.py는 RL hyperparameter와 train/eval 실행 설정만 관리한다.
    """

    # ------------------------------------------------------------------
    # 1) 실행 모드 / seed / device
    # ------------------------------------------------------------------
    phase: str = "pretrain"
    mode: str = "train"  # train | eval
    seed: int = 2026
    deterministic_torch: bool = True
    device: str = "cuda"  # cuda | cuda:0 | cpu | auto
    fail_if_cuda_unavailable: bool = True

    # ------------------------------------------------------------------
    # 2) run directory / checkpoint
    # ------------------------------------------------------------------
    # 상대경로면 proposed/ 아래에 생성된다.
    output_root: str = "fast"
    run_name: Optional[str] = None

    # None이면 fast_train.py에서 자동 이름 생성
    # Joint DPP는 현재 기준 학습이 완료된 fast policy로 candidate expected cost 평가.
    # cluster 실행 시 JOINT_FAST_CHECKPOINT에 trusted checkpoint 경로를 지정.
    checkpoint: Optional[str] = None

    # resume=True이면 model/optimizer/normalizer를 모두 복원한다.
    # dpp warm start에서는 보통 False로 두고 model/normalizer만 가져온다.
    resume: bool = False
    legacy_transfer: bool = False
    load_optimizer_on_warm_start: bool = False
    require_pretrained_fast_for_dpp: bool = False
    segment_id: int = 1
    save_latest_every_episode: bool = True
    allow_existing_run_dir: bool = False

    # ------------------------------------------------------------------
    # 3) episode / rollout
    # ------------------------------------------------------------------
    # H2 default:
    # 기존 H1 200 episodes에서 PPO trust-region screening과
    # categorical specialization 부족이 확인되었으므로,
    # 작은 actor update를 더 오랫동안 수행하도록 400 episodes로 확장한다.
    #
    # segmented execution을 사용할 경우 num_episodes만 segment 길이에
    # 맞게 override하고 target_total_episodes=400은 유지할 수 있다.
    num_episodes: int = 400
    target_total_episodes: int = 400
    eval_episodes: int = 5

    rounds_per_episode: int = 10
    eval_rounds_per_episode: int = 5

    # Joint mode에서는 반드시 EnvConfig.slow_T와 동일해야 한다.
    rollout_slots: int = 3600

    save_every_episodes: int = 5
    plot_every_episodes: int = 5
    plot_smooth_window: int = 5

    # ------------------------------------------------------------------
    # 4) PPO hyperparameters
    # ------------------------------------------------------------------
    batch_size: int = 450
    update_epochs: int = 4

    # Queue/Battery 장기 pressure를 보려면 gamma=0.90은 너무 짧다.
    # 환경 reward는 raw DPP 단위로 유지하고, rollout buffer 저장 직전에
    # ppo_reward_scale만 적용한다.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # H1:
    # actor_lr=1.5e-5에서 post-update KL/clip fraction이 과도했으며,
    # repository H1 diagnostic 역시 H2 actor_lr=7.5e-6을 권고한다.
    #
    # Critic은 기존 3e-5를 유지하여 H1->H2 변경 원인을
    # actor update magnitude에 한정한다.
    actor_lr: float = 7.5e-6
    critic_lr: float = 3e-5

    clip_coef: float = 0.15
    target_kl: Optional[float] = None

    # ppo_reward_scale 적용 후 critic loss가 actor를 압도하지 않도록
    # value_coef는 0.5로 유지한다
    value_coef: float = 0.5
    categorical_entropy_coef: float = 1e-4
    power_entropy_coef: float = 1e-4

    max_grad_norm: float = 0.5
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # 기존 -1.5는 std≈0.22로 탐색이 좁다.
    # raw reward 유지 + large action space에서는 -1.0부터 시작 권장.
    init_log_std: float = -1.0

    # ------------------------------------------------------------------
    # 5) normalization
    # ------------------------------------------------------------------
    obs_norm: bool = True
    adv_norm: bool = True

    reward_norm: bool = False
    reward_scale: Optional[float] = None
    reward_clip: Optional[float] = None

    # reward scaling 변수
    ppo_reward_scale: float = 1e-4

    # Round 동안 observation-normalizer 통계도 고정하고,
    # round 종료 뒤 실제 observation batch로 한 번 갱신한다.
    freeze_obs_norm_within_round: bool = True

    # ------------------------------------------------------------------
    # 6) Critic 안정화 옵션
    # ------------------------------------------------------------------
    # fast_train.py / fast_agent.py에서 지원하도록 반영하면 좋다.
    # 지원하지 않는 경우에도 config 저장용으로 문제 없음.
    use_value_huber_loss: bool = True
    use_value_clip: bool = False

    # PPO에 저장되는 scaled reward/value 단위의 value clipping 폭.
    value_clip_coef: float = 0.5

    # ------------------------------------------------------------------
    # 7) Fast-only 학습용 slow decision 생성 방식
    # ------------------------------------------------------------------
    slow_decision_mode: str = "random"  # random | dpp

    # for random
    random_rsu_user_prob: float = 0.50
    random_uav_hire_prob: float = 0.70
    random_uav_user_prob: float = 0.80

    # DPP mode: current Fast policy를 고정하여 complete round forecast
    dpp_forecast_horizon: int = 3600
    dpp_forecast_scenarios: int = 1

    dpp_candidate_batch_size: int = 256
    dpp_forecast_workers: int = 18

    # 전체 region Cartesian product는 사용하지 않는다.
    # 완전한 global action을 평가하는 region-coordinate minimization으로 고정.
    dpp_coordinate_sweeps: int = 3
    dpp_max_region_candidates: int = 8192
    dpp_improvement_tolerance: float = 1.0

    # 논문처럼 candidate cost 평가 시 greedy/mean Fast action 사용.
    dpp_deterministic_fast_forecast: bool = True

    # Candidate 간 RNG 호출 순서 차이를 제거하기 위해 forecast RSU channel은
    # Rayleigh/shadowing의 mean power gain을 사용한다. 실제 round는 stochastic.
    dpp_use_mean_rsu_channel: bool = True

    # 한 scenario라도 battery outage가 발생하면 해당 slow candidate를 infeasible 처리.
    dpp_reject_forecast_outage: bool = True

    # Scenario 7.1: schedule user가 없는 UAV hiring candidate는 생성하지 않음.
    dpp_forbid_empty_hiring: bool = True

    # Ablation:
    # True  -> RSU-first + residual UAV DPP-MWM
    # False -> RSU-only DPP-MWM
    #
    # Evaluation ablation용이며 기본 proposed algorithm은 True.
    dpp_enable_uav: bool = True

    # ------------------------------------------------------------------
    # 8) Mobility curriculum
    # ------------------------------------------------------------------
    mobility_curriculum: Tuple[Tuple[int, float], ...] = ((1, 1e-4),)

    # ------------------------------------------------------------------
    # 9) Debug
    # ------------------------------------------------------------------
    fail_on_nan: bool = True
    fail_on_outage: bool = True
    battery_soc_tolerance: float = 0.05
    # True이면 매 round마다 전체 GPU parameter를 CPU로 복사해 SHA를 계산한다.
    # production 학습에서는 GPU synchronization을 유발하므로 False로 둔다.
    audit_runtime_invariants: bool = False

    def __post_init__(self) -> None:
        if self.phase not in {
            "pretrain",
            "joint_dpp",
            "eval_pretrain",
            "eval_joint",
        }:
            raise ValueError(
                f"Unsupported phase: {self.phase!r}."
            )

        if self.mode not in {"train", "eval"}:
            raise ValueError(
                "mode must be 'train' or 'eval'."
            )

        if self.slow_decision_mode not in {
            "random",
            "dpp",
        }:
            raise ValueError(
                "slow_decision_mode must be "
                "'random' or 'dpp'."
            )

        for name in (
            "num_episodes",
            "target_total_episodes",
            "rounds_per_episode",
            "eval_episodes",
            "eval_rounds_per_episode",
            "rollout_slots",
            "batch_size",
            "update_epochs",
            "dpp_forecast_horizon",
            "dpp_forecast_scenarios",
            "dpp_candidate_batch_size",
            "dpp_coordinate_sweeps",
            "dpp_max_region_candidates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if int(self.dpp_forecast_workers) < 0:
            raise ValueError(
                "dpp_forecast_workers must be >= 0."
            )

        if int(self.num_episodes) > int(
            self.target_total_episodes
        ):
            raise ValueError(
                "num_episodes cannot exceed "
                "target_total_episodes."
            )

        if self.rollout_slots < self.batch_size:
            raise ValueError(
                "rollout_slots must be >= batch_size."
            )

        if self.rollout_slots % self.batch_size != 0:
            raise ValueError(
                "rollout_slots must be divisible by batch_size."
            )

        if (
            not self.hidden_dims
            or any(
                int(value) <= 0
                for value in self.hidden_dims
            )
        ):
            raise ValueError(
                "hidden_dims must contain positive integers."
            )

        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                "gamma must be in (0,1]."
            )

        if not (0.0 < self.gae_lambda <= 1.0):
            raise ValueError(
                "gae_lambda must be in (0,1]."
            )

        if self.actor_lr <= 0.0:
            raise ValueError(
                "actor_lr must be positive."
            )

        if self.critic_lr <= 0.0:
            raise ValueError(
                "critic_lr must be positive."
            )

        if self.clip_coef <= 0.0:
            raise ValueError(
                "clip_coef must be positive."
            )

        if self.max_grad_norm <= 0.0:
            raise ValueError(
                "max_grad_norm must be positive."
            )

        if self.value_clip_coef <= 0.0:
            raise ValueError(
                "value_clip_coef must be positive."
            )

        if self.ppo_reward_scale <= 0.0:
            raise ValueError(
                "ppo_reward_scale must be positive."
            )

        if self.battery_soc_tolerance < 0.0:
            raise ValueError(
                "battery_soc_tolerance must be non-negative."
            )

        if self.dpp_improvement_tolerance < 0.0:
            raise ValueError(
                "dpp_improvement_tolerance "
                "must be non-negative."
            )

        for name in (
            "value_coef",
            "categorical_entropy_coef",
            "power_entropy_coef",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if (
            self.target_kl is not None
            and self.target_kl <= 0.0
        ):
            raise ValueError(
                "target_kl must be None or positive."
            )

        if self.reward_norm:
            raise ValueError(
                "reward_norm must remain False."
            )

        if self.reward_scale is not None:
            raise ValueError(
                "reward_scale must remain None."
            )

        if self.reward_clip is not None:
            raise ValueError(
                "reward_clip must remain None."
            )

        for name in (
            "random_rsu_user_prob",
            "random_uav_hire_prob",
            "random_uav_user_prob",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"{name} must be in [0,1]."
                )

        if self.resume and self.legacy_transfer:
            raise ValueError(
                "resume and legacy_transfer "
                "cannot both be True."
            )

        if self.mode == "eval" and self.resume:
            raise ValueError(
                "Evaluation must not use resume=True."
            )

        if int(self.segment_id) <= 0:
            raise ValueError(
                "segment_id must be positive."
            )

        expected_slow_mode = (
            "dpp"
            if self.phase
            in {"joint_dpp", "eval_joint"}
            else "random"
        )

        if (
            self.slow_decision_mode
            != expected_slow_mode
        ):
            raise ValueError(
                f"phase={self.phase!r} requires "
                f"slow_decision_mode="
                f"{expected_slow_mode!r}."
            )

        checkpoint_required = (
            self.resume
            or self.legacy_transfer
            or self.mode == "eval"
            or (
                self.slow_decision_mode == "dpp"
                and self.require_pretrained_fast_for_dpp
            )
        )

        if (
            checkpoint_required
            and self.checkpoint is None
        ):
            raise ValueError(
                "A checkpoint path is required "
                "for the selected mode."
            )

        if self.slow_decision_mode == "dpp":
            if (
                self.rollout_slots
                != self.dpp_forecast_horizon
            ):
                raise ValueError(
                    "Joint DPP mode requires "
                    "rollout_slots == "
                    "dpp_forecast_horizon."
                )

            if not self.dpp_deterministic_fast_forecast:
                raise ValueError(
                    "The finalized DPP forecast uses "
                    "deterministic Fast policy."
                )

            if not self.dpp_use_mean_rsu_channel:
                raise ValueError(
                    "The finalized implementation "
                    "requires mean RSU channel "
                    "during forecast to preserve "
                    "common exogenous paths."
                )

        prev_episode = 0

        for (
            start_episode,
            move_prob,
        ) in self.mobility_curriculum:
            if int(start_episode) <= prev_episode:
                raise ValueError(
                    "mobility_curriculum episodes "
                    "must be strictly increasing."
                )

            if not (
                0.0
                <= float(move_prob)
                <= 1.0
            ):
                raise ValueError(
                    "move_prob must be in [0,1]."
                )

            prev_episode = int(start_episode)

        if (
            len(self.mobility_curriculum) == 0
            or int(
                self.mobility_curriculum[0][0]
            )
            != 1
        ):
            raise ValueError(
                "mobility_curriculum must start "
                "from episode 1."
            )

        if int(
            self.dpp_candidate_batch_size
        ) < int(self.dpp_forecast_workers):
            raise ValueError(
                "dpp_candidate_batch_size must "
                "be at least dpp_forecast_workers."
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def make_run_dir(
        self,
        proposed_root: Path,
    ) -> Path:
        output_root_path = Path(
            self.output_root
        )

        if not output_root_path.is_absolute():
            output_root_path = (
                proposed_root
                / output_root_path
            )

        run_name = (
            str(self.run_name)
            if self.run_name is not None
            else "fast_pretrain_h2"
        )

        run_dir = (
            output_root_path
            / run_name
        )
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return run_dir


def get_fast_ppo_config() -> FastTrainConfig:
    phase = (
        _env_text(
            "FAST_PPO_PHASE",
            "pretrain",
        )
        or "pretrain"
    ).lower()

    if phase not in {
        "pretrain",
        "joint_dpp",
        "eval_pretrain",
        "eval_joint",
    }:
        raise ValueError(
            "FAST_PPO_PHASE must be "
            "pretrain, joint_dpp, "
            "eval_pretrain, or eval_joint; "
            f"got {phase!r}."
        )

    seed = _env_int(
        "FAST_PPO_SEED",
        2026,
    )
    segment_id = _env_int(
        "FAST_PPO_SEGMENT_ID",
        1,
    )

    is_eval = phase.startswith(
        "eval_"
    )
    is_dpp = phase in {
        "joint_dpp",
        "eval_joint",
    }

    if phase == "pretrain":
        resume_checkpoint = _env_text(
            "FAST_PRETRAIN_RESUME_CHECKPOINT"
        )
        checkpoint = resume_checkpoint
        resume = (
            resume_checkpoint is not None
        )
        output_root = "fast"

        default_name = (
            f"fast_pretrain_h2_"
            f"seed{seed}_ep400_v1"
        )
        default_save_every = 5

    elif phase == "joint_dpp":
        resume_checkpoint = _env_text(
            "JOINT_RESUME_CHECKPOINT"
        )
        warm_start_checkpoint = _env_text(
            "JOINT_FAST_CHECKPOINT"
        )

        if (
            resume_checkpoint is not None
            and warm_start_checkpoint is not None
        ):
            raise ValueError(
                "Set only one of "
                "JOINT_RESUME_CHECKPOINT and "
                "JOINT_FAST_CHECKPOINT."
            )

        checkpoint = (
            resume_checkpoint
            or warm_start_checkpoint
        )
        resume = (
            resume_checkpoint is not None
        )
        output_root = "joint"

        default_name = (
            f"joint_dpp_fastppo_seed{seed}"
        )
        default_save_every = 1

    else:
        checkpoint = _env_text(
            "FAST_PPO_CHECKPOINT"
        )
        resume = False
        output_root = "eval"

        kind = (
            "joint"
            if is_dpp
            else "pretrain"
        )

        default_name = (
            f"eval_{kind}_seed{seed}"
        )
        default_save_every = 1

    explicit_resume = _env_text(
        "FAST_PPO_RESUME"
    )

    if explicit_resume is not None:
        requested_resume = _env_bool(
            "FAST_PPO_RESUME",
            resume,
        )

        if requested_resume != resume:
            raise ValueError(
                "FAST_PPO_RESUME conflicts "
                "with the selected checkpoint "
                "environment variable."
            )

    run_name = _env_text(
        "FAST_PPO_RUN_NAME",
        default_name,
    )

    output_root = (
        _env_text(
            "FAST_PPO_OUTPUT_ROOT",
            output_root,
        )
        or output_root
    )

    return FastTrainConfig(
        phase=phase,
        mode=(
            "eval"
            if is_eval
            else "train"
        ),
        seed=seed,
        deterministic_torch=_env_bool(
            "FAST_PPO_DETERMINISTIC_TORCH",
            True,
        ),
        device=(
            _env_text(
                "FAST_PPO_DEVICE",
                "cuda",
            )
            or "cuda"
        ),
        fail_if_cuda_unavailable=_env_bool(
            "FAST_PPO_FAIL_IF_CUDA_UNAVAILABLE",
            True,
        ),
        output_root=output_root,
        run_name=run_name,
        checkpoint=checkpoint,
        resume=resume,
        legacy_transfer=False,
        load_optimizer_on_warm_start=False,
        require_pretrained_fast_for_dpp=is_dpp,
        segment_id=segment_id,
        allow_existing_run_dir=_env_bool(
            "ALLOW_EXISTING_RUN_DIR",
            False,
        ),

        # H2 default training horizon
        num_episodes=_env_int(
            "FAST_PPO_NUM_EPISODES",
            400,
        ),
        target_total_episodes=_env_int(
            "FAST_PPO_TARGET_TOTAL_EPISODES",
            400,
        ),

        eval_episodes=_env_int(
            "FAST_PPO_EVAL_EPISODES",
            5,
        ),
        rounds_per_episode=_env_int(
            "FAST_PPO_ROUNDS_PER_EPISODE",
            10,
        ),
        eval_rounds_per_episode=_env_int(
            "FAST_PPO_EVAL_ROUNDS_PER_EPISODE",
            5,
        ),
        rollout_slots=_env_int(
            "FAST_PPO_ROLLOUT_SLOTS",
            3600,
        ),

        batch_size=_env_int(
            "FAST_PPO_BATCH_SIZE",
            450,
        ),
        update_epochs=_env_int(
            "FAST_PPO_UPDATE_EPOCHS",
            4,
        ),

        gamma=_env_float(
            "FAST_PPO_GAMMA",
            0.99,
        ),
        gae_lambda=_env_float(
            "FAST_PPO_GAE_LAMBDA",
            0.95,
        ),

        # H2: H1 actor LR의 1/2
        actor_lr=_env_float(
            "FAST_PPO_ACTOR_LR",
            7.5e-6,
        ),
        critic_lr=_env_float(
            "FAST_PPO_CRITIC_LR",
            3e-5,
        ),

        max_grad_norm=_env_float(
            "FAST_PPO_MAX_GRAD_NORM",
            0.5,
        ),
        clip_coef=_env_float(
            "FAST_PPO_CLIP_COEF",
            0.15,
        ),
        target_kl=_env_optional_float(
            "FAST_PPO_TARGET_KL",
            None,
        ),

        value_coef=_env_float(
            "FAST_PPO_VALUE_COEF",
            0.5,
        ),

        categorical_entropy_coef=_env_float(
            "FAST_PPO_CAT_ENTROPY_COEF",
            1e-4,
        ),
        power_entropy_coef=_env_float(
            "FAST_PPO_POWER_ENTROPY_COEF",
            1e-4,
        ),

        hidden_dims=_env_int_list(
            "FAST_PPO_HIDDEN_DIMS",
            [256, 256],
        ),
        init_log_std=_env_float(
            "FAST_PPO_INIT_LOG_STD",
            -1.0,
        ),

        obs_norm=_env_bool(
            "FAST_PPO_OBS_NORM",
            True,
        ),
        adv_norm=_env_bool(
            "FAST_PPO_ADV_NORM",
            True,
        ),

        ppo_reward_scale=_env_float(
            "FAST_PPO_REWARD_SCALE",
            1e-4,
        ),

        freeze_obs_norm_within_round=_env_bool(
            "FAST_PPO_FREEZE_OBS_NORM_WITHIN_ROUND",
            True,
        ),

        use_value_huber_loss=_env_bool(
            "FAST_PPO_USE_VALUE_HUBER",
            True,
        ),
        use_value_clip=_env_bool(
            "FAST_PPO_USE_VALUE_CLIP",
            False,
        ),
        value_clip_coef=_env_float(
            "FAST_PPO_VALUE_CLIP_COEF",
            0.5,
        ),

        random_rsu_user_prob=_env_float(
            "FAST_PPO_RANDOM_RSU_USER_PROB",
            0.50,
        ),
        random_uav_hire_prob=_env_float(
            "FAST_PPO_RANDOM_UAV_HIRE_PROB",
            0.70,
        ),
        random_uav_user_prob=_env_float(
            "FAST_PPO_RANDOM_UAV_USER_PROB",
            0.80,
        ),

        mobility_curriculum=(
            _env_mobility_curriculum(
                "FAST_PPO_MOBILITY_CURRICULUM",
                ((1, 1e-4),),
            )
        ),

        fail_on_nan=_env_bool(
            "FAST_PPO_FAIL_ON_NAN",
            True,
        ),
        fail_on_outage=_env_bool(
            "FAST_PPO_FAIL_ON_OUTAGE",
            True,
        ),
        battery_soc_tolerance=_env_float(
            "FAST_PPO_BATTERY_SOC_TOLERANCE",
            0.05,
        ),

        audit_runtime_invariants=_env_bool(
            "FAST_PPO_AUDIT_RUNTIME_INVARIANTS",
            False,
        ),

        save_every_episodes=_env_int(
            "FAST_PPO_SAVE_EVERY_EPISODES",
            default_save_every,
        ),
        plot_every_episodes=_env_int(
            "FAST_PPO_PLOT_EVERY_EPISODES",
            default_save_every,
        ),
        save_latest_every_episode=_env_bool(
            "FAST_PPO_SAVE_LATEST_EVERY_EPISODE",
            True,
        ),

        slow_decision_mode=(
            "dpp"
            if is_dpp
            else "random"
        ),

        dpp_forecast_workers=(
            _env_int(
                "FAST_PPO_DPP_FORECAST_WORKERS",
                18,
            )
            if is_dpp
            else 0
        ),
        dpp_enable_uav=_env_bool(
            "FAST_PPO_DPP_ENABLE_UAV",
            True,
        ),
    )