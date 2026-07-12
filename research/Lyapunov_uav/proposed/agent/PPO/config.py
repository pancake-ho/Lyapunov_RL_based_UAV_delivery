from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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
    mode: str = "train"  # train | eval
    seed: int = 2026
    deterministic_torch: bool = False
    device: str = "cuda"  # cuda | cuda:0 | cpu | auto

    # ------------------------------------------------------------------
    # 2) run directory / checkpoint
    # ------------------------------------------------------------------
    # 상대경로면 proposed/ 아래에 생성된다.
    output_root: str = "fast"

    # None이면 fast_train.py에서 자동 이름 생성
    run_name: Optional[str] = (
        "fast_ppo_qdeg_mask_v1"
    )

    checkpoint: Optional[str] = None
    resume: bool = False

    # ------------------------------------------------------------------
    # 3) episode / rollout
    # ------------------------------------------------------------------
    # 현재 fast-only 학습에서는 episode 하나를 slow_T slot짜리 round 하나로 본다.
    # 즉, num_episodes=50000이면 slow_T slot짜리 round를 50000번 학습한다.
    num_episodes: int = 5000
    eval_episodes: int = 20

    rounds_per_episode: int = 20
    eval_rounds_per_episode: int = 50

    # PPO update 1번에 모을 slot 수
    rollout_slots: int = 4096

    # checkpoint / plot 저장 주기
    save_every_episodes: int = 5_000
    plot_every_episodes: int = 1_000
    plot_smooth_window: int = 200

    # ------------------------------------------------------------------
    # 4) PPO hyperparameters
    # ------------------------------------------------------------------
    batch_size: int = 512
    update_epochs: int = 4

    # Queue/Battery 장기 pressure를 보려면 0.90은 너무 짧다.
    # reward scaling은 하지 않고 horizon만 길게 본다.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # action_dim이 큰 continuous PPO라 actor lr는 낮게 유지
    lr: float = 3e-5
    clip_coef: float = 0.10

    # raw reward를 그대로 쓰므로 value loss가 커질 수 있다.
    # critic loss가 actor를 압도하지 않도록 value_coef는 낮게 둔다.
    value_coef: float = 0.5

    # 기존 1e-5는 action_dim=1000 기준으로 거의 영향이 없다.
    entropy_coef: float = 1e-3

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

    # ------------------------------------------------------------------
    # 6) Critic 안정화 옵션
    # ------------------------------------------------------------------
    # fast_train.py / fast_agent.py에서 지원하도록 반영하면 좋다.
    # 지원하지 않는 경우에도 config 저장용으로 문제 없음.
    use_value_huber_loss: bool = True
    use_value_clip: bool = True

    # raw reward 기준 value prediction 변화폭 clipping.
    value_clip_coef: float = 20.0

    # reward scaling 변수
    ppo_reward_scale: float = 1e-4

    # ------------------------------------------------------------------
    # 7) Fast-only 학습용 slow decision 생성 방식
    # ------------------------------------------------------------------
    slow_decision_mode: str = "random"

    # 기존 0.70 / 0.35 / 0.50은 나쁘지 않지만,
    # 처음 본격 학습에서는 active link를 너무 많이 만들면 action credit assignment가 어려워진다.
    # 여기서는 약간 완화한다.
    random_rsu_user_prob: float = 0.50
    random_uav_hire_prob: float = 0.35
    random_uav_user_prob: float = 0.40

    # ------------------------------------------------------------------
    # 8) logging / eval
    # ------------------------------------------------------------------
    eval_every_episodes: int = 1_000
    deterministic_eval: bool = True

    log_action_stats: bool = True
    log_queue_stats: bool = True
    log_battery_stats: bool = True
    log_reward_terms: bool = True

    # ------------------------------------------------------------------
    # 9) debug
    # ------------------------------------------------------------------
    fail_on_nan: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            object.__setattr__(self, "hidden_dims", [256, 256])

        if self.mode not in {"train", "eval"}:
            raise ValueError("mode must be one of {'train', 'eval'}.")

        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive.")

        if self.rounds_per_episode <= 0:
            raise ValueError("rounds_per_episode must be positive.")
        if self.eval_rounds_per_episode <= 0:
            raise ValueError("eval_rounds_per_episode must be positive.")

        if self.rollout_slots <= 0:
            raise ValueError("rollout_slots must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.rollout_slots < self.batch_size:
            raise ValueError("rollout_slots must be >= batch_size.")
        if self.rollout_slots % self.batch_size != 0:
            raise ValueError("rollout_slots must be divisible by batch_size.")

        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive.")

        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        if not (0.0 < self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in (0, 1].")

        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.clip_coef <= 0.0:
            raise ValueError("clip_coef must be positive.")
        if self.value_coef < 0.0:
            raise ValueError("value_coef must be non-negative.")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative.")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")

        if self.value_clip_coef <= 0.0:
            raise ValueError("value_clip_coef must be positive.")

        if self.slow_decision_mode != "random":
            raise ValueError("현재 fast-only 학습에서는 slow_decision_mode='random'만 지원합니다.")

        for name in [
            "random_rsu_user_prob",
            "random_uav_hire_prob",
            "random_uav_user_prob",
        ]:
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")

        if self.save_every_episodes < 0:
            raise ValueError("save_every_episodes must be >= 0.")
        if self.plot_every_episodes < 0:
            raise ValueError("plot_every_episodes must be >= 0.")
        if self.plot_smooth_window <= 0:
            raise ValueError("plot_smooth_window must be positive.")
        if self.eval_every_episodes <= 0:
            raise ValueError("eval_every_episodes must be positive.")

        if self.reward_scale is not None:
            raise ValueError("현재 설정은 reward scaling을 사용하지 않습니다. reward_scale은 None이어야 합니다.")
        if self.reward_clip is not None:
            raise ValueError("현재 설정은 reward clipping을 사용하지 않습니다. reward_clip은 None이어야 합니다.")
        if self.reward_norm:
            raise ValueError("현재 설정은 reward normalization을 사용하지 않습니다. reward_norm은 False여야 합니다.")
        if self.ppo_reward_scale <= 0.0:
            raise ValueError("ppo_reward_scale must be positive.")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def make_run_dir(self, proposed_root: Path) -> Path:
        output_root_path = Path(self.output_root)
        if not output_root_path.is_absolute():
            output_root_path = proposed_root / output_root_path

        if self.run_name is None:
            run_name = "fast_ppo_raw_reward_long"
        else:
            run_name = str(self.run_name)

        run_dir = output_root_path / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def get_fast_ppo_config() -> FastTrainConfig:
    """
    fast_train.py에서 호출하는 단일 config entry point.
    argparse 대신 여기 값을 수정해서 학습 설정을 바꾼다.
    """
    return FastTrainConfig()