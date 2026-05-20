from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FastPPOConfig:
    """
    Fast-timescale PPO 학습용 hyperparameter config.

    주의:
    - 이 파일은 RL/AI hyperparameter만 관리한다.
    - env/config.py의 시스템 상수, 채널 상수, UAV 배터리 상수와 섞지 않는다.
    - fast policy는 slot-level decision만 학습한다.
    - slow decision(mu, y, phi)은 fast 학습 시 random/fixed/external condition으로 주입된다.
    """

    # ------------------------------------------------------------------
    # 1) 기본 실행 설정
    # ------------------------------------------------------------------
    mode: str = "train"
    seed: int = 2026
    device: str = "cuda"

    # Seraph / Ubuntu 22.04 LTS 기준으로 /data/$USER 아래 사용 권장
    project_root: Optional[str] = None
    output_root: str = "runs/fast_ppo"
    run_name: str = "fast_ppo_stable_v1"

    checkpoint: Optional[str] = None
    resume: bool = False

    # ------------------------------------------------------------------
    # 2) Fast-timescale episode / rollout 설정
    # ------------------------------------------------------------------
    # 현 시나리오에서 한 episode는 fast policy 검증을 위해 여러 round를 포함할 수 있지만,
    # env 내부 step 단위는 slot이고 slow_T마다 slow decision이 갱신된다.
    num_episodes: int = 50_000
    eval_episodes: int = 20
    save_every_episodes: int = 5_000

    # rollout_slots는 PPO update 1번에 모을 slot 수.
    # action_dim이 큰 편이므로 너무 작으면 advantage 추정이 흔들린다.
    rollout_slots: int = 2048

    # mini-batch
    batch_size: int = 256

    # ------------------------------------------------------------------
    # 3) PPO 핵심 hyperparameter
    # ------------------------------------------------------------------
    # fast reward가 Lyapunov one-slot reward이고, slow_T가 짧으므로 gamma를 너무 크게 둘 필요는 없다.
    # 다만 slot-level delivery 결과가 다음 queue에 영향을 주므로 0.90~0.95가 적절.
    gamma: float = 0.90
    gae_lambda: float = 0.85

    # 현재 action_dim이 큰 연속 action PPO라면 lr을 낮게 잡는 게 안정적이다.
    # 기존 5e-5도 가능하지만, 초기 안정성은 3e-5가 더 안전하다.
    lr: float = 3e-5

    # 큰 action_dim에서 policy가 한 번에 너무 많이 변하면 KL/clip이 튄다.
    clip_coef: float = 0.12

    # reward scale이 큰 Lyapunov reward에서는 value loss가 쉽게 커진다.
    # value_coef를 0.1 이하로 낮게 두는 편이 안정적이다.
    value_coef: float = 0.05

    # action_dim이 크므로 entropy 총합이 커질 수 있다.
    # 너무 크면 power/layer/chunk action이 계속 흔들린다.
    entropy_coef: float = 1e-5

    # gradient 안정화
    max_grad_norm: float = 0.5

    # PPO update epoch.
    # action_dim이 500 수준이면 update_epochs=4보다 2가 더 안전한 경우가 많다.
    update_epochs: int = 2

    # target KL. None이면 early stop 안 함.
    # 너무 작게 잡으면 학습이 자주 멈추므로 0.03 정도로 둔다.
    target_kl: Optional[float] = 0.03

    # ------------------------------------------------------------------
    # 4) Actor/Critic network 설정
    # ------------------------------------------------------------------
    hidden_dims: List[int] = None

    # Gaussian policy 초기 log_std.
    # -1.5는 std≈0.22라서 action_dim이 큰 환경에서 무난하다.
    # 너무 크면 전송량/power가 과격하게 흔들리고, 너무 작으면 탐색이 부족하다.
    init_log_std: float = -1.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0

    # orthogonal initialization 사용 여부
    orthogonal_init: bool = True

    # ------------------------------------------------------------------
    # 5) Normalization / scaling 설정
    # ------------------------------------------------------------------
    # observation normalization은 권장.
    # Z, B, distance, channel capacity, scheduling mask scale이 다르기 때문.
    obs_norm: bool = True
    obs_norm_eps: float = 1e-8
    obs_clip: float = 10.0

    # reward normalization은 Lyapunov reward 해석을 흐릴 수 있으므로 기본은 False.
    # 대신 PPO 내부 loss 안정화를 위해 reward_scale만 사용한다.
    reward_norm: bool = False

    # 중요:
    # 환경 reward 자체는 그대로 기록하고, PPO buffer에 넣는 학습용 reward만 scale한다.
    # 기존 reward mean이 수백 단위이고 value loss가 매우 크게 나온 경우,
    # 0.01부터 시작하는 것이 안전하다.
    # reward_scale: float = 0.01

    # advantage normalization은 PPO에서 거의 필수.
    adv_norm: bool = True
    adv_norm_eps: float = 1e-8

    # value target clipping.
    # critic update 폭을 제한해 value loss 폭주를 줄인다.
    value_clip: bool = True
    value_clip_coef: float = 0.2

    # ------------------------------------------------------------------
    # 6) Fast-only training에서 slow decision 생성 방식
    # ------------------------------------------------------------------
    # 현재 목표는 fast policy만 먼저 학습/검증하는 것.
    # 따라서 slow policy는 아직 학습하지 않고 round마다 random slow decision을 생성하는 것을 기본값으로 둔다.
    slow_decision_mode: str = "random"  # random | fixed | external

    # RSU-user scheduling random 확률.
    # RSU가 기본 infra이므로 너무 낮게 두면 UAV가 과도하게 중요해진다.
    random_rsu_user_prob: float = 0.70

    # UAV hiring random 확률.
    # UAV는 residual user 보조용이므로 0.5 이상으로 너무 높게 두면 hiring cost 학습 구조가 왜곡된다.
    # fast-only 학습에서는 UAV action을 충분히 경험해야 하므로 0.30~0.45 권장.
    random_uav_hire_prob: float = 0.35

    # UAV-user candidate scheduling 확률.
    # hiring된 UAV가 residual/candidate user를 어느 정도 보게 해야 fast action 학습 가능.
    random_uav_user_prob: float = 0.50

    # fixed mode용 옵션
    fixed_uav_hire: bool = True

    # ------------------------------------------------------------------
    # 8) Logging / evaluation
    # ------------------------------------------------------------------
    log_interval_updates: int = 1
    eval_interval_episodes: int = 5_000

    # moving average window
    reward_ma_window: int = 1000

    # CSV / plot 저장
    save_csv: bool = True
    save_plots: bool = True

    # TensorBoard는 서버 환경에서 선택적으로 사용
    use_tensorboard: bool = False

    # ------------------------------------------------------------------
    # 9) Debug / safety
    # ------------------------------------------------------------------
    detect_anomaly: bool = False
    fail_on_nan: bool = True
    print_model_summary: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            object.__setattr__(self, "hidden_dims", [256, 256])

        if self.rollout_slots <= 0:
            raise ValueError("rollout_slots must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.rollout_slots < self.batch_size:
            raise ValueError("rollout_slots must be >= batch_size.")
        if self.rollout_slots % self.batch_size != 0:
            raise ValueError("rollout_slots must be divisible by batch_size.")

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
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive.")

        if self.slow_T <= 0:
            raise ValueError("slow_T must be positive.")

        if self.slow_decision_mode not in {"random", "fixed", "external"}:
            raise ValueError(
                "slow_decision_mode must be one of {'random', 'fixed', 'external'}."
            )

        for name in [
            "random_rsu_user_prob",
            "random_uav_hire_prob",
            "random_uav_user_prob",
        ]:
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")

        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if self.obs_norm_eps <= 0.0:
            raise ValueError("obs_norm_eps must be positive.")
        if self.adv_norm_eps <= 0.0:
            raise ValueError("adv_norm_eps must be positive.")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def make_run_dir(self, proposed_root: Path) -> Path:
        """
        run directory 생성용 helper.

        proposed_root:
            보통 /data/$USER/.../Lyapunov_uav/proposed
        """
        output_root_path = Path(self.output_root)
        if not output_root_path.is_absolute():
            output_root_path = proposed_root / output_root_path

        run_dir = output_root_path / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def get_fast_ppo_config() -> FastPPOConfig:
    """
    fast_train.py에서 기본 config를 가져올 때 사용.
    """
    return FastPPOConfig()