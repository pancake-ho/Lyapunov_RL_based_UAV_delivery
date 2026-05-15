from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    재현성을 위한 random seed setting.

    deterministic = True 설정은 CUDA 연산을 더 결정적으로 만들지만,
    학습 속도가 느려질 수는 있음.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> torch.device:
    """
    호환성을 위한 device selection setting.

    현재 시나리오 기준:
        device:
            None / "auto"   : cuda 가능 시 cuda, 아니면 cpu
                "cpu"       : cpu
                'cuda"      : cuda:0
                "cuda:1"    : cuda:1
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA가 요청되었지만 사용 불가 상태입니다. CPU로 동작합니다.")
        return torch.device("cpu")
    
    return torch.device(device)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """
    용이한 Path 탐색 및 생성을 위한 directory setting.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def to_tensor(
    x: Any,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    임의로 들어오는 input data에 대해 torch.Tensor type으로 반환 및 device/dtype 설정.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_numpy(x: Any) -> np.ndarray:
    """
    임의로 들어오는 input data에 대해 numpy array type으로 반환.
    또한 input data가 torch.Tensor type일 경우 detach().cpu() 추가 설정.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def explained_var(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Critic Value Function 품질 확인용 지표.

    1에 가까울수록 value prediction이 return을 잘 설명하고,
    0에 가까울수록 critic이 거의 의미 없다는 신호일 수 있음.
    """
    pred = to_numpy(y_pred).reshape(-1)
    true = to_numpy(y_true).reshape(-1)

    var_y = np.var(true)
    if var_y < eps:
        return 0.0
    
    return float(1.0 - np.var(true - pred) / (var_y + eps))


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Input으로 주어지는 모델에 대한 전체 parameter 수 반환.
    trainable_only = True 설정 시 학습 가능한 parameter 수에 대해서만 반환.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    """
    JSON 파일 기록 및 저장용.
    """
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    JSON 파일 반환용.
    """
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)
    

def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    학습 중단 상황을 대비한 save model checkpoint.

    현재 시나리오 기준:
        기본적으로 model/optimizer parameter를 저장하고,
        optional로 주어지는 extra에는 다음을 넣을 수 있음.
            - update step
            - obs normalizer step
            - config dict
            - reward logs
    """
    path_obj = Path(path)
    ensure_dir(path_obj.parent)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra is not None:
        checkpoint["extra"] = extra
    
    torch.save(checkpoint, path_obj)


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    학습 중단 후 재개되는 상황을 대비한 load model checkpoint.

    현재 시나리오 기준:
        return:
            checkpoint 전체 dict.
            extra의 경우 checkpoint.get("extra", {})로 접근.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class ScalarLogger:
    """
    Minimal CSV logger.

    TensorBoard 고려하여, 붙이기 전 fast PPOsmoke test 단계에서 사용을 대비
    """

    def __init__(self, log_path: str | os.PathLike[str]) -> None:
        self.log_path = Path(log_path)
        ensure_dir(self.log_path.parent)
        self._header_written = self.log_path.exists() and self.log_path.stat().st_size > 0

    def write(self, row: Dict[str, Any]) -> None:
        if len(row) == 0:
            return

        keys = list(row.keys())

        if not self._header_written:
            with self.log_path.open("w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
            self._header_written = True

        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.10g}")
            else:
                values.append(str(value))

        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(",".join(values) + "\n")