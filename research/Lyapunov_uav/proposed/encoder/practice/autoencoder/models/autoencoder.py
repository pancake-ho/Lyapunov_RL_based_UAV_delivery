from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop


@dataclass(frozen=True)
class AutoencoderConfig:
    z_size: int
    c_hid: int = 32
    num_image_channels: int = 3
    learning_rate: float = 1e-4
    flatten_size: int = 6144


class PrintShape(nn.Module):
    """
    Debug helper for inspecting intermediate tensor shapes.
    """
    def forward(self, x: TeNSOR)