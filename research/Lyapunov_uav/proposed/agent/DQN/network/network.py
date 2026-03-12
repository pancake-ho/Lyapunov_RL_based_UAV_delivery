from __future__ import annotations

from typing import Dict, Mapping

import sys
import numpy as np
import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    Multi-head Q-Network 클래스를 위한 기본 뼈대 신경망 클래스
    """
    def __init__(self, in_dim: int, hidden_dims=(256, 256), dropout: float=0.0):
        super().__init__()

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        
        self.network = nn.Sequential(*layers)
        self.out_dim = prev_dim
    
    def forwar(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class BranchingQNetwork(nn.Module):
    """
    DQN용 multi-head Q-Network 클래스
    """
    def __init__(
        self,
        state_dim: int,
        action_dims: Mapping[str, int],
        branch_action_bins: Mapping[str, int],
        hidden_dims=(256, 256),
        hidden_dim_val: int = 128,
        hidden_dim_adv: int = 128,
        dropout: float=0.0,
        dueling: bool=True,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.action_dims = {k: int(v) for k, v in action_dims.items()}
        self.branch_action_bins = {k: int(v) for k, v in branch_action_bins.items()}
        self.branch_names = list(self.action_dims.keys())
        self.dueling = bool(dueling)

        # 예외 처리
        for k, v in self.branch_names:
            if k not in self.branch_action_bins:
                raise ValueError(f"Missing action bin info for branch: {k}")
            if self.action_dims[k] <= 0:
                raise ValueError(f"action_dims[{k}] must be positive") 
            if self.branch_action_bins[k] <= 1:
                raise ValueError(f"branch_action_bins[{k}] mush be larger than 2")

        # 기본 신경망 로딩       
        self.net_backbone = MLPBlock(
            in_dim=self.state_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        backbone_out_dim = self.net_backbone.out_dim
        
        # value와 adv head에 추가적인 신경망 결합
        if self.dueling:
            self.value_heads = nn.ModuleDict()
            self.adv_heads = nn.ModuleDict()

            for name in self.branch_names:
                branch_dim = self.action_dims[name]
                num_bins = self.branch_action_bins[name]

                self.value_heads[name] = nn.Sequential(
                    nn.Linear(backbone_out_dim, hidden_dim_val),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim_val, branch_dim),
                )

                self.adv_heads[name] = nn.Sequential(
                    nn.Linear(backbone_out_dim, hidden_dim_adv),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim_adv, branch_dim * num_bins),
                )
        else: # q head에만 추가적인 신경망 결합
            self.q_heads = nn.ModuleDict()
            
            for name in self.branch_names:
                branch_dim = self.action_dims[name]
                num_bins = self.branch_action_bins[name]

                self.q_heads[name] = nn.Sequential(
                    nn.Linear(backbone_out_dim, hidden_dim_adv),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim_adv, branch_dim * num_bins),
                )
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 예외 처리
        if x.dim() != 2 or x.size(-1) != self.state_dim:
            raise ValueError(
                f"Expected input shape [B, {self.state_dim}], got {tuple(x.shape)}"
            )
        
        # backbone network에서 output 뽑기
        backbone_out = self.net_backbone(x)
        q_dict: Dict[str, torch.Tensor] = {}

        if self.dueling:
            for name in self.branch_names:
                branch_dim = self.action_dims[name]
                num_bins = self.branch_action_bins[name]

                # 이전에 만든 value와 adv head에서 output 뽑기
                value = self.value_heads[name](backbone_out) # [B, branch_dim]
                adv = self.adv_heads[name](backbone_out) # [B, branch_dim * num_bins]
                adv = adv.view(-1, branch_dim, num_bins)
                
                value = value.unsqueeze(-1) # [B, branch_dim, 1]
                q = value + adv - adv.mean(dim=-1, keepdim=True)
                q_dict[name] = q
        else:
            for name in self.branch_names:
                branch_dim = self.action_dims[name]
                num_bins = self.branch_action_bins[name]

                # 이전에 만든 q head에서 output 뽑기
                q = self.q_heads[name](backbone_out)
                q = q.view(-1, branch_dim, num_bins)
                q_dict[name] = q
        
        return q_dict
    
    @torch.no_grad()
    def act(self, state: torch.Tensor, action_mask: Dict[str, torch.Tensor] | None=None, epsilon: float=0.0) -> Dict[str, torch.Tensor]:
        """
        위에서 정의한 각 branch에 대해 Epsilon-greedy action을 선택하는 함수
        """
        self.eval()
        q_dict = self.forward(state)
        batch_size = state.size(0)
        device = self.device

        action_dict: Dict[str, torch.Tensor] = []

        for name in self.branch_names:
            q = q_dict[name] # [B, D, A]
            B, D, A = q.shape

            if action_mask is not None and name in action_mask:
                mask = action_mask[name].to(device=device, dtype=q.dtype)

                # mask[0]에 대한 차원 일치 검증
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).expand(B, -1, -1)
                elif mask.dim() == 3:
                    if mask.size(0) == 1 and B > 1:
                        mask = mask.expand(B, -1, -1)
                    elif mask.size(0) != B:
                        raise ValueError(
                            f"Mask batch mismatch for branch [{name}]: "
                            f"q batch={B}, mask batch={mask.size(0)}"
                        )
                else:
                    raise ValueError(
                        f"Mask for branch [{name}] mush have shape [D, A] or [B, D, A]"
                    ) 

                # mask[1]과 mask[2]에 대한 차원 일치 검증
                if mask.shape[1] != D or mask.shape[2] != A:
                    raise ValueError(
                        f"Mask shape mismatch for branch [{name}]: "
                        f"expected [B, {D}, {A}] compatible, got {tuple(mask.shape)}"
                    )
                
                masked_q = q.masked_fill(mask <= 0.0, float("-inf"))
                greedy_action = masked_q.argmax(dim=-1) # [B, D]

                valid_any = (mask > 0.0).any(dim=-1) # [B, D]
                # 검증
                if not valid_any.all():
                    fallback = torch.zeros_like(greedy_action)
                    greedy_action = torch.where(valid_any, greedy_action, fallback)

            else:
                greedy_action = q.argmax(dim=-1)
            
            # epsilon-greedy action
            if epsilon > 0.0:
                random_pick = torch.rand(B, D, device=device) < epsilon

                if action_mask is not None and name in action_mask:
                    mask = action_mask[name].to(device=device, dtype=torch.float32)
                    if mask.dim() != 2:
                        mask = mask.unsqueeze(0).expand(B, -1, -1)
                    elif mask.dim() == 3 and mask.size(0) == 1 and B > 1:
                        mask = mask.expand(B, -1, -1)
                    
                    random_action = self._sample_masked_random_action(mask)
                else:
                    random_action = torch.randint(
                        low=0, high=A, size=(B, D), device=device
                    )

                action = torch.where(random_pick, random_action, greedy_action)
            else:
                action = greedy_action
            
            action_dict[name] = action.long()
        
        return action_dict
    
    @staticmethod
    def _sample_masked_random_action(mask: torch.Tensor) -> torch.Tensor:
        B, D, A = mask.shape
        device = mask.device

        probs = mask.float()
        denom = probs.sum(dim=-1, keepdim=True) # [B, D, 1]
        no_valid = denom.unsqueeze(-1) <= 0.0

        probs = torch.where(
            denom > 0.0,
            probs / denom.clamp_min(1e-12),
            torch.full_like(probs, 1.0 / A),
        )