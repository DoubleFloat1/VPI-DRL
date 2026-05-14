from __future__ import annotations

from typing import List
import torch
from torch import Tensor
from typing import Dict, Any
from numpy import ndarray

class RLModel:
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99):
        self.state_size: List[int] = state_size
        self.actions_amount: int = actions_amount
        self.gamma: float = gamma
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"

    
    def get_next_action(self, state: Tensor) -> int | ndarray:
        raise NotImplementedError
    
    def inference_next_action(self, state: Tensor) -> int | ndarray:
        raise NotImplementedError
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, terminated: bool, truncated: bool, env_id: int = 0) -> None:
        raise NotImplementedError
    
    def get_params_dict(self) -> Dict[str, Any]:
        return {"gamma": self.gamma}
    
    def new(self) -> RLModel:
        raise NotImplementedError
    
    def save(self) -> None:
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        raise NotImplementedError
        