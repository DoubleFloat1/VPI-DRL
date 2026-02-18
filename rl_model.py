from typing import List
import torch

class RLModel:
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99):
        self.state_size: List[int] = state_size
        self.actions_amount: int = actions_amount
        self.gamma: float = gamma
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"

    
    def get_next_action(self, state: List[float]) -> int:
        raise NotImplementedError
    
    def inference_next_action(self, state: List[float]) -> int:
        raise NotImplementedError
    
    def improve(self, state: List[float], action: int, reward: float, next_state: List[float], episode_terminated: bool) -> None:
        raise NotImplementedError