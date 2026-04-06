from models.VPIDQN import VPIDQN, VPIValueModelManager
import torch
from torch import Tensor
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
from models.DQN import DQN
from models.experience_replay import ExperienceManagerInterface, ExperienceManager, PrioritizedExperienceManager
from models.vpi_distribution import DistributionStrategy, UniformStrategy, NormalStrategy
from models.params import VPIDQNParams

class OptunaVPIValueModelManager(VPIValueModelManager):
    def __init__(self, state_size: List[int], actions_amount: int, params: VPIDQNParams, device: torch.device):
        self.q_value_models: NormalStrategy
        super().__init__(state_size, actions_amount, params, device)
    
    def get_inference_action(self, state, use_mu: bool):
        return self.q_value_models.get_inference_action(state, use_mu)

class OptunaVPIDQN(VPIDQN):
    def __init__(self, state_size: List[int], actions_amount: int, params: VPIDQNParams, load_model_path: str = None):
        self.value_model_manager: OptunaVPIValueModelManager
        super().__init__(state_size, actions_amount, params, load_model_path)
    
    def createValueModelManager(self):
        return OptunaVPIValueModelManager(self.state_size, self.actions_amount, self.params, self.device)
    
    def inference_next_action(self, state, use_mu: bool):
        return self.value_model_manager.get_inference_action(state, use_mu)