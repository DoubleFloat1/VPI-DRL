from __future__ import annotations

import torch
from torch import Tensor
import torch.distributions as dist
from models.continuous_network import ContinuousQValueModel, ContinuousPolicyModel
from typing import List, Tuple, Dict, Any
from torch.optim import Optimizer, Adam
import numpy as np
from numpy import ndarray
from models.rl_model import RLModel
from models.experience_replay import ExperienceManagerInterface, ExperienceManager
from models.params import DDPGParams
import copy

class DDPG(RLModel):
    def __init__(self, state_size: List[int], action_size: int, action_range_min: float, action_range_max: float, params: DDPGParams):
        super().__init__(state_size, action_size, params.gamma)
        self.action_range_min: float = action_range_min
        self.action_range_max: float = action_range_max
        self.params: DDPGParams = params

        self.q_value_model: ContinuousQValueModel = ContinuousQValueModel(state_size, action_size).to(self.device)
        self.target_q_value_model: ContinuousQValueModel = copy.deepcopy(self.q_value_model)
        self.q_value_optim: Optimizer = Adam(self.q_value_model.parameters(), lr=params.q_value_lr)

        self.policy_model: ContinuousPolicyModel = ContinuousPolicyModel(state_size, action_size).to(self.device)
        self.target_policy_model: ContinuousPolicyModel = copy.deepcopy(self.policy_model)
        self.policy_optim: Optimizer = Adam(self.policy_model.parameters(), lr=params.policy_lr)

        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

        self.experience_replay: ExperienceManager = ExperienceManager(params.experience_replay_max_size, params.batch_size, self.state_size,
                                                                      params.experience_replay_state_to_uint8, self.device, action_size=action_size)
        
        self.eps: float = params.initial_eps
        self.step_count: int = 0
    
    def get_next_action(self, state: Tensor) -> ndarray:
        if self.step_count < self.params.uniform_start_steps:
            return np.random.uniform(low=self.action_range_min, high=self.action_range_max, size=(self.actions_amount))

        with torch.no_grad():
            norm: dist.Normal = dist.Normal(loc=0.0, scale=self.eps)
            state = state.to(self.device)
            action: Tensor = self.policy_model(state) * (self.action_range_max - self.action_range_min) + self.action_range_min
            action = action.cpu()
            noise: Tensor = norm.sample([self.actions_amount])
            return torch.clamp(action + noise, min=self.action_range_min, max=self.action_range_max).numpy()
    
    def inference_next_action(self, state: Tensor) -> ndarray:
        with torch.no_grad():
            state = state.to(self.device)
            action: Tensor = self.policy_model(state) * (self.action_range_max - self.action_range_min) + self.action_range_min
            return action.cpu().numpy()
    
    def improve(self, state: Tensor, action: int | Tensor, reward: float, next_state: Tensor, terminated: bool, truncated: bool, env_id: int = 0) -> None:
        self.experience_replay.add_experience(state, action, reward, next_state, terminated)
        self.step_count += 1
        if self.eps > self.params.min_eps:
            proportion: float = self.step_count / self.params.total_steps_of_eps_decay
            self.eps = proportion * self.params.min_eps + (1.0 - proportion) * self.params.initial_eps
        if self.experience_replay.can_get_batch() and (self.step_count % self.params.steps_per_update == 0):
            self.update_models()
    
    def update_models(self) -> None:
        experience_tensors: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = self.experience_replay.get_batch(discrete_action=False)
        
        state_tensor: Tensor = experience_tensors[0]
        action_tensor: Tensor = experience_tensors[1]
        reward_tensor: Tensor = experience_tensors[2]
        next_state_tensor: Tensor = experience_tensors[3]
        episode_terminated_tensor: Tensor = experience_tensors[4]

        with torch.no_grad():
            predicted_next_actions: Tensor = self.target_policy_model(next_state_tensor)
            predicted_next_q_value: Tensor = self.target_q_value_model(next_state_tensor, predicted_next_actions)
            td_target: Tensor = reward_tensor + self.params.gamma * (1 - episode_terminated_tensor) * predicted_next_q_value
        
        predicted_q_value: Tensor = self.q_value_model(state_tensor, action_tensor)
        q_model_loss: Tensor = self.loss_function(predicted_q_value, td_target)
        self.q_value_optim.zero_grad()
        q_model_loss.backward()
        self.q_value_optim.step()

        predicted_actions: Tensor = self.policy_model(state_tensor)
        policy_loss: Tensor = -self.q_value_model(state_tensor, predicted_actions)
        policy_loss = policy_loss.mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.target_q_value_model.polyak_update(self.q_value_model, self.params.polyak)
        self.target_policy_model.polyak_update(self.policy_model, self.params.polyak)

    
    def get_params_dict(self) -> Dict[str, Any]:
        return self.params.to_dict()
    
    def new(self) -> RLModel:
        return DDPG(self.state_size, self.actions_amount, self.action_range_min, self.action_range_max, self.params)
    
    def save(self) -> None:
        torch.save(self.q_value_model.state_dict(), "./saved_models/ddpg_q.pth")
        torch.save(self.policy_model.state_dict(), "./saved_models/ddpg_policy.pth")
    
    def load(self, path: str) -> None:
        pass