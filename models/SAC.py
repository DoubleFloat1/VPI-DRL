from __future__ import annotations

import torch
from torch import Tensor
import torch.distributions as dist
from torch.distributions import MultivariateNormal
from models.continuous_network import ContinuousQValueModel, SACPolicyModel
from typing import List, Tuple, Dict, Any
from torch.optim import Optimizer, Adam
import numpy as np
from numpy import ndarray
from models.rl_model import RLModel
from models.experience_replay import ExperienceManagerInterface, ExperienceManager
from models.params import SACParams
import copy

class SAC(RLModel):
    def __init__(self, state_size: List[int], action_size: int, action_range_min: float, action_range_max: float, params: SACParams):
        super().__init__(state_size, action_size, params.gamma)
        self.action_range_min: float = action_range_min
        self.action_range_max: float = action_range_max
        self.params: SACParams = params

        self.q_value_model1: ContinuousQValueModel = ContinuousQValueModel(state_size, action_size).to(self.device)
        self.target_q_value_model1: ContinuousQValueModel = copy.deepcopy(self.q_value_model1)
        self.q_value_optim1: Optimizer = Adam(self.q_value_model1.parameters(), lr=params.q_value_lr)

        self.q_value_model2: ContinuousQValueModel = ContinuousQValueModel(state_size, action_size).to(self.device)
        self.target_q_value_model2: ContinuousQValueModel = copy.deepcopy(self.q_value_model2)
        self.q_value_optim2: Optimizer = Adam(self.q_value_model2.parameters(), lr=params.q_value_lr)

        self.policy_model: SACPolicyModel = SACPolicyModel(state_size, action_size).to(self.device)
        self.policy_optim: Optimizer = Adam(self.policy_model.parameters(), lr=params.policy_lr)

        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

        self.experience_replay: ExperienceManager = ExperienceManager(params.experience_replay_max_size, params.batch_size, self.state_size,
                                                                      params.experience_replay_state_to_uint8, self.device, action_size=action_size)
        
        self.step_count: int = 0
        self.multi_normal: MultivariateNormal = MultivariateNormal(loc=torch.zeros(self.actions_amount), covariance_matrix=torch.eye(self.actions_amount))
    
    def get_next_action(self, state: Tensor) -> ndarray:
        if self.step_count < self.params.uniform_start_steps:
            return np.random.uniform(low=self.action_range_min, high=self.action_range_max, size=(self.actions_amount))

        with torch.no_grad():
            state = state.to(self.device)
            mu: Tensor
            sigma: Tensor
            mu, sigma = self.policy_model(state)
            mu = mu.cpu()
            sigma = sigma.cpu()
            
            action: Tensor = torch.sigmoid(mu + sigma * self.multi_normal.sample())
            action = action * (self.action_range_max - self.action_range_min) + self.action_range_min
            return action.numpy()
    
    def inference_next_action(self, state: Tensor) -> ndarray:
        with torch.no_grad():
            state = state.to(self.device)
            mu: Tensor
            mu, _ = self.policy_model(state)

            action: Tensor = torch.sigmoid(mu)
            action = action * (self.action_range_max - self.action_range_min) + self.action_range_min
            return action.cpu().numpy()
    
    def _policy_action_and_log_prob(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        mu: Tensor
        sigma: Tensor
        mu, sigma = self.policy_model(state)

        sample: Tensor = self.multi_normal.sample([self.params.batch_size]).to(self.device)
        norm_action: Tensor = mu + sigma * sample
        action = torch.sigmoid(norm_action) * (self.action_range_max - self.action_range_min) + self.action_range_min

        norm: dist.Normal = dist.Normal(loc=mu, scale=sigma)
        delta: float = 1e-4 # To avoid exploding the value by divisions close to 0
        u: Tensor = norm_action
        derivative: Tensor = (self.action_range_max - self.action_range_min) / ((action - self.action_range_min) * (self.action_range_max - action) + delta)
        log_prob: Tensor = torch.clamp(norm.log_prob(u), min=np.log(delta)) + torch.log(derivative) # [batch_size x action_dim]
        log_prob = log_prob.sum(dim=-1, keepdim=True) # [batch_size x 1]

        return action, log_prob
    
    def improve(self, state: Tensor, action: int | Tensor, reward: float, next_state: Tensor, terminated: bool, truncated: bool, env_id: int = 0) -> None:
        self.experience_replay.add_experience(state, action, reward, next_state, terminated)
        self.step_count += 1
        if self.experience_replay.can_get_batch() and (self.step_count % self.params.steps_per_update == 0):
            self.update_models()
    
    def update_models(self) -> None:
        for _ in range(self.params.repeats_per_update):
            experience_tensors: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = self.experience_replay.get_batch(discrete_action=False)
            
            state_tensor: Tensor = experience_tensors[0]
            action_tensor: Tensor = experience_tensors[1]
            reward_tensor: Tensor = experience_tensors[2]
            next_state_tensor: Tensor = experience_tensors[3]
            episode_terminated_tensor: Tensor = experience_tensors[4]

            with torch.no_grad():
                pred_next_actions: Tensor
                pred_next_actions_log_probs: Tensor
                pred_next_actions, pred_next_actions_log_probs = self._policy_action_and_log_prob(next_state_tensor)

                pred_next_q_value1: Tensor = self.target_q_value_model1(next_state_tensor, pred_next_actions)
                pred_next_q_value2: Tensor = self.target_q_value_model2(next_state_tensor, pred_next_actions)
                temp: Tensor = torch.min(pred_next_q_value1, pred_next_q_value2) - self.params.entropy_const * pred_next_actions_log_probs
                td_target: Tensor = reward_tensor + self.params.gamma * (1 - episode_terminated_tensor) * temp
            
            predicted_q_value1: Tensor = self.q_value_model1(state_tensor, action_tensor)
            q_model_loss1: Tensor = self.loss_function(predicted_q_value1, td_target)
            self.q_value_optim1.zero_grad()
            q_model_loss1.backward()
            self.q_value_optim1.step()

            predicted_q_value2: Tensor = self.q_value_model2(state_tensor, action_tensor)
            q_model_loss2: Tensor = self.loss_function(predicted_q_value2, td_target)
            self.q_value_optim2.zero_grad()
            q_model_loss2.backward()
            self.q_value_optim2.step()

            predicted_actions: Tensor
            predicted_actions_log_probs: Tensor
            predicted_actions, predicted_actions_log_probs = self._policy_action_and_log_prob(state_tensor)
            q_val1: Tensor = self.q_value_model1(state_tensor, predicted_actions)
            q_val2: Tensor = self.q_value_model2(state_tensor, predicted_actions)
            policy_loss: Tensor = -(torch.min(q_val1, q_val2) - self.params.entropy_const * predicted_actions_log_probs)
            policy_loss = policy_loss.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            self.target_q_value_model1.polyak_update(self.q_value_model1, self.params.polyak)
            self.target_q_value_model2.polyak_update(self.q_value_model2, self.params.polyak)

    
    def get_params_dict(self) -> Dict[str, Any]:
        return self.params.to_dict()
    
    def new(self) -> RLModel:
        return SAC(self.state_size, self.actions_amount, self.action_range_min, self.action_range_max, self.params)
    
    def save(self) -> None:
        torch.save(self.q_value_model1.state_dict(), "./saved_models/sac_q1.pth")
        torch.save(self.q_value_model2.state_dict(), "./saved_models/sac_q2.pth")
        torch.save(self.policy_model.state_dict(), "./saved_models/sac_policy.pth")
    
    def load(self, path: str) -> None:
        pass