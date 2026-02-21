from __future__ import annotations

import torch
from torch import Tensor, FloatTensor, LongTensor
from models.network import ValueModel
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
import copy

class ExperienceManager:
    def __init__(self, max_size: int, batch_size: int, state_size: List[int], device: torch.device):
        self.device: torch.device = device

        self.state_size: List[int] = state_size
        self.max_size: int = max_size
        self.batch_size: int = batch_size

        self.state_batch_dimensions: List[int] = [max_size] + state_size

        self.state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=torch.float32).to(device)
        self.action_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.long).to(device)
        self.reward_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.float32).to(device)
        self.next_state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=torch.float32).to(device)
        self.episode_terminated_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.float32).to(device)
        
        self.multinomial_weights: Tensor = torch.zeros(max_size, dtype=torch.float32)
        self.current_size: int = 0
        self.next_index: int = 0

    def add_experience(self, state: List[float], action: int, reward: float, next_state: List[float], episode_terminated: bool) -> None:
        self.state_batch[self.next_index] = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.action_batch[self.next_index] = torch.tensor([action], dtype=torch.long).to(self.device)
        self.reward_batch[self.next_index] = torch.tensor([reward], dtype=torch.float32).to(self.device)
        self.next_state_batch[self.next_index] = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        self.episode_terminated_batch[self.next_index] = torch.tensor([episode_terminated], dtype=torch.float32).to(self.device)

        if self.current_size < self.max_size:
            self.multinomial_weights[self.current_size] = 1.0
            self.current_size += 1
        self.next_index = (self.next_index + 1) % self.max_size
    
    def can_get_batch(self) -> bool:
        return self.current_size >= self.batch_size
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indexes: Tensor = torch.multinomial(self.multinomial_weights, self.batch_size)
        s: Tensor = self.state_batch[indexes]
        a: Tensor = self.action_batch[indexes]
        r: Tensor = self.reward_batch[indexes]
        ns: Tensor = self.next_state_batch[indexes]
        et: Tensor = self.episode_terminated_batch[indexes]
        return (s, a, r, ns, et)

    def empty_experience(self) -> None:
        self.state_batch = torch.zeros(self.state_batch_dimensions, dtype=torch.float32).to(self.device)
        self.action_batch = torch.zeros(self.max_size, 1, dtype=torch.long).to(self.device)
        self.reward_batch = torch.zeros(self.max_size, 1, dtype=torch.float32).to(self.device)
        self.next_state_batch = torch.zeros(self.state_batch_dimensions, dtype=torch.float32).to(self.device)
        self.episode_terminated_batch = torch.zeros(self.max_size, 1, dtype=torch.float32).to(self.device)
        
        self.multinomial_weights = torch.zeros(self.max_size, dtype=torch.float32)
        self.current_size = 0
        self.next_index = 0

    
# TODO: implement n-step return
# TODO: implement target network
class ValueModelManager:
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float, batch_size: int, learning_rate: float, 
                 experience_replay_max_size: int, updates_to_renew_target_network: int, device: torch.device):
        self.device: torch.device = device
        self.gamma: float = gamma

        self.policy_network: ValueModel = ValueModel(state_size, actions_amount).to(device)
        self.target_network: ValueModel = copy.deepcopy(self.policy_network)

        self.optimizer: Optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

        self.experience_manager: ExperienceManager = ExperienceManager(experience_replay_max_size, batch_size, state_size, device)
        self.updates_to_renew_target_network: int = updates_to_renew_target_network
        self.update_count: int = 0

    def get_state_q_values(self, state: List[float]) -> Tensor:
        state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.policy_network(state_tensor)
    
    def inference_get_state_q_values(self, state: List[float]) -> Tensor:
        with torch.no_grad():
            state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            return self.policy_network(state_tensor)

    def improve(self, state: List[float], action: int, reward: float, next_state: List[float], episode_terminated: bool) -> None:
        self.experience_manager.add_experience(state, action, reward, next_state, episode_terminated)
        if self.experience_manager.can_get_batch():
            self.update_model()
            #self.experience_manager.empty_experience()
    
    def update_model(self) -> None:
        experience_tensors = self.experience_manager.get_batch()

        state_tensor: Tensor = experience_tensors[0]
        action_tensor: Tensor = experience_tensors[1]
        reward_tensor: Tensor = experience_tensors[2]
        next_state_tensor: Tensor = experience_tensors[3]
        episode_terminated_tensor: Tensor = experience_tensors[4]

        predicted_rewards: Tensor = self.policy_network(state_tensor)
        predicted_action_rewards: Tensor = predicted_rewards.gather(dim=-1, index=action_tensor)

        predicted_next_rewards: Tensor = self.target_network(next_state_tensor)
        predicted_next_max_reward: Tensor = predicted_next_rewards.max(dim=-1, keepdim=True)[0].detach()

        expected_action_rewards: Tensor = reward_tensor + self.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
        loss: Tensor = self.loss_function(predicted_action_rewards, expected_action_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count >= self.updates_to_renew_target_network:
            self.target_network = copy.deepcopy(self.policy_network)
            self.update_count = 0


# TODO: make eps variable with respect to time step
class DQN(RLModel):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99, value_lr: float = 6e-4, value_batch_size: int = 32,
                 experience_replay_max_size: int = 8192, updates_to_renew_target_network: int = 128):
        super().__init__(state_size, actions_amount, gamma)
        self.value_model_manager: ValueModelManager = ValueModelManager(state_size, actions_amount, gamma, value_batch_size, value_lr,
                                                                        experience_replay_max_size, updates_to_renew_target_network, self.device)
        self.value_lr: float = value_lr
        self.value_batch_size: int = value_batch_size
        self.experience_replay_max_size: int = experience_replay_max_size
        self.updates_to_renew_target_network: int = updates_to_renew_target_network

    
    def get_next_action(self, state: List[float]) -> int:
        with torch.no_grad():
            eps: float = 0.1
            if np.random.random() <= eps:
                return np.random.randint(0, self.actions_amount)
            
            q_values: Tensor = self.value_model_manager.get_state_q_values(state)
            return q_values.argmax(dim=-1).item()
    
    def inference_next_action(self, state: List[float]) -> int:
        q_values: Tensor = self.value_model_manager.inference_get_state_q_values(state)
        return q_values.argmax(dim=-1).item()
    
    def improve(self, state: List[float], action: int, reward: float, next_state: List[float], episode_terminated: bool) -> None:
        self.value_model_manager.improve(state, action, reward, next_state, episode_terminated)

    def get_params_dict(self):
        param_dict: Dict = {
            "name": "DQN",
            "gamma": self.gamma,
            "value_lr": self.value_lr,
            "value_batch_size": self.value_batch_size,
            "experience_replay_max_size": self.experience_replay_max_size,
            "updates_to_renew_target_network": self.updates_to_renew_target_network
        }
        return param_dict
    
    def new(self) -> DQN:
        return DQN(self.state_size, self.actions_amount, self.gamma, self.value_lr, self.value_batch_size, self.experience_replay_max_size,
                   self.updates_to_renew_target_network)
