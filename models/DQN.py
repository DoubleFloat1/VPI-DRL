from __future__ import annotations

import torch
from torch import Tensor
from models.network import ValueModel
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
from models.experience_replay import ExperienceManagerInterface, ExperienceManager
import copy


    
# TODO: implement n-step return
class ValueModelManager:
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float, batch_size: int, learning_rate: float, 
                 experience_replay_max_size: int, experience_replay_state_to_uint8: bool, updates_to_renew_target_network: int, device: torch.device):
        self.state_size: List[int] = state_size
        self.actions_amount: int = actions_amount
        self.device: torch.device = device
        self.gamma: float = gamma

        self.policy_network: ValueModel = ValueModel(self.state_size, self.actions_amount).to(self.device)
        self.target_network: ValueModel = copy.deepcopy(self.policy_network)

        self.optimizer: Optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function: torch.nn.MSELoss = torch.nn.HuberLoss()

        self.experience_manager: ExperienceManagerInterface = ExperienceManager(experience_replay_max_size, batch_size, state_size, 
                                                                                experience_replay_state_to_uint8, device)
        self.updates_to_renew_target_network: int = updates_to_renew_target_network
        self.update_count: int = 0

    def get_state_q_values(self, state: Tensor) -> Tensor:
        return self.policy_network(state)
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            return self.target_network(state)
    
    def inference_get_state_q_values(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            return self.policy_network(state)

    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool) -> None:
        self.experience_manager.add_experience(state, action, reward, next_state, episode_terminated)
        if self.experience_manager.can_get_batch():
            self.update_model()
    
    def post_update(self) -> None:
        if self.update_count >= self.updates_to_renew_target_network:
            self.target_network = copy.deepcopy(self.policy_network)
            self.update_count = 0

    def get_expected_action_rewards(self, reward_tensor: Tensor, next_state_tensor: Tensor, episode_terminated_tensor: Tensor, use_ddqn: bool = True) -> Tensor:
        with torch.no_grad():
            if not use_ddqn:
                predicted_next_rewards: Tensor = self.get_target_q_values(next_state_tensor)
                predicted_next_max_reward: Tensor = predicted_next_rewards.max(dim=-1, keepdim=True)[0].detach()
                expected_action_rewards: Tensor = reward_tensor + self.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
                return expected_action_rewards
            
            policy_next_state_rewards: Tensor = self.get_state_q_values(next_state_tensor)
            policy_argmax_actions: Tensor = policy_next_state_rewards.argmax(dim=-1, keepdim=True)
            target_next_state_rewards: Tensor = self.get_target_q_values(next_state_tensor)
            predicted_next_max_reward: Tensor = target_next_state_rewards.gather(dim=-1, index=policy_argmax_actions)
            expected_action_rewards: Tensor = reward_tensor + self.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
            return expected_action_rewards
        


    def update_model(self) -> None:
        experience_tensors = self.experience_manager.get_batch()

        state_tensor: Tensor = experience_tensors[0]
        action_tensor: Tensor = experience_tensors[1]
        reward_tensor: Tensor = experience_tensors[2]
        next_state_tensor: Tensor = experience_tensors[3]
        episode_terminated_tensor: Tensor = experience_tensors[4]

        predicted_rewards: Tensor = self.get_state_q_values(state_tensor)
        predicted_action_rewards: Tensor = predicted_rewards.gather(dim=-1, index=action_tensor)

        expected_action_rewards: Tensor = self.get_expected_action_rewards(reward_tensor, next_state_tensor, episode_terminated_tensor)
        loss: Tensor = self.loss_function(predicted_action_rewards, expected_action_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count >= self.updates_to_renew_target_network:
            self.target_network = copy.deepcopy(self.policy_network)
            self.update_count = 0


class DQN(RLModel):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99, value_lr: float = 3e-4, value_batch_size: int = 32,
                 experience_replay_max_size: int = 4096, experience_replay_state_to_uint8: bool = False, updates_to_renew_target_network: int = 256, 
                 initial_eps: float = 0.5, min_eps: float = 0.1, total_steps_of_eps_decay: int = 100000, load_model_path: str = None):
        super().__init__(state_size, actions_amount, gamma)
        self.value_lr: float = value_lr
        self.value_batch_size: int = value_batch_size
        self.experience_replay_max_size: int = experience_replay_max_size
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.updates_to_renew_target_network: int = updates_to_renew_target_network

        self.initial_eps: float = initial_eps
        self.min_eps: float = min_eps
        self.total_steps_of_eps_decay: int = total_steps_of_eps_decay
        self.eps_decay_step_count: int = 0

        self.eps: float = initial_eps

        self.value_model_manager: ValueModelManager = self.createValueModelManager()
        if load_model_path != None:
            self.load(load_model_path)
        

    def createValueModelManager(self) -> ValueModelManager:
        return ValueModelManager(
            self.state_size,
            self.actions_amount,
            self.gamma,
            self.value_batch_size,
            self.value_lr,
            self.experience_replay_max_size,
            self.experience_replay_state_to_uint8,
            self.updates_to_renew_target_network,
            self.device
        )
    
    def get_next_action(self, state: Tensor) -> int:
        with torch.no_grad():
            if np.random.random() <= self.eps:
                return np.random.randint(0, self.actions_amount)
            
            state = state.to(self.device)
            q_values: Tensor = self.value_model_manager.get_state_q_values(state)
            return q_values.argmax(dim=-1).item()
    
    def inference_next_action(self, state: Tensor) -> int:
        state = state.to(self.device)
        q_values: Tensor = self.value_model_manager.inference_get_state_q_values(state)
        return q_values.argmax(dim=-1).item()
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool) -> None:
        self.value_model_manager.improve(state, action, reward, next_state, episode_terminated)
        if self.eps > self.min_eps:
            self.eps_decay_step_count += 1
            proportion: float = self.eps_decay_step_count / self.total_steps_of_eps_decay
            self.eps = proportion * self.min_eps + (1 - proportion) * self.initial_eps

    def get_params_dict(self):
        param_dict: Dict = {
            "name": "DQN",
            "gamma": self.gamma,
            "value_lr": self.value_lr,
            "value_batch_size": self.value_batch_size,
            "experience_replay_max_size": self.experience_replay_max_size,
            "updates_to_renew_target_network": self.updates_to_renew_target_network,
            "initial_eps": self.initial_eps,
            "min_eps": self.min_eps,
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay
        }
        return param_dict
    
    def new(self) -> DQN:
        return DQN(self.state_size, 
                   self.actions_amount, 
                   self.gamma, 
                   self.value_lr, 
                   self.value_batch_size, 
                   self.experience_replay_max_size,
                   self.experience_replay_state_to_uint8, 
                   self.updates_to_renew_target_network, 
                   self.initial_eps, 
                   self.min_eps, 
                   self.total_steps_of_eps_decay,
                   None
                   )

    def save(self) -> None:
        torch.save(self.value_model_manager.policy_network.state_dict(), "./saved_models/dqn.pth")

    def load(self, path: str) -> None:
        self.value_model_manager.policy_network.load_state_dict(torch.load(path, weights_only=True))
