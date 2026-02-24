from __future__ import annotations

import torch
from torch import Tensor
from models.network import BayesTransitionModel, BayesValueModel, BayesValueModelUniform
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
import copy
from models.DQN import DQN, ValueModelManager, ExperienceManager

# TODO: implement n-step return
class VPIValueModelManager(ValueModelManager):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float, batch_size: int, learning_rate: float, kl_weight: float,
                 updates_to_pass_posterior: int, experience_replay_max_size: int, experience_replay_state_to_uint8: bool, updates_to_renew_target_network: int, 
                 device: torch.device):
        self.kl_weight: float = kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.policy_network: BayesValueModelUniform
        self.target_network: BayesValueModelUniform
        super().__init__(
            state_size,
            actions_amount,
            gamma,
            batch_size,
            learning_rate,
            experience_replay_max_size,
            experience_replay_state_to_uint8,
            updates_to_renew_target_network,
            device
        )

    def initialize_models(self) -> None:
        self.policy_network = BayesValueModelUniform(self.state_size, self.actions_amount).to(self.device)
        self.policy_network.prior_to_device(self.device)
        self.target_network = copy.deepcopy(self.policy_network)

    def get_state_q_values(self, state: Tensor) -> Tensor:
        q_values: Tensor
        q_values, _, _ = self.policy_network(state)
        return q_values
    
    def get_target_q_values(self, state):
        with torch.no_grad():
            q_values: Tensor
            q_values, _, _ = self.target_network(state)
            return q_values
    
    def get_detailed_state_q_values(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            return self.policy_network(state)
    
    def inference_get_state_q_values(self, state: List[float]) -> Tensor:
        with torch.no_grad():
            state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values: Tensor
            q_values, _, _ = self.policy_network(state_tensor)
            return q_values
        
    def model_loss(self) -> Tensor:
        return self.policy_network.kl_loss() * self.kl_weight
    
    def post_update(self) -> None:
        if self.update_count % self.updates_to_pass_posterior == 0:
            self.policy_network.pass_posterior_to_prior()
        if self.update_count % self.updates_to_renew_target_network == 0:
            self.target_network = copy.deepcopy(self.policy_network)



class VPIDQN(DQN):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99, value_lr: float = 3e-4, value_kl_weight: float = 0.1, 
                 value_batch_size: int = 32, updates_to_pass_posterior: int = 512, 
                 experience_replay_max_size: int = 4096,  experience_replay_state_to_uint8: bool = False,
                 updates_to_renew_target_network: int = 256, initial_eps: float = 0.5, min_eps: float = 0.1, total_steps_of_eps_decay: int = 100000,
                 load_model_path: str = None):
        self.value_kl_weight: float = value_kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.value_model_manager: VPIValueModelManager
        super().__init__(state_size, 
                         actions_amount, 
                         gamma,
                         value_lr, 
                         value_batch_size,
                         experience_replay_max_size,
                         experience_replay_state_to_uint8,
                         updates_to_renew_target_network,
                         initial_eps,
                         min_eps,
                         total_steps_of_eps_decay,
                         load_model_path)

    
    def createValueModelManager(self):
        return VPIValueModelManager(
            self.state_size,
            self.actions_amount,
            self.gamma,
            self.value_batch_size,
            self.value_lr,
            self.value_kl_weight,
            self.updates_to_pass_posterior,
            self.experience_replay_max_size,
            self.experience_replay_state_to_uint8,
            self.updates_to_renew_target_network,
            self.device
        )

    def get_next_action(self, state: Tensor) -> int:
        state = state.to(self.device)
        if np.random.random() > self.eps:
            q_values: Tensor = self.value_model_manager.get_state_q_values(state)
            return q_values.argmax(dim=-1).item()

        sample_q_values, q_value_r1, q_value_r2 = self.value_model_manager.get_detailed_state_q_values(state)
        #q_value_r1 = mu - 2 * sigma
        #q_value_r2 = mu + 2 * sigma
        for i in range(self.actions_amount):
            if q_value_r1[i] > q_value_r2[i]:
                temp = q_value_r1[i]
                q_value_r1[i] = q_value_r2[i]
                q_value_r2[i] = temp
        
        q_value_means = (q_value_r1 + q_value_r2) / 2.0
        sorted_index_q_value_means = q_value_means.sort()[1]
        greedy_action = sorted_index_q_value_means[-1]
        second_greedy_action = sorted_index_q_value_means[-2]

        vpis = torch.tensor([1e-5 for _ in range(self.actions_amount)], dtype=torch.float32).to(self.device)
        for a in range(self.actions_amount):
            if a != greedy_action:
                action_range_min = q_value_r1[a]
                action_range_max = q_value_r2[a]
                greedy_action_mean = q_value_means[greedy_action]
                if action_range_max > greedy_action_mean:
                    prob_density: Tensor = 1.0 / (action_range_max - action_range_min)
                    if action_range_min < greedy_action_mean:
                        vpis[a] += (action_range_max - greedy_action_mean)**2 * prob_density / 2.0
                    else:
                        vpis[a] += (action_range_max - action_range_min)**2 * prob_density / 2.0
            else:
                greedy_range_min = q_value_r1[greedy_action]
                greedy_range_max = q_value_r2[greedy_action]
                second_greedy_action_mean = q_value_means[second_greedy_action]
                if greedy_range_min < second_greedy_action_mean:
                    prob_density = 1.0 / (greedy_range_max - greedy_range_min)
                    if second_greedy_action_mean < greedy_range_max:
                        vpis[a] += (second_greedy_action_mean - greedy_range_min)**2 * prob_density / 2.0
                    else:
                        vpis[a] += (greedy_range_max - greedy_range_min)**2 * prob_density / 2.0
        
        return vpis.multinomial(1).item()


    def inference_next_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = state.to(self.device)
            _, q_value_r1, q_value_r2 = self.value_model_manager.get_detailed_state_q_values(state)
            return ((q_value_r1 + q_value_r2) / 2.0).argmax(dim=-1).item()

    def get_params_dict(self):
        param_dict: Dict = {
            "name": "VPIDQN",
            "gamma": self.gamma,
            "value_lr": self.value_lr,
            "value_kl_weight": self.value_kl_weight,
            "value_batch_size": self.value_batch_size,
            "uptades_to_pass_posterior": self.updates_to_pass_posterior,
            "experience_replay_max_size": self.experience_replay_max_size,
            "updates_to_renew_target_network": self.updates_to_renew_target_network,
            "initial_eps": self.initial_eps,
            "min_eps": self.min_eps,
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay
        }
        return param_dict

    def new(self) -> VPIDQN:
        return VPIDQN(self.state_size, 
                      self.actions_amount, 
                      self.gamma, 
                      self.value_lr, 
                      self.value_kl_weight, 
                      self.value_batch_size,
                      self.updates_to_pass_posterior, 
                      self.experience_replay_max_size, 
                      self.experience_replay_state_to_uint8,
                      self.updates_to_renew_target_network,
                      self.initial_eps, 
                      self.min_eps, 
                      self.total_steps_of_eps_decay,
                      None)
    
    def save(self) -> None:
        torch.save(self.value_model_manager.policy_network.state_dict(), "./saved_models/vpidqn.pth")

