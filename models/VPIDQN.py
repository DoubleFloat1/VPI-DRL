from __future__ import annotations

import torch
from torch import Tensor
from models.network import BayesValueModelNormal, BayesValueModelUniform, BayesModel, BayesValueModelNormal2
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
import copy
from models.DQN import DQN, ValueModelManager, ExperienceManager
import torch.distributions as dist


class DistributionStrategy:
    def __init__(self, state_size: List[int], actions_amount: int, device: torch.device):
        self.state_size: List[int] = state_size
        self.actions_amount: int = actions_amount
        self.device: torch.device = device

        self.policy_network: BayesModel
        self.target_network: BayesModel

    def policy_kl_loss(self) -> Tensor:
        return self.policy_network.kl_loss()
    
    def pass_posterior_to_prior(self) -> None:
        self.policy_network.pass_posterior_to_prior()

    def copy_policy_to_target(self) -> None:
        self.target_network = copy.deepcopy(self.policy_network)

    def get_state_q_values(self, state: Tensor) -> Tensor:
        raise NotImplementedError
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        raise NotImplementedError
    
    def get_inference_action(self, state: Tensor) -> int:
        raise NotImplementedError
    
    def get_state_actions_vpi(self, state: Tensor) -> Tensor:
        raise NotImplementedError
    

class UniformStrategy(DistributionStrategy):
    def __init__(self, state_size: List[int], actions_amount: int, device: torch.device):
        super().__init__(state_size, actions_amount, device)
        self.policy_network: BayesValueModelUniform = BayesValueModelUniform(self.state_size, self.actions_amount).to(self.device)
        self.policy_network.prior_to_device(self.device)
        self.target_network = copy.deepcopy(self.policy_network)

    def get_state_q_values(self, state: Tensor) -> Tensor:
        q_values: Tensor
        q_values, _, _ = self.policy_network(state)
        return q_values
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            q_values: Tensor
            q_values, _, _ = self.target_network(state)
            return q_values
        
    def get_inference_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = state.to(self.device)
            _, q_value_r1, q_value_r2 = self.policy_network(state)
            return ((q_value_r1 + q_value_r2) / 2.0).argmax(dim=-1).item()
        
    def get_state_actions_vpi(self, state: Tensor) -> Tensor:
        sample_q_values, q_value_r1, q_value_r2 = self.policy_network(state)
        for i in range(self.actions_amount):
            if q_value_r1[i] > q_value_r2[i]:
                temp = q_value_r1[i]
                q_value_r1[i] = q_value_r2[i]
                q_value_r2[i] = temp
        
        q_value_means: Tensor = (q_value_r1 + q_value_r2) / 2.0
        sorted_index_q_value_means = q_value_means.sort()[1]
        greedy_action = sorted_index_q_value_means[-1]
        second_greedy_action = sorted_index_q_value_means[-2]

        vpis: Tensor = torch.tensor([1e-5 for _ in range(self.actions_amount)], dtype=torch.float32).to(self.device)
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

        return vpis


class NormalStrategy(DistributionStrategy):
    def __init__(self, state_size: List[int], actions_amount: int, device: torch.device):
        super().__init__(state_size, actions_amount, device)
        self.policy_network: BayesValueModelNormal2 = BayesValueModelNormal2(self.state_size, self.actions_amount).to(self.device)
        self.policy_network.prior_to_device(self.device)
        self.target_network = copy.deepcopy(self.policy_network)

        self.normal: dist.Normal = dist.Normal(loc=0.0, scale=1.0)

    def get_state_q_values(self, state: Tensor) -> Tensor:
        q_values: Tensor
        q_values, _, _ = self.policy_network(state)
        return q_values
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            q_values: Tensor
            q_values, _, _ = self.target_network(state)
            return q_values
        
    def get_inference_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = state.to(self.device)
            mu: Tensor
            _, mu, _ = self.policy_network(state)
            return mu.argmax(dim=-1).item()

    def get_state_actions_vpi(self, state: Tensor) -> Tensor:
        mu: Tensor
        _, mu, sigma = self.policy_network(state)
        
        sorted_index_q_value_means = mu.sort()[1]
        greedy_action = sorted_index_q_value_means[-1]
        greedy_action_mean = mu[greedy_action]
        second_greedy_action = sorted_index_q_value_means[-2]
        second_greedy_action_mean = mu[second_greedy_action]


        z: Tensor = (greedy_action_mean - mu) / sigma
        z[greedy_action] = (second_greedy_action_mean - mu[greedy_action]) / sigma[greedy_action]
        pdf: Tensor = self.normal.log_prob(z).exp()
        cdf: Tensor = self.normal.cdf(z)

        auxiliary_tensor: Tensor = mu - greedy_action_mean
        auxiliary_tensor2: Tensor = 1.0 - cdf
        auxiliary_tensor[greedy_action] = second_greedy_action_mean - mu[greedy_action]
        auxiliary_tensor2[greedy_action] = cdf[greedy_action]

        vpis: Tensor = sigma * pdf + auxiliary_tensor * auxiliary_tensor2
        vpis += 1e-5
        return vpis

# TODO: implement n-step return
class VPIValueModelManager(ValueModelManager):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float, use_uniform_distribution: bool, batch_size: int, learning_rate: float, 
                 kl_weight: float, updates_to_pass_posterior: int, experience_replay_max_size: int, experience_replay_state_to_uint8: bool, 
                 updates_to_renew_target_network: int, device: torch.device):
        self.kl_weight: float = kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.use_uniform_distribution: bool = use_uniform_distribution
        self.q_value_models: DistributionStrategy
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
        if self.use_uniform_distribution:
            self.q_value_models = UniformStrategy(self.state_size, self.actions_amount, self.device)
        else:
            self.q_value_models = NormalStrategy(self.state_size, self.actions_amount, self.device)

        # TODO: Refactor later:
        self.policy_network = self.q_value_models.policy_network
        self.target_network = self.q_value_models.target_network

    def get_state_q_values(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_state_q_values(state)
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_target_q_values(state)
        
    def model_loss(self) -> Tensor:
        return self.q_value_models.policy_kl_loss() * self.kl_weight
    
    def post_update(self) -> None:
        if self.update_count % self.updates_to_pass_posterior == 0:
            self.q_value_models.pass_posterior_to_prior()
        if self.update_count % self.updates_to_renew_target_network == 0:
            self.q_value_models.copy_policy_to_target()

    def get_inference_action(self, state: Tensor) -> int:
        return self.q_value_models.get_inference_action(state)
    
    def get_state_actions_vpi(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_state_actions_vpi(state)



class VPIDQN(DQN):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99, value_lr: float = 3e-4, value_kl_weight: float = 0.1, 
                 value_batch_size: int = 32, updates_to_pass_posterior: int = 512, 
                 experience_replay_max_size: int = 4096,  experience_replay_state_to_uint8: bool = False,
                 updates_to_renew_target_network: int = 256, initial_eps: float = 0.5, min_eps: float = 0.1, total_steps_of_eps_decay: int = 100000,
                 use_uniform_distribution: bool = False, load_model_path: str = None):
        self.value_kl_weight: float = value_kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.use_uniform_distribution: bool = use_uniform_distribution
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
            self.use_uniform_distribution,
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

        vpis: Tensor = self.value_model_manager.get_state_actions_vpi(state)
        return vpis.multinomial(1).item()

    def inference_next_action(self, state: Tensor) -> int:
        return self.value_model_manager.get_inference_action(state)

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
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay,
            "use_uniform_distribution": self.use_uniform_distribution
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
                      self.use_uniform_distribution,
                      None)
    
    def save(self) -> None:
        torch.save(self.value_model_manager.policy_network.state_dict(), "./saved_models/vpidqn.pth")

