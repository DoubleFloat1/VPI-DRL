from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
from models.DQN import DQN
from models.experience_replay import ExperienceManagerInterface, ExperienceManager, PrioritizedExperienceManager, PrioritizedExperienceManager2
from models.vpi_distribution import DistributionStrategy, UniformStrategy, NormalStrategy



# TODO: implement n-step return
class VPIValueModelManager:
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float, use_uniform_distribution: bool, vpi_batch_size: int, rand_batch_size: int, 
                 learning_rate: float, kl_weight: float, updates_to_pass_posterior: int, experience_replay_max_size: int, experience_replay_state_to_uint8: bool, 
                 updates_to_renew_target_network: int, alpha: float, initial_beta: float, total_steps_of_beta_growth: int, device: torch.device):
        self.kl_weight: float = kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.use_uniform_distribution: bool = use_uniform_distribution
        self.q_value_models: DistributionStrategy
        
        self.state_size: List[int] = state_size
        self.actions_amount: int = actions_amount
        self.device: torch.device = device
        self.gamma: float = gamma

        self.loss_function: torch.nn.MSELoss = torch.nn.HuberLoss(reduction="none")

        self.experience_manager: PrioritizedExperienceManager = PrioritizedExperienceManager(experience_replay_max_size, vpi_batch_size, rand_batch_size, state_size, 
                                                                                             experience_replay_state_to_uint8, alpha, initial_beta,
                                                                                             total_steps_of_beta_growth, device)
        
        if self.use_uniform_distribution:
            self.q_value_models = UniformStrategy(self.state_size, self.actions_amount, self.device)
        else:
            self.q_value_models = NormalStrategy(self.state_size, self.actions_amount, self.device, pem=self.experience_manager)

        self.optimizer: Optimizer = Adam(self.q_value_models.policy_network.parameters(), lr=learning_rate)

        self.updates_to_renew_target_network: int = updates_to_renew_target_network
        self.update_count: int = 0

    def get_state_q_values(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_state_q_values(state)
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_target_q_values(state)

    def get_inference_action(self, state: Tensor) -> int:
        return self.q_value_models.get_inference_action(state)
    
    def get_actions_vpi_from_state(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_actions_vpi_from_state(state)
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool, priority: float = 1.0) -> None:
        self.experience_manager.add_experience(state, action, reward, next_state, episode_terminated, priority=priority)
        if self.experience_manager.can_get_batch():
            self.update_model()

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

        predicted_action_rewards: Tensor = self.q_value_models.get_predicted_state_action_rewards(state_tensor, action_tensor)

        expected_action_rewards: Tensor = self.get_expected_action_rewards(reward_tensor, next_state_tensor, episode_terminated_tensor)
        loss: Tensor = self.loss_function(predicted_action_rewards, expected_action_rewards)

        weights: Tensor = self.experience_manager.get_last_batch_weights()
        weights = weights / weights.max()
        loss = (weights * loss.squeeze()).mean()

        loss += self.q_value_models.policy_kl_loss() * self.kl_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.updates_to_pass_posterior == 0:
            self.q_value_models.pass_posterior_to_prior()
        if self.update_count % self.updates_to_renew_target_network == 0:
            self.q_value_models.copy_policy_to_target()



class VPIDQN(DQN):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99, value_lr: float = 1e-5, value_kl_weight: float = 0.1, 
                 value_vpi_batch_size: int = 16, value_rand_batch_size: int = 16, updates_to_pass_posterior: int = 512, 
                 experience_replay_max_size: int = 4096,  experience_replay_state_to_uint8: bool = False,
                 updates_to_renew_target_network: int = 256, initial_eps: float = 1.0, min_eps: float = 0.1, total_steps_of_eps_decay: int = 1000000,
                 use_uniform_distribution: bool = False, alpha: float = 1.0, initial_beta: float = 1.0, total_steps_of_beta_growth: int = 1000000, 
                 load_model_path: str = None):
        self.value_vpi_batch_size: int = value_vpi_batch_size
        self.value_rand_batch_size: int = value_rand_batch_size
        self.value_kl_weight: float = value_kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.use_uniform_distribution: bool = use_uniform_distribution
        self.value_model_manager: VPIValueModelManager
        self.prev_state_action_vpi: float = None
        self.alpha: float = alpha
        self.initial_beta: float = initial_beta
        self.total_steps_of_beta_growth: int = total_steps_of_beta_growth
        super().__init__(state_size, 
                         actions_amount, 
                         gamma,
                         value_lr, 
                         value_vpi_batch_size + value_rand_batch_size,
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
            self.value_vpi_batch_size,
            self.value_rand_batch_size,
            self.value_lr,
            self.value_kl_weight,
            self.updates_to_pass_posterior,
            self.experience_replay_max_size,
            self.experience_replay_state_to_uint8,
            self.updates_to_renew_target_network,
            self.alpha,
            self.initial_beta,
            self.total_steps_of_beta_growth,
            self.device
        )

    def get_next_action(self, state: Tensor) -> int:
        state = state.to(self.device)
        if np.random.random() > self.eps:
            q_values: Tensor = self.value_model_manager.get_state_q_values(state)
            return q_values.argmax(dim=-1).item()

        vpis: Tensor = self.value_model_manager.get_actions_vpi_from_state(state)
        action: int = vpis.multinomial(1).item()
        self.prev_state_action_vpi = vpis[action].item()
        return action

    def inference_next_action(self, state: Tensor) -> int:
        return self.value_model_manager.get_inference_action(state)
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool) -> None:
        self.value_model_manager.improve(state, action, reward, next_state, episode_terminated, priority=self.prev_state_action_vpi)
        if self.eps > self.min_eps:
            self.eps_decay_step_count += 1
            proportion: float = self.eps_decay_step_count / self.total_steps_of_eps_decay
            self.eps = proportion * self.min_eps + (1 - proportion) * self.initial_eps

    def get_params_dict(self):
        param_dict: Dict = {
            "name": "VPIDQN",
            "gamma": self.gamma,
            "value_lr": self.value_lr,
            "value_kl_weight": self.value_kl_weight,
            "value_vpi_batch_size": self.value_vpi_batch_size,
            "value_rand_batch_size": self.value_rand_batch_size,
            "uptades_to_pass_posterior": self.updates_to_pass_posterior,
            "experience_replay_max_size": self.experience_replay_max_size,
            "updates_to_renew_target_network": self.updates_to_renew_target_network,
            "initial_eps": self.initial_eps,
            "min_eps": self.min_eps,
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay,
            "use_uniform_distribution": self.use_uniform_distribution,
            "alpha": self.alpha,
            "initial_beta": self.initial_beta,
            "total_steps_of_beta_growth": self.total_steps_of_beta_growth
        }
        return param_dict

    def new(self) -> VPIDQN:
        return VPIDQN(self.state_size, 
                      self.actions_amount, 
                      self.gamma, 
                      self.value_lr, 
                      self.value_kl_weight, 
                      self.value_vpi_batch_size,
                      self.value_rand_batch_size,
                      self.updates_to_pass_posterior, 
                      self.experience_replay_max_size, 
                      self.experience_replay_state_to_uint8,
                      self.updates_to_renew_target_network,
                      self.initial_eps, 
                      self.min_eps, 
                      self.total_steps_of_eps_decay,
                      self.use_uniform_distribution,
                      self.alpha,
                      self.initial_beta,
                      self.total_steps_of_beta_growth,
                      None)
    
    def save(self) -> None:
        torch.save(self.value_model_manager.q_value_models.policy_network.state_dict(), "./saved_models/vpidqn.pth")

