from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
from models.DQN import DQN
from models.experience_replay import ExperienceManagerInterface, ExperienceManager, PrioritizedExperienceManager
from models.vpi_distribution import DistributionStrategy, NormalStrategy
from models.params import VPIDQNParams



# TODO: implement n-step return
class VPIValueModelManager:
    def __init__(self, state_size: List[int], actions_amount: int, params: VPIDQNParams, device: torch.device):
        self.state_size: List[int] = state_size
        self.actions_amount: int = actions_amount
        self.params: VPIDQNParams = params
        self.device: torch.device = device
        self.loss_function: torch.nn = torch.nn.MSELoss(reduction="none")

        self.experience_manager: PrioritizedExperienceManager = PrioritizedExperienceManager(self.state_size, self.params, device)

        self.q_value_models: NormalStrategy = NormalStrategy(self.state_size, self.actions_amount, self.device, pem=self.experience_manager)

        self.optimizer: Optimizer = Adam(self.q_value_models.policy_network.parameters(), lr=self.params.value_lr)
        self.update_count: int = 0

    def get_state_q_values(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_state_q_values(state)
    
    def get_greedy_action(self, state: Tensor) -> Tuple[int, float]:
        return self.q_value_models.get_greedy_action(state)
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        return self.q_value_models.get_target_q_values(state)

    def get_inference_action(self, state: Tensor, use_mu: bool) -> int:
        return self.q_value_models.get_inference_action(state, use_mu)
    
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
                expected_action_rewards: Tensor = reward_tensor + self.params.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
                return expected_action_rewards
            
            policy_next_state_rewards: Tensor = self.get_state_q_values(next_state_tensor)
            policy_argmax_actions: Tensor = policy_next_state_rewards.argmax(dim=-1, keepdim=True)
            target_next_state_rewards: Tensor = self.get_target_q_values(next_state_tensor)
            predicted_next_max_reward: Tensor = target_next_state_rewards.gather(dim=-1, index=policy_argmax_actions)
            expected_action_rewards: Tensor = reward_tensor + self.params.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
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

        loss += self.q_value_models.policy_kl_loss() * self.params.value_kl_weight

        #self.optimizer.zero_grad()
        for param in self.q_value_models.policy_network.parameters():
            param.grad = None

        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.params.updates_to_pass_posterior == 0:
            self.q_value_models.pass_posterior_to_prior()
        if self.update_count % self.params.updates_to_renew_target_network == 0:
            self.q_value_models.copy_policy_to_target()



class VPIDQN(RLModel):
    def __init__(self, state_size: List[int], actions_amount: int, params: VPIDQNParams, load_model_path: str = None):
        super().__init__(state_size, actions_amount, params.gamma)
        self.params: VPIDQNParams = params
        self.prev_state_action_vpi: float = None
        self.eps_decay_step_count: int = 0
        self.eps: float = self.params.initial_eps

        self.value_model_manager: VPIValueModelManager = self.createValueModelManager()
        self.exception_ocurred: bool = False

    def createValueModelManager(self):
        return VPIValueModelManager(self.state_size, self.actions_amount, self.params, self.device)

    def set_inference_use_mu(self, use_mu: bool) -> None:
        self.params.inference_use_mu = use_mu

    def get_next_action(self, state: Tensor) -> int:
        state = state.to(self.device)
        if np.random.random() > self.eps:
            action, vpi = self.value_model_manager.get_greedy_action(state)
            self.prev_state_action_vpi = vpi
            return action

        vpis: Tensor = self.value_model_manager.get_actions_vpi_from_state(state)

        try:
            action: int = vpis.multinomial(1).item()
            self.prev_state_action_vpi = vpis[action].item()
        except Exception as e:
            if not self.exception_ocurred:
                print("exception:", e)
                self.exception_ocurred = True

            action, vpi = self.value_model_manager.get_greedy_action(state)
            self.prev_state_action_vpi = vpi
            
        return action

    def inference_next_action(self, state: Tensor) -> int:
        return self.value_model_manager.get_inference_action(state, self.params.inference_use_mu)
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, terminated: bool, truncated: bool, env_id: int = 0) -> None:
        self.value_model_manager.improve(state, action, reward, next_state, terminated, priority=self.prev_state_action_vpi)
        if self.eps > self.params.min_eps:
            self.eps_decay_step_count += 1
            proportion: float = self.eps_decay_step_count / self.params.total_steps_of_eps_decay
            self.eps = proportion * self.params.min_eps + (1 - proportion) * self.params.initial_eps

    def get_params_dict(self):
        return self.params.to_dict()

    def new(self) -> VPIDQN:
        return VPIDQN(self.state_size, self.actions_amount, self.params, None)
    
    def save(self) -> None:
        torch.save(self.value_model_manager.q_value_models.policy_network.state_dict(), "./saved_models/vpidqn.pth")

