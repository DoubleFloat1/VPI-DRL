from __future__ import annotations

import torch
from torch import Tensor
from models.network import BayesTransitionModel, BayesValueModel, BayesValueModelUniform
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
import copy

class ExperienceManager:
    def __init__(self, max_size: int, batch_size: int, state_size: List[int], state_to_uint8: bool, device: torch.device):
        self.device: torch.device = device
        self.state_size: List[int] = state_size
        self.max_size: int = max_size
        self.batch_size: int = batch_size

        self.state_batch_dimensions: List[int] = [max_size] + state_size
        self.state_dtype: torch.dtype = torch.uint8 if state_to_uint8 else torch.float32

        self.state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(device)
        self.action_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.int32).to(device)
        self.reward_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.float32).to(device)
        self.next_state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(device)
        self.episode_terminated_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.uint8).to(device)
        
        self.current_size: int = 0
        self.next_index: int = 0

    def add_experience(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool) -> None:
        if self.state_dtype == torch.uint8:
            state = state.detach().clone().mul(255.0).to(torch.uint8)
            next_state = next_state.detach().clone().mul(255.0).to(torch.uint8)

        self.state_batch[self.next_index] = state.to(self.device)
        self.action_batch[self.next_index] = torch.tensor([action], dtype=torch.long).to(self.device)
        self.reward_batch[self.next_index] = torch.tensor([reward], dtype=torch.float32).to(self.device)
        self.next_state_batch[self.next_index] = next_state.to(self.device)
        self.episode_terminated_batch[self.next_index] = torch.tensor([episode_terminated], dtype=torch.float32).to(self.device)

        if self.current_size < self.max_size:
            self.current_size += 1
        self.next_index = (self.next_index + 1) % self.max_size
    
    def can_get_batch(self) -> bool:
        return self.current_size >= self.batch_size
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indexes: Tensor = torch.tensor(np.random.randint(0, self.current_size, self.batch_size), dtype=torch.int32).to(self.device)
        s: Tensor = self.state_batch[indexes]
        a: Tensor = self.action_batch[indexes]
        r: Tensor = self.reward_batch[indexes]
        ns: Tensor = self.next_state_batch[indexes]
        et: Tensor = self.episode_terminated_batch[indexes]

        if self.state_dtype == torch.uint8:
            s = s.detach().clone().float().div(255.0)
            ns = ns.detach().clone().float().div(255.0)

        return (s, a, r, ns, et)

    def empty_experience(self) -> None:
        self.state_batch = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(self.device)
        self.action_batch = torch.zeros(self.max_size, 1, dtype=torch.int32).to(self.device)
        self.reward_batch = torch.zeros(self.max_size, 1, dtype=torch.float32).to(self.device)
        self.next_state_batch = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(self.device)
        self.episode_terminated_batch = torch.zeros(self.max_size, 1, dtype=torch.uint8).to(self.device)
        
        self.current_size = 0
        self.next_index = 0


# TODO: implement n-step return
class ValueModelManager:
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float, batch_size: int, learning_rate: float, kl_weight: float,
                 updates_to_pass_posterior: int, experience_replay_max_size: int, experience_replay_state_to_uint8: bool, updates_to_renew_target_network: int, 
                 device: torch.device):
        self.device: torch.device = device
        self.gamma: float = gamma

        self.policy_network: BayesValueModelUniform = BayesValueModelUniform(state_size, actions_amount).to(device)
        self.policy_network.prior_to_device(device)
        self.target_network: BayesValueModelUniform = copy.deepcopy(self.policy_network)

        self.optimizer: Optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()
        self.kl_weight: float = kl_weight

        self.experience_manager: ExperienceManager = ExperienceManager(experience_replay_max_size, batch_size, state_size, experience_replay_state_to_uint8, device)
        self.updates_to_renew_target_network: int = updates_to_renew_target_network

        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.updates_count: int = 0

    def sample_state_q_values(self, state: Tensor) -> Tensor:
        state_tensor: Tensor = state.to(self.device)
        q_values: Tensor
        q_values, _, _ = self.policy_network(state_tensor)
        return q_values
    
    def detailed_sample_state_q_values(self, state: Tensor) -> Tensor:
        state = state.to(self.device)
        return self.policy_network(state)
    
    def multisample_state_q_values(self, state: List[float], sample_size: int) -> Tensor:
        samples: List[Tensor] = [self.sample_state_q_values(state) for _ in range(sample_size)]
        return torch.stack(samples, dim=0)
    
    def inference_sample_state_q_values(self, state: List[float]) -> Tensor:
        with torch.no_grad():
            state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values: Tensor
            q_values, _, _ = self.policy_network(state_tensor)
            return q_values

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

        predicted_rewards: Tensor 
        predicted_rewards, _, _ = self.policy_network(state_tensor)
        predicted_action_rewards: Tensor = predicted_rewards.gather(dim=-1, index=action_tensor)

        predicted_next_rewards: Tensor 
        predicted_next_rewards, _, _ = self.target_network(next_state_tensor)
        predicted_next_max_reward: Tensor = predicted_next_rewards.max(dim=-1, keepdim=True)[0].detach()

        expected_action_rewards: Tensor = reward_tensor + self.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
        loss: Tensor = self.loss_function(predicted_action_rewards, expected_action_rewards)
        loss += self.policy_network.kl_loss() * self.kl_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates_count += 1
        if self.updates_count % self.updates_to_pass_posterior == 0:
            self.policy_network.pass_posterior_to_prior()
        if self.updates_count % self.updates_to_renew_target_network == 0:
            self.target_network = copy.deepcopy(self.policy_network)



class VPIDQN(RLModel):
    def __init__(self, state_size: List[int], actions_amount: int, gamma: float = 0.99, value_lr: float = 3e-4, value_kl_weight: float = 0.1, 
                 value_batch_size: int = 32, updates_to_pass_posterior: int = 512, 
                 experience_replay_max_size: int = 4096,  experience_replay_state_to_uint8: bool = False,
                 updates_to_renew_target_network: int = 256, initial_eps: float = 0.5, min_eps: float = 0.1, total_steps_of_eps_decay: int = 100000):
        super().__init__(state_size, actions_amount, gamma)
        self.value_model_manager: ValueModelManager = ValueModelManager(state_size, actions_amount, gamma, value_batch_size, value_lr, value_kl_weight, 
                                                                        updates_to_pass_posterior, experience_replay_max_size, experience_replay_state_to_uint8,
                                                                        updates_to_renew_target_network, self.device)
        self.value_lr: float = value_lr
        self.value_kl_weight: float = value_kl_weight
        self.value_batch_size: int = value_batch_size
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.experience_replay_max_size: int = experience_replay_max_size
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.updates_to_renew_target_network: int = updates_to_renew_target_network

        self.initial_eps: float = initial_eps
        self.min_eps: float = min_eps
        self.total_steps_of_eps_decay: int = total_steps_of_eps_decay
        self.eps_decay_step_count: int = 0

        self.eps: float = initial_eps
        print(f"eps will go from {self.initial_eps} to {self.min_eps} in {self.total_steps_of_eps_decay} steps")
    
    def get_next_action(self, state: Tensor) -> int:
        if np.random.random() > self.eps:
            q_values: Tensor = self.value_model_manager.sample_state_q_values(state)
            return q_values.argmax(dim=-1).item()

        sample_q_values, q_value_r1, q_value_r2 = self.value_model_manager.detailed_sample_state_q_values(state)
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
            sample_q_value, q_value_r1, q_value_r2 = self.value_model_manager.detailed_sample_state_q_values(state)
            return ((q_value_r1 + q_value_r2) / 2.0).argmax(dim=-1).item()
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool) -> None:
        self.value_model_manager.improve(state, action, reward, next_state, episode_terminated)
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
                      self.total_steps_of_eps_decay)







    

class TempVPIDQN2(VPIDQN):
    def __init__(self, state_size: int, actions_amount: int, gamma: float = 0.99, value_lr: float = 6e-4, value_kl_weight: float = 0.1, value_batch_size: int = 32,
                 updates_to_pass_posterior: int = 512, experience_replay_max_size: int = 8192, updates_to_renew_target_network: int = 256, vpi_const: float = 0.5):
        super().__init__(state_size, actions_amount, gamma, value_lr, value_kl_weight, value_batch_size,
                         updates_to_pass_posterior, experience_replay_max_size, updates_to_renew_target_network)
        self.vpi_const: float = vpi_const
        
    def get_next_action(self, state: List[float]) -> int:
        sample_q_values, q_value_r1, q_value_r2 = self.value_model_manager.detailed_sample_state_q_values(state)

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
        
        min_val: Tensor = (sample_q_values + vpis * self.vpi_const).min()
        return (sample_q_values + vpis * self.vpi_const - min_val + 1e-5).multinomial(1).item()
    
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
            "vpi_const": self.vpi_const
        }
        return param_dict