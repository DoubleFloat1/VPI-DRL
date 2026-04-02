import torch
from torch import Tensor
from typing import List, Tuple
import copy
import torch.distributions as dist
from models.network import BayesValueModelNormal, BayesValueModelUniform, BayesModel, BayesValueModelNormal, BayesValueModelPure
from models.experience_replay import PrioritizedExperienceManager

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
    
    def get_greedy_action(self, state: Tensor) -> Tuple[int, float]:
        raise NotImplementedError
    
    def get_predicted_state_action_rewards(self, state: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        raise NotImplementedError
    
    def get_inference_action(self, state: Tensor) -> int:
        raise NotImplementedError
    
    def get_actions_vpi_from_state(self, state: Tensor) -> Tensor:
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
        
    def get_actions_vpi_from_state(self, state: Tensor) -> Tensor:
        sample_q_values, q_value_r1, q_value_r2 = self.policy_network(state)
        for i in range(self.actions_amount):
            if q_value_r1[i] > q_value_r2[i]:
                temp = q_value_r1[i].detach().clone()
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
    def __init__(self, state_size: List[int], actions_amount: int, device: torch.device, pem: PrioritizedExperienceManager = None):
        super().__init__(state_size, actions_amount, device)
        self.policy_network: BayesValueModelPure = BayesValueModelPure(self.state_size, self.actions_amount).to(self.device)
        self.policy_network.prior_to_device(self.device)
        self.target_network = copy.deepcopy(self.policy_network)

        self.normal: dist.Normal = dist.Normal(loc=0.0, scale=1.0)
        self.prioritized_experience_manager: PrioritizedExperienceManager = pem

    def get_state_q_values(self, state: Tensor) -> Tensor:
        q_values, _, _ = self.policy_network(state)
        return q_values
    
    def get_greedy_action(self, state: Tensor) -> Tuple[int, float]:
        q_values: Tensor
        q_values, mu, sigma = self.policy_network(state)
        greedy_action: Tensor = q_values.argmax(dim=-1)
        vpi: Tensor = self._get_state_action_vpi(mu, sigma, greedy_action.unsqueeze(0))
        return (greedy_action.item(), vpi.item())
    
    def get_predicted_state_action_rewards(self, state: Tensor, action: Tensor) -> Tensor:
        predicted_rewards: Tensor
        mu: Tensor
        sigma: Tensor
        predicted_rewards, mu, sigma = self.policy_network(state)
        predicted_action_rewards: Tensor = predicted_rewards.gather(dim=-1, index=action)
        self._update_experience_priority(mu.detach(), sigma.detach(), action)
        return predicted_action_rewards
    
    def get_target_q_values(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            q_values: Tensor
            q_values, _, _ = self.target_network(state)
            return q_values
        
    def get_inference_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = state.to(self.device)
            q_values: Tensor
            mu: Tensor
            sigma: Tensor
            q_values, mu, sigma = self.policy_network(state)
            return mu.argmax(dim=-1).item()

    def get_actions_vpi_from_state(self, state: Tensor) -> Tensor:
        mu: Tensor
        sigma: Tensor
        with torch.no_grad():
            _, mu, sigma = self.policy_network(state)
            return self._get_actions_vpi_from_state(mu, sigma)
        
        
    
    def _get_actions_vpi_from_state(self, mu: Tensor, sigma: Tensor) -> Tensor:
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
    
    def _get_state_action_vpi(self, mu: Tensor, sigma: Tensor, action: Tensor) -> Tensor:
        action_mu: Tensor = mu.gather(dim=-1, index=action).squeeze()
        action_sigma: Tensor = sigma.gather(dim=-1, index=action).squeeze()

        sorted_mu: Tensor
        sorted_mu_indexes: Tensor
        sorted_mu, sorted_mu_indexes = mu.sort()

        greedy_action_mu: Tensor = torch.select(sorted_mu, dim=-1, index=-1)
        second_greedy_action_mu: Tensor = torch.select(sorted_mu, dim=-1, index=-2)

        if mu.dim() == 2:
            is_greedy_action = torch.zeros_like(action, dtype=torch.uint8).squeeze()
            for i in range(len(action)):
                if action[i] == sorted_mu_indexes[i][-1]:
                    is_greedy_action[i] = 1
        else:
            is_greedy_action = torch.zeros_like(action, dtype=torch.uint8)
            if action == sorted_mu_indexes[-1]:
                is_greedy_action[0] = 1
        
        z: Tensor = ((is_greedy_action * second_greedy_action_mu + (1 - is_greedy_action) * greedy_action_mu) - action_mu) / action_sigma
        pdf: Tensor = self.normal.log_prob(z).exp()
        cdf: Tensor = self.normal.cdf(z)

        aux_tensor1: Tensor = is_greedy_action * (second_greedy_action_mu - action_mu) + (1 - is_greedy_action) * (action_mu - greedy_action_mu)
        aux_tensor2: Tensor = is_greedy_action * cdf + (1 - is_greedy_action) * (1.0 - cdf)

        vpis: Tensor = action_sigma * pdf + aux_tensor1 * aux_tensor2
        vpis += 1e-5
        return vpis
        
    def _update_experience_priority(self, mu: Tensor, sigma: Tensor, action: Tensor) -> Tensor:
        vpis: Tensor = self._get_state_action_vpi(mu, sigma, action)
        if self.prioritized_experience_manager != None:
            self.prioritized_experience_manager.update_priorities(vpis)
        
        return vpis