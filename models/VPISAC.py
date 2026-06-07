from __future__ import annotations

import torch
from torch import Tensor
import torch.distributions as dist
from torch.distributions import MultivariateNormal
from models.continuous_network import VPISACQValueModel, SACPolicyModel, VPIDistributionModel
from typing import List, Tuple, Dict, Any
from torch.optim import Optimizer, Adam
import numpy as np
from numpy import ndarray
from models.rl_model import RLModel
from models.experience_replay import ExperienceManagerInterface, ExperienceManager
from models.params import VPISACParams
import copy

class VPISAC(RLModel):
    def __init__(self, state_size: List[int], action_size: int, action_range_min: float, action_range_max: float, params: VPISACParams):
        super().__init__(state_size, action_size, params.gamma)
        self.action_range_min: float = action_range_min
        self.action_range_max: float = action_range_max
        self.params: VPISACParams = params

        self.q_value_model1: VPISACQValueModel = VPISACQValueModel(state_size, action_size).to(self.device)
        self.target_q_value_model1: VPISACQValueModel = copy.deepcopy(self.q_value_model1)
        self.q_value_optim1: Optimizer = Adam(self.q_value_model1.parameters(), lr=params.q_value_lr)

        self.q_value_model2: VPISACQValueModel = VPISACQValueModel(state_size, action_size).to(self.device)
        self.target_q_value_model2: VPISACQValueModel = copy.deepcopy(self.q_value_model2)
        self.q_value_optim2: Optimizer = Adam(self.q_value_model2.parameters(), lr=params.q_value_lr)

        self.policy_model: SACPolicyModel = SACPolicyModel(state_size, action_size).to(self.device)
        self.policy_optim: Optimizer = Adam(self.policy_model.parameters(), lr=params.policy_lr)

        self.vpi_dist_model: VPIDistributionModel = VPIDistributionModel(state_size, action_size, output_norms_amount=1).to(self.device)
        self.vpi_dist_optim: Optimizer = Adam(self.vpi_dist_model.parameters(), lr=params.vpi_lr)

        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

        self.experience_replay: ExperienceManager = ExperienceManager(params.experience_replay_max_size, params.batch_size, self.state_size,
                                                                      params.experience_replay_state_to_uint8, self.device, action_size=action_size)
        
        self.step_count: int = 0
        self.multi_normal: MultivariateNormal = MultivariateNormal(loc=torch.zeros(self.actions_amount), covariance_matrix=torch.eye(self.actions_amount))
        self.normal: dist.Normal = dist.Normal(loc=0.0, scale=1.0)
    
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
    
    def normal_prob(self, tensor: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        exponent: Tensor = -0.5 * ((tensor - loc) / scale)**2
        frac: Tensor = 1.0 / (scale * np.sqrt(2 * np.pi))
        return frac * exponent.exp()

    def _sample_q_value(self, model: VPISACQValueModel, state: Tensor, action: Tensor) -> Tensor:
        mu: Tensor
        sigma: Tensor
        mu, sigma = model(state, action)
        return mu + sigma * self.normal.sample(sigma.size()).to(self.device)
    
    def _policy_action_and_log_prob(self, state: Tensor, delta: float = 1e-4) -> Tuple[Tensor, Tensor, Tensor]:
        # Policy action
        mu: Tensor # [batch_size x action_dim]
        sigma: Tensor # [batch_size x action_dim]
        mu, sigma = self.policy_model(state)

        sample: Tensor = self.multi_normal.sample([self.params.batch_size]).to(self.device)
        norm_action: Tensor = mu + sigma * sample
        action = torch.sigmoid(norm_action) * (self.action_range_max - self.action_range_min) + self.action_range_min

        # Policy log prob
        norm: dist.Normal = dist.Normal(loc=mu, scale=sigma)
        u: Tensor = norm_action
        derivative: Tensor = (self.action_range_max - self.action_range_min) / ((action - self.action_range_min) * (self.action_range_max - action) + delta)
        policy_log_prob: Tensor = torch.log(torch.exp(norm.log_prob(u)) + delta) + torch.log(derivative) # [batch_size x action_dim]
        policy_log_prob = policy_log_prob.sum(dim=-1, keepdim=True) # [batch_size x 1]

        # VPI distribution log prob
        all_mu, all_sigma = self.vpi_dist_model(state)
        mu_stack: Tensor = torch.stack(torch.chunk(all_mu, self.params.vpi_output_norms_amount, dim=-1)) # [norm_amount x batch_size x action_dim]
        sigma_stack: Tensor = torch.stack(torch.chunk(all_sigma, self.params.vpi_output_norms_amount, dim=-1)) # [norm_amount x batch_size x action_dim]
        vpi_norms: dist.Normal = dist.Normal(loc=mu_stack, scale=sigma_stack)

        u_copies: Tensor = u.unsqueeze(0).expand(self.params.vpi_output_norms_amount, -1, -1) # [norm_amount x batch_size x action_dim]
        vpi_probs: Tensor = (torch.exp(vpi_norms.log_prob(u_copies)) + delta) * derivative # [norm_amount x batch_size x action_dim]
        vpi_probs = vpi_probs.prod(dim=-1, keepdim=True) # [norm_amount x batch_size x 1]
        vpi_probs = vpi_probs.sum(dim=0, keepdim=False) / self.params.vpi_output_norms_amount # [batch_size x 1]
        vpi_dist_log_prob = torch.log(vpi_probs + 1e-10)

        return action, policy_log_prob, vpi_dist_log_prob
        
    def _get_vpi_dist_model_loss(self, state: Tensor, delta: float = 1e-4) -> Tensor:
        # Sample VPI actions
        all_mu: Tensor # [batch_size x action_dim * output_norms_amount]
        all_sigma: Tensor # [batch_size x action_dim * output_norms_amount]
        all_mu, all_sigma = self.vpi_dist_model(state)
        mu_stack: Tensor = torch.stack(torch.chunk(all_mu, self.params.vpi_output_norms_amount, dim=-1)) # [norm_amount x batch_size x action_dim]
        sigma_stack: Tensor = torch.stack(torch.chunk(all_sigma, self.params.vpi_output_norms_amount, dim=-1)) # [norm_amount x batch_size x action_dim]

        rand: np.ndarray = np.random.randint(0, self.params.vpi_output_norms_amount, size=[self.params.batch_size])
        matrix_mu: Tensor = mu_stack[rand].diagonal(dim1=0, dim2=1).T # [batch_size x action_dim]
        matrix_sigma: Tensor = sigma_stack[rand].diagonal(dim1=0, dim2=1).T # [batch_size x action_dim]
        sample: Tensor = self.normal.sample([self.params.batch_size, self.actions_amount]).to(self.device) # [batch_size x action_dim]
        norm_action: Tensor = matrix_mu + matrix_sigma * sample # [batch_size x action_dim]
        action = torch.sigmoid(norm_action) * (self.action_range_max - self.action_range_min) + self.action_range_min
                
        # Get vpi distribution log probs
        norms: dist.Normal = dist.Normal(loc=mu_stack, scale=sigma_stack)
        u: Tensor = norm_action # [batch_size x action_dim]
        derivative: Tensor = (self.action_range_max - self.action_range_min) / ((action - self.action_range_min) * (self.action_range_max - action) + delta)
        u_copies: Tensor = u.unsqueeze(0).expand(self.params.vpi_output_norms_amount, -1, -1) # [norm_amount x batch_size x action_dim]
        probs: Tensor = (torch.exp(norms.log_prob(u_copies)) + delta) * derivative # [norm_amount x batch_size x action_dim]
        probs = probs.prod(dim=-1, keepdim=True) # [norm_amount x batch_size x 1]
        probs = probs.sum(dim=0, keepdim=False) / self.params.vpi_output_norms_amount # [batch_size x 1]

        vpi_dist_log_prob: Tensor = torch.log(probs + 1e-10) # [batch_size x 1]

        # Compute log of VPI
        optimal_action_value: Tensor # [batch_size x 1]
        with torch.no_grad():
            optimal_action: Tensor # [batch_size x action_dim]
            optimal_action, _ = self.policy_model(state)
            optimal_action = torch.sigmoid(optimal_action) * (self.action_range_max - self.action_range_min) + self.action_range_min

            q1, _ = self.q_value_model1(state, optimal_action)
            q2, _ = self.q_value_model2(state, optimal_action)
            optimal_action_value = torch.min(q1, q2)
        
        mu1, sigma1 = self.q_value_model1(state, action)
        mu2, sigma2 = self.q_value_model2(state, action)
        q_mu: Tensor = torch.min(mu1, mu2) # [batch_size x 1]
        q_sigma: Tensor =  torch.where(mu1 < mu2, sigma1, sigma2) # [batch_size x 1]

        z: Tensor = (optimal_action_value - q_mu) / (q_sigma + delta) # [batch_size x 1]
        pdf: Tensor = torch.exp(self.normal.log_prob(z)) # [batch_size x 1]
        cdf: Tensor = self.normal.cdf(z) # [batch_size x 1]
        vpi: Tensor = q_sigma * pdf + (q_mu - optimal_action_value) * (1.0 - cdf) # [batch_size x 1]
        vpi_log: Tensor = torch.log(vpi + delta) # [batch_size x 1]

        return (vpi_dist_log_prob - vpi_log).mean()


    def improve(self, state: Tensor, action: int | Tensor, reward: float, next_state: Tensor, terminated: bool, truncated: bool, env_id: int = 0) -> None:
        self.experience_replay.add_experience(state, action, reward, next_state, terminated)
        self.step_count += 1
        if self.experience_replay.can_get_batch() and (self.step_count % self.params.steps_per_update == 0):
            self.update_models()
    
    def update_models(self) -> None:
        for t in range(self.params.repeats_per_update):
            # Get batch of experiences
            experience_tensors: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = self.experience_replay.get_batch(discrete_action=False)
            
            state_tensor: Tensor = experience_tensors[0]
            action_tensor: Tensor = experience_tensors[1]
            reward_tensor: Tensor = experience_tensors[2]
            next_state_tensor: Tensor = experience_tensors[3]
            episode_terminated_tensor: Tensor = experience_tensors[4]

            # ompute the temporal difference target
            with torch.no_grad():
                pred_next_actions: Tensor
                pred_next_actions_log_probs: Tensor
                pred_next_actions, pred_next_actions_log_probs, vpi_dist_log_prob = self._policy_action_and_log_prob(next_state_tensor)

                pred_next_q_value1: Tensor = self._sample_q_value(self.target_q_value_model1, next_state_tensor, pred_next_actions)
                pred_next_q_value2: Tensor = self._sample_q_value(self.target_q_value_model2, next_state_tensor, pred_next_actions)
                temp: Tensor = torch.min(pred_next_q_value1, pred_next_q_value2)
                temp = temp - self.params.vpi_const * pred_next_actions_log_probs
                temp = temp + self.params.vpi_const * vpi_dist_log_prob
                td_target: Tensor = reward_tensor + self.params.gamma * (1 - episode_terminated_tensor) * temp
            
            # Update Q1 model parameters
            predicted_q_value1: Tensor = self._sample_q_value(self.q_value_model1, state_tensor, action_tensor)
            q_model_loss1: Tensor = self.loss_function(predicted_q_value1, td_target)
            self.q_value_optim1.zero_grad()
            q_model_loss1.backward()
            self.q_value_optim1.step()

            # Update Q2 model parameters
            predicted_q_value2: Tensor = self._sample_q_value(self.q_value_model2, state_tensor, action_tensor)
            q_model_loss2: Tensor = self.loss_function(predicted_q_value2, td_target)
            self.q_value_optim2.zero_grad()
            q_model_loss2.backward()
            self.q_value_optim2.step()

            # Update policy model parameters
            predicted_actions: Tensor
            predicted_actions_log_probs: Tensor
            predicted_actions, predicted_actions_log_probs, predicted_actions_vpi_dist_log_prob = self._policy_action_and_log_prob(state_tensor)
            q_val1: Tensor = self._sample_q_value(self.q_value_model1, state_tensor, predicted_actions)
            q_val2: Tensor = self._sample_q_value(self.q_value_model2, state_tensor, predicted_actions)
            policy_loss: Tensor = -torch.min(q_val1, q_val2)
            policy_loss = policy_loss + self.params.vpi_const * predicted_actions_log_probs
            policy_loss = policy_loss - self.params.vpi_const * predicted_actions_vpi_dist_log_prob
            policy_loss = policy_loss.mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Update VPI distribution model parameters
            vpi_dist_loss: Tensor = self._get_vpi_dist_model_loss(next_state_tensor)
            self.vpi_dist_optim.zero_grad()
            vpi_dist_loss.backward()
            self.vpi_dist_optim.step()

            # Polyak update target models
            self.target_q_value_model1.polyak_update(self.q_value_model1, self.params.polyak)
            self.target_q_value_model2.polyak_update(self.q_value_model2, self.params.polyak)

    
    def get_params_dict(self) -> Dict[str, Any]:
        return self.params.to_dict()
    
    def new(self) -> RLModel:
        return VPISAC(self.state_size, self.actions_amount, self.action_range_min, self.action_range_max, self.params)
    
    def save(self) -> None:
        torch.save(self.q_value_model1.state_dict(), "./saved_models/vpisac_q1.pth")
        torch.save(self.q_value_model2.state_dict(), "./saved_models/vpisac_q2.pth")
        torch.save(self.policy_model.state_dict(), "./saved_models/vpisac_policy.pth")
        torch.save(self.vpi_dist_model.state_dict(), "./saved_models/vpisac_vpi.pth")
    
    def load(self, path: str) -> None:
        pass