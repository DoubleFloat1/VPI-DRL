import torch
from torch import Tensor
from network import BayesTransitionModel, BayesValueModel, BayesValueModelUniform
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from rl_model import RLModel
from scipy.stats import norm
from numpy import ndarray

class TransitionModelWrapper:
    def __init__(self, state_size: int, batch_size: int, learning_rate: float, kl_weight: float, device: torch.device):
        self.device: torch.device = device
        self.model: BayesTransitionModel = BayesTransitionModel(state_size, state_size).to(device)
        self.model.prior_to_device(device)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()
        self.kl_weight: float = kl_weight

        self.batch_size: int = batch_size
        self.state_batch: List[List[float]] = []
        self.next_state_batch: List[List[float]] = []

        self.updates_count: int = 0

    def clear_batch(self) -> None:
        self.state_batch = []
        self.next_state_batch = []
    
    def improve(self, state: List[float], next_state: List[float]) -> None:
        self.state_batch.append(state)
        self.next_state_batch.append(next_state)
        if len(self.state_batch) >= self.batch_size:
            self.update_model()
    
    def update_model(self) -> None:
        self.state_batch = np.array(self.state_batch)
        input_tensor: Tensor = torch.tensor(self.state_batch, dtype=torch.float32).to(self.device)
        self.next_state_batch = np.array(self.next_state_batch)
        expected_output_tensor: Tensor = torch.tensor(self.next_state_batch, dtype=torch.float32).to(self.device)
        
        output: Tensor 
        output, _, _ = self.model(input_tensor)
        loss: Tensor = self.loss_function(output, expected_output_tensor)
        loss += self.model.kl_loss() * self.kl_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates_count += 1
        if self.updates_count % 1000 == 0:
            self.model.pass_posterior_to_prior()

        self.clear_batch()
    
    def sample_transition(self, initial_state: List[float]) -> Tensor:
        initial_state_tensor: Tensor = torch.tensor(initial_state, dtype=torch.float32).to(self.device)
        sampled_next_state: Tensor
        sampled_next_state, _, _ = self.model(initial_state_tensor)
        return sampled_next_state
    
    def detailed_sample_transition(self, initial_state: List[float]) -> Tuple[Tensor, Tensor, Tensor]:
        initial_state_tensor: Tensor = torch.tensor(initial_state, dtype=torch.float32).to(self.device)
        return self.model(initial_state_tensor)


class TransitionModelsManager:
    def __init__(self, state_size: int, actions_amount: int, batch_size: int, learning_rate: float, kl_weight: float, device: torch.device):
        self.models = [TransitionModelWrapper(state_size, batch_size, learning_rate, kl_weight, device) for _ in range(actions_amount)]
        self.device: torch.device = device
    
    def improve(self, state: List[float], action: int, next_state: List[float]) -> None:
        action_model: TransitionModelWrapper = self.models[action]
        action_model.improve(state, next_state)

    def sample_state_action_transition(self, state: List[float], action: int) -> Tensor:
        action_model: TransitionModelWrapper = self.models[action]
        return action_model.sample_transition(state)
    
    def multisample_state_action_transition(self, state: List[float], action: int, sample_size: int) -> Tensor:
        samples: List[Tensor] = [self.sample_state_action_transition(state, action) for _ in range(sample_size)]
        return torch.stack(samples, dim=0)
    
    def detailed_sample_state_action_transition(self, state: List[float], action: int) -> Tuple[Tensor, Tensor, Tensor]:
        action_model: TransitionModelWrapper = self.models[action]
        return action_model.detailed_sample_transition(state)

# TODO: implement n-step return
# TODO: implement target network
# TODO: implement replay memory
# TODO: make pass_posterior_to_prior more modular (some constants should be parameters)
class ValueModelManager:
    def __init__(self, state_size: int, actions_amount: int, gamma: float, batch_size: int, learning_rate: float, kl_weight: float, device: torch.device):
        self.device: torch.device = device
        self.gamma: float = gamma

        self.model: BayesValueModelUniform = BayesValueModelUniform(state_size, actions_amount).to(device)
        self.model.prior_to_device(device)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()
        self.kl_weight: float = kl_weight

        self.batch_size: int = batch_size
        self.state_batch: List[List[float]] = []
        self.action_batch: List[int] = []
        self.reward_batch: List[float] = []
        self.next_state_batch: List[List[float]] = []
        self.episode_terminated_batch: List[bool] = []

        self.updates_count: int = 0

    def sample_state_q_values(self, state: List[float]) -> Tensor:
        state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values: Tensor
        q_values, _, _ = self.model(state_tensor)
        return q_values
    
    def detailed_sample_state_q_values(self, state: List[float]) -> Tensor:
        state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.model(state_tensor)
    
    def multisample_state_q_values(self, state: List[float], sample_size: int) -> Tensor:
        samples: List[Tensor] = [self.sample_state_q_values(state) for _ in range(sample_size)]
        return torch.stack(samples, dim=0)
    
    def inference_sample_state_q_values(self, state: List[float]) -> Tensor:
        with torch.no_grad():
            state_tensor: Tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values: Tensor
            q_values, _, _ = self.model(state_tensor)
            return q_values

    def improve(self, state: List[float], action: int, reward: float, next_state: List[float], episode_terminated: bool) -> None:
        self.state_batch.append(state)
        self.action_batch.append(action)
        self.reward_batch.append(reward)
        self.next_state_batch.append(next_state)
        self.episode_terminated_batch.append(episode_terminated)
        if len(self.state_batch) >= self.batch_size:
            self.update_model()
    
    def update_model(self) -> None:
        self.state_batch = np.array(self.state_batch)
        state_tensor: Tensor = torch.tensor(self.state_batch, dtype=torch.float32).to(self.device)
        action_tensor: Tensor = torch.tensor(self.action_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_tensor: Tensor = torch.tensor(self.reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.next_state_batch = np.array(self.next_state_batch)
        next_state_tensor: Tensor = torch.tensor(self.next_state_batch, dtype=torch.float32).to(self.device)
        episode_terminated_tensor: Tensor = torch.tensor(self.episode_terminated_batch, dtype=torch.float32).unsqueeze(1).to(self.device)

        predicted_rewards: Tensor 
        predicted_rewards, _, _ = self.model(state_tensor)
        predicted_action_rewards: Tensor = predicted_rewards.gather(dim=-1, index=action_tensor)

        predicted_next_rewards: Tensor 
        predicted_next_rewards, _, _ = self.model(next_state_tensor)
        predicted_next_max_reward: Tensor = predicted_next_rewards.max(dim=-1, keepdim=True)[0]

        expected_action_rewards: Tensor = reward_tensor + self.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
        loss: Tensor = self.loss_function(predicted_action_rewards, expected_action_rewards)
        loss += self.model.kl_loss() * self.kl_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates_count += 1
        if self.updates_count % 1000 == 0:
            self.model.pass_posterior_to_prior()

        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.next_state_batch = []
        self.episode_terminated_batch = []

class VPIDQN2(RLModel):
    def __init__(self, state_size: int, actions_amount: int, gamma: float = 0.99, transition_lr: float = 1e-3, value_lr: float = 1e-3, 
                 transition_kl_weight: float = 0.1, value_kl_weight: float = 0.1, value_batch_size: int = 32, transition_batch_size: int = 32):
        super().__init__(state_size, actions_amount, gamma)
        self.value_model_manager: ValueModelManager = ValueModelManager(state_size, actions_amount, gamma, value_batch_size, 
                                                                        value_lr, value_kl_weight, self.device)
        self.transition_models_manager: TransitionModelsManager = TransitionModelsManager(state_size, actions_amount, transition_batch_size, 
                                                                                          transition_lr, transition_kl_weight, self.device)
    
    def get_next_action(self, state: List[float]) -> int:
        eps: float = 0.1
        if np.random.random() <= eps:
            return np.random.randint(0, self.actions_amount)
        
        q_values: Tensor = self.value_model_manager.sample_state_q_values(state)
        return q_values.argmax(dim=-1).item()

    def vpi_get_next_action(self, state: List[float]) -> int:
        eps: float = 0.1
        if np.random.random() > eps:
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

        vpis = torch.tensor([1e-5 for _ in range(self.actions_amount)], dtype=torch.float32)
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
        '''
        const: float = 1.0
        if sample_q_values.argmax().item() != (const * vpis + sample_q_values).argmax().item():
            print("q_vals", sample_q_values)
            print("vpis", const * vpis)
            self.vpi_trigger_count += 1
        else:
            self.vpi_non_trigger_count += 1
        
        return (const * vpis + sample_q_values).argmax().item()
        '''
        


    def inference_next_action(self, state: List[float]) -> int:
        with torch.no_grad():
            q_values, mu, sigma = self.value_model_manager.detailed_sample_state_q_values(state)
            return mu.argmax(dim=-1).item()
    
    def improve(self, state: List[float], action: int, reward: float, next_state: List[float], episode_terminated: bool) -> None:
        self.value_model_manager.improve(state, action, reward, next_state, episode_terminated)
        #self.transition_models_manager.improve(state, action, next_state)


# TEMPORARY FOR TESTING ONLY. DELETE LATER
class TestVPIDQN2(VPIDQN2):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def get_next_action(self, state):
        eps: float = 0.1
        if np.random.random() <= eps:
            return np.random.randint(0, self.actions_amount)
        
        q_values: Tensor = self.value_model_manager.sample_state_q_values(state)
        return q_values.argmax(dim=-1).item()
    
    def improve(self, state, action, reward, next_state, episode_terminated):
        self.value_model_manager.improve(state, action, reward, next_state, episode_terminated)
        #self.transition_models_manager.improve(state, action, next_state)