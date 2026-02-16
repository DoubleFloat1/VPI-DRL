import torch
from torch import Tensor
from network import ValueModel
from typing import List, Tuple, Dict
from torch.optim import Optimizer, Adam
import numpy as np
from rl_model import RLModel

# TODO: implement n-step return
# TODO: implement target network
# TODO: implement replay memory
# TODO: add pass_posterior_to_prior
class ValueModelManager:
    def __init__(self, state_size: int, actions_amount: int, gamma: float, batch_size: int, learning_rate: float):
        self.gamma: float = gamma

        self.model: ValueModel = ValueModel(state_size, actions_amount)
        self.optimizer: Optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

        self.batch_size: int = batch_size
        self.state_batch: List[List[float]] = []
        self.action_batch: List[int] = []
        self.reward_batch: List[float] = []
        self.next_state_batch: List[List[float]] = []
        self.episode_terminated_batch: List[bool] = []

    def get_state_q_values(self, state: List[float]) -> Tensor:
        state_tensor: Tensor = torch.FloatTensor(state)
        return self.model(state_tensor)
    
    def inference_get_state_q_values(self, state: List[float]) -> Tensor:
        with torch.no_grad():
            state_tensor: Tensor = torch.FloatTensor(state)
            return self.model(state_tensor)

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
        state_tensor: Tensor = torch.FloatTensor(self.state_batch)
        action_tensor: Tensor = torch.LongTensor(self.action_batch).unsqueeze(1)
        reward_tensor: Tensor = torch.FloatTensor(self.reward_batch).unsqueeze(1)
        self.next_state_batch = np.array(self.next_state_batch)
        next_state_tensor: Tensor = torch.FloatTensor(self.next_state_batch)
        episode_terminated_tensor: Tensor = torch.FloatTensor(self.episode_terminated_batch).unsqueeze(1)

        predicted_rewards: Tensor = self.model(state_tensor)
        predicted_action_rewards: Tensor = predicted_rewards.gather(dim=-1, index=action_tensor)

        predicted_next_rewards: Tensor = self.model(next_state_tensor)
        predicted_next_max_reward: Tensor = predicted_next_rewards.max(dim=-1, keepdim=True)[0]

        expected_action_rewards: Tensor = reward_tensor + self.gamma * predicted_next_max_reward * (1 - episode_terminated_tensor)
        loss: Tensor = self.loss_function(predicted_action_rewards, expected_action_rewards)
        #loss += self.model.kl_loss() * self.kl_weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.next_state_batch = []
        self.episode_terminated_batch = []

# TODO: make eps variable with respoect to time step
class DQN(RLModel):
    def __init__(self, state_size: int, actions_amount: int, gamma: float = 0.99, value_lr: float = 1e-3, value_batch_size: int = 32):
        super().__init__(state_size, actions_amount, gamma)
        self.value_model_manager: ValueModelManager = ValueModelManager(state_size, actions_amount, gamma, value_batch_size, value_lr)

    
    def get_next_action(self, state: List[float]) -> int:
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

