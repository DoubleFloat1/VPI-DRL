import torch
from torch import Tensor
from typing import List, Tuple
import numpy as np
from models.priority import Priority, StandardPriority, MinHeapPriority

class ExperienceManagerInterface:
    def add_experience(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool, priority: float = 1.0) -> None:
        raise NotImplementedError
    
    def can_get_batch(self) -> bool:
        raise NotImplementedError
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError
    
    def empty_experience(self) -> None:
        raise NotImplementedError


class ExperienceManager(ExperienceManagerInterface):
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

    def add_experience(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool, priority: float = 1.0) -> None:
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
        return self.current_size >= self.batch_size * 100 or self.current_size == self.max_size
    
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



class PrioritizedExperienceManager(ExperienceManagerInterface):
    def __init__(self, max_size: int, vpi_batch_size: int, rand_batch_size: int, state_size: List[int], state_to_uint8: bool, alpha: float, initial_beta: float, 
                 total_steps_of_beta_growth: int, device: torch.device):
        self.device: torch.device = device
        self.state_size: List[int] = state_size
        self.max_size: int = max_size

        self.alpha: float = alpha
        self.initial_beta: float = initial_beta
        self.total_steps_of_beta_growth: int = total_steps_of_beta_growth
        self.beta: float = initial_beta
        self.step_count: int = 0

        self.state_batch_dimensions: List[int] = [max_size] + state_size
        self.state_dtype: torch.dtype = torch.uint8 if state_to_uint8 else torch.float32

        self.state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(device)
        self.action_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.int32).to(device)
        self.reward_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.float32).to(device)
        self.next_state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(device)
        self.episode_terminated_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.uint8).to(device)
        
        self.priority: Priority = StandardPriority(max_size, vpi_batch_size, rand_batch_size, device)

    def add_experience(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool, priority: float = 1.0) -> None:
        exp_index: int = self.priority.add_new_priority(priority**self.alpha)
        self._add_experience(exp_index, state, action, reward, next_state, episode_terminated)
    
    def _add_experience(self, exp_index: int, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool) -> None:
        if self.state_dtype == torch.uint8:
            state = state.detach().clone().mul(255.0).to(torch.uint8)
            next_state = next_state.detach().clone().mul(255.0).to(torch.uint8)

        if exp_index != -1:
            self.state_batch[exp_index] = state.to(self.device)
            self.action_batch[exp_index] = torch.tensor([action], dtype=torch.long).to(self.device)
            self.reward_batch[exp_index] = torch.tensor([reward], dtype=torch.float32).to(self.device)
            self.next_state_batch[exp_index] = next_state.to(self.device)
            self.episode_terminated_batch[exp_index] = torch.tensor([episode_terminated], dtype=torch.float32).to(self.device)

        self.step_count += 1
        if self.beta < 1.0:
            prop: float = self.step_count / self.total_steps_of_beta_growth
            self.beta = prop + (1.0 - prop) * self.initial_beta
        
    
    def can_get_batch(self) -> bool:
        return self.priority.can_get_batch()
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indexes: Tensor = self.priority.get_batch_indexes()

        s: Tensor = self.state_batch[indexes]
        a: Tensor = self.action_batch[indexes]
        r: Tensor = self.reward_batch[indexes]
        ns: Tensor = self.next_state_batch[indexes]
        et: Tensor = self.episode_terminated_batch[indexes]

        if self.state_dtype == torch.uint8:
            s = s.detach().clone().float().div(255.0)
            ns = ns.detach().clone().float().div(255.0)

        return (s, a, r, ns, et)

    
    def update_priorities(self, priorities: Tensor) -> None:
        self.priority.update_last_batch_priorities(priorities**self.alpha)

    def get_last_batch_weights(self) -> Tensor:
        return self.priority.get_last_batch_weights()**(-self.beta)

