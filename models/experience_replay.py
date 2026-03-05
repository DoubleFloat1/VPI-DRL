import torch
from torch import Tensor
from typing import List, Tuple
import numpy as np

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



















class SumTree:
    def __init__(self, max_size: int):
        self.max_size: int = max_size
        self.tree: np.ndarray = np.zeros(2 * max_size, dtype=np.float32)
        self.next_add_index = 0

    def add(self, value: float) -> None:
        self.update(self.next_add_index, value)

        self.next_add_index += 1
        if self.next_add_index >= self.max_size:
            self.next_add_index = 0
    
    def update(self, tree_index: int, value: float) -> None:
        tree_index: int = self.max_size + tree_index
        delta: float = value - self.tree[tree_index]
        self.tree[tree_index] = value
        while tree_index > 1:
            tree_index = tree_index // 2
            self.tree[tree_index] += delta
    
    def get_leaf_index_from_value(self, value: float) -> int:
        tree_index: int = 1
        current_value: float = value
        while tree_index < self.max_size:
            left_index: int = 2 * tree_index
            right_index: int = left_index + 1
            if current_value <= self.tree[left_index]:
                tree_index = left_index
            else:
                tree_index = right_index
                current_value -= self.tree[left_index]
    
        return tree_index - self.max_size
    
    def sample(self) -> int:
        max_value: float = self.tree[1]
        rand = np.random.uniform(0, max_value)
        return self.get_leaf_index_from_value(rand)
    
    def batch_sample(self, batch_size: int) -> np.ndarray[int]:
        indexes: np.ndarray = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            sampled_index = self.sample()
            indexes[i] = sampled_index
        
        return indexes




class PrioritizedExperienceManager(ExperienceManagerInterface):
    def __init__(self, max_size: int, batch_size: int, state_size: List[int], state_to_uint8: bool, alpha: float, beta: float, device: torch.device):
        self.device: torch.device = device
        self.state_size: List[int] = state_size
        self.max_size: int = max_size
        self.batch_size: int = batch_size
        self.alpha: float = alpha
        self.beta: float = beta

        self.state_batch_dimensions: List[int] = [max_size] + state_size
        self.state_dtype: torch.dtype = torch.uint8 if state_to_uint8 else torch.float32

        self.state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(device)
        self.action_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.int32).to(device)
        self.reward_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.float32).to(device)
        self.next_state_batch: Tensor = torch.zeros(self.state_batch_dimensions, dtype=self.state_dtype).to(device)
        self.episode_terminated_batch: Tensor = torch.zeros(max_size, 1, dtype=torch.uint8).to(device)
        
        self.priorities: Tensor = torch.zeros(max_size, dtype=torch.float32).to(device)
        self.current_size: int = 0
        self.next_index: int = 0

        self.last_batch_indexes: Tensor = None
        self.last_batch_prob_weights: Tensor = None
        self.prob_weights_sum: Tensor = torch.zeros(1, dtype=torch.float32).to(device)

    def add_experience(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool, priority: float = 1.0) -> None:
        if self.state_dtype == torch.uint8:
            state = state.detach().clone().mul(255.0).to(torch.uint8)
            next_state = next_state.detach().clone().mul(255.0).to(torch.uint8)

        self.state_batch[self.next_index] = state.to(self.device)
        self.action_batch[self.next_index] = torch.tensor([action], dtype=torch.long).to(self.device)
        self.reward_batch[self.next_index] = torch.tensor([reward], dtype=torch.float32).to(self.device)
        self.next_state_batch[self.next_index] = next_state.to(self.device)
        self.episode_terminated_batch[self.next_index] = torch.tensor([episode_terminated], dtype=torch.float32).to(self.device)

        delta: Tensor = priority**self.alpha - self.priorities[self.next_index]
        self.priorities[self.next_index] = priority**self.alpha
        self.prob_weights_sum += delta

        if self.current_size < self.max_size:
            self.current_size += 1
        self.next_index = (self.next_index + 1) % self.max_size
    
    def can_get_batch(self) -> bool:
        return self.current_size >= self.batch_size
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # TODO: May be too much exploration with vpi.
        indexes: Tensor = self.priorities.multinomial(self.batch_size, replacement=True)
        self.last_batch_indexes = indexes
        self.last_batch_prob_weights = self.priorities[indexes]

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
        
        self.priorities: Tensor = torch.zeros(self.max_size, dtype=torch.float32).to(self.device)
        self.current_size = 0
        self.next_index = 0

        self.last_batch_indexes: Tensor = None
        self.last_batch_prob_weights: Tensor = None
        self.prob_weights_sum: Tensor = torch.zeros(1, dtype=torch.float32).to(self.device)
    
    def update_priorities(self, priorities: Tensor) -> None:
        for i in range(len(priorities)):
            index = self.last_batch_indexes[i]
            delta: Tensor = priorities[i]**self.alpha - self.priorities[index]
            self.priorities[index] = priorities[i]**self.alpha
            self.prob_weights_sum += delta

    def get_last_batch_weights(self) -> Tensor:
        return (self.current_size * self.last_batch_prob_weights / self.prob_weights_sum)**(-self.beta)












class PrioritizedExperienceManager2(ExperienceManagerInterface):
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
        
        self.sum_tree: SumTree = SumTree(max_size)
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
        return self.current_size >= self.batch_size
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indexes: Tensor = torch.tensor(self.sum_tree.batch_sample(self.batch_size), dtype=torch.int32).to(self.device)
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
        
        self.sum_tree: SumTree = SumTree(self.max_size)
        self.current_size = 0
        self.next_index = 0