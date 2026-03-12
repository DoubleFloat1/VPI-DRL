import torch
from torch import Tensor
from typing import List, Tuple
import numpy as np
from numpy import ndarray

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









class Priority:
    def __init__(self, max_size: int, vpi_batch_size: int, rand_batch_size: int, device: torch.device):
        self.max_size: int = max_size
        self.vpi_batch_size: int = vpi_batch_size
        self.rand_batch_size: int = rand_batch_size
        self.batch_size: int = vpi_batch_size + rand_batch_size
        self.device: torch.device = device
    
    def add_new_priority(self, priority: float) -> int:
        raise NotImplementedError

    def can_get_batch(self) -> bool:
        raise NotImplementedError

    def get_batch_indexes(self) -> Tensor:
        raise NotImplementedError

    def update_last_batch_priorities(self, priorities: Tensor) -> None:
        raise NotImplementedError
        
    def get_last_batch_weights(self) -> Tensor:
        raise NotImplementedError






class StandardPriority(Priority):
    def __init__(self, max_size: int, vpi_batch_size: int, rand_batch_size: int, device: torch.device):
        super().__init__(max_size, vpi_batch_size, rand_batch_size, device)

        self.priorities: Tensor = torch.zeros(max_size, dtype=torch.float32)
        self.current_size: int = 0
        self.next_index: int = 0

        self.last_batch_indexes: Tensor = -torch.ones(self.batch_size, dtype=torch.int32)
        self.last_batch_priorities: Tensor = torch.zeros(self.batch_size, dtype=torch.float32)
        self.priority_sum: Tensor = torch.zeros(1, dtype=torch.float32)
    
    def add_new_priority(self, priority: float) -> int:
        if self.current_size < self.max_size:
            self.priorities[self.current_size] = priority
            self.current_size += 1
            self.priority_sum += priority
            return self.current_size - 1
        else:
            replace_index: int = self.next_index
            self.next_index = (self.next_index + 1) % self.max_size

            delta: float = priority - self.priorities[replace_index]
            self.priorities[replace_index] = priority
            self.priority_sum += delta
            return replace_index

    def can_get_batch(self) -> bool:
        return self.current_size >= self.batch_size * 100

    def get_batch_indexes(self) -> Tensor:
        indexes1: Tensor = self.priorities.multinomial(self.vpi_batch_size, replacement=True)
        indexes2: Tensor = torch.tensor(np.random.randint(low=0, high=self.current_size, size=self.rand_batch_size), dtype=torch.int32)
        indexes: Tensor = torch.cat([indexes1, indexes2])

        self.last_batch_indexes = indexes
        self.last_batch_priorities = self.priorities[indexes]
        return indexes.to(self.device)

    def update_last_batch_priorities(self, priorities: Tensor) -> None:
        priorities = priorities.cpu()
        for i in range(len(priorities)):
            index: Tensor = self.last_batch_indexes[i]
            priority: Tensor = priorities[i]
            self._update_priority(index, priority)
        
    def get_last_batch_weights(self) -> Tensor:
        return (self.current_size * (self.last_batch_priorities / self.priority_sum)).to(self.device)

    def _update_priority(self, index: int, priority: float) -> None:
        delta: float = priority - self.priorities[index]
        self.priorities[index] = priority
        self.priority_sum += delta



class MinHeapPriority(Priority):
    def __init__(self, max_size: int, vpi_batch_size: int, rand_batch_size: int, device: torch.device):
        super().__init__(max_size, vpi_batch_size, rand_batch_size, device)

        self.priorities: Tensor = torch.zeros(max_size+1, dtype=torch.float32)
        self.exp_to_heap: Tensor = -np.ones(max_size, dtype=np.int32)
        self.heap_to_exp: Tensor = -np.ones(max_size+1, dtype=np.int32)
        self.current_size: int = 0

        self.last_batch_indexes: Tensor = -torch.ones(self.batch_size, dtype=torch.int32)
        self.last_batch_priorities: Tensor = torch.zeros(self.batch_size, dtype=torch.float32)
        self.priority_sum: Tensor = torch.zeros(1, dtype=torch.float32)
    
    def add_new_priority(self, priority: float) -> int:
        if self.current_size < self.max_size:
            self.priorities[self.current_size + 1] = priority
            self.exp_to_heap[self.current_size] = self.current_size + 1
            self.heap_to_exp[self.current_size + 1] = self.current_size

            self._float(self.current_size + 1)
            self.current_size += 1
            self.priority_sum += priority
            return self.current_size - 1
        else:
            if priority < self.priorities[1]:
                return -1
            
            exp_replaced_index: int = self.heap_to_exp[1]
            delta: float = priority - self.priorities[1]
            self.priorities[1] = priority
            self._sink(1)
            self.priority_sum += delta
            return exp_replaced_index

    def can_get_batch(self) -> bool:
        return self.current_size >= self.batch_size * 100

    def get_batch_indexes(self) -> Tensor:
        heap_indexes: Tensor = self.priorities.multinomial(self.vpi_batch_size, replacement=True)
        exp_indexes1: Tensor = torch.tensor(self.heap_to_exp[heap_indexes], dtype=torch.int32)
        exp_indexes2: Tensor = torch.tensor(np.random.randint(low=0, high=self.current_size, size=self.rand_batch_size), dtype=torch.int32)
        indexes: Tensor = torch.cat([exp_indexes1, exp_indexes2])

        self.last_batch_indexes = indexes
        self.last_batch_priorities = self.priorities[heap_indexes]
        return indexes.to(self.device)

    def update_last_batch_priorities(self, priorities: Tensor) -> None:
        priorities = priorities.cpu()
        for i in range(len(priorities)):
            exp_index: Tensor = self.last_batch_indexes[i]
            priority: Tensor = priorities[i]
            self._update_priority(exp_index, priority)
        
    def get_last_batch_weights(self) -> Tensor:
        return (self.current_size * (self.last_batch_priorities / self.priority_sum)).to(self.device)
    
    def _float(self, heap_index: int) -> None:
        while heap_index > 1 and self.priorities[heap_index // 2] > self.priorities[heap_index]:
            self._swap_priorities(heap_index, heap_index // 2)

    def _sink(self, heap_index: int) -> None:
        sink_to: int = heap_index
        while sink_to != -1:
            sink_to = -1
            min_value: float = self.priorities[heap_index]
            if heap_index * 2 <= self.current_size and self.priorities[heap_index * 2] < min_value:
                sink_to = heap_index * 2
                min_value = self.priorities[heap_index * 2]
            if heap_index * 2 + 1 <= self.current_size and self.priorities[heap_index * 2 + 1] < min_value:
                sink_to = heap_index * 2 + 1
                min_value = self.priorities[heap_index * 2] + 1
            
            if sink_to != -1:
                self._swap_priorities(heap_index, sink_to)
                heap_index = sink_to
    
    def _swap_priorities(self, heap_index1, heap_index2) -> None:
        self._swap_tensor_indexes(self.priorities, heap_index1, heap_index2)

        exp_index1: int = self.heap_to_exp[heap_index1]
        exp_index2: int = self.heap_to_exp[heap_index2]
        self.exp_to_heap[exp_index1], self.exp_to_heap[exp_index2] = self.exp_to_heap[exp_index2], self.exp_to_heap[exp_index1]
        self.heap_to_exp[heap_index1], self.heap_to_exp[heap_index2] = self.heap_to_exp[heap_index2], self.heap_to_exp[heap_index1]
    
    def _swap_tensor_indexes(self, tensor: Tensor, index1: int, index2: int) -> None:
        temp = tensor[index1].detach().clone()
        tensor[index1] = tensor[index2]
        tensor[index2] = temp

    def _update_priority(self, exp_index: int, priority: float) -> None:
        heap_index: int = self.exp_to_heap[exp_index]
        delta: float = priority - self.priorities[heap_index]
        self.priorities[heap_index] = priority
        if delta > 0:
            self._sink(heap_index)
        elif delta < 0:
            self._float(heap_index)
        
        self.priority_sum += delta





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
        
        self.min_priority_heap: Priority = StandardPriority(max_size, vpi_batch_size, rand_batch_size, device)

    def add_experience(self, state: Tensor, action: int, reward: float, next_state: Tensor, episode_terminated: bool, priority: float = 1.0) -> None:
        if self.state_dtype == torch.uint8:
            state = state.detach().clone().mul(255.0).to(torch.uint8)
            next_state = next_state.detach().clone().mul(255.0).to(torch.uint8)

        exp_index: int = self.min_priority_heap.add_new_priority(priority**self.alpha)
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
        return self.min_priority_heap.can_get_batch()
    
    def get_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indexes: Tensor = self.min_priority_heap.get_batch_indexes()

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
        self.min_priority_heap.update_last_batch_priorities(priorities**self.alpha)

    def get_last_batch_weights(self) -> Tensor:
        return self.min_priority_heap.get_last_batch_weights()**(-self.beta)

