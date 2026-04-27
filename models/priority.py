import torch
from torch import Tensor
import numpy as np
from numpy import ndarray

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
    def __init__(self, max_size: int, vpi_batch_size: int, rand_batch_size: int, adds_per_update: int, device: torch.device):
        super().__init__(max_size, vpi_batch_size, rand_batch_size, device)

        self.priorities: Tensor = torch.zeros(max_size, dtype=torch.float32)
        self.current_size: int = 0
        self.next_index: int = 0

        self.last_batch_indexes: Tensor = -torch.ones(self.batch_size, dtype=torch.int32)
        self.last_batch_priorities: Tensor = torch.zeros(self.batch_size, dtype=torch.float32)
        self.priority_sum: Tensor = torch.zeros(1, dtype=torch.float32)

        self.adds_per_update: int = adds_per_update
        self.priorities_added: int = 0
    
    def add_new_priority(self, priority: float) -> int:
        self.priorities_added += 1
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
        return ((self.current_size >= self.batch_size * 100) or (self.current_size == self.max_size)) and (self.priorities_added % self.adds_per_update == 0)

    def get_batch_indexes(self) -> Tensor:
        indexes1: Tensor = self.priorities.multinomial(self.vpi_batch_size, replacement=True)
        indexes2: Tensor = torch.tensor(np.random.randint(low=0, high=self.current_size, size=self.rand_batch_size), dtype=torch.int32)
        indexes: Tensor = torch.cat([indexes1, indexes2])

        self.last_batch_indexes = indexes
        self.last_batch_priorities = self.priorities[indexes]
        return indexes.to(self.device)

    def update_last_batch_priorities(self, priorities: Tensor) -> None:
        self._update_indexes_priorities(self.last_batch_indexes, priorities)
    
    def _update_indexes_priorities(self, indexes: Tensor, priorities: Tensor) -> None:
        priorities = priorities.cpu()
        for i in range(len(priorities)):
            index: Tensor = indexes[i]
            priority: Tensor = priorities[i]
            self._update_priority(index, priority)
        
    def get_last_batch_weights(self) -> Tensor:
        return (self.current_size * (self.last_batch_priorities / self.priority_sum)).to(self.device)

    def _update_priority(self, index: int, priority: float) -> None:
        delta: float = priority - self.priorities[index]
        self.priorities[index] = priority
        self.priority_sum += delta



class MinHeapPriority(Priority):
    def __init__(self, max_size: int, vpi_batch_size: int, rand_batch_size: int, adds_per_update: int, device: torch.device):
        super().__init__(max_size, vpi_batch_size, rand_batch_size, device)

        self.priorities: Tensor = torch.zeros(max_size+1, dtype=torch.float32)
        self.exp_to_heap: ndarray[int] = -np.ones(max_size, dtype=np.int32)
        self.heap_to_exp: ndarray[int] = -np.ones(max_size+1, dtype=np.int32)
        self.current_size: int = 0

        self.last_batch_exp_indexes: Tensor = -torch.ones(self.batch_size, dtype=torch.int32)
        self.last_batch_priorities: Tensor = torch.zeros(self.batch_size, dtype=torch.float32)
        self.priority_sum: Tensor = torch.zeros(1, dtype=torch.float32)

        self.adds_per_update: int = adds_per_update
        self.added_previous_priority: bool = True
        self.priorities_added: int = 0
        self.debug: int = 0
        self.debug2: int = 0
    
    def add_new_priority(self, priority: float) -> int:
        self.debug2 += 1
        if self.debug2 % 10000 == 9999:
            print("skipped experiences:", self.debug)

        if self.current_size < self.max_size:
            self.priorities[self.current_size + 1] = priority
            self.exp_to_heap[self.current_size] = self.current_size + 1
            self.heap_to_exp[self.current_size + 1] = self.current_size

            self._float(self.current_size + 1)
            self.current_size += 1
            self.priority_sum += priority

            self.priorities_added += 1
            return self.current_size - 1
        else:
            if priority < self.priorities[1]:
                self.added_previous_priority = False
                self.debug += 1
                return -1
            else:
                self.priorities_added += 1
                self.added_previous_priority = True
            
            exp_replaced_index: int = self.heap_to_exp[1]
            delta: float = priority - self.priorities[1]
            self.priorities[1] = priority
            self._sink(1)
            self.priority_sum += delta
            return exp_replaced_index

    def can_get_batch(self) -> bool:
        size_flag: bool = (self.current_size >= self.batch_size * 100) or (self.current_size == self.max_size)
        previous_priority_flag: bool = (self.priorities_added % self.adds_per_update == 0) and self.added_previous_priority
        return size_flag and previous_priority_flag

    def get_batch_indexes(self) -> Tensor:
        heap_indexes: Tensor = self.priorities.multinomial(self.vpi_batch_size, replacement=True)
        exp_indexes1: Tensor = torch.tensor(self.heap_to_exp[heap_indexes], dtype=torch.int32)
        exp_indexes2: Tensor = torch.tensor(np.random.randint(low=0, high=self.current_size, size=self.rand_batch_size), dtype=torch.int32)
        indexes: Tensor = torch.cat([exp_indexes1, exp_indexes2])

        self.last_batch_exp_indexes = indexes
        self.last_batch_priorities = self.priorities[heap_indexes]
        return indexes.to(self.device)

    def update_last_batch_priorities(self, priorities: Tensor) -> None:
        self._update_indexes_priorities(self.last_batch_exp_indexes, priorities)

    def _update_indexes_priorities(self, exp_indexes: Tensor, priorities: Tensor) -> None:
        priorities = priorities.cpu()
        for i in range(len(priorities)):
            exp_index: Tensor = exp_indexes[i]
            priority: Tensor = priorities[i]
            self._update_priority(exp_index, priority)
        
    def get_last_batch_weights(self) -> Tensor:
        return (self.current_size * (self.last_batch_priorities / self.priority_sum)).to(self.device)
    
    def _float(self, heap_index: int) -> None:
        while heap_index > 1 and self.priorities[heap_index // 2] > self.priorities[heap_index]:
            self._swap_priorities(heap_index, heap_index // 2)
            heap_index = heap_index // 2

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