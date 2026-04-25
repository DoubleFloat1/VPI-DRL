from __future__ import annotations

import torch
from torch import Tensor
from models.network import ActorNetwork, CriticNetwork
from typing import List, Tuple, Dict, Any, TypeVar, Generic
from torch.optim import Optimizer, Adam
import numpy as np
from models.rl_model import RLModel
from models.params import A2CParams
import copy


T = TypeVar("T")
class Node(Generic[T]):
    def __init__(self, content: T):
        self.content = content
        self.next: Node = None

class History(Generic[T]):
    def __init__(self, size_limit: int):
        self.size_limit: int = size_limit
        self.size: int = 0
        self.start: Node = None
        self.end: Node = None
    
    def __len__(self) -> int:
        return self.size
    
    def insert(self, element: T) -> None:
        new_node: Node = Node(element)
        if self.size == 0:
            self.start = new_node
            self.end = new_node
            self.size = 1
        elif self.size < self.size_limit:
            self.end.next = new_node
            self.end = new_node
            self.size += 1
        else:
            self.end.next = new_node
            self.end = new_node
            self.start = self.start.next
    
    def get_first_content(self) -> T:
        return self.start.content
    
    def get_all_content(self) -> List[T]:
        contents = []
        node: Node = self.start
        while node != None:
            contents.append(node.content)
            node = node.next
        return contents

class EnvironmentHist:
    def __init__(self, n_step_return: int):
        self.n_step_return: int = n_step_return
        self.reward_hist: History[float] = History[float](n_step_return)
        self.state_hist: History[Tensor] = History[Tensor](n_step_return)
        self.action_hist: History[int] = History[int](n_step_return)
    
    def reset(self):
        self.reward_hist = History[float](self.n_step_return)
        self.state_hist = History[Tensor](self.n_step_return)
        self.action_hist = History[int](self.n_step_return)
    
    def insert(self, state_tensor: Tensor, action: int, reward: float):
        self.state_hist.insert(state_tensor)
        self.action_hist.insert(action)
        self.reward_hist.insert(reward)
    
    def is_full(self) -> bool:
        return len(self.state_hist) == self.n_step_return
    
    def all_insert_empty(self):
        self.state_hist.insert(None)
        self.reward_hist.insert(0.0)
        self.action_hist.insert(None)

class StepBatch:
    def __init__(self, max_size: int, state_size: List[int], device: torch.device):
        self.device: torch.device = device
        self.max_batch_size = max_size
        self.current_batch_size = 0

        self.state_batch_dimensions: List[int] = [max_size] + state_size
        self.reference_state_batch: List[Tensor] = torch.zeros(self.state_batch_dimensions, dtype=torch.float32).to(device)
        self.td_target_batch: List[Tensor] = torch.zeros(max_size, 1, dtype=torch.float32).to(device)
        self.chosen_action_batch: List[Tensor] = torch.zeros(max_size, 1, dtype=torch.int32).to(device)
    
    def add(self, reference_state: Tensor, reference_action: int, td_target: Tensor) -> None:
        self.reference_state_batch[self.current_batch_size] = reference_state.to(self.device)
        self.td_target_batch[self.current_batch_size] = td_target.to(self.device)
        self.chosen_action_batch[self.current_batch_size] = reference_action
        self.current_batch_size += 1
    
    def is_batch_full(self) -> bool:
        return self.current_batch_size == self.max_batch_size
    
    def get_states_tensor(self) -> Tensor:
        return self.reference_state_batch
    
    def get_td_targets_tensor(self) -> Tensor:
        return self.td_target_batch
    
    def get_actions_tensor(self) -> Tensor:
        return self.chosen_action_batch
    
    def empty_batch(self) -> None:
        self.current_batch_size = 0


class A2C(RLModel):
    def __init__(self, state_size: List[int], actions_amount: int, params: A2CParams):
        super().__init__(state_size, actions_amount, params.gamma)

        self.params: A2CParams = params
        self.actor: ActorNetwork = ActorNetwork(state_size, actions_amount).to(self.device)
        self.actor_optim: Optimizer = Adam(self.actor.parameters(), lr=params.actor_lr)
        self.critic: CriticNetwork = CriticNetwork(state_size).to(self.device)
        self.critic_optim: Optimizer = Adam(self.critic.parameters(), lr=params.critic_lr)

        self.environments_hist: List[EnvironmentHist] = [EnvironmentHist(params.n_step_return) for _ in range(params.envs_amount)]
        self.batch: StepBatch = StepBatch(params.batch_size, state_size, self.device)
        self.loss_function: torch.nn = torch.nn.MSELoss()
        
    
    def get_next_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = state.to(self.device)
            action_probs: Tensor = self.actor(state)
            return action_probs.multinomial(1).item()
    
    def inference_next_action(self, state: Tensor) -> int:
        with torch.no_grad():
            state = state.to(self.device)
            action_probs: Tensor = self.actor(state)
            return action_probs.multinomial(1).item()
    
    def improve(self, state: Tensor, action: int, reward: float, next_state: Tensor, terminated: bool, truncated: bool, env_id: int = 0) -> None:
        env_hist: EnvironmentHist = self.environments_hist[env_id]
        env_hist.insert(state, action, reward)

        if terminated:
            self.episode_end_improve(env_id)
            return
        
        if not env_hist.is_full():
            return
        
        reference_state: Tensor = env_hist.state_hist.get_first_content()
        reference_action: int = env_hist.action_hist.get_first_content()

        with torch.no_grad():
            next_state_value: Tensor = self.critic(next_state.to(self.device))
            rewards: List[float] = env_hist.reward_hist.get_all_content()
            reward_sum: float = sum([(self.params.gamma**i) * rewards[i] for i in range(self.params.n_step_return)])
            td_target: Tensor = reward_sum + (self.gamma**self.params.n_step_return) * next_state_value

        self.add_to_batch(reference_state, reference_action, td_target)

        if truncated:
            env_hist.reset()
        

    def episode_end_improve(self, env_id: int) -> None:
        env_hist: EnvironmentHist = self.environments_hist[env_id]
        hist_len: int = len(env_hist.reward_hist)
        for i in range(hist_len):
            reference_state: Tensor = env_hist.state_hist.get_first_content()
            reference_action: int = env_hist.action_hist.get_first_content()

            rewards: List[float] = env_hist.reward_hist.get_all_content()
            reward_sum: float = sum([(self.gamma**k) * rewards[k] for k in range(hist_len - i)])
            td_target: Tensor = torch.tensor([reward_sum], dtype=torch.float32)

            self.add_to_batch(reference_state, reference_action, td_target)
            env_hist.all_insert_empty()

        env_hist.reset()

    def add_to_batch(self, reference_state: Tensor, reference_action: int, td_target: Tensor) -> None:
        self.batch.add(reference_state, reference_action, td_target)
        if self.batch.is_batch_full():
            self.improve_models()

    def improve_models(self):
        states: Tensor = self.batch.get_states_tensor()
        actions: Tensor = self.batch.get_actions_tensor()
        td_targets: Tensor = self.batch.get_td_targets_tensor()
        self.batch.empty_batch()

        predicted_values: Tensor = self.critic(states)
        advantage: Tensor = td_targets - predicted_values

        critic_loss: Tensor = self.loss_function(predicted_values, td_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        predicted_action_probs: Tensor = self.actor(states)
        selected_action_probs: Tensor = predicted_action_probs.gather(dim=-1, index=actions)
        log_probs = torch.log(selected_action_probs)
        actor_loss: Tensor = torch.mean(-log_probs * advantage.detach())
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    
    def get_params_dict(self) -> Dict[str, Any]:
        return self.params.to_dict()
    
    def new(self) -> RLModel:
        return A2C(self.state_size, self.actions_amount, self.params)
    
    def save(self) -> None:
        torch.save(self.actor.state_dict(), "./saved_models/a2c_actor.pth")
        torch.save(self.critic.state_dict(), "./saved_models/a2c_critic.pth")
    
    def load(self, path: str) -> None:
        self.actor.load_state_dict(torch.load(path, weights_only=True))