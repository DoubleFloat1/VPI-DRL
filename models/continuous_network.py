from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

class PolyakModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.model: nn.Module
	
	def polyak_update(self, other: PolyakModel, polyak: float) -> None:
		with torch.no_grad():
			self_modules: List[Tuple[str, Parameter]] = [m for m in self.named_parameters()]
			other_modules:  List[Tuple[str, Parameter]] = [m for m in other.named_parameters()]
			for i in range(len(self_modules)):
				self_params: Parameter = self_modules[i][1]
				other_params: Parameter = other_modules[i][1]
				self_params.copy_(polyak * self_params + (1.0 - polyak) * other_params)

class ContinuousQValueModel(PolyakModel):
	def __init__(self, state_size: List[int], actions_size: int):
		super().__init__()
		hidden_layers_size: int = 256
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0] + actions_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, 1)
			)
		elif len(state_size) == 3:
			raise NotImplementedError
	
	def forward(self, x, y):
		z = torch.cat([x, y], dim=-1)
		return self.model(z)

class ContinuousPolicyModel(PolyakModel):
	def __init__(self, state_size: List[int], actions_size: int):
		super().__init__()
		hidden_layers_size: int = 256
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0], hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, actions_size),
				nn.Sigmoid()
			)
		elif len(state_size) == 3:
			raise NotImplementedError
	
	def forward(self, x):
		return self.model(x)

class SACPolicyModel(nn.Module):
	def __init__(self, state_size: List[int], actions_size: int):
		super().__init__()
		hidden_layers_size: int = 256
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0], hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, actions_size * 2),
			)
		elif len(state_size) == 3:
			raise NotImplementedError
	
	def forward(self, x):
		mu, log_sigma = torch.chunk(self.model(x), chunks=2, dim=-1)
		return mu, torch.exp(log_sigma)


class VPISACQValueModel(PolyakModel):
	def __init__(self, state_size: List[int], actions_size: int):
		super().__init__()
		hidden_layers_size: int = 256
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0] + actions_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, 2),
			)
		elif len(state_size) == 3:
			raise NotImplementedError
	
	def forward(self, x, y):
		z = torch.cat([x, y], dim=-1)
		mu, log_sigma = torch.chunk(self.model(z), chunks=2, dim=-1)
		return mu, torch.exp(log_sigma)


class VPIDistributionModel(nn.Module):
	def __init__(self, state_size: List[int], actions_size: int, output_norms_amount: int = 1):
		super().__init__()
		self.output_norms_amount: int = output_norms_amount
		hidden_layers_size: int = 256
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0], hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, 2 * actions_size * output_norms_amount),
			)
		elif len(state_size) == 3:
			raise NotImplementedError
	
	def forward(self, x):
		mu, log_sigma = torch.chunk(self.model(x), chunks=2, dim=-1)
		weights: Tensor = mu.new_ones(mu.shape[:-1] + (self.output_norms_amount,))
		weights = weights / self.output_norms_amount
		return mu, torch.exp(log_sigma), weights

class VPIDistributionModel2(nn.Module):
	def __init__(self, state_size: List[int], actions_size: int, output_norms_amount: int = 1):
		super().__init__()
		self.output_normal_part_size: int = 2 * actions_size * output_norms_amount
		self.output_norms_amount: int = output_norms_amount
		hidden_layers_size: int = 256
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0], hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, hidden_layers_size),
				nn.ReLU(),
				nn.Linear(hidden_layers_size, 2 * actions_size * output_norms_amount + output_norms_amount),
			)
		elif len(state_size) == 3:
			raise NotImplementedError
	
	def forward(self, x):
		output: Tensor = self.model(x)
		mu_and_log_sigma: Tensor = output[..., :self.output_normal_part_size]
		mu, log_sigma = torch.chunk(mu_and_log_sigma, chunks=2, dim=-1)
		weights: Tensor = output[..., self.output_normal_part_size:]
		return mu, torch.exp(log_sigma), torch.softmax(weights, dim=-1)