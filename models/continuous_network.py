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
		hidden_layers_size: int = 1024
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
		hidden_layers_size: int = 1024
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