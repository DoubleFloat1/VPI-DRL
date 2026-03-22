import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

class ValueModel(nn.Module):
	def __init__(self, state_size: List[int], actions_amount: int):
		super().__init__()
		if len(state_size) == 1:
			self.model = nn.Sequential(
				nn.Linear(state_size[0], 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, actions_amount)
			)
		elif len(state_size) == 3:
			height: int = state_size[1]
			width: int = state_size[2]
			height = height // 4 - 1
			width = width // 4 - 1
			height = height // 2 - 1
			width = width // 2 - 1
			height = height - 2
			width = width - 2

			flatten_size: int = 64 * height * width

			self.model = nn.Sequential(
				nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Flatten(start_dim=-3),
				nn.Linear(flatten_size, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, actions_amount)
			)
	
	def forward(self, x):
		return self.model(x)

class BayesLinear(nn.Module):
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""
	
	def __init__(self, in_features: int, out_features: int, prior_weight_mu: float = 0.0, prior_weight_sigma: float = 0.01, 
			     prior_bias_mu: float = 0.0, prior_bias_sigma: float = 0.01, bias: bool = True):
		super(BayesLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
        
		self.prior_weight_mu: Tensor = prior_weight_mu * torch.ones(out_features, in_features)
		self.prior_weight_log_sigma: Tensor = np.log(prior_weight_sigma) * torch.ones(out_features, in_features)

		self.weight_mu: Parameter = Parameter(prior_weight_mu * torch.ones(out_features, in_features))
		self.weight_log_sigma: Parameter = Parameter(np.log(prior_weight_sigma) * torch.ones(out_features, in_features))
		self.register_buffer('weight_eps', None)
                
		if bias is None or bias is False :
			self.bias = False
		else :
			self.bias = True
            
		if self.bias:
			self.prior_bias_mu: Tensor = prior_bias_mu * torch.ones(out_features)
			self.prior_bias_log_sigma: Tensor = np.log(prior_bias_sigma) * torch.ones(out_features)

			self.bias_mu = Parameter(prior_bias_mu * torch.ones(out_features))
			self.bias_log_sigma = Parameter(np.log(prior_bias_sigma) * torch.ones(out_features))
			self.register_buffer('bias_eps', None)
		else:
			self.register_parameter('bias_mu', None)
			self.register_parameter('bias_log_sigma', None)
			self.register_buffer('bias_eps', None)
    
	def freeze(self):
		self.weight_eps = torch.randn_like(self.weight_log_sigma)
		if self.bias:
			self.bias_eps = torch.randn_like(self.bias_log_sigma)
        
	def unfreeze(self):
		self.weight_eps = None
		if self.bias:
			self.bias_eps = None

	def kl_loss(self) -> Tensor:
		weight_kl: Tensor = self.prior_weight_log_sigma - self.weight_log_sigma + (torch.exp(self.weight_log_sigma)**2 + (self.weight_mu - self.prior_weight_mu)**2) / (2 * torch.exp(self.prior_weight_log_sigma)**2) - 0.5
		kl: Tensor = weight_kl.sum()
		if self.bias:
			bias_kl: Tensor = self.prior_bias_log_sigma - self.bias_log_sigma + (torch.exp(self.bias_log_sigma)**2 + (self.bias_mu - self.prior_bias_mu)**2) / (2 * torch.exp(self.prior_bias_log_sigma)**2) - 0.5
			kl += bias_kl.sum()
		return kl
	
	def distributions_amount(self) -> int:
		amount: int = len(self.weight_mu.view(-1))
		if self.bias:
			amount += len(self.bias_mu.view(-1))
		return amount
	
	def prior_to_device(self, device: torch.device) -> None:
		self.prior_weight_mu = self.prior_weight_mu.to(device)
		self.prior_weight_log_sigma = self.prior_weight_log_sigma.to(device)
		if self.bias:
			self.prior_bias_mu = self.prior_bias_mu.to(device)
			self.prior_bias_log_sigma = self.prior_bias_log_sigma.to(device)

	def pass_posterior_to_prior(self) -> None:
		self.prior_weight_mu = self.weight_mu.detach().clone()
		self.prior_weight_log_sigma = self.weight_log_sigma.detach().clone()
		if self.bias:
			self.prior_bias_mu = self.bias_mu.detach().clone()
			self.prior_bias_log_sigma = self.bias_log_sigma.detach().clone()

	def forward(self, input):
		if self.weight_eps is None:
			weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
		else:
			weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
		if self.bias:
			if self.bias_eps is None:
				bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
			else:
				bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps                
		else:
			bias = None
            
		return F.linear(input, weight, bias)
	
	def get_output_mean_and_std(self, input: Tensor) -> Tuple[Tensor, Tensor]:
		bias_mu: Tensor = self.bias_mu if self.bias else None
		bias_var: Tensor = torch.exp(self.bias_log_sigma)**2 if self.bias else None

		mu: Tensor = F.linear(input, self.weight_mu, bias_mu)
		var: Tensor = F.linear(input**2, torch.exp(self.weight_log_sigma)**2, bias_var)
		return mu, torch.sqrt(var)



class BayesModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.device: torch.device = "cpu"

	def forward(self, x: Tensor) -> Tensor:
		raise NotImplementedError
	
	def kl_loss(self) -> Tensor:
		kl_sum: Tensor = torch.tensor([0.0], dtype=torch.float32).to(self.device)
		n: int = 0
		for m in self.modules():
			if isinstance(m, BayesLinear):
				kl_sum += m.kl_loss()
				n += m.distributions_amount()
		
		kl_sum = kl_sum.squeeze()
		return kl_sum / n
	
	def prior_to_device(self, device: torch.device) -> None:
		self.device = device
		for m in self.modules():
			if isinstance(m, BayesLinear):
				m.prior_to_device(device)

	def pass_posterior_to_prior(self) -> None:
		for m in self.modules():
			if isinstance(m, BayesLinear):
				m.pass_posterior_to_prior()



class BayesModelUniform(BayesModel):
	def __init__(self):
		super().__init__()
		self.range1 = None
		self.range2 = None
	
	def forward(self, x: Tensor) -> Tensor:
		r1 = self.range1(x)
		r2 = self.range2(x)
		y: Tensor = self.reparameterize(r1, r2)
		return y, r1, r2
	
	def reparameterize(self, r1: Tensor, r2: Tensor) -> Tensor:
		return torch.rand_like(r1) * (r2 - r1) + r1



class BayesValueModelUniform(BayesModelUniform):
	def __init__(self, state_size: List[int], actions_amount: int):
		super().__init__()

		if len(state_size) == 1:
			self.range1 = nn.Sequential(
				nn.Linear(state_size[0], 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				#nn.Linear(64, out_features)
				BayesLinear(512, actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
			)

			self.range2 = nn.Sequential(
				nn.Linear(state_size[0], 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				#nn.Linear(64, out_features)
				BayesLinear(512, actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
			)
		else:
			height: int = state_size[1]
			width: int = state_size[2]
			height = height // 4 - 1
			width = width // 4 - 1
			height = height // 2 - 1
			width = width // 2 - 1
			height = height - 2
			width = width - 2

			flatten_size: int = 64 * height * width

			self.range1 = nn.Sequential(
				nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Flatten(start_dim=-3),
				nn.Linear(flatten_size, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				BayesLinear(512, actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
			)

			self.range2 = nn.Sequential(
				nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Flatten(start_dim=-3),
				nn.Linear(flatten_size, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				BayesLinear(512, actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
			)




class BayesModelNormal(BayesModel):
	def __init__(self):
		super().__init__()
		self.mu_log_sigma_model = None
		# TODO: Is there a way to make a loss function take more into account the variance of the data?
		# Works well for small variance. It may be best to assume we know a upper and lower bound to normalize the data to [0, 1] range
		# It seems necessary to use the pass_posterior_to_prior function every once in a while

	def forward(self, x: Tensor) -> Tensor:
		mu_log_sigma: Tensor
		mu_log_sigma = self.mu_log_sigma_model(x)
		actions_amount = mu_log_sigma.shape[-1] // 2

		mu, log_sigma = torch.split(mu_log_sigma, actions_amount, dim=-1)
		y: Tensor = self.reparameterize(mu, log_sigma)
		return y, mu, torch.exp(log_sigma)
	
	def reparameterize(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
		return mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)


class BayesValueModelNormal(BayesModelNormal):
	def __init__(self, state_size: List[int], actions_amount: int):
		super().__init__()

		if len(state_size) == 1:
			self.mu_log_sigma_model = nn.Sequential(
				nn.Linear(state_size[0], 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				#nn.Linear(64, out_features)
				BayesLinear(512, 2 * actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
			)
		else:
			height: int = state_size[1]
			width: int = state_size[2]
			height = height // 4 - 1
			width = width // 4 - 1
			height = height // 2 - 1
			width = width // 2 - 1
			height = height - 2
			width = width - 2

			flatten_size: int = 64 * height * width

			self.mu_log_sigma_model = nn.Sequential(
				nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Flatten(start_dim=-3),
				nn.Linear(flatten_size, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				BayesLinear(512, 2 * actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
			)


# EXPERIMENTAL MODULES

class BayesModelPure(BayesModel):
	def __init__(self):
		super().__init__()
		self.pre_bayes_model: nn.Sequential = None
		self.bayes_linear: BayesLinear = None

	def forward(self, x: Tensor) -> Tensor:
		z: Tensor = self.pre_bayes_model(x)
		y: Tensor = self.bayes_linear(z)
		mu, sigma = self.bayes_linear.get_output_mean_and_std(z)
		return y, mu, sigma
	
class BayesValueModelPure(BayesModelPure):
	def __init__(self, state_size: List[int], actions_amount: int):
		super().__init__()
		self.bayes_linear = BayesLinear(512, actions_amount, prior_weight_sigma=0.01, prior_bias_sigma=0.01)

		if len(state_size) == 1:
			self.pre_bayes_model = nn.Sequential(
				nn.Linear(state_size[0], 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
			)
		else:
			height: int = state_size[1]
			width: int = state_size[2]
			height = height // 4 - 1
			width = width // 4 - 1
			height = height // 2 - 1
			width = width // 2 - 1
			height = height - 2
			width = width - 2

			flatten_size: int = 64 * height * width

			self.pre_bayes_model = nn.Sequential(
				nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Flatten(start_dim=-3),
				nn.Linear(flatten_size, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
			)