import numpy as np
import math
import random

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchbnn as bnn
from torch import Tensor
from scipy.stats import norm

class ValueModel(nn.Module):
	def __init__(self, state_size: int, actions_amount: int):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, actions_amount)
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
		self.prior_weight_mu = self.weight_mu.clone().detach()
		self.prior_weight_log_sigma = self.weight_log_sigma.clone().detach()
		if self.bias:
			self.prior_bias_mu = self.bias_mu.clone().detach()
			self.prior_bias_log_sigma = self.bias_log_sigma.clone().detach()

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

class BayesModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.mu_model = None
		self.log_sigma_model = None
		self.range_min: float = 0.0
		self.range_max: float = 1.0
		self.loss: nn.MSELoss = nn.MSELoss()
		# TODO: Is there a way to make a loss function take more into account the variance of the data?
		# Works well for small variance. It may be best to assume we know a upper and lower bound to normalize the data to [0, 1] range
		# It seems necessary to use the pass_posterior_to_prior function every once in a while
	
	def pre_process(self, x: Tensor) -> Tensor:
		return (x - self.range_min) / (self.range_max - self.range_min)
	
	def post_process(self, x: Tensor) -> Tensor:
		return (x * (self.range_max - self.range_min)) + self.range_min
	
	def forward(self, x: Tensor) -> Tensor:
		mu = self.mu_model(x)
		log_sigma = self.log_sigma_model(x)
		y: Tensor = self.reparameterize(mu, log_sigma)
		y = self.post_process(y)
		return y, mu, torch.exp(log_sigma)
	
	def reparameterize(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
		return mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)
	
	def reconstruction_loss(self, output: Tensor, expected_output: Tensor) -> Tensor:
		normalized_output: Tensor = self.pre_process(output)
		normalized_expected_output: Tensor = self.pre_process(expected_output)
		return self.loss(normalized_output, normalized_expected_output)

	
	def kl_loss(self) -> Tensor:
		kl_sum: Tensor = 0.0
		n: int = 0
		for m in self.modules():
			if isinstance(m, BayesLinear):
				kl_sum += m.kl_loss()
				n += m.distributions_amount()
		
		kl_sum = kl_sum.squeeze()
		return kl_sum / n
	
	def prior_to_device(self, device: torch.device) -> None:
		for m in self.modules():
			if isinstance(m, BayesLinear):
				m.prior_to_device(device)

	def pass_posterior_to_prior(self) -> None:
		for m in self.modules():
			if isinstance(m, BayesLinear):
				m.pass_posterior_to_prior()

class BayesTransitionModel(BayesModel):
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		self.mu_model = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			#nn.Linear(64, out_features)
			BayesLinear(64, out_features, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
		)

		self.log_sigma_model = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			#nn.Linear(64, out_features)
			BayesLinear(64, out_features, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
		)

		self.range_min: float = 0.0
		self.range_max: float = 1.0
		self.loss: nn.MSELoss = nn.MSELoss()

class BayesValueModel(BayesModel):
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		self.mu_model = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			#nn.Linear(64, out_features)
			BayesLinear(64, out_features, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
		)

		self.log_sigma_model = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			#nn.Linear(64, out_features)
			BayesLinear(64, out_features, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
		)

		self.range_min: float = 0.0
		self.range_max: float = 1.0
		self.loss: nn.MSELoss = nn.MSELoss()










# EXPERIMENTAL MODULES
class BayesModelUniform(nn.Module):
	def __init__(self):
		super().__init__()
		self.range1 = None
		self.range2 = None
		self.range_min: float = 0.0
		self.range_max: float = 1.0
		self.loss: nn.MSELoss = nn.MSELoss()
		# TODO: Is there a way to make a loss function take more into account the variance of the data?
		# Works well for small variance. It may be best to assume we know a upper and lower bound to normalize the data to [0, 1] range
		# It seems necessary to use the pass_posterior_to_prior function every once in a while
	
	def pre_process(self, x: Tensor) -> Tensor:
		return (x - self.range_min) / (self.range_max - self.range_min)
	
	def post_process(self, x: Tensor) -> Tensor:
		return (x * (self.range_max - self.range_min)) + self.range_min
	
	def forward(self, x: Tensor) -> Tensor:
		r1 = self.range1(x)
		r2 = self.range2(x)
		y: Tensor = self.reparameterize(r1, r2)
		y = self.post_process(y)
		return y, r1, r2
	
	def reparameterize(self, r1: Tensor, r2: Tensor) -> Tensor:
		return torch.rand_like(r1) * (r2 - r1) + r1
	
	def reconstruction_loss(self, output: Tensor, expected_output: Tensor) -> Tensor:
		normalized_output: Tensor = self.pre_process(output)
		normalized_expected_output: Tensor = self.pre_process(expected_output)
		return self.loss(normalized_output, normalized_expected_output)

	
	def kl_loss(self) -> Tensor:
		kl_sum: Tensor = torch.tensor([0.0], dtype=torch.float32)
		n: int = 0
		for m in self.modules():
			if isinstance(m, BayesLinear):
				kl_sum += m.kl_loss()
				n += m.distributions_amount()
		
		kl_sum = kl_sum.squeeze()
		return kl_sum / n
	
	def prior_to_device(self, device: torch.device) -> None:
		for m in self.modules():
			if isinstance(m, BayesLinear):
				m.prior_to_device(device)

	def pass_posterior_to_prior(self) -> None:
		for m in self.modules():
			if isinstance(m, BayesLinear):
				m.pass_posterior_to_prior()



class BayesValueModelUniform(BayesModelUniform):
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		self.range1 = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			#nn.Linear(64, out_features)
			BayesLinear(64, out_features, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
		)

		self.range2 = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			#nn.Linear(64, out_features)
			BayesLinear(64, out_features, prior_weight_sigma=0.01, prior_bias_sigma=0.01)
		)

		self.range_min: float = 0.0
		self.range_max: float = 1.0
		self.loss: nn.MSELoss = nn.MSELoss()