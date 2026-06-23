from typing import List, Tuple, Dict, Any

class VPIDQNParams:
    def __init__(self, gamma: float = 0.99, value_lr: float = 1e-5, value_kl_weight: float = 0.1, 
                 value_vpi_batch_size: int = 16, value_rand_batch_size: int = 16, updates_to_pass_posterior: int = 512, 
                 experience_replay_max_size: int = 4096,  experience_replay_state_to_uint8: bool = False,
                 updates_to_renew_target_network: int = 256, initial_eps: float = 1.0, min_eps: float = 0.1, total_steps_of_eps_decay: int = 1000000,
                 alpha: float = 1.0, initial_beta: float = 0.0, max_beta: float = 1.0, 
                 total_steps_of_beta_growth: int = 1000000, use_heap_experience_replay: bool = False,
                 experience_replays_adds_per_update: int = 64, inference_use_mu: bool = True):
        self.gamma: float = gamma
        self.value_lr: float = value_lr
        self.experience_replay_max_size: int = experience_replay_max_size
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.updates_to_renew_target_network: int = updates_to_renew_target_network
        self.initial_eps: float = initial_eps
        self.min_eps: float = min_eps
        self.total_steps_of_eps_decay: int = total_steps_of_eps_decay
        self.value_vpi_batch_size: int = value_vpi_batch_size
        self.value_rand_batch_size: int = value_rand_batch_size
        self.value_kl_weight: float = value_kl_weight
        self.updates_to_pass_posterior: int = updates_to_pass_posterior
        self.alpha: float = alpha
        self.initial_beta: float = initial_beta
        self.max_beta: float = max_beta
        self.total_steps_of_beta_growth: int = total_steps_of_beta_growth
        self.use_heap_experience_replay: bool = use_heap_experience_replay
        self.experience_replays_adds_per_update: int = experience_replays_adds_per_update
        self.inference_use_mu: bool = inference_use_mu
    
    def to_dict(self) -> Dict[str, Any]:
        param_dict: Dict = {
            "name": "VPIDQN",
            "gamma": self.gamma,
            "value_lr": self.value_lr,
            "value_kl_weight": self.value_kl_weight,
            "value_vpi_batch_size": self.value_vpi_batch_size,
            "value_rand_batch_size": self.value_rand_batch_size,
            "uptades_to_pass_posterior": self.updates_to_pass_posterior,
            "experience_replay_max_size": self.experience_replay_max_size,
            "updates_to_renew_target_network": self.updates_to_renew_target_network,
            "initial_eps": self.initial_eps,
            "min_eps": self.min_eps,
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay,
            "alpha": self.alpha,
            "initial_beta": self.initial_beta,
            "max_beta": self.max_beta,
            "total_steps_of_beta_growth": self.total_steps_of_beta_growth,
            "use_heap_experience_replay": self.use_heap_experience_replay,
            "experience_replays_adds_per_update": self.experience_replays_adds_per_update,
            "inference_use_mu": self.inference_use_mu
        }
        return param_dict

class A2CParams:
    def __init__(self, gamma: float = 0.99, actor_lr: float = 1e-4, critic_lr: float = 1e-4, batch_size: int = 1024, n_step_return: int = 5,
                 envs_amount: int = 4):
        self.gamma: float = gamma
        self.actor_lr: float = actor_lr
        self.critic_lr: float = critic_lr
        self.batch_size: int = batch_size
        self.n_step_return: int = n_step_return
        self.envs_amount: int = envs_amount
    
    def to_dict(self) -> Dict[str, Any]:
        params_dict: Dict = {
            "name": "A2C",
            "gamma": self.gamma,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "batch_size": self.batch_size,
            "n_step_return": self.n_step_return,
            "envs_amount": self.envs_amount
        }
        return params_dict

class DDPGParams:
    def __init__(self, gamma: float = 0.99, q_value_lr: float = 1e-4, policy_lr: float = 1e-4, batch_size: int = 64, experience_replay_max_size: int = 500000,
                 polyak: float = 0.995, steps_per_update: int = 32, repeats_per_update: int = 8, experience_replay_state_to_uint8: bool = False,
                 initial_eps: float = 1.0, min_eps: float = 0.1, total_steps_of_eps_decay: int = 1000000, uniform_start_steps: int = 10000):
        self.gamma: float = gamma
        self.q_value_lr: float = q_value_lr
        self.policy_lr: float = policy_lr
        self.batch_size: int = batch_size
        self.experience_replay_max_size: int = experience_replay_max_size
        self.polyak: float = polyak
        self.steps_per_update: int = steps_per_update
        self.repeats_per_update: int = repeats_per_update
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.initial_eps: float = initial_eps
        self.min_eps: float = min_eps
        self.total_steps_of_eps_decay: int = total_steps_of_eps_decay
        self.uniform_start_steps: int = uniform_start_steps
    
    def to_dict(self) -> Dict[str, Any]:
        params_dict: Dict = {
            "name": "DDPG",
            "gamma": self.gamma,
            "q_value_lr": self.q_value_lr,
            "policy_lr": self.policy_lr,
            "batch_size": self.batch_size,
            "experience_replay_max_size": self.experience_replay_max_size,
            "polyak": self.polyak,
            "steps_per_update": self.steps_per_update,
            "repeats_per_update": self.repeats_per_update,
            "experience_replay_state_to_uint8": self.experience_replay_state_to_uint8,
            "initial_eps": self.initial_eps,
            "min_eps": self.min_eps,
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay,
            "uniform_start_steps": self.uniform_start_steps
        }
        return params_dict

class TD3Params:
    def __init__(self, gamma: float = 0.99, q_value_lr: float = 1e-4, policy_lr: float = 1e-4, batch_size: int = 64, experience_replay_max_size: int = 500000,
                 polyak: float = 0.995, steps_per_update: int = 32, repeats_per_update: int = 8, experience_replay_state_to_uint8: bool = False,
                 initial_eps: float = 0.5, min_eps: float = 0.1, total_steps_of_eps_decay: int = 1000000, 
                 target_noise_std: float = 0.2, noise_clip: float = 0.5, policy_delay: int = 2, uniform_start_steps: int = 10000):
        self.gamma: float = gamma
        self.q_value_lr: float = q_value_lr
        self.policy_lr: float = policy_lr
        self.batch_size: int = batch_size
        self.experience_replay_max_size: int = experience_replay_max_size
        self.polyak: float = polyak
        self.steps_per_update: int = steps_per_update
        self.repeats_per_update: int = repeats_per_update
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.initial_eps: float = initial_eps
        self.min_eps: float = min_eps
        self.total_steps_of_eps_decay: int = total_steps_of_eps_decay
        self.target_noise_std: float = target_noise_std
        self.noise_clip: float = noise_clip
        self.policy_delay: int = policy_delay
        self.uniform_start_steps: int = uniform_start_steps
    
    def to_dict(self) -> Dict[str, Any]:
        params_dict: Dict = {
            "name": "TD3",
            "gamma": self.gamma,
            "q_value_lr": self.q_value_lr,
            "policy_lr": self.policy_lr,
            "batch_size": self.batch_size,
            "experience_replay_max_size": self.experience_replay_max_size,
            "polyak": self.polyak,
            "steps_per_update": self.steps_per_update,
            "repeats_per_update": self.repeats_per_update,
            "experience_replay_state_to_uint8": self.experience_replay_state_to_uint8,
            "initial_eps": self.initial_eps,
            "min_eps": self.min_eps,
            "total_steps_of_eps_decay": self.total_steps_of_eps_decay,
            "target_noise_std": self.target_noise_std,
            "noise_clip": self.noise_clip,
            "policy_delay": self.policy_delay,
            "uniform_start_steps": self.uniform_start_steps
        }
        return params_dict
    
class SACParams:
    def __init__(self, gamma: float = 0.99, q_value_lr: float = 1e-4, policy_lr: float = 1e-4, batch_size: int = 64, experience_replay_max_size: int = 500000,
                 polyak: float = 0.995, steps_per_update: int = 32, repeats_per_update: int = 8, experience_replay_state_to_uint8: bool = False,
                 entropy_const: float = 0.2, uniform_start_steps: int = 10000):
        self.gamma: float = gamma
        self.q_value_lr: float = q_value_lr
        self.policy_lr: float = policy_lr
        self.batch_size: int = batch_size
        self.experience_replay_max_size: int = experience_replay_max_size
        self.polyak: float = polyak
        self.steps_per_update: int = steps_per_update
        self.repeats_per_update: int = repeats_per_update
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.entropy_const: float = entropy_const
        self.uniform_start_steps: int = uniform_start_steps
    
    def to_dict(self) -> Dict[str, Any]:
        params_dict: Dict = {
            "name": "SAC",
            "gamma": self.gamma,
            "q_value_lr": self.q_value_lr,
            "policy_lr": self.policy_lr,
            "batch_size": self.batch_size,
            "experience_replay_max_size": self.experience_replay_max_size,
            "polyak": self.polyak,
            "steps_per_update": self.steps_per_update,
            "repeats_per_update": self.repeats_per_update,
            "experience_replay_state_to_uint8": self.experience_replay_state_to_uint8,
            "entropy_const": self.entropy_const,
            "uniform_start_steps": self.uniform_start_steps
        }
        return params_dict


class VPISACParams:
    def __init__(self, gamma: float = 0.99, q_value_lr: float = 1e-4, policy_lr: float = 1e-4, vpi_lr: float = 1e-4, batch_size: int = 100,
                 experience_replay_max_size: int = 1000000, polyak: float = 0.995, steps_per_update: int = 50, repeats_per_update: int = 50,
                 experience_replay_state_to_uint8: bool = False, vpi_output_norms_amount: int = 1, initial_vpi_const: float = 0.2, min_vpi_const: float = 0.05,
                 total_steps_of_vpi_const_decay: int = 3000000, uniform_start_steps: int = 10000, weighted_normals: bool = True):
        self.gamma: float = gamma
        self.q_value_lr: float = q_value_lr
        self.policy_lr: float = policy_lr
        self.vpi_lr: float = vpi_lr
        self.batch_size: int = batch_size
        self.experience_replay_max_size: int = experience_replay_max_size
        self.polyak: float = polyak
        self.steps_per_update: int = steps_per_update
        self.repeats_per_update: int = repeats_per_update
        self.experience_replay_state_to_uint8: bool = experience_replay_state_to_uint8
        self.vpi_output_norms_amount: int = vpi_output_norms_amount
        self.initial_vpi_const: float = initial_vpi_const
        self.min_vpi_const: float = min_vpi_const
        self.total_steps_of_vpi_const_decay: int = total_steps_of_vpi_const_decay
        self.uniform_start_steps: int = uniform_start_steps
        self.weighted_normals: bool = weighted_normals

    
    def to_dict(self) -> Dict[str, Any]:
        params_dict: Dict = {
            "name": "VPI-SAC",
            "gamma": self.gamma,
            "q_value_lr": self.q_value_lr,
            "policy_lr": self.policy_lr,
            "vpi_lr": self.vpi_lr,
            "batch_size": self.batch_size,
            "experience_replay_max_size": self.experience_replay_max_size,
            "polyak": self.polyak,
            "steps_per_update": self.steps_per_update,
            "repeats_per_update": self.repeats_per_update,
            "experience_replay_state_to_uint8": self.experience_replay_state_to_uint8,
            "vpi_output_norms_amount": self.vpi_output_norms_amount,
            "initial_vpi_const": self.initial_vpi_const,
            "min_vpi_const": self.min_vpi_const,
            "total_steps_of_vpi_const_decay": self.total_steps_of_vpi_const_decay,
            "uniform_start_steps": self.uniform_start_steps,
            "weighted_normals": self.weighted_normals
        }
        return params_dict