from typing import List, Tuple, Dict, Any
from numpy import ndarray
import numpy as np
import gymnasium as gym
import torch
import torchvision.transforms as T
from torch import Tensor
from gym_wrappers import AtariWrapper, ContinuousActionDiscretization

class GymEnv:
    def __init__(self):
        self.gym_name: str = ""
        self.state_size: List[int] = []
        self.actions_amount: int = -1
        self.image_state: bool = False

    def preprocess(self, x: ndarray) -> Tensor:
        return torch.tensor(x, dtype=torch.float32)
    
    def end_episode(self) -> None:
        pass

    def create_environment(self) -> gym.Env:
        return gym.make(self.gym_name, render_mode=None)
    
    def get_params_dict(self) -> Dict[str, Any]:
        return {"gym_id": self.gym_name}
    
class ImageGymEnv(GymEnv):
    def __init__(self):
        super().__init__()
        self.image_state = True
        self.frame_stack_size: int = 4
        self.frames: List[Tensor] = None
    
    def initialize_frame_stack(self) -> None:
        frame_size = self.state_size.copy()
        frame_size[0] = frame_size[0] // self.frame_stack_size
        self.frames = [torch.zeros(frame_size, dtype=torch.uint8) for _ in range(self.frame_stack_size)]
        self.transformations: T.Compose = T.Compose([T.ToPILImage(), T.ToTensor()])
    
    def add_frame(self, x) -> None:
        for i in range(self.frame_stack_size - 1):
            self.frames[i] = self.frames[i+1]
        self.frames[-1] = x
    
    def get_frame_stack(self) -> Tensor:
        return torch.cat(self.frames)
    
    def end_episode(self):
        self.initialize_frame_stack()

    def preprocess(self, x: ndarray) -> Tensor:
        y: Tensor = self.transformations(x)
        self.add_frame(y)
        return self.get_frame_stack()


class Polecart(GymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "CartPole-v1"
        self.state_size = [4]
        self.actions_amount = 2

class LunarLander(GymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "LunarLander-v3"
        self.state_size = [8]
        self.actions_amount = 4

class MarioBros(ImageGymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "ALE/MarioBros-v5"
        self.state_size = [self.frame_stack_size * 1, 84, 84]
        self.actions_amount = 18
        self.initialize_frame_stack()
    
    def preprocess(self, x: ndarray):
        x = np.transpose(x, (2, 0, 1)) / 255.0
        self.add_frame(x)
        return self.get_frame_stack()

    
class Highway(GymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "highway-v0"
        self.state_size = [25]
        self.actions_amount = 5
    
    def preprocess(self, x: ndarray):
        return x.flatten()


class CarRacing(ImageGymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "CarRacing-v3"
        self.state_size = [self.frame_stack_size * 1, 84, 84]
        self.actions_amount = 5
        self.initialize_frame_stack()
    
    def create_environment(self):
        return gym.make(self.gym_name, render_mode=None, continuous=False)
    

class RaceTrack(GymEnv):
    def __init__(self, track_index: int = 0):
        super().__init__()
        self.gym_name = "custom_envs/RaceTrack-v0"
        self.state_size = [4]
        self.actions_amount = 9

        self.track_index = track_index
    
    def preprocess(self, x):
        if self.track_index == 0:
            x[0] = x[0] / 30.0
            x[1] = x[1] / 33.0
            x[2] = x[2] / 30.0
            x[3] = x[3] / 33.0
        elif self.track_index == 1:
            x[0] = x[0] / 30.0
            x[1] = x[1] / 80.0
            x[2] = x[2] / 30.0
            x[3] = x[3] / 80.0
        return x

    def create_environment(self):
        return gym.make(self.gym_name, render_mode=None, track_index=self.track_index)



class AtariEnv(ImageGymEnv):
    def __init__(self, gym_name: str, actions_amount: int, noop_max: int = 5, frame_skip: int = 2, screen_size: int = 84, clip_reward: bool = True,
                 repeat_action_probability: float = 0.25, continuous_actions: bool = False, angles_amount: int = 8, magnitudes_amount: int = 2):
        super().__init__()
        self.gym_name = gym_name
        self.state_size = [self.frame_stack_size * 1, screen_size, screen_size]
        self.actions_amount = actions_amount
        if continuous_actions:
            self.actions_amount = 2 * angles_amount * (magnitudes_amount - 1) + 2

        self.repeat_action_probability: float = repeat_action_probability
        self.noop_max: int = noop_max
        self.frame_skip: int = frame_skip
        self.screen_size: int = screen_size
        self.clip_reward: bool = clip_reward

        self.continuous_actions: bool = continuous_actions
        self.angles_amount: int = angles_amount
        self.magnitudes_amount: int = magnitudes_amount

        self.initialize_frame_stack()
    
    def create_environment(self) -> gym.Env:
        env = gym.make(self.gym_name, render_mode="rgb_array", frameskip=1, repeat_action_probability=self.repeat_action_probability, 
                       continuous=self.continuous_actions)
        return AtariWrapper(env, 
                            noop_max=self.noop_max, 
                            frame_skip=self.frame_skip,
                            screen_size=self.screen_size,
                            clip_reward=self.clip_reward,
                            continuous_actions=self.continuous_actions,
                            angles_amount=self.angles_amount,
                            magnitudes_amount=self.magnitudes_amount)
    
    def get_params_dict(self) -> Dict[str, Any]:
        return {"gym_id": self.gym_name,
                "state_size": self.state_size,
                "actions_amount": self.actions_amount,
                "repeat_action_probability": self.repeat_action_probability,
                "noop_max": self.noop_max,
                "frame_skip": self.frame_skip,
                "screen_size": self.screen_size,
                "clip_reward": self.clip_reward,
                "continuous_actions": self.continuous_actions,
                "angles_amount": self.angles_amount,
                "magnitudes_amount": self.magnitudes_amount}



class SpaceInvaders(AtariEnv):
    def __init__(self, noop_max: int = 5, frame_skip: int = 2, screen_size: int = 84, clip_reward: bool = True, 
                 repeat_action_probability: float = 0.25, continuous_actions: bool = False, angles_amount: int = 8, magnitudes_amount: int = 2):
        super().__init__("ALE/SpaceInvaders-v5", 6, noop_max, frame_skip, screen_size, clip_reward, repeat_action_probability, 
                         continuous_actions, angles_amount, magnitudes_amount)
        
class Breakout(AtariEnv):
    def __init__(self, noop_max: int = 5, frame_skip: int = 2, screen_size: int = 84, clip_reward: bool = True, 
                 repeat_action_probability: float = 0.25, continuous_actions: bool = False, angles_amount: int = 8, magnitudes_amount: int = 2):
        super().__init__("ALE/Breakout-v5", 4, noop_max, frame_skip, screen_size, clip_reward, repeat_action_probability, 
                         continuous_actions, angles_amount, magnitudes_amount)
    
class Asteroid(AtariEnv):
    def __init__(self, noop_max: int = 5, frame_skip: int = 2, screen_size: int = 84, clip_reward: bool = True, 
                 repeat_action_probability: float = 0.25, continuous_actions: bool = False, angles_amount: int = 8, magnitudes_amount: int = 2):
        super().__init__("ALE/Asteroids-v5", 14, noop_max, frame_skip, screen_size, clip_reward, repeat_action_probability, 
                         continuous_actions, angles_amount, magnitudes_amount)

class Asterisk(AtariEnv):
    def __init__(self, noop_max: int = 5, frame_skip: int = 2, screen_size: int = 84, clip_reward: bool = True, 
                 repeat_action_probability: float = 0.25, continuous_actions: bool = False, angles_amount: int = 8, magnitudes_amount: int = 2):
        super().__init__("ALE/Asterix-v5", 9, noop_max, frame_skip, screen_size, clip_reward, repeat_action_probability, 
                         continuous_actions, angles_amount, magnitudes_amount)
        



class MujucoEnv(GymEnv):
    def __init__(self, gym_name: str, state_size: List[int], action_dimension: int, action_range_min: float, action_range_max: float, discretization_factor: int):
        super().__init__()
        self.gym_name = gym_name
        self.state_size: List[int] = state_size
        self.action_dimension: int = action_dimension
        self.action_range_min: float = action_range_min
        self.action_range_max: float = action_range_max
        self.discretization_factor: int = discretization_factor

        self.actions_amount = discretization_factor**action_dimension
    
    def create_environment(self) -> gym.Env:
        env = gym.make(self.gym_name, render_mode=None)
        return ContinuousActionDiscretization(env, self.action_dimension, self.action_range_min, self.action_range_max, self.discretization_factor)
    
    def get_params_dict(self) -> Dict[str, Any]:
        return {"gym_id": self.gym_name,
                "state_size": self.state_size,
                "action_dimension": self.action_dimension,
                "action_range_min": self.action_range_min,
                "action_range_max": self.action_range_max,
                "discretization_factor": self.discretization_factor,
                "actions_amount": self.actions_amount
                }

class MujucoAnt(MujucoEnv):
    def __init__(self, discretization_factor: int = 3):
        super().__init__("Ant-v5", [105], 8, -1.0, 1.0, discretization_factor)

class MujucoHalfCheetah(MujucoEnv):
    def __init__(self, discretization_factor: int = 4):
        super().__init__("HalfCheetah-v5", [17], 6, -1.0, 1.0, discretization_factor)

class MujucoHopper(MujucoEnv):
    def __init__(self, discretization_factor: int = 18):
        super().__init__("Hopper-v5", [11], 3, -1.0, 1.0, discretization_factor)