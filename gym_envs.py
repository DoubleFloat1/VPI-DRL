from typing import List, Tuple, Dict
from numpy import ndarray
import numpy as np

class GymEnv:
    def __init__(self):
        self.gym_name: str = ""
        self.state_size: List[int] = []
        self.actions_amount: int = -1

    def preprocess(self, x):
        return x
    
    def end_episode(self) -> None:
        pass
    
class ImageGymEnv(GymEnv):
    def __init__(self):
        super().__init__()
        self.frame_stack_size: int = 4
        self.frames: List[ndarray] = None
    
    def initialize_frame_stack(self) -> None:
        frame_size = self.state_size.copy()
        frame_size[0] = frame_size[0] // self.frame_stack_size
        self.frames = [np.zeros(frame_size) for _ in range(self.frame_stack_size)]
    
    def add_frame(self, x) -> None:
        for i in range(self.frame_stack_size - 1):
            self.frames[i] = self.frames[i+1]
        self.frames[-1] = x
    
    def get_frame_stack(self) -> ndarray:
        return np.concat(self.frames)
    
    def end_episode(self):
        self.initialize_frame_stack()


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
        self.state_size = [self.frame_stack_size * 3, 210, 160]
        self.actions_amount = 18
        self.initialize_frame_stack()
    
    def preprocess(self, x: ndarray):
        x = np.transpose(x, (2, 0, 1)) / 255.0
        self.add_frame(x)
        return self.get_frame_stack()
    
class CollectHealth(ImageGymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "MiniWorld-CollectHealth-v0"
        self.state_size = [self.frame_stack_size * 3, 60, 80]
        self.actions_amount = 8
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
    
class SpaceInvaders(ImageGymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "ALE/SpaceInvaders-v5"
        self.state_size = [self.frame_stack_size * 3, 210, 160]
        self.actions_amount = 6
        self.initialize_frame_stack()
    
    def preprocess(self, x: ndarray):
        x = np.transpose(x, (2, 0, 1)) / 255.0
        self.add_frame(x)
        return self.get_frame_stack()
