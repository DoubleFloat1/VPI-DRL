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

class MarioBros(GymEnv):
    def __init__(self):
        super().__init__()
        self.gym_name = "ALE/MarioBros-v5"
        self.state_size = [3, 210, 160]
        self.actions_amount = 18
    
    def preprocess(self, x: ndarray):
        return np.transpose(x, (2, 0, 1)) / 255.0