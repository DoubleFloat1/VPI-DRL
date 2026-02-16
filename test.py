from VPIDQN import VPIDQN, TestVPIDQN
from VPIDQN2 import VPIDQN2, TestVPIDQN2
from DQN import DQN
from rl_model import RLModel
import gymnasium as gym
from gymnasium import Env
from typing import List
from numpy import ndarray
import time

ENV_NAME: str = "CartPole-v1"

class EnvironmentWrapper:
    def __init__(self, env: Env):
        self.env: Env = env
        self.current_state: ndarray[float] = None
        self.episode_reward: float = 0.0
        self.reset()
    
    def reset(self) -> float:
        self.current_state, _ = self.env.reset()

        total_episode_reward = self.episode_reward
        self.episode_reward = 0.0
        return total_episode_reward
    
    def step(self, action: int):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.current_state = next_state
        self.episode_reward += reward
        return next_state, reward, terminated, truncated
    
    def get_current_state(self) -> ndarray[float]:
        return self.current_state

class TrainManager:
    def __init__(self, envs_amount: int):
        self.envs: List[EnvironmentWrapper] = [EnvironmentWrapper(gym.make(ENV_NAME, render_mode=None)) for _ in range(envs_amount)]
    
    def train_model(self, model: RLModel, steps_amount: int) -> None:
        for t in range(steps_amount):
            if t % 1000 == 0:
                pass
                print(t)

            for env in self.envs:
                state = env.get_current_state()
                action = model.get_next_action(state)

                next_state, reward, terminated, truncated = env.step(action)
                model.improve(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    env.reset()

class TestManager:
    def __init__(self):
        self.test_env: EnvironmentWrapper = EnvironmentWrapper(gym.make(ENV_NAME, render_mode=None))
    
    def test_model(self, model: RLModel, episodes_amount: int) -> List[float]:
        episode_count = 0
        episode_total_reward = 0.0
        episode_total_reward_list = []

        self.test_env.reset()
        while episode_count < episodes_amount:
            state = self.test_env.get_current_state()
            action = model.inference_next_action(state)
            _, reward, terminated, truncated = self.test_env.step(action)
            episode_total_reward += reward
            if terminated or truncated:
                episode_total_reward_list.append(episode_total_reward)
                episode_total_reward = 0.0
                episode_count += 1
                self.test_env.reset()
        
        return episode_total_reward_list


        

model = VPIDQN2(4, 2)
train_manager: TrainManager = TrainManager(4)
test_manager: TestManager = TestManager()

rewards = test_manager.test_model(model, 100)
print(sum(rewards) / len(rewards))

for i in range(10):
    print(f"Epoch {i}")

    start = time.perf_counter()
    train_manager.train_model(model, 10000)
    end = time.perf_counter()
    print(f"{end - start:.3f}s")

    rewards = test_manager.test_model(model, 100)
    print(sum(rewards) / len(rewards))

