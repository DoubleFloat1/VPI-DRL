from VPIDQN import VPIDQN
from DQN import DQN
from rl_model import RLModel
import gymnasium as gym
from gymnasium import Env
from typing import List
from numpy import ndarray
import time
from gym_envs import *
import ale_py

# TODO: add multiple frames per network pass
gym.register_envs(ale_py)

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
    def __init__(self, gym_env: GymEnv, envs_amount: int):
        self.gym_env: GymEnv = gym_env
        self.envs: List[EnvironmentWrapper] = [EnvironmentWrapper(gym.make(gym_env.gym_name, render_mode=None)) for _ in range(envs_amount)]
    
    def train_model(self, model: VPIDQN, steps_amount: int) -> None:
        for t in range(steps_amount):
            print(f"{100 * t / (steps_amount - 1):.2f}%", end="\r")

            for env in self.envs:
                state = env.get_current_state()
                state = self.gym_env.preprocess(state)
                action = model.get_next_action(state)

                next_state, reward, terminated, truncated = env.step(action)
                next_state = self.gym_env.preprocess(next_state)
                model.improve(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    env.reset()
        print()
    
    def vpi_train_model(self, model: VPIDQN, steps_amount: int) -> None:
        for t in range(steps_amount):
            print(f"{100 * t / (steps_amount - 1):.2f}%", end="\r")

            for env in self.envs:
                state = env.get_current_state()
                state = self.gym_env.preprocess(state)
                action = model.vpi_get_next_action(state)

                next_state, reward, terminated, truncated = env.step(action)
                next_state = self.gym_env.preprocess(next_state)
                model.improve(state, action, reward, next_state, terminated)

                if terminated or truncated:
                    env.reset()
        print()

class TestManager:
    def __init__(self, gym_env: GymEnv):
        self.gym_env: GymEnv = gym_env
        self.test_env: EnvironmentWrapper = EnvironmentWrapper(gym.make(gym_env.gym_name, render_mode=None))
    
    def test_model(self, model: RLModel, episodes_amount: int) -> List[float]:
        episode_count = 0
        episode_total_reward = 0.0
        episode_total_reward_list = []

        self.test_env.reset()
        step_count = 0
        while episode_count < episodes_amount:
            step_count += 1
            if step_count % 1000 == 0:
                pass
                #print(step_count)
            state = self.test_env.get_current_state()
            state = self.gym_env.preprocess(state)
            action = model.inference_next_action(state)
            _, reward, terminated, truncated = self.test_env.step(action)
            episode_total_reward += reward
            if terminated or truncated:
                episode_total_reward_list.append(episode_total_reward)
                episode_total_reward = 0.0
                episode_count += 1
                self.test_env.reset()
        
        return episode_total_reward_list


class ResultWriter:
    def __init__(self, file_name: str):
        self.trials_rewards = []
        self.file_name: str = file_name
        open(self.file_name, 'w').close()
    
    def add_trial_rewards(self, rewards):
        self.trials_rewards.append(rewards)
    
    def write(self):
        with open(self.file_name, 'w') as file:
            for rewards in self.trials_rewards:
                string = self.list_to_str(rewards)
                file.write(string + "\n")
    
    def write_line(self, rewards):
        with open(self.file_name, 'a') as file:
            string = self.list_to_str(rewards)
            file.write(string + "\n")
    
    def list_to_str(self, array):
        n = len(array)
        string = ""
        for i in range(n):
            string += str(array[i])
            if i < n-1:
                string += " "
        return string


gym_env: GymEnv = LunarLander()

normal_training_epochs: int = 0
vpi_training_epochs: int = 20

writer = ResultWriter("vpidqn2.txt")

for t in range(10):
    print(f"trial {t}")

    model = VPIDQN(gym_env.state_size, gym_env.actions_amount)
    train_manager: TrainManager = TrainManager(gym_env, 4)
    test_manager: TestManager = TestManager(gym_env)

    mean_reward_history = []
    rewards = test_manager.test_model(model, 100)
    print(sum(rewards) / len(rewards))
    mean_reward_history.append(sum(rewards) / len(rewards))
    for i in range(normal_training_epochs):
        print(f"Epoch {i}")

        start = time.perf_counter()
        train_manager.train_model(model, 2000)
        end = time.perf_counter()
        print(f"{end - start:.3f}s")

        rewards = test_manager.test_model(model, 100)
        print(sum(rewards) / len(rewards))
        mean_reward_history.append(sum(rewards) / len(rewards))

    print("Now with VPI")

    for i in range(normal_training_epochs, normal_training_epochs + vpi_training_epochs):
        print(f"Epoch {i}")

        start = time.perf_counter()
        train_manager.vpi_train_model(model, 2000)
        end = time.perf_counter()
        print(f"{end - start:.3f}s")

        rewards = test_manager.test_model(model, 100)
        print(sum(rewards) / len(rewards))
        mean_reward_history.append(sum(rewards) / len(rewards))

    writer.add_trial_rewards(mean_reward_history)
    writer.write_line(mean_reward_history)

writer.write()



gym_env: GymEnv = LunarLander()

normal_training_epochs: int = 20
vpi_training_epochs: int = 0

writer = ResultWriter("dqn.txt")

for t in range(10):
    print(f"trial {t}")

    model = DQN(gym_env.state_size, gym_env.actions_amount)
    train_manager: TrainManager = TrainManager(gym_env, 4)
    test_manager: TestManager = TestManager(gym_env)

    mean_reward_history = []
    rewards = test_manager.test_model(model, 100)
    print(sum(rewards) / len(rewards))
    mean_reward_history.append(sum(rewards) / len(rewards))
    for i in range(normal_training_epochs):
        print(f"Epoch {i}")

        start = time.perf_counter()
        train_manager.train_model(model, 2000)
        end = time.perf_counter()
        print(f"{end - start:.3f}s")

        rewards = test_manager.test_model(model, 100)
        print(sum(rewards) / len(rewards))
        mean_reward_history.append(sum(rewards) / len(rewards))

    print("Now with VPI")

    for i in range(normal_training_epochs, normal_training_epochs + vpi_training_epochs):
        print(f"Epoch {i}")

        start = time.perf_counter()
        train_manager.vpi_train_model(model, 2000)
        end = time.perf_counter()
        print(f"{end - start:.3f}s")

        rewards = test_manager.test_model(model, 100)
        print(sum(rewards) / len(rewards))
        mean_reward_history.append(sum(rewards) / len(rewards))

    writer.add_trial_rewards(mean_reward_history)
    writer.write_line(mean_reward_history)

writer.write()