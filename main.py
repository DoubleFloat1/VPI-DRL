import copy
from models.VPIDQN import VPIDQN
from models.DQN import DQN
from models.rl_model import RLModel
import gymnasium as gym
from gymnasium import Env
from typing import List, Tuple, Dict, Any
from numpy import ndarray
import time
from gym_envs import *
from custom_envs.race_track import RacetrackEnv

import ale_py
import highway_env

gym.register_envs(ale_py)
gym.register_envs(highway_env)
gym.register(id="custom_envs/RaceTrack-v0", entry_point=RacetrackEnv)

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

# TODO: Frame stacking only works for 1 training environment
class TrainManager:
    def __init__(self, gym_env: GymEnv):
        self.gym_env: GymEnv = gym_env
        self.env: EnvironmentWrapper = EnvironmentWrapper(gym_env.create_environment())
        self.preprocessed_state: Tensor = None
        self.reset_environment()
    
    def reset_environment(self) -> None:
        self.env.reset()
        self.gym_env.end_episode()
        self.preprocessed_state = self.gym_env.preprocess(self.env.get_current_state())

    def train_model(self, model: VPIDQN, steps_amount: int) -> None:
        for t in range(steps_amount):
            print(f"\rTraining: {100 * t / (steps_amount - 1):.2f}%", end="")

            state = self.preprocessed_state
            action = model.get_next_action(state)

            next_state, reward, terminated, truncated = self.env.step(action)
            next_state = self.gym_env.preprocess(next_state)
            model.improve(state, action, reward, next_state, terminated)

            self.preprocessed_state = next_state

            if terminated or truncated:
                self.reset_environment()
        print()


class TestManager:
    def __init__(self, gym_env: GymEnv):
        self.gym_env: GymEnv = gym_env
        self.test_env: EnvironmentWrapper = EnvironmentWrapper(gym_env.create_environment())
    
    def test_model(self, model: RLModel, episodes_amount: int) -> List[float]:
        episode_count = 0
        episode_total_reward = 0.0
        episode_total_reward_list = []

        self.test_env.reset()
        print(f"\rTesting: {episode_count} / {episodes_amount}", end="")
        while episode_count < episodes_amount:
            state = self.test_env.get_current_state()
            state = self.gym_env.preprocess(state)
            action = model.inference_next_action(state)
            _, reward, terminated, truncated = self.test_env.step(action)
            episode_total_reward += reward
            if terminated or truncated:
                episode_total_reward_list.append(episode_total_reward)
                episode_total_reward = 0.0
                episode_count += 1
                print(f"\rTesting: {episode_count} / {episodes_amount}", end="")
                self.test_env.reset()
                self.gym_env.end_episode()
        
        print()

        return episode_total_reward_list


class ResultWriter:
    def __init__(self, file_name: str, gym_env: GymEnv, model: RLModel, train_step_amount: int, create_new_file: bool = True):
        self.trials_rewards: List[List[float]] = []
        self.file_name: str = file_name
        if create_new_file:
            open(self.file_name, 'w').close()
        
        self.gym_env: GymEnv = gym_env
        self.model: RLModel = model
        self.train_step_amount: int = train_step_amount
    
    def set_gym_env(self, gym_env: GymEnv) -> None:
        self.gym_env = gym_env
    
    def set_model(self, model: RLModel) -> None:
        self.model = model

    def set_train_step_amount(self, amount: int) -> None:
        self.train_step_amount = amount

    def add_trial_rewards(self, rewards: List[float]) -> None:
        self.trials_rewards.append(rewards)
    
    def write(self) -> None:
        with open(self.file_name, 'w') as file:
            file.write(f"{self.train_step_amount}\n")
            if len(self.trials_rewards) > 0:
                file.write(f"{len(self.trials_rewards)} {len(self.trials_rewards[0])}\n")
            else:
                file.write("0 0\n")

            for rewards in self.trials_rewards:
                string = self.list_to_str(rewards)
                file.write(string + "\n")
            
            file.write("\nEnvironment params:\n")
            env_params_dict: Dict[str, Any] = self.gym_env.get_params_dict()
            for param_name in env_params_dict.keys():
                param = env_params_dict[param_name]
                file.write(f"{param_name} = {param}\n")

            file.write("\nModel params:\n")
            model_params_dict: Dict[str, Any] = self.model.get_params_dict()
            for param_name in model_params_dict.keys():
                param = model_params_dict[param_name]
                file.write(f"{param_name} = {param}\n")
    
    def write_line(self, rewards: List[float]):
        with open(self.file_name, 'a') as file:
            string = self.list_to_str(rewards)
            file.write(string + "\n")
    
    def list_to_str(self, array: List[float]) -> str:
        n = len(array)
        string = ""
        for i in range(n):
            string += str(array[i])
            if i < n-1:
                string += " "
        return string


def main(gym_env: GymEnv, model: RLModel, data_file: str, create_new_data_file: bool = True, training_epochs: int = 20,
         test_episode_amount: int = 100, train_step_amount: int = 2000, trials_amount: int = 10):
    writer = ResultWriter(data_file, gym_env, model, train_step_amount, create_new_data_file)
    writer.write()
    for t in range(trials_amount):
        train_manager: TrainManager = TrainManager(copy.deepcopy(gym_env))
        test_manager: TestManager = TestManager(copy.deepcopy(gym_env))

        mean_reward_history = []
        rewards = test_manager.test_model(model, test_episode_amount)
        mean_reward: float = sum(rewards) / len(rewards)
        print(mean_reward)
        mean_reward_history.append(mean_reward)
        best_mean_reward: float = mean_reward
        for i in range(training_epochs):
            print(f"Trial {t} - Epoch {i}")

            start = time.perf_counter()
            train_manager.train_model(model, train_step_amount)
            end = time.perf_counter()
            print(f"{end - start:.3f}s")

            rewards = test_manager.test_model(model, test_episode_amount)
            mean_reward = sum(rewards) / len(rewards)
            print(mean_reward)
            mean_reward_history.append(mean_reward)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                model.save()

        writer.add_trial_rewards(mean_reward_history)
        writer.write()

        if t < trials_amount - 1:
            model = model.new()


if __name__ == "__main__":
    gym_env = Breakout()

    train_step_amount: int = 80000
    training_epochs: int = 100
    test_episode_amount: int = 100
    trials_amount: int = 1

    total_steps_of_eps_decay: int = round(0.125 * train_step_amount * training_epochs)
    #dqn = DQN(gym_env.state_size, gym_env.actions_amount, 
    #          experience_replay_max_size=750000,
    #          experience_replay_state_to_uint8=gym_env.image_state,
    #          updates_to_renew_target_network=2500,
    #          value_lr=1e-5,
    #          initial_eps=1.0,
    #          min_eps=0.1,
    #          total_steps_of_eps_decay=total_steps_of_eps_decay,
    #          load_model_path=None
    #          )
    
    vpidqn = VPIDQN(gym_env.state_size, gym_env.actions_amount, 
                    experience_replay_max_size=750000,
                    experience_replay_state_to_uint8=gym_env.image_state,
                    value_lr=1e-5,
                    updates_to_renew_target_network=2500,
                    value_kl_weight=0.1,
                    updates_to_pass_posterior=2500,
                    initial_eps=1.0,
                    min_eps=0.1,
                    total_steps_of_eps_decay=total_steps_of_eps_decay,
                    use_uniform_distribution=False,
                    load_model_path=None
                    )

    #main(gym_env, dqn, "dqn.txt", train_step_amount=train_step_amount, training_epochs=training_epochs, test_episode_amount=test_episode_amount,trials_amount=trials_amount)
    main(gym_env, vpidqn, "vpidqn.txt", train_step_amount=train_step_amount, training_epochs=training_epochs, test_episode_amount=test_episode_amount,trials_amount=trials_amount)