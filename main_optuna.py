import copy
from models.optuna_VPIDQN import OptunaVPIDQN as VPIDQN
from models.params import VPIDQNParams
from models.DQN import DQN
from models.rl_model import RLModel
import gymnasium as gym
from main_util import TrainManager, ResultWriter, EnvironmentWrapper
import time
import datetime
from gym_envs import *
from custom_envs.race_track import RacetrackEnv

import torch
import optuna
from optuna.trial import Trial
from optuna.study import Study
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os

import ale_py
import highway_env

gym.register_envs(ale_py)
gym.register_envs(highway_env)
gym.register(id="custom_envs/RaceTrack-v0", entry_point=RacetrackEnv)

STUDY_NAME: str = "mujuco_halfcheetah_vpidqn"

class TestManager:
    def __init__(self, gym_env: GymEnv):
        self.gym_env: GymEnv = gym_env
        self.test_env: EnvironmentWrapper = EnvironmentWrapper(gym_env.create_environment())
    
    def test_model(self, model: VPIDQN, episodes_amount: int, use_mu: bool) -> List[float]:
        episode_count = 0
        episode_total_reward = 0.0
        episode_total_reward_list = []

        self.test_env.reset()
        #print(f"\rTesting: {episode_count} / {episodes_amount}", end="")
        while episode_count < episodes_amount:
            state = self.test_env.get_current_state()
            state = self.gym_env.preprocess(state)
            action = model.inference_next_action(state, use_mu)
            _, reward, terminated, truncated = self.test_env.step(action)
            episode_total_reward += reward
            if terminated or truncated:
                episode_total_reward_list.append(episode_total_reward)
                episode_total_reward = 0.0
                episode_count += 1
                #print(f"\rTesting: {episode_count} / {episodes_amount}", end="")
                self.test_env.reset()
                self.gym_env.end_episode()
        
        #print()

        return episode_total_reward_list

def main(gym_env: GymEnv, model: RLModel, data_file: str, create_new_data_file: bool = True, training_epochs: int = 20,
         test_episode_amount: int = 100, train_step_amount: int = 2000, trials_amount: int = 10, trial: Trial = None) -> List[float]:
    mu_writer = ResultWriter(data_file[:-4] + "_mu.txt", gym_env, model, train_step_amount, create_new_data_file)
    q_writer = ResultWriter(data_file[:-4] + "_q.txt", gym_env, model, train_step_amount, create_new_data_file)
    mu_writer.write()
    q_writer.write()
    for t in range(trials_amount):
        train_manager: TrainManager = TrainManager(copy.deepcopy(gym_env))
        test_manager: TestManager = TestManager(copy.deepcopy(gym_env))

        mu_mean_reward_history = []
        rewards = test_manager.test_model(model, test_episode_amount, True)
        mu_mean_reward: float = sum(rewards) / len(rewards)
        mu_mean_reward_history.append(mu_mean_reward)

        q_mean_reward_history = []
        rewards = test_manager.test_model(model, test_episode_amount, False)
        q_mean_reward: float = sum(rewards) / len(rewards)
        q_mean_reward_history.append(q_mean_reward)
        
        print(f"Start | mu: {mu_mean_reward} | q: {q_mean_reward}")

        consecutive_prune_reports: int = 0
        for i in range(training_epochs):
            start = time.perf_counter()
            train_manager.train_model(model, train_step_amount)
            end = time.perf_counter()

            rewards = test_manager.test_model(model, test_episode_amount, True)
            mu_mean_reward = sum(rewards) / len(rewards)
            mu_mean_reward_history.append(mu_mean_reward)

            rewards = test_manager.test_model(model, test_episode_amount, False)
            q_mean_reward = sum(rewards) / len(rewards)
            q_mean_reward_history.append(q_mean_reward)

            print(f"Epoch {i} | {end - start:.3f}s | mu: {mu_mean_reward} | q: {q_mean_reward}")

            if len(mu_mean_reward_history) >= 10:
                report_value = max(sum(mu_mean_reward_history[-10:]) / 10.0, sum(q_mean_reward_history[-10:]) / 10.0)
                trial.report(report_value, len(mu_mean_reward_history) - 10)
            
            if trial.should_prune():
                consecutive_prune_reports += 1
                if consecutive_prune_reports >= 5:
                    mu_writer.delete()
                    q_writer.delete()
                    del model
                    del train_manager
                    del test_manager
                    del mu_writer
                    del q_writer
                    print("deleted model in main")
                    raise optuna.exceptions.TrialPruned
            else:
                if consecutive_prune_reports > 0:
                    print(f"false alarm. consecutive_prune_reports was at {consecutive_prune_reports}")
                consecutive_prune_reports = 0

        mu_writer.add_trial_rewards(mu_mean_reward_history)
        mu_writer.write()

        q_writer.add_trial_rewards(q_mean_reward_history)
        q_writer.write()

        if t < trials_amount - 1:
            model = model.new()
    
    if sum(mu_mean_reward_history[-10:]) / 10.0 > sum(q_mean_reward_history[-10:]) / 10.0:
        return mu_mean_reward_history
    return q_mean_reward_history


def main_vpidqn(gym_env: GymEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int, 
                params: VPIDQNParams, trial: Trial) -> List[float]:
    vpidqn = VPIDQN(gym_env.state_size, gym_env.actions_amount, params, load_model_path=None)
    
    time_before: datetime = datetime.datetime.now()
    try:
        mean_reward_history: List[float] = main(gym_env, vpidqn, data_file, train_step_amount=train_step_amount, training_epochs=training_epochs, test_episode_amount=test_episode_amount,trials_amount=trials_amount, trial=trial)
    except Exception as e:
        time_after: datetime = datetime.datetime.now()
        delta: datetime.timedelta = time_after - time_before
        with open("interrupt_optuna_runtime.txt", "a") as file:
            file.write(f"{time_before},{time_after},{delta.seconds}s\n")
        
        del vpidqn
        print("deleted model in main_vpidqn")
        raise e
    else:
        time_after: datetime = datetime.datetime.now()
        delta: datetime.timedelta = time_after - time_before
        with open("optuna_runtime.txt", "a") as file:
            file.write(f"{time_before},{time_after},{delta.seconds}s\n")

        del vpidqn
        print("done")
    
    return mean_reward_history



def objective(trial: Trial):
    gamma = 0.99
    experience_replay_max_size = 250000
    value_lr = trial.suggest_float("value_lr", 1e-7, 1e-5, log=True)
    updates_to_renew_target_network = 2500
    updates_to_pass_posterior = trial.suggest_int("updates_to_pass_posterior", 1000, 10000)
    value_kl_weight = trial.suggest_float("value_kl_weight", -1.0, 1.0)
    min_eps = trial.suggest_float("min_eps", 0.1, 1.0)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    initial_beta = trial.suggest_float("initial_beta", 0.0, 1.0)
    heap_type = trial.suggest_categorical("heap_type", ["standard", "heap"])
    print(f"Running trial {trial.number=}. Parameters: {trial.params}")

    use_heap_experience_replay: bool = True if (heap_type == "heap") else False

    gym_env = MujucoHalfCheetah(discretization_factor=4)
    #gym_env = LunarLander()

    train_step_amount: int = 10000
    training_epochs: int = 100
    test_episode_amount: int = 10
    trials_amount: int = 1

    total_steps_of_eps_decay: int = 125000
    total_steps_of_beta_growth: int = 1000000
    #total_steps_of_eps_decay = round(0.125 * train_step_amount * training_epochs)
    #total_steps_of_beta_growth = train_step_amount * training_epochs
    params: VPIDQNParams = VPIDQNParams(gamma=gamma,
                    value_vpi_batch_size=32,
                    value_rand_batch_size=0,
                    experience_replay_max_size=experience_replay_max_size,
                    experience_replay_state_to_uint8=gym_env.image_state,
                    value_lr=value_lr,
                    updates_to_renew_target_network=updates_to_renew_target_network,
                    value_kl_weight=value_kl_weight,
                    updates_to_pass_posterior=updates_to_pass_posterior,
                    initial_eps=1.0,
                    min_eps=min_eps,
                    total_steps_of_eps_decay=total_steps_of_eps_decay,
                    use_uniform_distribution=False,
                    alpha=alpha,
                    initial_beta=initial_beta,
                    total_steps_of_beta_growth=total_steps_of_beta_growth,
                    use_heap_experience_replay=use_heap_experience_replay
                    )

    mean_reward_history: List[float] = main_vpidqn(gym_env, f"results/{STUDY_NAME}/trial{trial.number}.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount, params, trial)
    return sum(mean_reward_history[-10:]) / 10.0


def test_hyperparameters():
    study: Study = optuna.create_study(direction='maximize', 
                                       study_name=STUDY_NAME,
                                       sampler=optuna.samplers.TPESampler(),
                                       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10),
                                       storage=JournalStorage(JournalFileBackend(file_path=f"./results/{STUDY_NAME}/{STUDY_NAME}.log")),
                                       load_if_exists=True)
    
    study.optimize(objective, n_trials=10)

    print(study.best_params)

if __name__ == "__main__":
    try:
        os.mkdir(f"./results/{STUDY_NAME}")
        print(f"Directory './results/{STUDY_NAME}' created successfully.")
    except FileExistsError:
        print(f"Directory './results/{STUDY_NAME}' already exists.")

    test_hyperparameters()