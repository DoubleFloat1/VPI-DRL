import copy
from models.VPIDQN import VPIDQN
from models.params import VPIDQNParams
from models.DQN import DQN
from models.rl_model import RLModel
import gymnasium as gym
from main_util import TrainManager, TestManager, ResultWriter
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

STUDY_NAME: str = "asteroid_vpidqn"

def main(gym_env: GymEnv, model: RLModel, data_file: str, create_new_data_file: bool = True, training_epochs: int = 20,
         test_episode_amount: int = 100, train_step_amount: int = 2000, trials_amount: int = 10, trial: Trial = None) -> List[float]:
    writer = ResultWriter(data_file, gym_env, model, train_step_amount, create_new_data_file)
    writer.write()
    for t in range(trials_amount):
        train_manager: TrainManager = TrainManager(copy.deepcopy(gym_env))
        test_manager: TestManager = TestManager(copy.deepcopy(gym_env))

        mean_reward_history = []
        rewards = test_manager.test_model(model, test_episode_amount)
        mean_reward: float = sum(rewards) / len(rewards)
        mean_reward_history.append(mean_reward)
        consecutive_prune_reports: int = 0
        for i in range(training_epochs):
            start = time.perf_counter()
            train_manager.train_model(model, train_step_amount)
            end = time.perf_counter()

            rewards = test_manager.test_model(model, test_episode_amount)
            mean_reward = sum(rewards) / len(rewards)
            print(f"Epoch {i} | {end - start:.3f}s | {mean_reward}")
            mean_reward_history.append(mean_reward)

            if len(mean_reward_history) >= 10:
                trial.report(sum(mean_reward_history[-10:]) / 10.0, len(mean_reward_history) - 10)
            
            if trial.should_prune():
                consecutive_prune_reports += 1
                if consecutive_prune_reports >= 5:
                    writer.delete()
                    del model
                    del train_manager
                    del test_manager
                    print("deleted model in main")
                    raise optuna.exceptions.TrialPruned
            else:
                if consecutive_prune_reports > 0:
                    print(f"false alarm. consecutive_prune_reports was at {consecutive_prune_reports}")
                consecutive_prune_reports = 0

        writer.add_trial_rewards(mean_reward_history)
        writer.write()

        if t < trials_amount - 1:
            model = model.new()
    
    return mean_reward_history


def main_dqn(gym_env: GymEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    total_steps_of_eps_decay: int = round(0.125 * train_step_amount * training_epochs)

    dqn = DQN(gym_env.state_size, gym_env.actions_amount, 
            experience_replay_max_size=100000,
            experience_replay_state_to_uint8=gym_env.image_state,
            updates_to_renew_target_network=2500,
            value_lr=1e-5,
            initial_eps=1.0,
            min_eps=0.1,
            total_steps_of_eps_decay=total_steps_of_eps_decay,
            load_model_path=None
            )
    
    main(gym_env, dqn, data_file, train_step_amount=train_step_amount, training_epochs=training_epochs, test_episode_amount=test_episode_amount,trials_amount=trials_amount)


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
    print(f"Running trial {trial.number=}")
    gamma = trial.suggest_float("gamma", 0.98, 1.0)
    experience_replay_max_size = trial.suggest_int("experience_replay_max_size", 100000, 500000)
    value_lr = trial.suggest_float("value_lr", 1e-7, 1e-5, log=True)
    updates_to_renew_target_network = trial.suggest_int("updates_to_renew_target_network", 1000, 10000)
    updates_to_pass_posterior = trial.suggest_int("updates_to_pass_posterior", 1000, 10000)
    value_kl_weight = trial.suggest_float("value_kl_weight", -1.0, 1.0)
    min_eps = trial.suggest_float("min_eps", 0.1, 0.5)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    initial_beta = trial.suggest_float("initial_beta", 0.0, 1.0)

    gym_env = SpaceInvaders(noop_max=5, frame_skip=2)
    #gym_env = LunarLander()

    train_step_amount: int = 20000
    training_epochs: int = 100
    test_episode_amount: int = 100
    trials_amount: int = 1

    total_steps_of_eps_decay: int = 1000000
    total_steps_of_beta_growth: int = 8000000
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
                    total_steps_of_beta_growth=total_steps_of_beta_growth
                    )

    #main_dqn(gym_env, "dqn.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)
    mean_reward_history: List[float] = main_vpidqn(gym_env, f"results/{STUDY_NAME}/trial{trial.number}.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount, params, trial)
    return sum(mean_reward_history[-10:]) / 10.0


def test_hyperparameters():
    study: Study = optuna.create_study(direction='maximize', 
                                       study_name=STUDY_NAME,
                                       sampler=optuna.samplers.TPESampler(),
                                       pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=10),
                                       storage=JournalStorage(JournalFileBackend(file_path=f"./results/{STUDY_NAME}/{STUDY_NAME}.log")),
                                       load_if_exists=True)
    
    study.enqueue_trial({
        "gamma": 0.99,
        "experience_replay_max_size": 250000,
        "value_lr": 3e-6,
        "updates_to_renew_target_network": 1500,
        "value_kl_weight": -0.3,
        "updates_to_pass_posterior": 8000,
        "min_eps": 0.1,
        "alpha": 0.4,
        "initial_beta": 0.1,
    })
    study.optimize(objective, n_trials=10)

    print(study.best_params)

if __name__ == "__main__":
    try:
        os.mkdir(f"./results/{STUDY_NAME}")
        print(f"Directory './results/{STUDY_NAME}' created successfully.")
    except FileExistsError:
        print(f"Directory './results/{STUDY_NAME}' already exists.")

    test_hyperparameters()