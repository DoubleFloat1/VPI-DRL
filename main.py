import copy
from models.VPIDQN import VPIDQN
from models.DQN import DQN
from models.rl_model import RLModel
import gymnasium as gym
from main_util import TrainManager, TestManager, ResultWriter
import time
import datetime
from gym_envs import *
from custom_envs.race_track import RacetrackEnv

import ale_py
import highway_env

gym.register_envs(ale_py)
gym.register_envs(highway_env)
gym.register(id="custom_envs/RaceTrack-v0", entry_point=RacetrackEnv)



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


def main_vpidqn(gym_env: GymEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    total_steps_of_eps_decay: int = round(0.125 * train_step_amount * training_epochs)
    total_steps_of_beta_growth: int = train_step_amount * training_epochs

    vpidqn = VPIDQN(gym_env.state_size, gym_env.actions_amount, 
                    value_vpi_batch_size=32,
                    value_rand_batch_size=0,
                    experience_replay_max_size=5000,
                    experience_replay_state_to_uint8=gym_env.image_state,
                    value_lr=1e-5,
                    updates_to_renew_target_network=2500,
                    value_kl_weight=0.1,
                    updates_to_pass_posterior=2500,
                    initial_eps=1.0,
                    min_eps=0.1,
                    total_steps_of_eps_decay=total_steps_of_eps_decay,
                    use_uniform_distribution=False,
                    alpha=0.6,
                    initial_beta=0.4,
                    total_steps_of_beta_growth=total_steps_of_beta_growth,
                    load_model_path=None
                    )
    
    main(gym_env, vpidqn, data_file, train_step_amount=train_step_amount, training_epochs=training_epochs, test_episode_amount=test_episode_amount,trials_amount=trials_amount)


if __name__ == "__main__":
    time_before: datetime = datetime.datetime.now()
    try:
        #gym_env = Asteroid(noop_max=5, frame_skip=2)
        gym_env = LunarLander()

        train_step_amount: int = 800
        training_epochs: int = 100
        test_episode_amount: int = 100
        trials_amount: int = 1

        #main_dqn(gym_env, "dqn.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)
        main_vpidqn(gym_env, "vpidqn.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)
    except KeyboardInterrupt:
        time_after: datetime = datetime.datetime.now()
        delta: datetime.timedelta = time_after - time_before
        with open("runtime.txt", "a") as file:
            file.write(f"{time_before},{time_after},{delta.seconds}s,Interrupted\n")
        
        raise KeyboardInterrupt
    else:
        time_after: datetime = datetime.datetime.now()
        delta: datetime.timedelta = time_after - time_before
        with open("runtime.txt", "a") as file:
            file.write(f"{time_before},{time_after},{delta.seconds}s,Completed\n")