import copy
from models.VPIDQN import VPIDQN
from models.params import *
from models.DQN import DQN
from models.A2C import A2C
from models.DDPG import DDPG
from models.TD3 import TD3
from models.SAC import SAC
from models.VPISAC import VPISAC
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
         test_episode_amount: int = 100, train_step_amount: int = 2000, trials_amount: int = 1, training_envs_amount: int = 1):
    writer = ResultWriter(data_file, gym_env, model, train_step_amount, create_new_data_file)
    writer.write()
    for t in range(trials_amount):
        train_manager: TrainManager = TrainManager(copy.deepcopy(gym_env), envs_amount=training_envs_amount)
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

def begin_training(gym_env: GymEnv, model: RLModel, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, 
                   trials_amount: int, training_envs_amount: int = 1):
    time_before: datetime = datetime.datetime.now()
    try:
        main(gym_env, model, data_file, train_step_amount=train_step_amount, training_epochs=training_epochs, test_episode_amount=test_episode_amount,
             trials_amount=trials_amount, training_envs_amount=training_envs_amount)
    except Exception as e:
        time_after: datetime = datetime.datetime.now()
        delta: datetime.timedelta = time_after - time_before
        with open("interrupt_runtime.txt", "a") as file:
            file.write(f"{time_before},{time_after},{delta.seconds}s\n")
        
        raise e
    else:
        time_after: datetime = datetime.datetime.now()
        delta: datetime.timedelta = time_after - time_before
        with open("runtime.txt", "a") as file:
            file.write(f"{time_before},{time_after},{delta.seconds}s\n")

        del model
        print("done")

def main_dqn(gym_env: GymEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    total_steps_of_eps_decay: int = round(0.125 * train_step_amount * training_epochs)

    dqn = DQN(gym_env.state_size, gym_env.actions_amount, 
            experience_replay_max_size=250000,
            experience_replay_state_to_uint8=gym_env.image_state,
            updates_to_renew_target_network=2500,
            value_batch_size=128,
            value_lr=4e-7,
            initial_eps=1.0,
            min_eps=0.2,
            total_steps_of_eps_decay=total_steps_of_eps_decay,
            load_model_path=None
            )
    
    begin_training(gym_env, dqn, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount)


def main_vpidqn(gym_env: GymEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    total_steps_of_eps_decay: int = round(train_step_amount * training_epochs / 8)
    total_steps_of_beta_growth: int = train_step_amount * training_epochs
    params: VPIDQNParams = VPIDQNParams(gamma=0.99,
                    value_vpi_batch_size=128,
                    value_rand_batch_size=0,
                    experience_replay_max_size=250000,
                    experience_replay_state_to_uint8=gym_env.image_state,
                    value_lr=1e-4,
                    updates_to_renew_target_network=250,
                    value_kl_weight=0.5,
                    updates_to_pass_posterior=600,
                    initial_eps=1.0,
                    min_eps=0.2,
                    total_steps_of_eps_decay=total_steps_of_eps_decay,
                    alpha=0.1,
                    initial_beta=1.0,
                    max_beta=1.0,
                    total_steps_of_beta_growth=total_steps_of_beta_growth,
                    use_heap_experience_replay=True,
                    experience_replays_adds_per_update=64,
                    inference_use_mu=False
                    )

    vpidqn = VPIDQN(gym_env.state_size, gym_env.actions_amount, params, load_model_path=None)
    begin_training(gym_env, vpidqn, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount)

def main_a2c(gym_env: GymEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    params: A2CParams = A2CParams(
        gamma=0.99,
        actor_lr=1e-4,
        critic_lr=2e-4,
        batch_size=512,
        n_step_return=5,
        envs_amount=4
    )

    a2c = A2C(gym_env.state_size, gym_env.actions_amount, params)
    begin_training(gym_env, a2c, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount, training_envs_amount=params.envs_amount)

def main_ddpg(gym_env: MujocoEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    total_steps_of_eps_decay = int(train_step_amount * training_epochs / 8)
    uniform_start_steps = int(train_step_amount * training_epochs / 100)
    params: DDPGParams = DDPGParams(
        gamma=0.99,
        q_value_lr=1e-3,
        policy_lr=1e-3,
        batch_size=100,
        experience_replay_max_size=1000000,
        polyak=0.995,
        steps_per_update=50,
        repeats_per_update=50,
        initial_eps=0.1,
        min_eps=0.1,
        total_steps_of_eps_decay=total_steps_of_eps_decay,
        uniform_start_steps=10000
    )

    ddpg = DDPG(gym_env.state_size, gym_env.action_dimension, gym_env.action_range_min, gym_env.action_range_max, params)
    begin_training(gym_env, ddpg, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount, training_envs_amount=1)

def main_td3(gym_env: MujocoEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    total_steps_of_eps_decay = int(train_step_amount * training_epochs / 8)
    uniform_start_steps = int(train_step_amount * training_epochs / 100)
    params: TD3Params = TD3Params(
        gamma=0.99,
        q_value_lr=1e-3,
        policy_lr=1e-3,
        batch_size=100,
        experience_replay_max_size=1000000,
        polyak=0.995,
        steps_per_update=50,
        repeats_per_update=50,
        initial_eps=0.1,
        min_eps=0.1,
        total_steps_of_eps_decay=total_steps_of_eps_decay,
        target_noise_std=0.2,
        noise_clip=0.5,
        policy_delay=2,
        uniform_start_steps=10000
    )

    td3 = TD3(gym_env.state_size, gym_env.action_dimension, gym_env.action_range_min, gym_env.action_range_max, params)
    begin_training(gym_env, td3, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount, training_envs_amount=1)

def main_sac(gym_env: MujocoEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    uniform_start_steps = int(train_step_amount * training_epochs / 100)
    params: SACParams = SACParams(
        gamma=0.99,
        q_value_lr=1e-4,
        policy_lr=1e-4,
        batch_size=100,
        experience_replay_max_size=1000000,
        polyak=0.995,
        steps_per_update=50,
        repeats_per_update=50,
        entropy_const=0.2,
        uniform_start_steps=10000
    )

    sac = SAC(gym_env.state_size, gym_env.action_dimension, gym_env.action_range_min, gym_env.action_range_max, params)
    begin_training(gym_env, sac, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount, training_envs_amount=1)

def main_vpisac(gym_env: MujocoEnv, data_file: str, train_step_amount: int, training_epochs: int, test_episode_amount: int, trials_amount: int):
    params: VPISACParams = VPISACParams(
        gamma=0.99,
        q_value_lr=1e-4,
        policy_lr=1e-4,
        vpi_lr=1e-4,
        batch_size=100,
        experience_replay_max_size=1000000,
        polyak=0.995,
        steps_per_update=50,
        repeats_per_update=50,
        vpi_output_norms_amount=16,
        vpi_const=0.2,
        uniform_start_steps=10000,
        weighted_normals=True
    )

    vpisac = VPISAC(gym_env.state_size, gym_env.action_dimension, gym_env.action_range_min, gym_env.action_range_max, params)
    begin_training(gym_env, vpisac, data_file, train_step_amount, training_epochs, test_episode_amount, trials_amount, training_envs_amount=1)

if __name__ == "__main__":
    #gym_env = MujucoHalfCheetah(discretization_factor=None)
    #gym_env = MujucoAnt(discretization_factor=None, ctrl_cost_weight=0.5, healthy_reward=1.0, contact_cost_weight=5e-4)
    gym_env = MujucoWalker2D(discretization_factor=None, healthy_reward=1.0, ctrl_cost_weight=1e-3, terminate_when_unhealthy=True)

    train_step_amount: int = 10000
    training_epochs: int = 200
    test_episode_amount: int = 10
    trials_amount: int = 1

    #main_ddpg(gym_env, "results/ddpg.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)
    #main_td3(gym_env, "results/td3.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)
    #main_sac(gym_env, "results/sac.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)
    main_vpisac(gym_env, "results/vpisac.txt", train_step_amount, training_epochs, test_episode_amount, trials_amount)