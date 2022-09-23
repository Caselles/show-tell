import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
from teacher.teacher import Teacher
import random
from mpi4py import MPI
from language.build_dataset import sentence_from_configuration
from utils import get_instruction, generate_goals_demonstrator
from arguments import get_args
import pickle as pkl

#from mujoco_py import GlfwContext
#GlfwContext(offscreen=True)  # Create a window to init GLFW.


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def run_test(args, model_path=None):

    ################ SETTING UP TESTING ENVIRONMENT ################

    stable_prediction_accuracy = []
    stable_success_rate = []

    if args.compute_statistically_significant_results:
        nb_runs_predictability = 30
        nb_runs_reachability = 30
    else:
        nb_runs_predictability = 1
        nb_runs_reachability = 1

    ### CREATE ENV
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    args.env_params = get_env_params(env)

    for i in range(10000):

        env.render()
        observation_new, r, _, _ = env.step(np.array([0.,0.,0.,0.]))


    

if __name__ == '__main__':

    args = get_args()
    args.env_name = 'FetchManipulate3Objects-v0'

    run_test(args)


    
