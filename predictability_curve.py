import torch
import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
from rollout import RolloutWorker
from temporary_lg_goal_sampler import LanguageGoalSampler
from goal_sampler import GoalSampler
from goal_sampler_teacher import GoalSamplerTeacher
from teacher.teacher import Teacher
from utils import init_storage, get_instruction, get_eval_goals, generate_goals_demonstrator
import time
from mpi_utils import logger
from language.build_dataset import sentence_from_configuration
from test_learner_mpi import run_test_mpi
import matplotlib.pyplot as plt
from plots_v2 import setup_figure


def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def predictability_curve(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    language_goal = None
    goal_sampler = GoalSampler(args)
    goal_sampler_teacher = GoalSamplerTeacher(args)
    all_achievable_goals = generate_goals_demonstrator()

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # Initialize Teacher
    teacher = Teacher(args)

    prs = []
    for config in ['naive_literal', 'naive_pragmatic', 'pedagogical_literal', 'pedagogical_pragmatic']:
        if 'naive' in config:
            args.teacher_mode = 'naive'
        if 'pedagogical' in config:
            args.teacher_mode = 'pedagogical'
        pr = []
        sr = []
        for pt_nb in range(0,110,10):
            model_path = args.learner_to_test + config + '/' + os.listdir(args.learner_to_test + config)[0] + '/'
            predictability, reachability = run_test_mpi(args, model_path=model_path, write=False, pt_nb=pt_nb)
            pr.append(predictability)
            print(pr)

        prs.append(pr)

    #prs = [np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) + np.random.rand() for i in range(4)]
    colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0.929, 0.04, 0.125],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]

    # Plotting
    if rank == 0:
        artists, ax = setup_figure(xlabel='Epochs (x10)',
                                   # xlabel='Epochs',
                                   ylabel='Prediction Accuracy',
                                   xlim=[-0.2, len(prs[0])],
                                   ylim=[-0.02, 1 + 0.2])

        for i in range(len(prs)):
            plt.plot(range(len(prs[0])), prs[i], color=colors[i], marker='x', markersize=30, linewidth=10)
            #plt.fill_between(len(pr), pr, sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)
        plt.savefig(model_path + 'predictability_curve.pdf')

if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    # launch script
    predictability_curve(args)