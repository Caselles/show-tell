import sys 

sys.path.append('../')

import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from utils import generate_goals_demonstrator, generate_simple_goals_demonstrator
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import random
from mpi4py import MPI
import pickle
from copy import deepcopy
from arguments import get_args
import os
from tell_policy.tell_policy import TellPolicy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Teacher:
    def __init__(self, args):

        self.policy = None
        self.demo_dataset = None
        self.all_goals = generate_goals_demonstrator()
        if args.teacher_action_mode == 'naive':
            self.path_demos = '/media/gohu/backup_data/postdoc/show-tell/demos_datasets/naive_teacher_1000/'
        if args.teacher_action_mode == 'pedagogical':
            self.path_demos = '/media/gohu/backup_data/postdoc/show-tell/demos_datasets/pedagogical_teacher_1000/'

        self.nb_available_demos = len(os.listdir(self.path_demos+'goal_0/'))

        self.initialize_tell_policy(args)


    def get_demo_for_goals(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                ind = np.random.randint(self.nb_available_demos)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos


    def get_demo_for_goals_demo_encoder_training(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                ind = np.random.randint(9000)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos

    def get_demo_for_goals_demo_encoder_testing(self, goals, saved=False):

        demos = []

        for goal in goals:

            if saved:

                goal_ind = self.all_goals.index(goal.tolist())

                ind = np.random.randint(9000, 10000)

                with open(self.path_demos + 'goal_' + str(goal_ind) + '/demo_' + str(ind) + '.pkl', 'rb') as f:
                    demo = pickle.load(f)

                # check if we get demo for the right goal
                assert (goal == demo['g'][-1]).all()

                demos.append(demo)

            else:
                raise NotImplementedError

        return demos

    def initialize_tell_policy(self, args):

        self.language_tools = TellPolicy(generate_simple_goals_demonstrator(), args.teacher_language_mode)

        return 

