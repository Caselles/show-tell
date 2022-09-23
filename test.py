import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from language.build_dataset import sentence_from_configuration
from utils import get_instruction, generate_goals_demonstrator
from arguments import get_args
import pickle as pkl

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1


    agent_to_test = 'custom'

    if agent_to_test == 'naive_teacher':
        path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING_70%_PEDAGOGICAL_SR/models/'
        model_path = path + 'model_150.pt'

    if agent_to_test == 'pedagogical_teacher':
        path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-11 17:01:34_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_DONE_WORKS/models/'
        model_path = path + 'model_150.pt'

    if agent_to_test == 'custom':
        path = ''
        model_path = path + 'model_xx.pt'

    # with open(path + 'config.json', 'r') as f:
    #     params = json.load(f)
    # args = SimpleNamespace(**params)
    args = get_args()

    if args.algo == 'continuous':
        args.env_name = 'FetchManipulate3ObjectsContinuous-v0'
        args.multi_criteria_her = True
    else:
        args.env_name = 'FetchManipulate3Objects-v0'

    # Make the environment
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

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # eval_goals = goal_sampler.valid_goals
    #eval_goals, eval_masks = goal_sampler.generate_eval_goals()
    eval_goals = np.array(generate_goals_demonstrator())
    proba_goals = eval_goals.copy()
    no_noise = False
    eval_masks = np.zeros((len(eval_goals), 9))
    if args.algo == 'language':
        language_goal = get_instruction()
        eval_goals = np.array([goal_sampler.valid_goals[0] for _ in range(len(language_goal))])
    else:
        language_goal = None
    inits = [None] * len(eval_goals)
    all_results = []

    illustrative_example = False

    if illustrative_example:
        # one is 0 over 1, the other is 0 above 1 and 0 close to 1
        eval_goals = np.array([[1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.], [-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]])
        proba_goals = eval_goals.copy()
        # the trick is that the initialization will gave 0 close to 1 already in one case, and not the other

        no_noise = False # we need to have a stochastic policy


    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, eval_masks, self_eval=no_noise, true_eval=no_noise, biased_init=True, animated=False, 
            language_goal=language_goal, verbose=True, return_proba=proba_goals, illustrative_example=illustrative_example)
        if args.algo == 'language':
            results = np.array([e['language_goal'] in sentence_from_configuration(e['ag'][-1], all=True) for e in episodes]).astype(np.int)
        elif args.algo == 'continuous':
            results = np.array([e['rewards'][-1] == 3. for e in episodes])
        else:
            results = np.array([e['rewards'][-1] == 3. for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))

