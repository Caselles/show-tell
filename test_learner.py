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

    goal_sampler = GoalSampler(args)

    ### INITIALIZE TEACHER
    teacher = Teacher(args)

    ### LIST OF POSSIBLE AGENTS
    if model_path==None:

        if args.learner_to_test == 'naive_teacher':
            path = '/home/gohu/workspace/postdoc/show-tell/gangstr_predicates_instructions/output_teacher/2022-06-30 20:42:26_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_2_ATTRIBUTES/models/'
            model_path = path + 'model_170.pt'

        if args.learner_to_test == 'pedagogical_teacher':
            path = '/home/gohu/Downloads/'
            model_path = path + 'model_50azi.pt'

        if args.learner_to_test == 'pedagogical_teacher_original':
            path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output_teacher/2022-01-11 17:01:34_FetchManipulate3Objects-v0_gnn_per_object_PEDAGOGICAL_TEACHER_DONE_WORKS/models/'
            model_path = path + 'model_170.pt'

    else:

        path = model_path
        model_path = path + '/models/model_1.pt'

    ### CREATE AND LOAD THE AGENT
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    ### ROLLOUT WORKER
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    ### INITIALIZE EVALUATION GOALS
    eval_goals = np.array(generate_goals_demonstrator())
    proba_goals = eval_goals.copy()
    eval_goals = np.array([[1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.], [-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.], [-1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.]])
    no_noise = False
    eval_masks = np.zeros((len(eval_goals), 9))
    language_goal = None
    ambiguity_analysis = True

    ### ENABLE ILLUSTRATIVE EXAMPLE SETUP
    if args.illustrative_example:
        # one is 0 over 1, the other is 0 above 1 and 0 close to 1
        eval_goals = np.array([[1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.], [-1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]])
        proba_goals = eval_goals.copy()
        # the trick is that the initialization will gave 0 close to 1 already in one case, and not the other

    ################ PREDICTABILITY RESULTS ################
    if args.predictability:
        for i in range(nb_runs_predictability):
            demos = teacher.get_demo_for_goals(eval_goals, saved=True)
            predicted_goals = rollout_worker.predict_goal_from_demos(demos, list(eval_goals))
            self_eval = False
            prediction_success = [1 if (eval_goals[pg_ind] == predicted_goal).all() else 0 for pg_ind, predicted_goal in enumerate(predicted_goals)]
            if ambiguity_analysis:
                for pg_ind, pred in enumerate(prediction_success):
                    if not pred:
                        print('pursued goal in the demo =', eval_goals[pg_ind])
                        print('achieved goal in the demo =', demos[pg_ind]['ag'][-1])
                        print('is pursued goal is the same as achieved goal?', (eval_goals[pg_ind]==demos[pg_ind]['ag'][-1]).all())
                        print('predicted goal =', predicted_goals[pg_ind])
            print(prediction_success, 'Prediction success on this run.')
            print(np.mean(prediction_success), 'Mean predictability on this run.')
            stable_prediction_accuracy.append(prediction_success)

        if args.compute_statistically_significant_results:
            print(stable_prediction_accuracy, 'All prediction accuracy results.')
            print(np.mean(stable_prediction_accuracy), 'Average prediction accuracy.')

    #eval_goals = np.array([np.array([ 1.,  1.,  1., -1., -1., -1., -1., -1., -1.])])

    ################ REACHABILITY RESULTS ################
    if args.reachability:
        for i in range(nb_runs_reachability):
            episodes = rollout_worker.generate_rollout(eval_goals, eval_masks, self_eval=no_noise, true_eval=no_noise, biased_init=True, animated=True, 
                language_goal=language_goal, verbose=False, return_proba=None, illustrative_example=args.illustrative_example, save_ambiguous_episodes=False)
            
            success_rate = np.array([int(e['success'][-1]) for e in episodes])
            stable_success_rate.append(success_rate)
            print(success_rate, 'Successes on this run.')
            print(success_rate.mean(), 'Mean success rate on this run.')

        if args.compute_statistically_significant_results:
            print(stable_success_rate, 'All success rate results.')
            print(np.mean(stable_success_rate), 'Average success rate.')

        sr_ambiguity = []
        for e in episodes:
            active_predicates_goal = np.where(e['g'][-1] == 1)
            sr_ambiguity.append((e['ag'][-1] == e['g'][-1]).all())
        print(sr_ambiguity, 'All ambiguity results.')
        print(np.mean(sr_ambiguity), 'Average ambiguity rate.')

    if 'models' in path:
        save_path = path.split('models')[0]
    else:
        save_path = path + '/'

    '''with open(save_path + 'results_predictability_reachability.txt', 'w') as f:
        f.write('Predictability:' + str(np.mean(stable_prediction_accuracy)))
        f.write('\n')
        f.write('Reachability:' + str(np.mean(stable_success_rate)))'''

    return np.mean(stable_prediction_accuracy), np.mean(stable_success_rate)

if __name__ == '__main__':

    args = get_args()
    args.env_name = 'FetchManipulate3Objects-v0'

    run_test(args)


    