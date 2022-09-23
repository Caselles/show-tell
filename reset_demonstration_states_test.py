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

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):

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

    # Fine-tuning for pedagogical teacher
    #if args.pedagogical_teacher:
        #ckpt_path = 
        #'/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output/2022-01-07 12:36:50_FetchManipulate3Objects-v0_gnn_per_object_NAIVE_TEACHER_FOR_FINETUNING/models/'
        #saved_model_path = ckpt_path + 'model_150.pt'
        #ckpt_path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output/2022-01-09 18:18:11_FetchManipulate3Objects-v0_gnn_per_object/models/'
        #saved_model_path = ckpt_path + 'model_60.pt'''
        #ckpt_path = '/media/gohu/backup_data/postdoc/gangstr_predicates_instructions/output/2022-01-10 16:52:14_FetchManipulate3Objects-v0_gnn_per_object/models/'
        #saved_model_path = ckpt_path + 'model_60.pt'''
        #policy.load(saved_model_path, args)
        #goal_sampler.set_goal_sampler_for_fine_tuning()
        #policy.buffer.load_buffer(ckpt_path + 'buffer_60.pkl')

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # Initialize Teacher
    teacher = Teacher(args)

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        time_dict = dict(goal_sampler=0,
                         rollout=0,
                         gs_update=0,
                         store=0,
                         norm_update=0,
                         policy_train=0,
                         policy_train_bc=0,
                         policy_train_sil=0,
                         eval=0,
                         epoch=0)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Annealing SQIL ratio of experience vs demos in replay buffer
        if epoch == -1 and args.sqil:
            policy.her_module.demos_vs_exp = 0.9

            print(policy.her_module.demos_vs_exp, ' SQIL ratio changed!')

        # Cycles loop
        for _ in range(args.n_cycles):

            # Sample goals
            t_i = time.time()
            '''sampled_goals = goal_sampler_teacher.sample_goals(all_achievable_goals, nb_goals=args.num_rollouts_per_mpi)
            sampled_masks = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(len(sampled_goals))])
            self_eval = False'''

            sampled_goals = goal_sampler_teacher.sample_goals(all_achievable_goals, nb_goals=args.num_rollouts_per_mpi)

            # fetch demo for the sampled goals
            demos = teacher.get_demo_for_goals(sampled_goals, saved=True)

            if args.reset_from_demos:
                reset_states = [x['obs'][0] for x in demos]

            predicted_goals = rollout_worker.predict_goal_from_demos(demos, goal_sampler_teacher.discovered_goals)
            prediction_success = [1 if (sampled_goals[pg_ind] == predicted_goal).all() else 0 for pg_ind, predicted_goal in enumerate(predicted_goals)]
            resampled_predicted_goals = goal_sampler_teacher.resample_incorrect_goal_predictions(sampled_goals, prediction_success)

            all_prediction_success = MPI.COMM_WORLD.gather(prediction_success, root=0)
            if rank == 0: print(np.mean(np.array(all_prediction_success)), 'PREDICTION SUCCESS RATE')
            
            demos_to_be_added_buffer = [demos[demo_ind] for demo_ind, success in enumerate(prediction_success) if success]
            resampled_predicted_masks = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(len(resampled_predicted_goals))])

            language_goal_ep = None
            time_dict['goal_sampler'] += time.time() - t_i

            # Control biased initializations
            if epoch < args.start_biased_init:
                biased_init = False
            else:
                biased_init = args.biased_init

            # Environment interactions
            t_i = time.time()

            self_eval = False
            
            '''if args.cuda:
                policy.model.actor.cuda()
                policy.model.critic.cuda()
                policy.model.critic_target.cuda()'''

            episodes = rollout_worker.generate_rollout(goals=resampled_predicted_goals,  # list of goal configurations
                                                       masks=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(len(resampled_predicted_goals))]),  # list of masks to be applied
                                                       self_eval=self_eval,  # whether the agent performs self-evaluations
                                                       true_eval=False,  # these are not offline evaluation episodes
                                                       biased_init=biased_init,  # whether initializations should be biased.
                                                       language_goal=language_goal_ep,
                                                       reset_from_demos=reset_states)   # ignore if no language used

if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    ### RUN TRAINING
    logdir = launch(args)