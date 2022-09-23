import numpy as np 


import sys
sys.path.append('.')

from utils import generate_complex_goals_demonstrator, get_eval_goals, close_above_associations, get_all_subgoals, get_above_cube

class TellPolicy():

	def __init__(self, goal_space, agent_mode):

		self.agent_mode = agent_mode
		if self.agent_mode in ['naive', 'shapes_preference', 'colors_preference', 'pedagogical', 'shapes_preference_R1', 'colors_preference_R1']:
			self.is_teacher = True
		elif self.agent_mode in ['literal', 'pragmatic']:
			self.is_teacher = False
		else:
			self.is_teacher = None
	
		self.goal_space = np.array([g for g in goal_space])

		self.shapes = ['plain', 'striped']
		self.colors = ['blue', 'red']

		self.items_index_attributes = {0: ['striped', 'blue'], 1: ['plain', 'red'], 2: ['plain', 'blue']}

		self.close_predicates_cubes_involved = [[0,1], [1,2], [0,2]] # for close predicates
		self.above_predicates_cubes_involved = [[1,0], [1,2], [0,1], [0,2], [2,1], [2,0]] # for close predicates
		self.associations_close_above = close_above_associations() # for linking close and above predicates

		self.all_subgoals = np.array(get_all_subgoals())
		self.all_subgoals_str = [self.convert_goallist_to_goalstr(x) for x in self.all_subgoals]

		self.create_all_instructions_per_subgoals()

		self.nb_iter_pedagogical_teacher = 10000
		self.proba_boost = 0.1
		self.minimum_proba = 0.001


		if self.is_teacher:
			self.initialize_policy_teacher()
		else:
			from_scratch = False
			#self.initialize_policy_learner(from_scratch)


		

	def get_instruction_from_attributes(self, attribute1, attribute2, close=False, above=False, pyramid=False, attribute3=False):

		if close:
			instruction = 'Put ' + attribute1 + ' next to ' + attribute2 + '.'

		elif above:
			instruction = 'Put ' + attribute1 + ' above ' + attribute2 + '.'

		elif pyramid:
			instruction = 'Put ' + attribute1 + ' above ' + attribute2 + ' and ' + attribute3 + '.'

		else:
			raise NotImplementedError

		return instruction


	def create_instructions_for_subgoal(self, subgoal):

		instructions = []

		subgoal_binary = (np.array(subgoal) + 1)/ 2

		if sum(subgoal_binary) == 1 or sum(subgoal_binary) == 2: # close and above

			close_predicates = subgoal[:3]

			for idx, predicate in enumerate(close_predicates):
				if predicate==1:

					attributes_involved = [self.items_index_attributes[x] for x in self.close_predicates_cubes_involved[idx]]
					for att1 in attributes_involved[0]:
						for att2 in attributes_involved[1]:

							if sum(subgoal_binary) == 1:
								instructions.append(self.get_instruction_from_attributes(att1, att2, close=True))

								if att1 != att2: # close symmetry
									instructions.append(self.get_instruction_from_attributes(att2, att1, close=True))

							elif sum(subgoal_binary) == 2:
								instructions.append(self.get_instruction_from_attributes(att1, att2, above=True))


		elif sum(subgoal_binary) == 4: # pyramid

			close_predicates = subgoal[:3]

			for idx, predicate in enumerate(close_predicates):
				if predicate==-1:
					idx_above_cube = get_above_cube(self.close_predicates_cubes_involved[idx])


					attributes_involved = [self.items_index_attributes[idx_above_cube]]

					for x in self.close_predicates_cubes_involved[idx]:
						attributes_involved.append(self.items_index_attributes[x])

					for att1 in attributes_involved[0]:
						for att2 in attributes_involved[1]:
							for att3 in attributes_involved[2]:
								instructions.append(self.get_instruction_from_attributes(att1, att2, pyramid=True, attribute3=att3))

								if att2 != att3: # close symmetry
									instructions.append(self.get_instruction_from_attributes(att1, att3, pyramid=True, attribute3=att2))



		else: 
			raise NotImplementedError


		return instructions

	def convert_goallist_to_goalstr(self, goal):

		goalstr = ''

		for predicate in goal:
			goalstr += str(int(predicate))

		return goalstr
			

	def create_all_instructions_per_subgoals(self):

		self.all_instructions_per_goal = {}

		for subgoal in self.all_subgoals:

			subgoalstr = self.convert_goallist_to_goalstr(subgoal)

			self.all_instructions_per_goal[subgoalstr] = {}

			self.all_instructions_per_goal[subgoalstr]['instructions'] = self.create_instructions_for_subgoal(subgoal)

			nb_instructions = len(self.all_instructions_per_goal[subgoalstr]['instructions'])

			self.all_instructions_per_goal[subgoalstr]['probas'] = np.ones(nb_instructions) / nb_instructions

		print(self.all_instructions_per_goal)

		return


	def curriculumify_subgoals(self, subgoals):
		# cumulatively add goals

		curriculumified_subgoals = []

		if len(subgoals) == 1:

			curriculumified_subgoals = subgoals

		else:

			curriculum = np.zeros(9)

			for subgoal in subgoals:

				curriculum += (subgoal + 1)/2.

				curriculumified_subgoals.append(curriculum)

			curriculumified_subgoals = [(goal - 0.5)*2. for goal in curriculumified_subgoals]

		return curriculumified_subgoals[0]

	def decompose_goal_into_subgoals(self, goal):

		goal_binary = (goal+1)/2 # to from 1,-1 to 1,0

		subgoals = []

		# first close predicates

		close_predicates = goal_binary[:3]
		above_predicates = goal_binary[3:]

		if sum(above_predicates) == 0:

			if sum(close_predicates) == 0: # case 1 (all far apart)
				subgoals = [goal] 

			elif sum(close_predicates) == 1: # case 2 (1 close)
				subgoals = [goal]

			elif sum(close_predicates) == 2: # case 3 (close 2 a 2)
				active_predicates_index = np.where(goal == 1.)[0]
				for idx in active_predicates_index:
					subgoal = -1*np.ones(9)
					subgoal[idx] = 1.
					subgoals.append(subgoal)

				order = True
				# order counts but should be handled by the agent (common cube should not be moved on step 2)

			elif sum(close_predicates) == 3: # case 4 (all close)
				active_predicates_index = np.where(goal == 1.)[0]
				for idx in active_predicates_index:
					subgoal = -1*np.ones(9)
					subgoal[idx] = 1.
					subgoals.append(subgoal)

				order = True
				# order counts but should be handled by the agent (common cube should not be moved on step 2)

		elif sum(above_predicates) == 1:

			if sum(close_predicates) == 1: # case 5 (stack of 2)
				subgoals = [goal]

			elif sum(close_predicates) == 2: # case 6 (pyramid L)
				

				# first define goal for stack of two
				active_above_predicates_index = np.where(above_predicates == 1.)[0][0]
				first_goal = np.zeros(9)
				first_goal[3 + active_above_predicates_index] = 1.
				first_goal[self.associations_close_above[active_above_predicates_index]] = 1.

				# then create goal for close to complete the pyramid
				second_goal = goal_binary - first_goal

				# retrieve +1,-1 format
				subgoals = [(first_goal - 0.5)*2., (second_goal - 0.5)*2.]

		elif sum(above_predicates) == 2:

			if sum(close_predicates) == 2: # case 8 (stack of 3)

				# first goal is take one random active above predicate and its associated close
				active_above_predicates_index = np.where(above_predicates == 1.)[0]
				first_goal = np.zeros(9)
				rdn_above_predicate = np.random.choice(active_above_predicates_index)
				first_goal[3 + rdn_above_predicate] = 1.
				first_goal[self.associations_close_above[rdn_above_predicate]] = 1.

				# then complete goal
				second_goal = goal_binary - first_goal

				# retrieve +1,-1 format
				subgoals = [(first_goal - 0.5)*2., (second_goal - 0.5)*2.]

			elif sum(close_predicates) == 3: # case 9 (pyramid T)

				# first goal is put the base together
				first_goal = np.zeros(9)
				active_above_predicates_index = np.where(above_predicates == 1.)[0]
				self.above_predicates_cubes_involved[active_above_predicates_index[0]]
				common_cube = (set(self.above_predicates_cubes_involved[active_above_predicates_index[0]]) & set(self.above_predicates_cubes_involved[active_above_predicates_index[1]])).pop()
				for idx, cubes in enumerate(self.close_predicates_cubes_involved):
					if common_cube not in cubes:
						first_goal[idx] = 1.

				# then complete goal
				second_goal = goal_binary - first_goal

				# retrieve +1,-1 format
				subgoals = [(first_goal - 0.5)*2., (second_goal - 0.5)*2.]

		return subgoals

	def tell_subgoals(self, subgoals):

		instructions = []

		for subgoal in subgoals:
			# tell method from simple policy with options
			subgoalstr = self.convert_goallist_to_goalstr(subgoal)

			instruction_idx = np.random.choice(range(len(self.all_instructions_per_goal[subgoalstr]['instructions'])), p=self.all_instructions_per_goal[subgoalstr]['probas'])

			instruction = self.all_instructions_per_goal[subgoalstr]['instructions'][instruction_idx]

			instructions.append(instruction)

		return instructions

	def initialize_policy_teacher(self):

		self.tell_policy = {}

		if self.agent_mode == 'naive':
			pass

		elif self.agent_mode == 'pedagogical':
			self.learning = True
			self.learn_pedagogical_policy(self.learning)


		return 

	
	def learn_pedagogical_policy(self, learning=False):

		if learning:

			# for loop of training to infer own goals from instructions

			for it in range(self.nb_iter_pedagogical_teacher):

				if it % 1000 == 0:
					print(it, '/', str(self.nb_iter_pedagogical_teacher))

				# sample random goal

				sampled_subgoal = self.sample_subgoals(size=1)[0]

				# sample instruction

				sampled_instruction = self.tell_subgoals([sampled_subgoal])[0]

				# P(G/I)

				proba_goal_instruction = []

				for g in self.all_subgoals_str:

					if sampled_instruction in self.all_instructions_per_goal[g]['instructions']:

						#proba_instruction = self.tell_policy[g][self.all_instructions_per_goal[g].index(sampled_instruction)]
						proba_instruction = 1

					else:

						proba_instruction = 0

					proba_goal_instruction.append(proba_instruction)

				proba_goal_instruction = np.array(proba_goal_instruction)

				proba_goal_instruction = proba_goal_instruction / sum(proba_goal_instruction)

				# infer g by sampling P(G/I)

				subgoal_inferred = np.random.choice(self.all_subgoals_str, p=proba_goal_instruction)

				# if correct, reinforce, if not correct, lower

				sampled_subgoal_str = self.convert_goallist_to_goalstr(sampled_subgoal)

				instruction_proba_index = self.all_instructions_per_goal[sampled_subgoal_str]['instructions'].index(sampled_instruction)

				if subgoal_inferred == sampled_subgoal:

					self.all_instructions_per_goal[sampled_subgoal_str]['probas'][instruction_proba_index] += self.proba_boost

				else:

					if self.all_instructions_per_goal[sampled_subgoal_str]['probas'][instruction_proba_index] > self.proba_boost:
						self.all_instructions_per_goal[sampled_subgoal_str]['probas'][instruction_proba_index] -= self.proba_boost
					else:
						self.all_instructions_per_goal[sampled_subgoal_str]['probas'][instruction_proba_index] = 0


				# normalize

				self.normalize_tell_policy()

		else:


			for g in self.all_subgoals_str:

				for i_idx, instruction in enumerate(self.all_instructions_per_goal[g]['instructions']):

					for g2 in self.all_subgoals_str:

						if g != g2:

							if instruction in self.all_instructions_per_goal[g2]['instructions']:

								self.all_instructions_per_goal[g]['probas'][i_idx] = 0

			self.normalize_tell_policy()


		return

	def normalize_tell_policy(self):

		for g in self.all_subgoals_str:

			sum_probas = np.sum(self.all_instructions_per_goal[g]['probas'])
		
			self.all_instructions_per_goal[g]['probas'] = self.all_instructions_per_goal[g]['probas'] / sum_probas

		return

	def sample_goals(self, size):

		sampled_goals = self.goal_space[np.random.choice(range(len(self.goal_space)), size=size)]

		return sampled_goals

	def sample_subgoals(self, size):

		sampled_subgoals = self.all_subgoals[np.random.choice(range(len(self.all_subgoals)), size=size)]

		return sampled_subgoals

	def tell(self, goals):

		subgoals_and_instructions = []

		for _, goal in enumerate(goals):
			subgoals = self.decompose_goal_into_subgoals(goal)
			print('SUBGOALS')
			print(_)
			print(subgoals)
			instructions = self.tell_subgoals(subgoals)
			print(instructions)
			print('instructions')
			curriculum = self.curriculumify_subgoals(subgoals)
			assert (curriculum == goal).all()
			print('curriculum and original goal match.')
			print('\n')
			subgoals_and_instructions.append([curriculum, instructions])
			
		return subgoals_and_instructions

 

agent_mode = 'pedagogical'

goal_space = generate_complex_goals_demonstrator()

tell_policy = TellPolicy(goal_space, agent_mode)

goals = tell_policy.sample_goals(3)

#tell_policy.tell(goals)

ss = tell_policy.tell([np.array(goal_space)[25]])

from plots import visualize_tell_policy_complex_goals

visualize_tell_policy_complex_goals('', tell_policy.all_instructions_per_goal, teacher_mode=agent_mode)


import pdb;pdb.set_trace()