import numpy as np 


class TellPolicy():

	def __init__(self, goal_space, agent_mode):

		self.agent_mode = agent_mode
		if self.agent_mode in ['naive', 'shapes_preference', 'colors_preference', 'pedagogical', 'shapes_preference_R1', 'colors_preference_R1']:
			self.is_teacher = True
		elif self.agent_mode in ['literal', 'pragmatic']:
			self.is_teacher = False
		else:
			self.is_teacher = None
	
		self.goal_space = [g[:3] for g in goal_space]

		self.shapes = ['plain', 'striped']
		self.colors = ['blue', 'red']

		self.items_index_attributes = {0: ['striped', 'blue'], 1: ['plain', 'red'], 2: ['plain', 'blue']}

		self.predicates_cubes_involved = [[0,1], [1,2], [0,2]] # attention à l'ordre!!!

		self.create_all_instructions_per_goal()
		self.goal_space_str = [g for g in self.all_instructions_per_goal.keys()]

		if self.is_teacher:
			self.initialize_policy_teacher()
		else:
			from_scratch = False
			self.initialize_policy_learner(from_scratch)


		if self.agent_mode == 'pedagogical':
			self.nb_iter_pedagogical_teacher = 1000
			self.learn_pedagogical_policy()


		self.proba_boost = 0.1
		self.minimum_proba = 0.001

	def get_instruction_from_attributes(self, attribute1, attribute2):

		instruction = 'Put ' + attribute1 + ' next to ' + attribute2 + '.'

		return instruction

	def get_attributes_from_instruction(self, instruction):

		attributes = instruction.split('Put ')[1].split('.')[0]

		attribute1 = attributes.split(' next to ')[0]
		attribute2 = attributes.split(' next to ')[1]

		return [attribute1, attribute2]

	def convert_goallist_to_goalstr(self, goal, nine_predicates=False):

		goalstr = ''
		if nine_predicates:
			for predicate in goal[:3]:
				goalstr+= str(int(predicate))
		else:
			for predicate in goal:
				goalstr+= str(int(predicate))

		return goalstr

	def create_all_instructions_per_goal(self):

		self.all_instructions_per_goal = {}

		for g in self.goal_space:

			instructions = []
			
			for idx, predicate in enumerate(g):
				if predicate==1:

					attributes_involved = [self.items_index_attributes[x] for x in self.predicates_cubes_involved[idx]]
					for att1 in attributes_involved[0]:
						for att2 in attributes_involved[1]:
							instructions.append(self.get_instruction_from_attributes(att1, att2))

							if att1 != att2:
								instructions.append(self.get_instruction_from_attributes(att2, att1))

			self.all_instructions_per_goal[self.convert_goallist_to_goalstr(g)] = instructions

		return

	def initialize_policy_teacher(self):

		self.tell_policy = {}

		for g in self.goal_space_str:
			instruction_probas = np.array([1]*len(self.all_instructions_per_goal[g])) / len(self.all_instructions_per_goal[g])
			self.tell_policy[g] = instruction_probas

		if self.agent_mode == 'naive' or self.agent_mode == 'pedagogical':

			return 

		elif self.agent_mode == 'shapes_preference' or self.agent_mode == 'colors_preference':

			# increase sampling probability for instructions referring to shapes
			for g in self.goal_space_str:
				for instruction_idx, instruction in enumerate(self.all_instructions_per_goal[g]):
					attributes_instruction = self.get_attributes_from_instruction(instruction)

					proba_boost = 0
					if self.agent_mode == 'shapes_preference':
						preferred_attributes = self.shapes
					elif self.agent_mode == 'colors_preference':
						preferred_attributes = self.colors

					for att in attributes_instruction:
						if att in preferred_attributes:
							proba_boost += 10

					self.tell_policy[g][instruction_idx] += proba_boost

			# normalize tell policy

			self.normalize_tell_policy()

			return

		elif self.agent_mode == 'shapes_preference_R1' or self.agent_mode == 'colors_preference_R1':

			if self.agent_mode == 'shapes_preference_R1':
				preferred_attributes = self.shapes
			elif self.agent_mode == 'colors_preference_R1':
				preferred_attributes = self.colors

			# increase sampling probability for instructions referring to shapes
			for goal_ind, g in enumerate(self.goal_space_str):

				if self.agent_mode == 'colors_preference_R1':

					if goal_ind == 0:

						valid_instructions = ['Put blue next to plain.', 'Put plain next to blue.']

					if goal_ind == 1:

						valid_instructions = ['Put blue next to red.', 'Put red next to blue.']

					if goal_ind == 2:

						valid_instructions = ['Put blue next to blue.']

				elif self.agent_mode == 'shapes_preference_R1':

					if goal_ind == 0:

						valid_instructions = ['Put blue next to plain.', 'Put plain next to blue.']

					if goal_ind == 1:

						valid_instructions = ['Put plain next to plain.']

					if goal_ind == 2:

						valid_instructions = ['Put striped next to plain.', 'Put plain next to striped.']

				for instruction_idx, instruction in enumerate(self.all_instructions_per_goal[g]):

					if instruction in valid_instructions:

						self.tell_policy[g][instruction_idx] = 1.

					else:

						self.tell_policy[g][instruction_idx] = 0.			


			self.normalize_tell_policy()


			return 

		return False

	def initialize_policy_learner(self, from_scratch=False):

		self.tell_policy = {}
		self.learner_goal_instruction_table = {}

		if not from_scratch:
			self.tell_policy = {}

			for g in self.goal_space_str:
				instruction_probas = np.array([1]*len(self.all_instructions_per_goal[g])) / len(self.all_instructions_per_goal[g])
				self.tell_policy[g] = instruction_probas

			self.learner_goal_instruction_table = self.all_instructions_per_goal

		return 

	def get_proba(self, goal, instruction, lookup_table):

		proba = self.tell_policy[goal][lookup_table[goal].index(instruction)]

		return proba

	def add_goal_instruction_policy_learner(self, goal, instruction):

		goal = self.convert_goallist_to_goalstr(goal, nine_predicates=True)

		if goal in self.learner_goal_instruction_table.keys():
			self.learner_goal_instruction_table[goal].append(instruction)
			self.tell_policy[goal].append(1/len(self.tell_policy[goal]))
		else:
			self.learner_goal_instruction_table[goal] = [instruction]
			self.tell_policy[goal] = [1.]

		self.normalize_tell_policy()

		return

	def bayesian_goal_inference(self, instructions):

		if self.is_teacher:
			lookup_table = self.all_instructions_per_goal
		else:
			lookup_table = self.learner_goal_instruction_table

		inferred_goals = []

		for instr in instructions:

			goal_candidates = []
			goal_proba_candidates = []

			for g in lookup_table.keys():
				if instr in lookup_table[g]:
					goal_candidates.append(g)
					goal_proba_candidates.append(self.get_proba(g, instr, lookup_table))

			if len(goal_candidates) == 0:

				# random goal
				inferred_goal = np.random.choice(self.tell_policy.keys())

			elif len(goal_candidates) == 1:

				# select goal
				inferred_goal = goal_candidates[0]

			elif len(goal_candidates) > 1:

				# fetch probas and bgi
				if sum(goal_proba_candidates) == 0.:
					normalized_goal_proba_candidates = [1/len(goal_proba_candidates) for _ in goal_proba_candidates]
				else:
					normalized_goal_proba_candidates = goal_proba_candidates/sum(goal_proba_candidates)
				inferred_goal = np.random.choice(goal_candidates, p=normalized_goal_proba_candidates)

			inferred_goals.append(inferred_goal)


		return inferred_goals

	def update_policy_learner_pragmatism(self, goal, instruction):

		#goal = self.convert_goallist_to_goalstr(goal, nine_predicates=True)

		instruction_proba_index = self.learner_goal_instruction_table[goal].index(instruction)

		if self.tell_policy[goal][instruction_proba_index] <= self.proba_boost:

			self.tell_policy[goal][instruction_proba_index] = self.minimum_proba

		else:

			self.tell_policy[goal][instruction_proba_index] -= self.proba_boost

		self.normalize_tell_policy()

		return

	def normalize_tell_policy(self):

		for g in self.tell_policy.keys():

			sum_probas = np.sum(self.tell_policy[g])
		
			self.tell_policy[g] = self.tell_policy[g] / sum_probas

		return
		
	def learn_pedagogical_policy(self, learning=False):

		if learning:

			# for loop of training to infer own goals from instructions

			for it in range(self.nb_iter_pedagogical_teacher):

				if it % 1000 == 0:
					print(it, '/', str(self.nb_iter_pedagogical_teacher))

				# sample random goal

				sampled_goal = self.sample_goals(size=1)[0]

				# sample instruction

				sampled_instruction = self.tell([sampled_goal])[0]

				# P(G/I)

				proba_goal_instruction = []

				for g in self.goal_space_str:

					if sampled_instruction in self.all_instructions_per_goal[g]:

						#proba_instruction = self.tell_policy[g][self.all_instructions_per_goal[g].index(sampled_instruction)]
						proba_instruction = 1

					else:

						proba_instruction = 0

					proba_goal_instruction.append(proba_instruction)

				proba_goal_instruction = np.array(proba_goal_instruction)

				proba_goal_instruction = proba_goal_instruction / sum(proba_goal_instruction)

				# infer g by sampling P(G/I)

				goal_inferred = np.random.choice(self.goal_space_str, p=proba_goal_instruction)

				# if correct, reinforce, if not correct, lower

				instruction_proba_index = self.all_instructions_per_goal[sampled_goal].index(sampled_instruction)

				if goal_inferred == sampled_goal:

					for idx, proba in enumerate(self.tell_policy[sampled_goal]):
						if idx == instruction_proba_index:
							self.tell_policy[sampled_goal][idx] += self.proba_boost
						'''else:
							if self.tell_policy[sampled_goal][idx] > self.proba_boost:
								self.tell_policy[sampled_goal][idx] -= self.proba_boost
							else:
								self.tell_policy[sampled_goal][idx] = 0'''

				else:

					for idx, proba in enumerate(self.tell_policy[sampled_goal]):
						if idx == instruction_proba_index:

							if self.tell_policy[sampled_goal][idx] > self.proba_boost:
								self.tell_policy[sampled_goal][idx] -= self.proba_boost
							else:
								self.tell_policy[sampled_goal][idx] = 0

						'''else:
							self.tell_policy[sampled_goal][idx] += self.proba_boost'''

				# normalize

				self.normalize_tell_policy()

		else:

			for g in self.goal_space_str:

				for instruction in self.all_instructions_per_goal[g]:

					instruction_proba_index = self.all_instructions_per_goal[g].index(instruction)

					for g2 in self.goal_space_str:

						if g != g2:

							if instruction in self.all_instructions_per_goal[g2]:

								self.tell_policy[g][instruction_proba_index] = 0

			self.normalize_tell_policy()

		return

	def sample_goals(self, size):

		sampled_goal = np.random.choice(self.goal_space_str, size=size)

		return sampled_goal

	def tell(self, goals, verbose=False):

		instructions = []

		for goal in goals:
			if not isinstance(goal, str):
				goal = self.convert_goallist_to_goalstr(goal, nine_predicates=True)
			instruction_idx = np.random.choice(range(len(self.tell_policy[goal])), p=self.tell_policy[goal])
			instruction = self.all_instructions_per_goal[goal][instruction_idx]

			if verbose:
				print(goal, 'goal')
				print('instructions:', self.all_instructions_per_goal[goal])
				print('instruction probas:', self.tell_policy[goal])

			instructions.append(instruction)
			

		return instructions

	#def evaluate_policy_inference(self, )


'''
print('Naive teacher tell policy ---------------------')
teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='naive')


# Sampling phase
sampled_goals = teacher.sample_goals(size=2)
print(sampled_goals)
instructions = teacher.tell(sampled_goals, verbose=True)
print(instructions)


print('Shape preference teacher tell policy ---------------------')
teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='shapes_preference')


# Sampling phase
sampled_goals = teacher.sample_goals(size=2)
print(sampled_goals)
instructions = teacher.tell(sampled_goals, verbose=True)
print(instructions)

print('Color preference teacher tell policy ---------------------')
teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='colors_preference')


# Sampling phase
sampled_goals = teacher.sample_goals(size=2)
print(sampled_goals)
instructions = teacher.tell(sampled_goals, verbose=True)
print(instructions)

print('Pedagogical teacher tell policy ---------------------')
teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='pedagogical')


# Sampling phase
print(teacher.tell_policy)
for g in teacher.goal_space_str:
	print(teacher.tell([g]))
print(all_instructions_per_goal)


# PLOTS ##########################

from plots import *
visualize_all_instructions(all_instructions_per_goal)

teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='naive')
visualize_tell_policy(all_instructions_per_goal, teacher.tell_policy, teacher.agent_mode)

teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='colors_preference')
visualize_tell_policy(all_instructions_per_goal, teacher.tell_policy, teacher.agent_mode)

teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='shapes_preference')
visualize_tell_policy(all_instructions_per_goal, teacher.tell_policy, teacher.agent_mode)

teacher = TeacherTellPolicy(all_instructions_per_goal, agent_mode='pedagogical')
visualize_tell_policy(all_instructions_per_goal, teacher.tell_policy, teacher.agent_mode)


'''
#import pdb;pdb.set_trace()