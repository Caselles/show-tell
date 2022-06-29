import numpy as np 


shapes = ['cube', 'sphere']
colors = ['blue', 'red']

items_index_attributes = {0: ['sphere', 'blue'], 1: ['cube', 'red'], 2: ['cube', 'blue']}

goal_space = [[1,0,0], [0,1,0], [0,0,1]]

predicates_cubes_involved = [[0,1], [1,2], [0,2]] # attention à l'ordre!!!

def get_instruction_from_attributes(attribute1, attribute2):

	instruction = 'Put ' + attribute1 + ' next to ' + attribute2 + '.'

	return instruction

def get_attributes_from_instruction(instruction):

	attributes = instruction.split('Put ')[1].split('.')[0]

	attribute1 = attributes.split(' next to ')[0]
	attribute2 = attributes.split(' next to ')[1]

	return [attribute1, attribute2]

def convert_goallist_to_goalstr(goal):

	goalstr = ''
	for predicate in goal:
		goalstr+= str(predicate)

	return goalstr

all_instructions_per_goal = {}

for g in goal_space:

	instructions = []
	
	for idx, predicate in enumerate(g):
		if predicate:

			attributes_involved = [items_index_attributes[x] for x in predicates_cubes_involved[idx]]
			for att1 in attributes_involved[0]:
				for att2 in attributes_involved[1]:
					instructions.append(get_instruction_from_attributes(att1, att2))

					if att1 != att2:
						instructions.append(get_instruction_from_attributes(att2, att1))

	all_instructions_per_goal[convert_goallist_to_goalstr(g)] = instructions


class TeacherSpeechPolicy():

	def __init__(self, all_instructions_per_goal, teacher_mode):
	
		self.all_instructions_per_goal = all_instructions_per_goal
		self.goal_space_str = [g for g in all_instructions_per_goal.keys()]
		self.teacher_mode = teacher_mode

		self.initialize_policy()

		if self.teacher_mode == 'pedagogical':
			self.nb_iter_pedagogical_teacher = 1000
			self.proba_boost = 0.1
			self.learn_pedagogical_policy()

	def initialize_policy(self):

		self.speech_policy = {}

		for g in self.goal_space_str:
			instruction_probas = np.array([1]*len(self.all_instructions_per_goal[g])) / len(self.all_instructions_per_goal[g])
			self.speech_policy[g] = instruction_probas

		if self.teacher_mode == 'naive' or self.teacher_mode == 'pedagogical':

			return 

		elif self.teacher_mode == 'shapes_preference' or self.teacher_mode == 'colors_preference':

			# increase sampling probability for instructions referring to shapes
			for g in self.goal_space_str:
				for instruction_idx, instruction in enumerate(self.all_instructions_per_goal[g]):
					attributes_instruction = get_attributes_from_instruction(instruction)

					proba_boost = 0
					if self.teacher_mode == 'shapes_preference':
						preferred_attributes = shapes
					elif self.teacher_mode == 'colors_preference':
						preferred_attributes = colors

					for att in attributes_instruction:
						if att in preferred_attributes:
							proba_boost += 10

					self.speech_policy[g][instruction_idx] += proba_boost

			# normalize speech policy

			self.normalize_speech_policy()

			return 

		return False

	def normalize_speech_policy(self):

		for g in self.goal_space_str:

			sum_probas = np.sum(self.speech_policy[g])
		
			self.speech_policy[g] = self.speech_policy[g] / sum_probas

		return
		
	def learn_pedagogical_policy(self, learning=True):

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

						#proba_instruction = self.speech_policy[g][self.all_instructions_per_goal[g].index(sampled_instruction)]
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

					for idx, proba in enumerate(self.speech_policy[sampled_goal]):
						if idx == instruction_proba_index:
							self.speech_policy[sampled_goal][idx] += self.proba_boost
						'''else:
							if self.speech_policy[sampled_goal][idx] > self.proba_boost:
								self.speech_policy[sampled_goal][idx] -= self.proba_boost
							else:
								self.speech_policy[sampled_goal][idx] = 0'''

				else:

					for idx, proba in enumerate(self.speech_policy[sampled_goal]):
						if idx == instruction_proba_index:

							if self.speech_policy[sampled_goal][idx] > self.proba_boost:
								self.speech_policy[sampled_goal][idx] -= self.proba_boost
							else:
								self.speech_policy[sampled_goal][idx] = 0

						'''else:
							self.speech_policy[sampled_goal][idx] += self.proba_boost'''

				# normalize

				self.normalize_speech_policy()

		else:

			for g in self.goal_space_str:

				for instruction in all_instructions_per_goal[g]:

					instruction_proba_index = self.all_instructions_per_goal[g].index(instruction)

					for g2 in self.goal_space_str:

						if g != g2:

							if instruction in self.all_instructions_per_goal[g2]:

								self.speech_policy[g][instruction_proba_index] = 0

			self.normalize_speech_policy()

		return

	def sample_goals(self, size):

		sampled_goal = np.random.choice(self.goal_space_str, size=size)

		return sampled_goal

	def tell(self, goals, verbose=False):

		instructions = []

		for goal in goals:
			instruction_idx = np.random.choice(range(len(self.speech_policy[goal])), p=self.speech_policy[goal])
			instruction = self.all_instructions_per_goal[goal][instruction_idx]

			if verbose:
				print('instructions:', self.all_instructions_per_goal[goal])
				print('instruction probas:', self.speech_policy[goal])

			instructions.append(instruction)
			

		return instructions

	#def evaluate_policy_inference(self, )



print('Naive teacher speech policy ---------------------')
teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='naive')


# Sampling phase
sampled_goals = teacher.sample_goals(size=2)
print(sampled_goals)
instructions = teacher.tell(sampled_goals, verbose=True)
print(instructions)


print('Shape preference teacher speech policy ---------------------')
teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='shapes_preference')


# Sampling phase
sampled_goals = teacher.sample_goals(size=2)
print(sampled_goals)
instructions = teacher.tell(sampled_goals, verbose=True)
print(instructions)

print('Color preference teacher speech policy ---------------------')
teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='colors_preference')


# Sampling phase
sampled_goals = teacher.sample_goals(size=2)
print(sampled_goals)
instructions = teacher.tell(sampled_goals, verbose=True)
print(instructions)

print('Pedagogical teacher speech policy ---------------------')
teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='pedagogical')


# Sampling phase
print(teacher.speech_policy)
for g in teacher.goal_space_str:
	print(teacher.tell([g]))
print(all_instructions_per_goal)


# PLOTS ##########################

from plots import *
visualize_all_instructions(all_instructions_per_goal)

teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='naive')
visualize_speech_policy(all_instructions_per_goal, teacher.speech_policy, teacher.teacher_mode)

teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='colors_preference')
visualize_speech_policy(all_instructions_per_goal, teacher.speech_policy, teacher.teacher_mode)

teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='shapes_preference')
visualize_speech_policy(all_instructions_per_goal, teacher.speech_policy, teacher.teacher_mode)

teacher = TeacherSpeechPolicy(all_instructions_per_goal, teacher_mode='pedagogical')
visualize_speech_policy(all_instructions_per_goal, teacher.speech_policy, teacher.teacher_mode)



#import pdb;pdb.set_trace()