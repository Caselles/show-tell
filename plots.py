import numpy as np
import matplotlib.pyplot as plt 

from matplotlib import cm

def visualize_speech_policy(all_instructions_per_goal, speech_policy, teacher_mode):

	greens = cm.get_cmap('Greens', 128)

	val1 = list(all_instructions_per_goal.keys())

	aux = [list(x) for x in all_instructions_per_goal.values()]
	auxx = [x for xs in aux for x in xs]
	unique_instructions = list(set(auxx))

	val2 = unique_instructions


	val3 = []
	cell_col = []
	for instruction in val2:

		aux = []
		aux_col = []
		for g in val1:
			if instruction in all_instructions_per_goal[g]:
				index_instruction = all_instructions_per_goal[g].index(instruction)
				aux.append(round(speech_policy[g][index_instruction], 3))
				aux_col.append(greens(1.5*round(speech_policy[g][index_instruction], 3)))
			else:
				aux.append('No')
				aux_col.append('lightcoral')
		val3.append(aux)
		cell_col.append(aux_col)


	   
	fig, ax = plt.subplots(figsize=(20,20)) 
	ax.set_axis_off() 
	table = ax.table( 
	    cellText = val3,  
	    rowLabels = val2,  
	    colLabels = val1, 
	    rowColours =["white"] * len(val2),  
	    colColours =["white"] * len(val1), 
	    cellLoc ='center', 
	    cellColours = cell_col,
	    loc ='upper left',
	    colWidths=[0.1 for x in range(len(val1))])         
	   
	#ax.set_title('matplotlib.axes.Axes.table() function Example', 
	#             fontweight ="bold")

	#table.set_fontsize(40)
	#table.scale(1.5, 1.5) 
	   
	plt.savefig(teacher_mode+'_speech_policy.pdf') 




	return

def visualize_all_instructions(all_instructions_per_goal):



	val1 = list(all_instructions_per_goal.keys())

	aux = [list(x) for x in all_instructions_per_goal.values()]
	auxx = [x for xs in aux for x in xs]
	unique_instructions = list(set(auxx))

	val2 = unique_instructions


	val3 = []
	cell_col = []
	for instruction in val2:

		aux = []
		aux_col = []
		for g in val1:
			if instruction in all_instructions_per_goal[g]:
				aux.append('Yes')
				aux_col.append('palegreen')
			else:
				aux.append('No')
				aux_col.append('lightcoral')
		val3.append(aux)
		cell_col.append(aux_col)


	   
	fig, ax = plt.subplots(figsize=(20,20)) 
	ax.set_axis_off() 
	table = ax.table( 
	    cellText = val3,  
	    rowLabels = val2,  
	    colLabels = val1, 
	    rowColours =["white"] * len(val2),  
	    colColours =["white"] * len(val1), 
	    cellLoc ='center', 
	    cellColours = cell_col,
	    loc ='upper left',
	    colWidths=[0.1 for x in range(len(val1))])         
	   
	ax.set_title('matplotlib.axes.Axes.table() function Example', 
	             fontweight ="bold")

	#table.set_fontsize(40)
	#table.scale(1.5, 1.5) 
	   
	plt.savefig('compatibility_goals_instructions.pdf') 



	return

