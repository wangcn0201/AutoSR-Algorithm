import random
#from ModuleRelation import *
import copy


def CodeInitialization(Parameters, Options, Relations):
	initial_code = {}
	SubMapping = {}
	for key_name in Parameters.keys():
		choice_types = Parameters[key_name]
		for choice_type in choice_types:
			SubMapping[choice_type] = key_name
		choice_type = choice_types[0]
		choice_value = Options[choice_type][0]
		initial_code[key_name] = [None, None]
	initial_code['Start'] = ['Start', None]
	initial_code['HIPM-WinSize'] = ['WinSize', 5]
	return initial_code, SubMapping
#initial_code, SubMapping = CodeInitialization()
#print("CodeInitialization:\n\tinitial_code: %s\n\n\tSubMapping: %s\n"%(initial_code, SubMapping))


def RandomCode(Number, Parameters, Options, Relations):
	'''
	Function: Randomly generate Number model_code
	Input: Number (the number of model_code generated)
	Output: random_code_list (a list with Number model_code)
		    model_code = {'Parameter': [SubParameter, SubParameter's value]}
		    model_path = {'Parameter': [SubParameter, SubParameter's value]}
	'''
	#global initial_code, SubMapping
	initial_code, SubMapping = CodeInitialization(Parameters, Options, Relations)
	random_code_list = []
	key_name_list = Parameters.keys()
	
	for i in range(Number):
		print(i)
		random_code = copy.deepcopy(initial_code)
		random_path = {'Start': ['Start', None]}
		pre_choice_type = 'Start'
		while True:
			choice_type = random.sample(Relations[pre_choice_type],1)[0]
			choice_value = random.sample(Options[choice_type],1)[0]
			key_name = SubMapping[choice_type]
			random_code[key_name] = [choice_type, choice_value]
			random_path[key_name] = [choice_type, choice_value]
			#print("\tkey_name: %s, choice_type: %s, choice_value: %s"%(key_name, choice_type, choice_value))
			pre_choice_type = choice_type
			if pre_choice_type == 'End':
				break
		if [random_code, random_path] not in random_code_list:
			random_code_list.append([random_code, random_path])
		print(len(random_code_list))
	return random_code_list
'''
print("\n\n\n")
random_code_list = RandomCode(2)
for i in range(len(random_code_list)):
	print(random_code_list[i][0])
	print()
	print(random_code_list[i][1])
	print("\n@@@\n")
print(random_code_list)
'''
