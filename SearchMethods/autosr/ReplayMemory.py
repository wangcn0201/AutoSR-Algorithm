import random


class ReplayMemory(object):
	def __init__(self, memory_size=10000):
		super(ReplayMemory, self).__init__()
		self.memory_size = memory_size
		self.memory = {
			'model_code':[],
			'model_path':[],
			'code_edge_index':[],
			'performance_score':[]
		}
		self.position = 0

	def length(self):
		return len(self.memory['model_code'])

	def code_existance(self, model_code):
		if model_code in self.memory['model_code']:
			index = self.memory['model_code'].index(model_code)
			return True, [self.memory['code_edge_index'][index], self.memory['performance_score'][index]]
		return False, None

	def push_data(self, model_code, model_path, code_edge_index, performance_score):
		if self.length() < self.memory_size:
			self.memory['model_code'].append(None)
			self.memory['model_path'].append(None)
			self.memory['code_edge_index'].append(None)
			self.memory['performance_score'].append(None)
		self.memory['model_code'][self.position] = model_code
		self.memory['model_path'][self.position] = model_path
		self.memory['code_edge_index'][self.position] = code_edge_index
		self.memory['performance_score'][self.position] = performance_score
		self.position = (self.position + 1) % self.memory_size
		return

	def sample_batch(self, batch_size, optimal_sample):
		index_list = [i for i in range(self.length())]
		if self.length() < batch_size:
			batch_data_index = index_list
		elif optimal_sample[0] == None:
			batch_data_index = random.sample(index_list, batch_size)
		else:
			batch_data_index = random.sample(index_list, batch_size-1)
		batch_data_code = [self.memory['code_edge_index'][batch_data_index[i]] for i in range(len(batch_data_index))]
		batch_data_score = [[self.memory['performance_score'][batch_data_index[i]]] for i in range(len(batch_data_index))]
		if optimal_sample[0] != None:
			batch_data_code.append(optimal_sample[1])
			batch_data_score.append([optimal_sample[0]])
		batch_data = [batch_data_code, batch_data_score]
		return batch_data
