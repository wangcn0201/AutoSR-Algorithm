import os
import sys
import logging
import shutil
import glob
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
from space.space_graph.BuildReverseGraph import *
from space.space_graph.CodeOnGraph import *
from space.space_graph.ModuleRelation import *
from space.space_graph.RandomCode import *
from autosr.EpsilonProcessor import *
from autosr.GraphModel_GCN import *
from autosr.ReplayMemory import *


class AutoSR_GCN(object):
	def __init__(self, config_path, data_name, gpu_device, evaluation_function):
		super(AutoSR_GCN, self).__init__()
		with open(config_path,"r") as f:
			config = json.load(f)
		config['code_dataset'] = data_name
		config['gpu_device'] = gpu_device

		self.environment_setting(config)

		self.dense_adj, self.edge_index, self.edge_index_dict, \
		self.edge_edgelist_dict, self.key_type_value_index, self.node_index_to_type_value, \
		self.key_name_list = BuildReverseGraph(Parameters, Options, Relations)
		self.edge_index = self.edge_index.cuda()
		
		self.graphcode = CodeOnGraph(self.edge_index_dict, \
			self.key_type_value_index)

		node_num = self.dense_adj.shape[0]
		self.model = GraphModel_GCN(node_num, Parameters, Options, Relations).cuda()
		self.memory = ReplayMemory()
		self.epsilon = EpsilonProcessor(config['autosr_epsilon_initial'], \
			config['autosr_epsilon_minimum'], config['autosr_epsilon_decay_rate'])
		
		self.evaluation = evaluation_function(config, self.logging_path)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["autosr_learning_rate"])
		self.rounds = config['autosr_rounds']
		self.batch_size = config['autosr_batch_size']
		self.update_frequency = config['autosr_update_frequency']
		return

	def create_exp_dir(self, path, scripts_to_save=None):
		if not os.path.exists(path):
			os.mkdir(path)

		if scripts_to_save is not None:
			os.mkdir(os.path.join(path, 'scripts'))
			for script in scripts_to_save:
				dst_file = os.path.join(path, 'scripts', os.path.basename(script))
				shutil.copyfile(script, dst_file)
		return
	def environment_setting(self, config):
		logging_dir = config['logging_dir']
		if not os.path.exists(logging_dir):
			os.mkdir(logging_dir)
		logging_path = logging_dir + '/NAS_AutoSR_GCN-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
		self.create_exp_dir(logging_path, scripts_to_save=glob.glob('*.py'))
		self.logging_path = logging_path

		log_format = '%(asctime)s %(message)s'
		logging.basicConfig(stream=sys.stdout, level=logging.INFO,
							format=log_format, datefmt='%m/%d %I:%M:%S %p')
		fh = logging.FileHandler(os.path.join(logging_path, 'log.txt'))
		fh.setFormatter(logging.Formatter(log_format))
		logging.getLogger().addHandler(fh)

		if not torch.cuda.is_available():
			logging.info('no gpu device available')
			sys.exit(1)

		np.random.seed(config['seed'])
		torch.cuda.set_device(config['gpu_device'])
		cudnn.benchmark = True
		torch.manual_seed(config['seed'])
		cudnn.enabled = True
		torch.cuda.manual_seed(config['seed'])
		logging.info('gpu device = %d' % config['gpu_device'])
		logging.info('config = %s', str(config))
		return

	def minibatch_learning(self, optimal_result):
		# train policy model using minibatch_learning method & update epsilon
		optimal_sample = [optimal_result[1], optimal_result[4]]
		self.model.train()
		batch_data = self.memory.sample_batch(self.batch_size, optimal_sample)
		#batch_data_code = torch.LongTensor(batch_data[0]).cuda()
		batch_data_code = [torch.LongTensor(batch_data[0][i]).cuda() for i in range(len(batch_data[0]))]
		batch_data_score = torch.LongTensor(batch_data[1]).cuda()

		self.optimizer.zero_grad()
		policy_value = self.model(self.edge_index, edge_edgelist_dict=self.edge_edgelist_dict, \
				batch_data=[batch_data_code, batch_data_score], function="Optimization")
		logging.info('policy_value: %s', str(policy_value))
		policy_value.backward()
		self.optimizer.step()
		return

	def main(self):
		start_time = time.time()
		optimal_result = [None, None, None, None, None] # epoch, performance_score, optimal_code, optimal_path, code_edge_index
		# initialize memory
		logging.info('\ninitialize memory......')
		random_code_list = RandomCode(self.batch_size, Parameters, Options, Relations)
		_, code_edge_index_list = self.graphcode.main(random_code_list)
		for i in range(len(random_code_list)):
			model_code, model_path = random_code_list[i]
			code_edge_index = code_edge_index_list[i]
			performance_score = self.evaluation.main(model_code)
			self.memory.push_data(model_code, model_path, code_edge_index, performance_score)
			logging.info('initialize %d: score -> %f, path -> %s, code -> %s, edge_index -> %s', i+1, performance_score, str(model_path), str(model_code), str(code_edge_index))
			logging.info('performance_score: %f, model_code: %s', performance_score, str(model_code))
			logging.info('update optimal result (initialization)......')
			if optimal_result[1] == None or optimal_result[1] < performance_score:
				optimal_result = ["initialize-"+str(i+1), performance_score, model_code, model_path, code_edge_index]
			logging.info('optimal_result: %s', str(optimal_result))
		logging.info('......initialize memory end\n')

		# Execute Graph-based RL Search Strategy
		for epoch in range(self.rounds):
			logging.info('\nround %d', epoch)

			# train policy model using minibatch_learning method & update epsilon
			if epoch % self.update_frequency == 0:
				logging.info('minibatch learning......')
				self.minibatch_learning(optimal_result)
			else:
				logging.info('no minibatch_learning')

			# recommend model_code
			logging.info('recommend model_code......')
			self.model.eval()
			epsilon_value = self.epsilon.get_epsilon()
			with torch.no_grad():
				optimal_code, optimal_path, optimal_code_node_index, \
				source_label = self.model(self.edge_index, \
					self.node_index_to_type_value, epsilon_value, \
					function="Recommendation")
			logging.info('epsilon: %f', epsilon_value)

			# duplicate removal & performance evaluation & memory update
			try_num = 0
			exist, info = self.memory.code_existance(optimal_code)
			while exist and try_num < 5:
				self.model.eval()
				with torch.no_grad():
					optimal_code, optimal_path, optimal_code_node_index, \
					source_label = self.model(self.edge_index, \
						self.node_index_to_type_value, epsilon_value, \
						function='Recommendation')
				exist, info = self.memory.code_existance(optimal_code)
				try_num += 1
			logging.info('try_num: %d, source_label: %s, optimal_code: %s', try_num, str(source_label), str(optimal_code))

			if exist:
				code_edge_index, performance_score = info
			else:
				_, code_edge_index_list = self.graphcode.main([[optimal_code, optimal_path]])
				#print("@@@")
				#print(code_edge_index_list)
				code_edge_index = code_edge_index_list[0]
				performance_score = self.evaluation.main(optimal_code)
			#logging.info('performance_score: %f', performance_score)
			logging.info('performance_score: %f, model_code: %s', performance_score, str(optimal_code))

			self.memory.push_data(optimal_code, optimal_path, code_edge_index, performance_score)
			
			logging.info('update optimal result and epsilon......')
			if optimal_result[1] == None or optimal_result[1] < performance_score:
				optimal_result = [epoch, performance_score, optimal_code, optimal_path, code_edge_index]
			logging.info('optimal_result: %s', str(optimal_result))
			running_time = time.time() - start_time
			logging.info('running_time: %f s', running_time)

			self.epsilon.epsilon_decay()
		return 
