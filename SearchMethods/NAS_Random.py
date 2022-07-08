import os
import sys
import logging
import shutil
import glob
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import json
from space.space_graph.RandomCode import *
from space.space_graph.ModuleRelation import *


class Random(object):
	def __init__(self, config_path, data_name, gpu_device, evaluation_function):
		super(Random, self).__init__()
		with open(config_path,"r") as f:
			config = json.load(f)
		config['code_dataset'] = data_name
		config['gpu_device'] = gpu_device

		self.environment_setting(config)
		
		self.evaluation = evaluation_function(config, self.logging_path)
		self.rounds = config['random_rounds']
		self.batch_size = config['autosr_batch_size']
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
		logging_dir = config["logging_dir"]
		if not os.path.exists(logging_dir):
			os.mkdir(logging_dir)
		logging_path = logging_dir + '/NAS_Random-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
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

		np.random.seed(config["seed"])
		torch.cuda.set_device(config["gpu_device"])
		cudnn.benchmark = True
		torch.manual_seed(config["seed"])
		cudnn.enabled = True
		torch.cuda.manual_seed(config["seed"])
		logging.info('gpu device = %d' % config["gpu_device"])
		logging.info("config = %s", str(config))
		return

	def main(self):
		start_time = time.time()
		optimal_result = [None, None, None, None] # epoch, performance_score, optimal_code, optimal_path
		logging.info('\ngenerate random codes......')
		print("logging info exist ... ...")
		random_code_list = RandomCode(self.rounds, Parameters, Options, Relations)
		logging.info('......RandomCode Finished, %d codes generated\n', len(random_code_list))
		initial_code_list = random_code_list[:min(self.batch_size,len(random_code_list))]
		logging.info('initial_code_list for AutoSR Algorithms: %s\n', len(initial_code_list))
		for epoch in range(len(random_code_list)):
			logging.info('round %d', epoch)
			logging.info('evaluate model code......')
			model_code, model_path = random_code_list[epoch]
			logging.info('model_code: %s', str(model_code))
			performance_score = self.evaluation.main(model_code)
			logging.info('performance_score: %f, model_code: %s', performance_score, str(model_code))
			logging.info('update optimal result......')
			if optimal_result[1] == None or optimal_result[1] < performance_score:
				optimal_result = [epoch, performance_score, model_code, model_path]
			logging.info('optimal_result: %s', str(optimal_result))
			running_time = time.time() - start_time
			logging.info('running_time %f s', running_time)
		return 
