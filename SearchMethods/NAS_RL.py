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
from space.space_cartesian.SpaceDict import *
from rl.RNNModel import *
import rl.utils as utils


class RL(object):
	def __init__(self, config_path, data_name, gpu_device, evaluation_function):
		super(RL, self).__init__()
		with open(config_path,"r") as f:
			config = json.load(f)
		config['code_dataset'] = data_name
		config['gpu_device'] = gpu_device

		self.environment_setting(config)

		action_list = list(HyperParameters.keys())
		self.model = RNNModel(HyperParameters, action_list).cuda()
		
		self.evaluation = evaluation_function(config, self.logging_path)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["rl_learning_rate"])
		self.rounds = config['rl_rounds']
		self.ema_baseline_decay = config['rl_ema_baseline_decay']
		self.cuda = True
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
		logging_path = logging_dir + '/NAS_RL-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
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

	def scale(self, value, history, last_k=10, scale_value=1):
		'''
		scale value into [-scale_value, scale_value], according last_k history
		'''
		max_reward = np.max(history[-last_k:])
		if max_reward == 0:
			return value
		return scale_value / max_reward * value

	def main(self):
		start_time = time.time()
		optimal_result = [None, None, None] # epoch, performance_score, optimal_code

		# Execute RNN-based RL Search Strategy
		history = []
		self.model.train()
		baseline = None
		total_loss = 0
		for epoch in range(self.rounds):
			logging.info('\nround %d', epoch)

			# sample graphnas
			logging.info('recommend model_code......')
			model_code_list, log_probs, entropies = self.model.sample(with_details=True)
			# calculate reward
			optimal_code = model_code_list[0]
			performance_score = self.evaluation.main(optimal_code)
			rewards = performance_score
			logging.info('performance_score: %f, model_code: %s', performance_score, str(optimal_code))
			#torch.cuda.empty_cache()

			logging.info('update optimal result......')
			if optimal_result[1] == None or optimal_result[1] < performance_score:
				optimal_result = [epoch, performance_score, optimal_code]
			logging.info('optimal_result: %s', str(optimal_result))
			running_time = time.time() - start_time
			logging.info('running_time: %f s', running_time)

			# moving average baseline
			if baseline is None:
				baseline = rewards
			else:
				decay = self.ema_baseline_decay
				baseline = decay * baseline + (1 - decay) * rewards
			adv = rewards - baseline
			history.append(adv)
			adv = self.scale(adv, history, scale_value=0.5)
			adv = torch.Tensor([adv])
			adv = utils.get_variable(adv, self.cuda, requires_grad=False)
			
			# policy loss
			logging.info('policy loss updating......')
			loss = -log_probs * adv
			loss = loss.sum()  # or loss.mean()
			logging.info('loss of the RNN controller: %f', loss)
			# update
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			total_loss += utils.to_item(loss.data)
			logging.info('total_loss of the RNN controller: %f', total_loss)
			sum_running_time = time.time() - start_time
			logging.info('sum_running_time: %f s', sum_running_time)
		return
