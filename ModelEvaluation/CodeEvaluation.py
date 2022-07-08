import os
import logging
import shutil
import glob
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
import math
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),"Process")) 
from evaluate_run import SRModelEvaluation


class CodeEvaluation(object):
	'''
	Function: Evaluate the performance of the target model_code
	Input: config (model evaluation training and dataset details)
		   model_code (dict)
	Output: ndcg (the validation performance score during the training)
	'''
	def __init__(self, config_path, data_name, model_name, gpu_device, model_code, mode, model_load_dir):
		super(CodeEvaluation, self).__init__()
		with open(config_path,"r") as f:
			config = json.load(f)
		config['model_name'] = model_name
		config['gpu_device'] = gpu_device
		config['model_code'] = model_code
		config['code_dataset'] = data_name
		config['code_mode'] = mode
		config['code_model_load_dir'] = model_load_dir

		self.environment_setting(config)

		self.config = config
		self.evaluater = SRModelEvaluation(self.config, self.logging_path)
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
		logging_path = logging_dir + '/Model-Evaluation-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
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

	def main(self, model_code):
		if isinstance(model_code[list(model_code.keys())[0]], list):
			for key in model_code.keys():
				model_code[key] = model_code[key][1]
		if 1==1:
			real_model_config = {
				"dataset": self.config['code_dataset'],
				"input_length": self.config['code_input_length'],
				"targetlength": self.config['code_target_length'],
				"neglength": self.config['code_neg_length'],
				"maxEpochnum": self.config['code_epoch_num'],
				"batchSize": self.config['code_batch_size'],
				"dropout": self.config['code_dropout'],
				"topk": self.config['code_topk'],
				"eval_item_num": self.config['code_eval_item_num'],

				"learning_rate": model_code['LR'],
				"optimizer": model_code['OF'],
				"numFactor": model_code['HIPM-EmbSize'],
				#HIPM_2
				"familiar_user_num": model_code['HIPM-WinSize'],
				#FExS_1
				"1st_attention_layers": model_code['1stFExS-k'],
				"2nd_attention_layers": model_code['2ndFExS-k'],
				#FExS_2
				"1st_hidden_layer": model_code['1stFExS-lhid'],
				"2nd_hidden_layer": model_code['2ndFExS-lhid'],
				"GRU_layers": 2,
				#FExS_4
				"1st_Pooling_method": model_code['1stFExS-Agg'], #Avg, Max
				"2nd_Pooling_method": model_code['2ndFExS-Agg'], #Avg, Max
				#FenS_1
				"1st_linear_layers": model_code['1stFEnS-L'],
				"2nd_linear_layers": model_code['2ndFEnS-L'],
				#LF
				"reg":0.0005,
				"kg_epochs": 50,
				#PS_2
				"numK": model_code['PS-K']
			}

			real_model_code = {
				"HIPM": model_code['HIPM'],
				"UVPM": model_code['UVPM'],
				"IRM": model_code['IRM'],
				"PS": model_code['PS'],
				"1stFExS": model_code['1stFExS'],
				"1stFEnS": model_code['1stFEnS'],
				"2ndFExS": model_code['2ndFExS'],
				"2ndFEnS": model_code['2ndFEnS'],
				"1stLF": model_code['1stLF'],
				"2ndLF": model_code['2ndLF']
			}

			self.evaluater.main(real_model_config, real_model_code)
		#except:
		#	print("Model Training Failed")
		return 
