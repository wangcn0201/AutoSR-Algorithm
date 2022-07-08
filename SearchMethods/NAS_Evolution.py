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
from space.space_cartesian.SpaceDict import *
import geatpy as ea


class Evolution(object):
	def __init__(self, config_path, data_name, gpu_device, evaluation_function):
		super(Evolution, self).__init__()
		with open(config_path,"r") as f:
			config = json.load(f)
		config['code_dataset'] = data_name
		config['gpu_device'] = gpu_device

		self.environment_setting(config)
		
		self.evaluation = evaluation_function(config, self.logging_path)
		self.rounds = config['evolution_rounds']
		self.population_num = config['evolution_population_num']
		self.HyperParameters = HyperParameters
		self.pars = list(self.HyperParameters.keys())
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
		logging_path = logging_dir + '/NAS_Evolution-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
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

	# 辅助函数：参数范围获取
	def ParamRange(self):
		param_range = {}
		continuous_list, integer_list = [], []
		for i in range(0,len(self.pars)):
			par = self.pars[i]
			content = self.HyperParameters[par]
			param_range[par] = (0,len(content)-1)
			integer_list.append(i)
		return param_range, continuous_list, integer_list

	class MyEvoProblem(ea.Problem): 
		def __init__(self, param_range, pars, continuous_list, integer_list, HyperParameters, evaluation_function):
			name = 'None' 
			M = 1
			maxormins = [1] * M # All objects are need to be minimized
			Dim = len(pars)
			varTypes = [] # Set the types of decision variables. 0 means continuous while 1 means discrete.
			lb, ub = [], []
			for i in range(Dim):
				par = pars[i]
				lb.append(param_range[par][0])
				ub.append(param_range[par][1])
				if i in continuous_list:
					varTypes.append(0)
				else:
					varTypes.append(1)
			lbin = [1] * Dim # Whether the lower boundary is included.
			ubin = [1] * Dim 
			ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
			self.pars = pars
			self.HyperParameters = HyperParameters
			self.evaluation = evaluation_function
			self.model_index = 0
			self.optimal_result = [None, None, None] # model_index, performance_score, optimal_code
			return

		def aimFunc(self, pop): 
			logging.info('\nevaluate model population......')
			Vars = pop.Phen 
			ObjV = []
			for i in range(len(Vars)):
				Var = Vars[i] 

				model_code = {}
				for j in range(0,len(self.pars)):
					par = self.pars[j]
					content = self.HyperParameters[par]
					model_code[par] = content[int(Var[j])]
					
				performance_score = self.evaluation.main(model_code)
				ObjV.append(performance_score*(-1))
				logging.info('model_index %d', self.model_index)
				logging.info('performance_score: %f, model_code: %s', performance_score, str(model_code))
				if self.optimal_result[1] == None or self.optimal_result[1] < performance_score:
					self.optimal_result = [self.model_index, performance_score, model_code]
				self.model_index += 1
				logging.info('update optimal result......')
				logging.info('optimal_result: %s', str(self.optimal_result))
			pop.ObjV = np.array([ObjV]).T
			logging.info('......evaluate model population end\n')
			return 

	def main(self):
		start_time = time.time()
		param_range, continuous_list, integer_list = self.ParamRange()
		problem = Evolution.MyEvoProblem(param_range, self.pars, continuous_list, integer_list, self.HyperParameters, self.evaluation)	 
		Encoding = 'RI'		  
		NIND = self.population_num		
		Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) 
		population = ea.Population(Encoding, Field, NIND) 

		myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population) 
		myAlgorithm.MAXGEN = self.rounds
		myAlgorithm.drawing = 1   
		myAlgorithm.mutOper.F = 0.5	# Set the F of DE
		myAlgorithm.recOper.XOVR = 0.2 # Set the Cr of DE (Here it is marked as XOVR)

		[population, obj_trace, var_trace] = myAlgorithm.run() # Run the algorithm templet
		best_gen = np.argmin(obj_trace[:, 1]) # Get the best generation
		best_individual = var_trace[best_gen]
		best_ndcg = np.min(obj_trace[:, 1])*(-1) # The objective value of the best solution

		running_time = time.time() - start_time
		logging.info('best_ndcg: %f, best_individual: %s', best_ndcg, best_individual)
		logging.info('running_time %f s', running_time)
		return 
