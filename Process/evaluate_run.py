import os
import sys
import copy
import pickle
import logging
import time
import torch
import torch.nn as nn
import torch.optim as opt
import torch.backends.cudnn as cudnn
import numpy as np
from State1_DataProcessing.HIPM import *
from State1_DataProcessing.UVPM import *
from State3_Prediction.IRM import *
from State3_Prediction.PS import *
from State2_FetureExtraction.FExS import *
from Training.LF import *
from MainModule import *
from Evaluator import *
import Datamodule


class SRModelEvaluation(object):
	def __init__(self, config, logging_path):
		super(SRModelEvaluation, self).__init__()
		self.dataset = config['code_dataset']
		self.input_length = config['code_input_length']
		self.target_length = config['code_target_length']
		self.neg_length = config['code_neg_length']
		self.epoch_num = config['code_epoch_num']
		self.batch_size = config['code_batch_size']
		self.dropout = config['code_dropout']
		self.topk = config['code_topk']
		self.eval_item_num = config['code_eval_item_num']
		self.mode = config['code_mode']
		self.model_load_dir = config['code_model_load_dir']
		self.config = config
		self.logging_path = logging_path

		self.environment_setting(config, logging_path)

		self.Data_model = Datamodule.Amazon(self.dataset)
		self.Data_model.generate_dataset()
		return

	def environment_setting(self, config, logging_path):
		if not os.path.exists(logging_path):
			os.mkdir(logging_path)

		log_format = '%(asctime)s %(message)s'
		logging.basicConfig(stream=sys.stdout, level=logging.INFO,
							format=log_format, datefmt='%m/%d %I:%M:%S %p')
		fh = logging.FileHandler(os.path.join(logging_path, 'Model-Evaluation-log.txt'))
		fh.setFormatter(logging.Formatter(log_format))
		logging.getLogger().addHandler(fh)

		if not torch.cuda.is_available():
			logging.info('no gpu device available')
			sys.exit(1)

		np.random.seed(config["seed"])
		torch.cuda.set_device(config["gpu_device"])
		self.device = torch.device("cuda:"+str(config["gpu_device"]) if torch.cuda.is_available() else "cpu")
		cudnn.benchmark = True
		torch.manual_seed(config["seed"])
		cudnn.enabled = True
		torch.cuda.manual_seed(config["seed"])
		logging.info('gpu device = %d' % config["gpu_device"])
		logging.info("config = %s", str(config))
		return

	def train(self, model_config, model_code):
		train_users = None
		train_sequences_input = None
		train_sequences_user_input = None
		train_sequences_target = None
		data_config = None
		neg_items = None
		familiar_num = model_config['familiar_user_num']

		logging.info('generating train (train+val) data......')
		begin_time = time.time()
		data_feature_name = str(familiar_num) + "+" \
			+ str(self.input_length) + "+" \
			+ str(self.target_length) + "+" \
			+ str(self.neg_length) + "_"
		BASE_DIR = os.path.abspath(os.path.dirname(__file__))
		data_for_train_path = BASE_DIR+"/Data/Amazon/" + self.dataset + "/" + data_feature_name + "train_data.pkl"

		if os.path.exists(data_for_train_path):
			logging.info('already been generated......')
			with open(data_for_train_path, "rb") as f:
				r_interval, cate_seq, neg_items, train_users, train_sequences_input, \
				train_sequences_user_input, \
				train_sequences_target, data_config = pickle.load(f)
		else:
			logging.info('generating......')
			r_interval, cate_seq, neg_items, train_users, \
			train_sequences_input, train_sequences_user_input, \
			train_sequences_target, data_config = Datamodule.get_data(self.Data_model, \
					familiar_num, "train", self.input_length, self.target_length, \
					self.neg_length)

			with open(data_for_train_path, "wb") as f:
				train_data = [r_interval, cate_seq, neg_items, train_users,
							 train_sequences_input, train_sequences_user_input,
							 train_sequences_target, data_config]
				pickle.dump(train_data, f)
		logging.info('time used: %f', time.time()-begin_time)
		logging.info('generate train (train+val) data finished......')

		trainsize = len(train_users)
		batchsize = self.batch_size
		num_batches = int(trainsize / batchsize) + 1

		logging.info('building model......')
		begin_time = time.time()
		model = BuildModule(model_code, model_config, data_config, self.device).to(self.device)
		logging.info('time used: %f', time.time()-begin_time)
		logging.info('build model finished......')

		if self.mode == 'train':
			if (model_code["IRM"] == "IRM_2"):
				kg_itemEmbedding, kg_r_embeddings, kg_betas, kg_mus, kg_sigmas = model.IRM.kg_train()
				#print("evaluate_run (trained kg)")
				#print(model.itemEmbedding[0][:10])
				model = BuildModule(model_code, model_config, data_config, self.device).to(self.device)
				#print("evaluate_run (before load)")
				#print(model.itemEmbedding[0][:10])
				model.itemEmbedding = Variable(kg_itemEmbedding.cuda(), requires_grad=True) 
				model.IRM.r_embeddings.weight.data.copy_(kg_r_embeddings)
				model.IRM.betas.weight.data.copy_(kg_betas)
				model.IRM.mus.weight.data.copy_(kg_mus)
				model.IRM.sigmas.weight.data.copy_(kg_sigmas)
				#print("evaluate_run (after load)")
				#print(model.itemEmbedding[0][:10])

			best_score = [0, None, None, None, None]
			optimizer = None
			if model_config['optimizer'] == "Adam":
				optimizer = opt.Adam(model.get_parameters(), lr=model_config['learning_rate'], betas=(0.9, 0.999))
			elif model_config['optimizer'] == "Adagrad":
				optimizer = opt.Adagrad(model.get_parameters(), lr=model_config['learning_rate'])

			logging.info('begin training......')
			no_change_num = 0
			for epoch in range(self.epoch_num):
				logging.info('epoch %d', epoch)
				begin_time = time.time()
				totalloss = 0
				model.train()
				for batchId in range(num_batches):
					start_idx = batchId * batchsize
					end_idx = start_idx + batchsize
					if end_idx > trainsize:
						end_idx = trainsize
						start_idx = end_idx - batchsize
					if end_idx == start_idx:
						start_idx = 0
						end_idx = start_idx + batchsize

					feed_dict = {
						"user_batch": train_users[start_idx:end_idx],  #(batchsize, numFactor)
						"input_seq_batch": train_sequences_input[start_idx:end_idx],  #(batchsize, input_length, numFactor)
						"input_user_seq_batch": train_sequences_user_input[start_idx:end_idx], #(batchsize, input_length, familiar, numFactor)
						"target_item_batch": train_sequences_target[start_idx:end_idx],  #(batchsize, targetlength)
						"neg_item_batch": neg_items[start_idx:end_idx],
						"r_interval_batch": r_interval[start_idx:end_idx],
						"cate_seq_batch": cate_seq[start_idx:end_idx]
					}
					optimizer.zero_grad()
					mode = "train"
					loss = model(feed_dict, mode)
					mean_loss = torch.mean(loss)
					#print("@@@")
					#print(mean_loss)
					#print(model.prediction)
					#print("batchId: %d, mean_loss: %f"%(batchId,mean_loss))
					totalloss += mean_loss
					mean_loss.backward()
					optimizer.step()

				end_time = time.time()
				logging.info('epoch %d: time = %lf, total_loss = %lf', epoch, end_time - begin_time, totalloss)

				precision, recall, ndcg, parameters = self.evaluate(model_config, model)
				if best_score[3] == None or best_score[3] < ndcg:
					best_score = [epoch, precision, recall, ndcg, parameters]
					model_save_dir = self.logging_path
					torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_best_weights.pt'))
					logging.info('better model found in epoch = %d, precision = %.5f: recall = %.5f, ndcg = %.5f, parameters = %.5f', best_score[0], best_score[1], best_score[2], best_score[3], best_score[4])
					no_change_num = 0
				else:
					no_change_num += 1
				logging.info('best_score: epoch = %d, precision = %.5f: recall = %.5f, ndcg = %.5f, parameters = %.5f', best_score[0], best_score[1], best_score[2], best_score[3], best_score[4])
				if no_change_num == 20:
					break
		return 

	def evaluate(self, model_config, model):
		test_users = None
		test_sequences_input = None
		test_sequences_user_input = None
		test_sequences_target = None
		data_config = None
		neg_items = None
		familiar_num = model_config["familiar_user_num"]

		logging.info('generating test (test) data......')
		data_feature_name = str(model_config['familiar_user_num']) + "+" \
			+ str(self.input_length) + "+" \
			+ str(self.target_length) + "+" \
			+ str(model_config['eval_item_num']) + "_"
		BASE_DIR = os.path.abspath(os.path.dirname(__file__))
		data_for_test_path = BASE_DIR+"/Data/Amazon/" + self.dataset + "/" + data_feature_name + "test_data.pkl"

		begin_time = time.time()
		if os.path.exists(data_for_test_path):
			logging.info('already been generated......')
			with open(data_for_test_path, "rb") as f:
				r_interval, cate_seq, neg_items, test_users, test_sequences_input, \
				test_sequences_user_input, \
				test_sequences_target, data_config = pickle.load(f)
		else:
			logging.info('generating......')
			r_interval, cate_seq, neg_items, test_users, \
			test_sequences_input, test_sequences_user_input, \
			test_sequences_target, data_config = Datamodule.get_data(self.Data_model, \
					model_config['familiar_user_num'], "test", self.input_length, self.target_length, \
					model_config['eval_item_num'])

			with open(data_for_test_path, "wb") as f:
				test_data = [r_interval, cate_seq, neg_items, test_users,
							 test_sequences_input, test_sequences_user_input,
							 test_sequences_target, data_config]
				pickle.dump(test_data, f)
		logging.info('time used: %f', time.time()-begin_time)
		logging.info('generate test (test) data finished......')

		testsize = len(test_users)
		batchsize = self.batch_size
		num_batches = int(testsize / batchsize) + 1
		topk = self.topk

		model.eval()
		with torch.no_grad():
			pred_list = None
			logging.info('begin evaluating......')
			for batchId in range(num_batches):
				start_idx = batchId * batchsize
				end_idx = start_idx + batchsize
				if end_idx > testsize:
					end_idx = testsize
					start_idx = end_idx - batchsize
				if end_idx == start_idx:
					start_idx = 0
					end_idx = start_idx + batchsize

				feed_dict = {
					"user_batch": test_users[start_idx:end_idx],
					"input_seq_batch": test_sequences_input[start_idx:end_idx],
					"input_user_seq_batch": test_sequences_user_input[start_idx:end_idx],
					"target_item_batch": test_sequences_target[start_idx:end_idx],
					"neg_item_batch": neg_items[start_idx:end_idx],
					"r_interval_batch": r_interval[start_idx:end_idx],
					"cate_seq_batch": cate_seq[start_idx:end_idx]
				}

				mode = "test"
				loss = model(feed_dict, mode)
				pred_score = model.prediction
				rating_pred = pred_score.data.cpu().numpy()

				eval_pred = np.array(rating_pred)
				ind = np.argpartition(eval_pred, -topk)  # ?????????topk?????????
				ind = ind[:, -topk:]
				arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
				arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
				batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
				"topK???index"

				if batchId == 0:
					pred_list = batch_pred_list
				elif end_idx == testsize:
					length = testsize % batchsize
					pred_list = np.append(pred_list, batch_pred_list[-length:], axis=0)
				else:
					pred_list = np.append(pred_list, batch_pred_list, axis=0)

		actual = [[0]] * testsize
		precision = precision_at_k(actual, pred_list, topk)
		recall = recall_at_k(actual, pred_list, topk)
		ndcg = ndcg_k(actual, pred_list, topk)
		parameters = get_nelement(model)
		"parameters"

		logging.info('precision = %.5f: recall = %.5f, ndcg = %.5f, parameters = %.5f', precision, recall, ndcg, parameters)
		return precision, recall, ndcg, parameters

	def main(self, model_config, module_code):
		self.train(model_config, module_code)
		return 
