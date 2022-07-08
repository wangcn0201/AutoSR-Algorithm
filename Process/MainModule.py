import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from State1_DataProcessing.HIPM import *
from State1_DataProcessing.UVPM import *
from State3_Prediction.IRM import *
from State3_Prediction.PS import *
from State2_FetureExtraction.FExS import *
from State2_FetureExtraction.FEnS import *
from Training.LF import *

class BuildModule(nn.Module):
	def __init__(self, module_code, config, data_config, device):
		super(BuildModule, self).__init__()


		self.module_code = module_code
		self.numFactor = config["numFactor"]
		self.numItem = data_config["numItem"]
		self.numUser = data_config["numUser"]
		self.numCate = data_config["numCate"]
		self.numK = config["numK"]
		self.input_length = config["input_length"]
		self.lr = config["learning_rate"]
		if self.module_code["IRM"] == "IRM2":
			self.lr = 0.1 * self.lr
			print(self.module_code["IRM"])

		self.trainBatchSize = config["batchSize"]
		self.maxEpoch = config["maxEpochnum"]
		self.userEmbedding = Variable(torch.randn((self.numUser, self.numFactor), dtype=torch.float32).cuda(),
									  requires_grad=True)
		self.itemEmbedding = Variable(torch.randn((self.numItem, self.numFactor), dtype=torch.float32).cuda(),
									  requires_grad=True)
		# self.userEmbedding = nn.Embedding(self.numUser, self.numFactor)
		# self.itemEmbedding = nn.Embedding(self.numItem, self.numFactor)

		module_config = {
			"module_code":module_code,
			"numItem":self.numItem,
			"numUser":self.numUser,
			"numCate":self.numCate,
			"userEmbedding":self.userEmbedding,
			"itemEmbedding":self.itemEmbedding
		}

		self.HIPM = eval(self.module_code["HIPM"])(config, module_config, device)
		self.UVPM = eval(self.module_code["UVPM"])(config, module_config, device)
		self.IRM = eval(self.module_code["IRM"])(config, module_config, device)
		self.FExS1 = eval(self.module_code["1stFExS"])(config, module_config, device, "1st")
		self.FEnS1 = eval(self.module_code["1stFEnS"])(config, module_config, device, "1st")
		self.FExS2 = eval(self.module_code["2ndFExS"])(config, module_config, device, "2nd")
		self.FEnS2 = eval(self.module_code["2ndFEnS"])(config, module_config, device, "2nd")
		self.PS = eval(self.module_code["PS"])(config, module_config, device)
		self.LF1 = eval(self.module_code["1stLF"])(config, module_config, device)
		self.LF2 = eval(self.module_code["2ndLF"])(config, module_config, device)

	def get_parameters(self):
		HIPM_para = self.HIPM.get_parameters()
		UVPM_para = self.UVPM.get_parameters()
		IRM_para = self.IRM.get_parameters()
		FExS1_para = self.FExS1.get_parameters()
		FEnS1_para = self.FEnS1.get_parameters()
		FExS2_para = self.FExS2.get_parameters()
		FEnS2_para = self.FEnS2.get_parameters()
		PS_para = self.PS.get_parameters()
		LF1_para = self.LF1.get_parameters()
		LF2_para = self.LF2.get_parameters()
		para = HIPM_para + UVPM_para + IRM_para + FExS1_para + FEnS1_para + FExS2_para + FEnS2_para + PS_para + LF1_para + LF2_para
		para.append({'params': self.userEmbedding})
		#para.append({'params': self.itemEmbedding, 'lr': self.lr})
		para.append({'params': self.itemEmbedding})
		return para

	def forward(self, feed_dict, mode):
		#the new user--item matrix accrodding to the history information
		user_em = self.UVPM.uvpm(feed_dict)
		"(batchsize, numFactor)"

		# target_itm, item_em = self.IPM.ipm(feed_dict)
		# "(batchsize)  (batchsize, input_length, numFactor)"

		history_item_em = self.HIPM.hipm(feed_dict, user_em)
		"(batchsize, input_length, numFactor)"

		#the new user embedding with the extracted item information in
		extracted_feature1 = self.FExS1.fexs(feed_dict, history_item_em, user_em)
		"(batchsize, numFactor)"

		enhanced_feature1 = self.FEnS1.fens(extracted_feature1)
		"(batchsize, numFactor)"

		extracted_feature2 = self.FExS2.fexs(feed_dict, history_item_em, user_em)
		"(batchsize, numFactor)"

		enhanced_feature2 = self.FEnS2.fens(extracted_feature2)
		"(batchsize, numFactor)"

		extracted_feature = (extracted_feature1 + extracted_feature2)/2
		enhanced_feature = (enhanced_feature1 + enhanced_feature2)/2

		item_batch, item_em = self.IRM.irm(feed_dict)
		"(batchsize, 2)   (batchsize, 2, numFactor)"

		#predict with the original user embedding, the new user embedding and the whole item embedding
		self.prediction, extra_info = \
			self.PS.ps(feed_dict, user_em, history_item_em, item_batch, item_em, extracted_feature, enhanced_feature, mode)
		"(batchsize, target_length+neg_length)"

		# LOSS = Loss(self, item_batch, self.prediction)
		# loss = eval("LOSS." + self.module_code["LOSS"] + "()")
		# return loss
		if mode == "test" or mode == "val_test":
			loss = None
		else:
			loss1 = self.LF1.lf(user_em, item_batch, item_em, self.prediction, extra_info)
			loss2 = self.LF2.lf(user_em, item_batch, item_em, self.prediction, extra_info)
			loss = (loss1 + loss2)/2
		return loss




