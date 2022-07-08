import os
import math
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),"Process")) 
from search_run import SRModelEvaluation


class CodeEvaluation(object):
	'''
	Function: Evaluate the performance of the target model_code
	Input: config (model evaluation training and dataset details)
		   logging_path (logging path)
		   model_code (dict)
	Output: ndcg (the validation performance score during the training)
	'''
	def __init__(self, config, logging_path):
		super(CodeEvaluation, self).__init__()
		self.config = config
		self.logging_path = logging_path
		self.evaluater = SRModelEvaluation(self.config, self.logging_path)
		return

	def main(self, model_code):
		if isinstance(model_code[list(model_code.keys())[0]], list):
			for key in model_code.keys():
				model_code[key] = model_code[key][1]
		try:
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
				"kg_epochs": 5,
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

			ndcg = self.evaluater.main(real_model_config, real_model_code)
		except:
			ndcg = 0
		return ndcg
