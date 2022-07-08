import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
from torch.autograd import Variable
import time
import pandas as pd

class IRM_1:
	def __init__(self, config, module, device):
		self.config = config
		self.module = module
		self.batchsize = config["batchSize"]
		self.targetlength = config["targetlength"]
		self.device = device

	def get_parameters(self):
		para = []
		return para

	def irm(self, feed_dict):
		"""
		:param batchId:
		"""

		target_item_batch = torch.tensor(feed_dict["target_item_batch"]).view(self.batchsize,
																			  self.targetlength).to(self.device)
		# negative_item_batch = torch.zeros((self.batchsize, self.targetlength),
		#								   dtype=torch.long).to(self.device)
		negative_item_batch = torch.tensor(feed_dict["neg_item_batch"]).view(self.batchsize, -1).to(self.device)
		item_batch = torch.cat([target_item_batch, negative_item_batch], 1)
		item_embedding = self.module["itemEmbedding"][item_batch]

		return item_batch, item_embedding

class IRM_2(nn.Module):
	def __init__(self, config, module, device):
		super(IRM_2, self).__init__()
		self.config = config
		self.module = module
		self.numFactor = config["numFactor"]
		self.numItem = module["numItem"]
		self.numCate = module["numCate"]
		self.lr = config['learning_rate']
		self.relation_num = 2

		self.r_embeddings = nn.Embedding(self.relation_num, self.numFactor).to(device)
		#print("IRM_2")
		#print(self.r_embeddings.weight[0][:10])
		self.kg_loss = nn.MarginRankingLoss(margin=1)
		self.trainBatchSize = config["batchSize"]
		self.batchsize = config["batchSize"]
		self.targetlength = config["targetlength"]
		self.kg_epochs = config["kg_epochs"]

		self.GetRelationINF()

		self.trainSize = len(self.head_ids)
		self.trainBatchNum = int(self.trainSize // self.trainBatchSize) + 1

		self.device = device

		# self.itemEmbedding = Variable(torch.randn((self.numItem, self.numFactor), dtype=torch.float32).to(device),
		#							   requires_grad=True)
		self.itemEmbedding = module["itemEmbedding"]

		self.category_num = module["numCate"]
		self.betas = nn.Embedding(self.category_num, self.relation_num).to(device)
		self.mus = nn.Embedding(self.category_num, self.relation_num).to(device)
		self.sigmas = nn.Embedding(self.category_num, self.relation_num).to(device)
		torch.nn.init.normal_(self.r_embeddings.weight, mean=0.0, std=0.01)
		torch.nn.init.normal_(self.betas.weight, mean=0.0, std=0.01)
		torch.nn.init.normal_(self.mus.weight, mean=0.0, std=0.01)
		torch.nn.init.normal_(self.sigmas.weight, mean=0.0, std=0.01)

		# self.betas = Variable(torch.randn((self.category_num, self.relation_num)).to(device),
		#					   requires_grad=True)
		# self.mus = Variable(torch.randn((self.category_num, self.relation_num)).to(device),
		#					   requires_grad=True)
		# self.sigmas = Variable(torch.randn((self.category_num, self.relation_num)).to(device),
		#					 requires_grad=True)

	def GetRelationINF(self):
		data_name = self.config["dataset"]
		BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
		self.relation_data = pd.read_csv(BASE_DIR + "/Data/Amazon/" + data_name + "/item_meta.csv", sep="\t")
		for col in self.relation_data.columns:
			if pd.api.types.is_string_dtype(self.relation_data[col]):
				self.relation_data[col] = self.relation_data[col].apply(lambda x: eval(str(x)))  # some list-value columns
	
		self.triplet_set = set()
		heads, relations, tails = [], [], []

		self.item_relations = [r for r in self.relation_data.columns if r.startswith('r_')]
		for idx in range(len(self.relation_data)):
			head_item = self.relation_data['item_id'].values[idx]
			for r_idx, r in enumerate(self.item_relations):
				for tail_item in self.relation_data[r].values[idx]:
					heads.append(head_item)
					tails.append(tail_item)
					relations.append(r_idx)  # idx 0 is reserved to be a virtual relation between items
					self.triplet_set.add((head_item, r_idx, tail_item))

		self.relation_df = pd.DataFrame()
		self.relation_df['head'] = heads
		self.relation_df['relation'] = relations
		self.relation_df['tail'] = tails
		self.relation_num = len(self.item_relations)
		self.numTriple = len(self.triplet_set)

		self.head_ids = []
		self.tail_ids = []
		self.relation_ids = []
		for i in range(self.numTriple):
			head, tail, relation = heads[i], tails[i], relations[i]
			neg_heads = np.random.randint(0, self.numItem)
			while (neg_heads, relation, tail) in self.triplet_set:
				neg_heads = np.random.randint(0, self.numItem)
			head_id = np.array([head, head, head, neg_heads])
			neg_tails = np.random.randint(0, self.numItem)
			while (head, relation, neg_tails) in self.triplet_set:
				neg_tails = np.random.randint(0, self.numItem)
			tail_id = np.array([tail, tail, neg_tails, tail])
			relation_id = np.array([relation] * 4)
			self.head_ids.append(head_id)
			self.tail_ids.append(tail_id)
			self.relation_ids.append(relation_id)

	def get_kg_parameters(self):
		para = list(self.parameters())
		para.append(self.itemEmbedding)
		return para

	def get_parameters(self):
		#para = list(self.parameters())
		#return para
		weight_p, kg_p, bias_p = [], [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			elif 'i_embeddings' in name or 'r_embeddings' in name:
				kg_p.append(p)
			else:
				weight_p.append(p)
		'''
		optimize_dict = [
			{'params': weight_p, 'lr': 1e-3},
			{'params': kg_p, 'lr': 0.1*self.lr},  # scale down the lr of pretrained embeddings
			{'params': bias_p, 'weight_decay': 0.0}
		]
		'''
		optimize_dict = [
			{'params': weight_p},
			{'params': kg_p},  # scale down the lr of pretrained embeddings
			{'params': bias_p}
		]
		return optimize_dict
											 
	def kg_forward(self, batchId):
		start_idx = batchId * self.trainBatchSize
		end_idx = start_idx + self.trainBatchSize

		if end_idx > self.trainSize:
			end_idx = self.trainSize
			start_idx = end_idx - self.trainBatchSize

		if end_idx == start_idx:
			start_idx = 0
			end_idx = start_idx + self.trainBatchSize

		head_ids = self.head_ids[start_idx:end_idx]
		tail_ids = self.tail_ids[start_idx:end_idx]
		relation_ids = self.relation_ids[start_idx:end_idx]

		relation_vectors = self.r_embeddings(torch.Tensor(relation_ids).long().cuda())
		head_vectors = self.itemEmbedding[head_ids]
		tail_vectors = self.itemEmbedding[tail_ids]

		prediction = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)
		return prediction

	def loss(self, batchId):
		predictions = self.kg_forward(batchId)
		batch_size = predictions.shape[0]
		pos_pred, neg_pred = predictions[:, :2].flatten(), predictions[:, 2:].flatten()
		target = torch.from_numpy(np.ones(batch_size * 2, dtype=np.float32)).to(self.device)
		loss = self.kg_loss(pos_pred, neg_pred, target)
		return loss

	def kg_train(self):
		epoch_num = self.kg_epochs
		optimizer = opt.Adam(self.get_kg_parameters(), lr=0.001, betas=(0.9, 0.999))
		print("Doing item embedding training.........")
		for epoch in range(epoch_num):
			totaloss = 0
			begin = time.time()
			loss_lst = []
			for batch_id in range(self.trainBatchNum):
				optimizer.zero_grad()
				loss = self.loss(batch_id)
				loss_lst.append(loss.detach().cpu().data.numpy())
				totaloss += loss
				loss.backward()
				optimizer.step()
			avg_loss = np.mean(loss_lst).item()
			end = time.time()
			print("epoch: %d, time: %lf, totaloss: %lf, avgloss: %lf" %(epoch, (end - begin), totaloss, avg_loss))
		# self.module.itemEmbedding = self.itemEmbedding
		print("Item embedding training finished")
		#self.itemEmbedding
		#self.r_embeddings
		kg_itemEmbedding = self.itemEmbedding.detach()
		kg_r_embeddings = self.r_embeddings.weight.data
		kg_betas = self.betas.weight.data
		kg_mus = self.mus.weight.data
		kg_sigmas = self.sigmas.weight.data
		#print("\n\n##@@@##")
		#print(type(kg_itemEmbedding))
		#print(kg_itemEmbedding.shape)
		#print(type(kg_r_embeddings))
		#print(kg_r_embeddings.shape)
		#print(self.betas.weight.data[:10])
		return kg_itemEmbedding, kg_r_embeddings, kg_betas, kg_mus, kg_sigmas

	def irm(self, feed_dict):
		# target_item_batch = torch.tensor(feed_dict["target_item_batch"]).view(self.batchsize,
		#																	   self.targetlength).to(self.device)
		# # negative_item_batch = torch.zeros((self.batchsize, self.targetlength),
		# #								   dtype=torch.long).to(self.device)
		# negative_item_batch = torch.tensor(feed_dict["neg_item_batch"]).view(self.batchsize, -1).to(self.device)
		# item_batch = torch.cat([target_item_batch, negative_item_batch], 1)
		# item_embedding = self.itemEmbedding[item_batch]
		#
		# return item_batch, item_embedding
		return self.irm_forward(feed_dict)

	def kernel_functions(self, r_interval, betas, sigmas, mus):
		"""
		Define kernel function for each relation (exponential distribution by default)
		:return [batch_size, -1, relation_num]
		"""
		decay_lst = list()
		for r_idx in range(self.relation_num):
			delta_t = r_interval[:, :, r_idx]
			beta, sigma, mu = betas[:, :, r_idx], sigmas[:, :, r_idx], mus[:, :, r_idx]
			if r_idx == 0:  # is_complement_of
				#print(torch.Tensor(beta.cpu()).isnan().tolist())
				#mask = (beta >= 0).float()
				norm_dist = torch.distributions.normal.Normal(0, beta)
				decay = norm_dist.log_prob(delta_t).exp()
			elif r_idx == 1:  # is_substitute_of
				#print(torch.Tensor(beta.cpu()).isnan().tolist())
				#mask = (beta >= 0).float()
				neg_norm_dist = torch.distributions.normal.Normal(0, beta)
				#mask = (sigma >= 0).float()
				norm_dist = torch.distributions.normal.Normal(mu, sigma)
				decay = -neg_norm_dist.log_prob(delta_t).exp() + norm_dist.log_prob(delta_t).exp()
			else:  # exponential by default
				exp_dist = torch.distributions.exponential.Exponential(beta)
				decay = exp_dist.log_prob(delta_t).exp()
			decay_lst.append(decay.clamp(-1, 1))
		return torch.stack(decay_lst, dim=2)

	def irm_forward(self, feed_dict):
		u_ids = feed_dict["user_batch"] #(batchsize)
		i_ids = feed_dict["input_seq_batch"] #(batchsize, input_length)
		c_ids = feed_dict["cate_seq_batch"]  #(batchsize, input_length)
		r_interval = feed_dict["r_interval_batch"] #(batchsize, input_length, 2)
		r_interval = torch.tensor(r_interval, dtype=torch.float32).to(self.device)

		target_item_batch = torch.tensor(feed_dict["target_item_batch"]).view(self.batchsize,
																			  self.targetlength).to(self.device)
		# negative_item_batch = torch.zeros((self.batchsize, self.targetlength),
		#								   dtype=torch.long).to(self.device)
		negative_item_batch = torch.tensor(feed_dict["neg_item_batch"]).view(self.batchsize, -1).to(self.device)
		item_batch = torch.cat([target_item_batch, negative_item_batch], 1)


		i_vectors = self.itemEmbedding[item_batch]  #(batchsize, 2, numFactor)

		c_ids = torch.LongTensor(c_ids).to(self.device)
		betas = (self.betas(c_ids) + 1).clamp(min=1e-10, max=10)
		sigmas = (self.sigmas(c_ids) + 1).clamp(min=1e-10, max=10)
		mus = self.mus(c_ids) + 1
		mask = (r_interval >= 0).float()
		temporal_decay = self.kernel_functions(r_interval * mask, betas, sigmas, mus)
		temporal_decay = temporal_decay * mask  # (batch_size, input_length, 2)

		self.relation_range = range(self.relation_num)
		r_vectors = self.r_embeddings(torch.Tensor(self.relation_range).long().cuda()) #(2, numFactor)
		ri_vectors = i_vectors[:, :, None, :] + r_vectors[None, None, :, :]  # (batch_size, input_length, 2, numFactor)
		chorus_vectors = i_vectors + (temporal_decay[:, :, :, None] * ri_vectors + 1e-8).sum(2)
		"(batchsize, input_length, numFactor)"
		return item_batch, chorus_vectors		

class IRM_3(nn.Module):
	def __init__(self, config, module, device):
		super(IRM_3, self).__init__()
		self.config = config
		self.module = module
		self.batchsize = config["batchSize"]
		self.targetlength = config["targetlength"]
		self.device = device
		self.numItem = module["numItem"]
		self.numFactor = config["numFactor"]
		self.W2 = nn.Embedding(self.numItem, self.numFactor, padding_idx=0).to(device)
		#self.W2 = Variable(torch.randn((self.numItem, self.numFactor), dtype=torch.float32).cuda(),
		#							  requires_grad=True)
		self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)

	def get_parameters(self):
		paras = list(self.parameters())
		optimize_dict = [{'params': paras}]
		return optimize_dict

	def irm(self, feed_dict):
		"""
		:param batchId:
		"""
		target_item_batch = torch.tensor(feed_dict["target_item_batch"]).view(self.batchsize,
																			  self.targetlength).to(self.device)
		# negative_item_batch = torch.zeros((self.batchsize, self.targetlength),
		#								   dtype=torch.long).to(self.device)
		negative_item_batch = torch.tensor(feed_dict["neg_item_batch"]).view(self.batchsize, -1).to(self.device)
		item_batch = torch.cat([target_item_batch, negative_item_batch], 1)
		item_embedding = self.W2(item_batch)
		return item_batch, item_embedding
