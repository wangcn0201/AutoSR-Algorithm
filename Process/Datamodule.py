import os
import pandas as pd
import numpy as np
import torch
# from sklearn.model_selection import train_test_split
import pickle
import math
from DataPreparing.DataForming import Former
import random
import csv

class DataSet(object):
	def load_pickle(self, name):
		with open(name, 'rb') as f:
			return pickle.load(f, encoding='latin1')

	def save_pickle(self, obj, name, protocol=3):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, protocol=protocol)

	def data_index_shift(self, lists, increase_by=2):
		"""
		Increase the item index to contain the pad_index
		:param lists:
		:param increase_by:
		:return:
		"""
		for seq in lists:
			for i, item_id in enumerate(seq):
				seq[i] = item_id + increase_by

		return lists

	def split_data_sequentially(self, user_records, test_radio=0.2):
		train_set = []
		test_set = []

		for item_list in user_records:
			len_list = len(item_list)
			num_test_samples = int(math.ceil(len_list * test_radio))
			train_sample = []
			test_sample = []
			for i in range(len_list - num_test_samples, len_list):
				test_sample.append(item_list[i])

			for place in item_list:
				if place not in set(test_sample):
					train_sample.append(place)

			train_set.append(train_sample)
			test_set.append(test_sample)

		return train_set, test_set

class Amazon(DataSet):
	def __init__(self, data_name):
		super(Amazon, self).__init__()
		# data_name = "Videos"
		BASE_DIR = os.path.abspath(os.path.dirname(__file__))
		self.dir_path = BASE_DIR+"/Data/Amazon/" + data_name + "/"
		self.user_record_file = data_name + "_item_sequences.pkl"
		self.user_mapping_file = data_name + '_user_mapping.pkl'
		self.item_mapping_file = data_name + '_item_mapping.pkl'

		self.data_name = data_name

		self.num_users = None
		self.num_items = None
		self.vocab_size = 0

		self.user_records = None
		self.user_mapping = None
		self.item_mapping = None

	def generate_dataset(self, index_shift=0):
		user_records = self.load_pickle(self.dir_path + self.user_record_file)
		user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
		item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
		col_names = ["item_id", "category", "r_complement", "r_substitute"]
		# item_meta = pd.read_csv(self.dir_path + "item_meta.csv", sep=',', names=col_names, engine='python')


		self.num_users = len(user_mapping)
		self.num_items = len(item_mapping)

		self.user_records = self.data_index_shift(user_records, increase_by=index_shift)

		# split dataset
		self.train_val_set, self.test_set = self.split_data_sequentially(user_records, test_radio=0.2)
		self.train_set, self.val_set = self.split_data_sequentially(self.train_val_set, test_radio=0.125)
		self.num_items += index_shift

		# self.neg_samples = generate_neg_samples(self.user_records, self.num_items, eval_item_num)







# def generate_neg_samples(user_records, num_items, eval_item_num):
#	 neg_user_samples = []
#	 for uid, item_list in enumerate(user_records):
#		 neg_samples = set()
#		 while(len(neg_samples) < eval_item_num):
#			 iid = random.randint(0, num_items-1)
#			 if iid not in item_list:
#				 neg_samples.add(iid)
#		 neg_samples = list(neg_samples)
#		 neg_user_samples.append(neg_samples)
#	 return neg_user_samples


def get_data(Data_model, familiar_num=5, mode="train", input_length=5, target_length=1, eval_item_num=999):
	data = Data_model
	# neg_samples = data.neg_samples
	if mode == "train":
		f = Former(data.train_val_set, data.num_users, data.num_items)
	elif mode == "test":
		f = Former(data.test_set, data.num_users, data.num_items)
	elif mode == "val_train":
		f = Former(data.train_set, data.num_users, data.num_items)
	elif mode == "val_test":
		f = Former(data.val_set, data.num_users, data.num_items)
	sequences, test_sequences, target_sequences, train_users = f.to_sequence(input_length, target_length)
	BASE_DIR = os.path.abspath(os.path.dirname(__file__))
	f.get_history_item_users(BASE_DIR+"/Data/Amazon/" + data.data_name + "/train.csv")
	train_sequences_input = f.sequences  # 每个user访问的input_lenth个item
	train_sequences_user_input = []  # 每个item的前k个用户
	train_sequences_target = target_sequences  # 每个user的target_lenth个item

	for i in range(f.num_subsequences):
		uid = f.sequences_users[i]
		item_hisusers_list = []
		for iid in sequences[i]:
			his_seq = [0] * familiar_num

			if iid == 0:
				item_hisusers_list.append(his_seq)
				continue

			time_position = f.time_for_users_to_item[iid][uid]
			if time_position < familiar_num:
				# his_seq = [1] * familiar_num
				his_seq[:time_position] = f.item_to_pastusers[iid][:time_position]
			else:
				his_seq = f.item_to_pastusers[iid][(time_position - familiar_num):time_position]

			item_hisusers_list.append(his_seq)
		train_sequences_user_input.append(item_hisusers_list)

	data_config = {
		"numUser": data.num_users,
		"numItem": data.num_items,
		"trainsize": f.num_subsequences
	}

	neg_items = generate_neg_samples(train_sequences_input, eval_item_num, data_config)

	item_batch = list(np.concatenate((train_sequences_target, np.array(neg_items)), axis=1))
	BASE_DIR = os.path.abspath(os.path.dirname(__file__))
	seq_data_path = BASE_DIR+"/Data/Amazon/" + data.data_name + "/seq_data.pickle"
	with open(seq_data_path, "wb") as f:
		pickle.dump([train_users, item_batch, data_config], f)

	BASE_DIR = os.path.abspath(os.path.dirname(__file__))
	dir_path = BASE_DIR+"/Data/Amazon/" + data.data_name
	if data.data_name == "ML100K":
		lensamples = len(train_users)
		numCate = lensamples
		r_interval = list(np.zeros((lensamples)))
		cate_seq = list(np.zeros((lensamples)))
	else:
		Relation = Relation_Former(dir_path)
		numCate, r_interval, cate_seq = Relation.init_data()
	data_config["numCate"] = numCate
	return r_interval, cate_seq, neg_items, train_users, train_sequences_input, \
		   train_sequences_user_input, train_sequences_target, data_config


def generate_neg_samples(train_sequences_input, eval_item_num, data_config):
	neg_sequence = []
	item_num = data_config["numItem"]
	for user, item_list in enumerate(train_sequences_input):
		neg_samples = set()
		while (len(neg_samples) < eval_item_num):
			iid = random.randint(0, item_num-1)
			if iid not in item_list:
				neg_samples.add(iid)
		neg_samples = list(neg_samples)
		neg_sequence.append(neg_samples)
	return neg_sequence

class Relation_Former:
	def __init__(self, dir_path):
		self.dir_path = dir_path
		self.seq_data_path = self.dir_path + "/seq_data.pickle"

	def init_data(self):
		with open(self.seq_data_path, "rb") as f:
			train_users, input_seq, self.data_config = pickle.load(f)
		self.generate_his_info()
		self.generate_relation_info()
		r_interval, cate_seq = self.generate_interval_info(train_users, input_seq)
		return self.numCate, r_interval, cate_seq

	def generate_his_info(self):
		self.numUser = self.data_config["numUser"]
		self.numItem = self.data_config["numItem"]
		# self.time_map = np.zeros((self.numUser, self.numItem))
		self.time_map = dict()
		# self.history_items = []
		# self.history_times = []

		rating_names = ["user_id", "item_id", "time"]
		self.ratings = pd.read_csv(self.dir_path + "/train.csv", sep="\t",
								   names=rating_names, engine='python')
		his_max = 20

		self.user_his_dict = dict()
		for row in range(len(self.ratings)):
			user_id = int(self.ratings["user_id"][row])
			item_id = int(self.ratings["item_id"][row])
			time = int(self.ratings["time"][row])

			# self.time_map[user_id][item_id] = time
			if user_id not in self.time_map:
				self.time_map[user_id] = dict()
				self.time_map[user_id][item_id] = time
			else:
				self.time_map[user_id][item_id] = time

			if user_id not in self.user_his_dict:
				self.user_his_dict[user_id] = []
			# for tup in self.user_his_dict[user_id]:
			#	 self.history_items.append(tup[0])
			#	 self.history_times.append(tup[1])
			self.user_his_dict[user_id].append((item_id, time))

	def generate_relation_info(self):
		self.relation_tuples = set()
		self.relation_data = pd.read_csv(self.dir_path + "/item_meta.csv", sep="\t")
		item_id = list(self.relation_data["item_id"])
		c_ids = list(self.relation_data["category"])
		r_com_list = list(self.relation_data["r_complement"])
		r_sub_list = list(self.relation_data["r_substitute"])

		cate_set = set()
		self.c_id_map = np.zeros((self.numItem))
		for i in range(len(item_id)):
			r_com_list[i] = eval(r_com_list[i])
			r_sub_list[i] = eval(r_sub_list[i])
			item = item_id[i]
			self.c_id_map[item] = c_ids[i]
			cate_set.add(c_ids[i])
		self.numCate = len(cate_set)

		for index in range(len(item_id)):
			item_head = item_id[index]
			r_com = r_com_list[index]
			r_sub = r_sub_list[index]
			for item_tail in r_com:
				self.relation_tuples.add( tuple([int(item_head), 0, int(item_tail)]) )
				self.relation_tuples.add( tuple([int(item_tail), 0, int(item_head)]) )
			for item_tail in r_sub:
				self.relation_tuples.add(tuple([int(item_head), 1, int(item_tail)]))
				self.relation_tuples.add(tuple([int(item_tail), 1, int(item_head)]))

	# def generate_interval_info(self):
	#	 for index in range(len(self.ratings)):
	def generate_interval_info(self, train_users, input_seq):
		#train_users: sequencesnum
		#input_seq: sequencesnum, input_length
		#r_interval: (sequencesnum, input_length, 2)


		'''
		# Collect information related to the target item:
			# - category id
			# - time intervals w.r.t. recent relational interactions (-1 if not existing)
			category_id = [self.item2cate[x] for x in feed_dict['item_id']]
			relational_interval = list()
			for i, target_item in enumerate(feed_dict['item_id']):
				interval = np.ones(self.model.relation_num, dtype=float) * -1
				# reserve the first dimension for the repeat consumption interval
				for j in range(len(history_item))[::-1]:
					if history_item[j] == target_item:
						interval[0] = (time - history_time[j]) / self.model.time_scalar
						break
				# the rest for relational intervals
				for r_idx in range(1, self.model.relation_num):
					for j in range(len(history_item))[::-1]:
						if (history_item[j], r_idx, target_item) in self.corpus.triplet_set:
							interval[r_idx] = (time - history_time[j]) / self.model.time_scalar
							break
				relational_interval.append(interval)
		'''


		self.r_interval = []
		history_items_all_dict = {}
		time_scalar = 60*60*24*100
		history_num = 20
		for i, uid in enumerate(train_users):
			intervals = []
			item_list = input_seq[i]
			#print(item_list)
			uidd = int(uid)
			#print(self.user_his_dict[0])
			#print(self.user_his_dict[int(uid)])
			#print(self.user_his_dict[uidd])
			target_item = int(item_list[0])
			if target_item not in self.time_map[uidd]:
				now_time = 0
			else:
				now_time = self.time_map[uidd][target_item]
			if uidd not in history_items_all_dict.keys():
				history_items_all = sorted(self.user_his_dict[uidd], key=lambda x: x[1])
				print(uidd)
				#print(history_items_all)
				history_items_all_dict[uidd] = list(history_items_all)
			else:
				history_items_all = history_items_all_dict[uidd]
			now_index = history_items_all.index((target_item, now_time))
			history_item = history_items_all[now_index-history_num:now_index]
			for now_item in item_list:
				interval = [-1] * 2
				for r_idx in range(0, 2):
					for j in range(len(history_item))[::-1]:
						if tuple([history_item[j][0], r_idx, int(now_item)]) in self.relation_tuples:
							interval[r_idx] = (now_time - history_item[j][1]) / time_scalar
							break
				intervals.append(interval)
			self.r_interval.append(intervals)

		self.cate_seq = []
		for i, uid in enumerate(train_users):
			item_list = input_seq[i]
			cate = [self.c_id_map[x] for x in item_list]
			self.cate_seq.append(cate)

		return self.r_interval, self.cate_seq


if __name__ == "__main__":
	"""
	a = Amazon()
	# data_records = a.get_history_item_users("./dataset/Amazon/train_1.csv")
	# print(data_records)
	train_set, val_set, train_val_set, test_set, num_users, num_items = a.generate_dataset()
	f = Former(train_val_set, num_users, num_items)
	print(f.user_ids, f.item_ids)
	print(len(f.user_ids), len(f.item_ids))
	print("numUser: %d, numItem: %d" % (f.num_users, f.num_items))
	sequences, test_sequences, target_sequences, sequences_users = f.to_sequence()
	print("numSequences: %d" % (f.num_subsequences))
	f.get_history_item_users("Data/Amazon/train.csv")
	train_users = f.user_ids  # user的编号
	train_sequences_input = f.sequences  # 每个user访问的input_lenth个item
	train_sequences_user_input = []  # 每个item的前k个用户
	train_sequences_target = []  # 每个user的target_lenth个item
	familiar_num = 5
	for i in range(f.num_subsequences):
		uid = f.user_ids[i]
		item_hisusers_list = []
		for iid in sequences[i]:
			his_seq = f.item_to_pastusers[iid][-familiar_num:]
			item_hisusers_list.append(his_seq)
		train_sequences_user_input.append(item_hisusers_list)

	print(train_sequences_input)
	print(len(train_sequences_input))
	print(train_sequences_user_input)
	# print(f.item_to_pastusers)
	# print(len(sequences))
	# a = train_val_set[:5]
	# b = test_set[:5]
	# print(a)
	# print(b)
	# print(len(train_val_set), len(test_set), num_users)
	# print(len(sequences))
	# print(num_users)
	"""
