import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import APPNP
import torch.nn.functional as F
import numpy as np
import random
from space.space_graph.RandomCode import CodeInitialization


class GraphModel(torch.nn.Module):
	def __init__(self, node_num, Parameters, Options, Relations, embedding_dim=32, output_dim=16, hidden_dim=8, K=10, alpha=0.1, dropout=0.5):
		super(GraphModel, self).__init__()
		self.embedding = nn.Embedding(node_num, embedding_dim)
		self.lin1 = Linear(embedding_dim, hidden_dim)
		self.lin2 = Linear(hidden_dim, output_dim)
		self.gcn = APPNP(K, alpha)
		self.nodes = torch.LongTensor([i for i in range(node_num)]).cuda()
		self.dropout = dropout
		self.Parameters, self.Options, self.Relations = Parameters, Options, Relations
		
	def OptimalCode(self, pred, edge_index, node_index_to_type_value, epsilon):
		pred = pred.data.cpu().numpy().tolist()
		tempt = edge_index.data.cpu().numpy()
		edge_index = []
		for i in range(len(tempt)):
			edge_index.append(tempt[i].tolist())

		node_connection_pred = {}
		for i in range(len(edge_index[0])):
			node1 = edge_index[0][i]
			node2 = edge_index[1][i]
			pred_value =  pred[i]
			if str(node2) not in node_connection_pred.keys():
				node_connection_pred[str(node2)] = []
			node_connection_pred[str(node2)].append([node1,pred_value])

		initial_code, _ = CodeInitialization(self.Parameters, self.Options, self.Relations)
		optimal_code = initial_code
		optimal_path = {'Start':['Start',None]}
		pre_node_index = 0
		source_label = []
		optimal_code_node_index = [0]
		while True:
			if str(pre_node_index) not in node_connection_pred.keys():
				break
			connections = node_connection_pred[str(pre_node_index)]
			
			if np.random.rand() < epsilon: # random selection
				best_node, best_pred = random.sample(connections,1)[0]
				source_label.append("Random")
			else: # best selection
				connections = sorted(connections, key=(lambda x: x[1]), reverse = True)
				best_node, best_pred = connections[0]
				source_label.append("Best")
			key_name, key_type, key_value = node_index_to_type_value[best_node]
			optimal_code[key_name] = [key_type, key_value]
			optimal_path[key_name] = [key_type, key_value]
			pre_node_index = best_node
			optimal_code_node_index.append(pre_node_index)
		return optimal_code, optimal_path, optimal_code_node_index, source_label

	def PolicyGradient(self, pred, edge_edgelist_dict, batch_data):
		# batch_data: batch_data_code -> edge_index list, batch_data_score -> [performance score] e.g., accuracy
		edge_num = pred.shape[0]
		pred_sum_list = {}
		possibility = []
		for i in range(edge_num):
			edge = i
			end_node = edge_edgelist_dict['edge_index_to_end_node'][str(edge)]
			if end_node not in pred_sum_list.keys():
				edge_indexes = torch.Tensor(edge_edgelist_dict['end_node_to_edge_indexes'][end_node]).cuda()
				pred_sum_list[end_node] = torch.sum(torch.index_select(pred, 0, edge_indexes.long()))
			possibility.append(pred[edge]*1.0/pred_sum_list[end_node])
		possibility = torch.stack(possibility, dim=0)
		#print(type(possibility))
		#print(possibility.size())
		#print(edge_num)

		batch_data_code = batch_data[0]
		batch_data_score = batch_data[1]
		code_num = batch_data_score.shape[0]
		for i in range(code_num):
			code = batch_data_code[i]
			perf_score = batch_data_score[i]*(-1)
			prob_scores = torch.index_select(possibility, 0, code)
			if i == 0:
				policy_value = torch.mean(prob_scores*perf_score)
			else:
				policy_value += torch.mean(prob_scores*perf_score)
		#print(type(policy_value))
		#print(policy_value.size())
		#print(policy_value)
		return policy_value

	def forward(self, edge_index, node_index_to_type_value=None, epsilon=None , edge_edgelist_dict=None, batch_data=None, function=None):
		x = self.embedding(self.nodes)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = F.relu(self.lin1(x))
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.lin2(x)
		x = self.gcn(x, edge_index)

		nodes_first = torch.index_select(x, 0, edge_index[0,:].long())
		nodes_second = torch.index_select(x, 0, edge_index[1,:].long())
		pred = torch.sum(nodes_first * nodes_second, dim=-1)
		if function == 'Recommendation' and node_index_to_type_value != None and epsilon != None:
			optimal_code, optimal_path, optimal_code_node_index, source_label = self.OptimalCode(pred, edge_index, node_index_to_type_value, epsilon)
			return optimal_code, optimal_path, optimal_code_node_index, source_label
		elif function == 'Optimization' and edge_edgelist_dict != None and batch_data != None:
			policy_value = self.PolicyGradient(pred, edge_edgelist_dict, batch_data)
			return policy_value
		return pred
