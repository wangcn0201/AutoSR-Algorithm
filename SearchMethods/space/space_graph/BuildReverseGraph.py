import torch


def BuildReverseGraph(Parameters, Options, Relations):
	# Build Reverse-Directed Relation Graph 
	dense_adj = [] # shape: [NodeNum, NodeNum]
	edge_index = [] # shape: [2, EdgeNum]
	edge_index_dict = {} # shape: [node1 index, node2 index]: edge index in edge_index

	edge_edgelist_dict = {'end_node_to_edge_indexes':{}, 'edge_index_to_end_node':{}}
	key_type_value_index = {}
	node_index_to_type_value = []
	index_num = 0
	key_name_list = list(Parameters.keys())
	for i in range(len(key_name_list)):
		key_name = key_name_list[i]
		key_types = Parameters[key_name]
		for j in range(len(key_types)):
			key_type = key_types[j]
			key_values = Options[key_type]
			for k in range(len(key_values)):
				key_value = key_values[k]
				key_type_value_index[str(key_type)+","+str(key_value)] = index_num
				node_index_to_type_value.append([key_name, key_type, key_value])
				index_num += 1
	key_type_value_index = key_type_value_index
	#print(key_type_value_index)
	NodeNum = len(list(key_type_value_index.keys()))
	dense_adj = [[0 for j in range(NodeNum)] for i in range(NodeNum)]

	for i in range(len(key_name_list)-1):
		key_name = key_name_list[i]
		key1_types = Parameters[key_name]
		for j in range(len(key1_types)):
			key1_type = key1_types[j]
			key1_values = Options[key1_type]
			node1_indexes = [key_type_value_index[str(key1_type)+","+str(key1_values[t])] for t in range(len(key1_values))]
			key2_types = Relations[key1_type]
			for k in range(len(key2_types)):
				key2_type = key2_types[k]
				key2_values = Options[key2_type]
				node2_indexes = [key_type_value_index[str(key2_type)+","+str(key2_values[t])] for t in range(len(key2_values))]
				for node1 in node1_indexes:
					for node2 in node2_indexes:
						dense_adj[node2][node1] = 1
						edge_index.append([node2, node1])
						edge_index_dict[str(node2)+","+str(node1)] = len(edge_index)-1
						edge_edgelist_dict['edge_index_to_end_node'][str(len(edge_index)-1)] = str(node1)
						if str(node1) not in edge_edgelist_dict['end_node_to_edge_indexes'].keys():
							edge_edgelist_dict['end_node_to_edge_indexes'][str(node1)] = []
						edge_edgelist_dict['end_node_to_edge_indexes'][str(node1)].append(len(edge_index)-1)
						#dense_adj[node1][node2] = 1
						#edge_index.append([node1, node2])
						#edge_index_dict[str(node1)+","+str(node2)] = len(edge_index)-1
	dense_adj = torch.Tensor(dense_adj) # shape: (node_num, node_num)
	edge_index = torch.IntTensor(edge_index).permute(1, 0).type(torch.LongTensor) # shape: (2, edge_num)
	return dense_adj, edge_index, edge_index_dict, edge_edgelist_dict, key_type_value_index, node_index_to_type_value, key_name_list
'''
from ModuleRelation import *
dense_adj, edge_index, edge_index_dict, edge_edgelist_dict, key_type_value_index, node_index_to_type_value, key_name_list = BuildReverseGraph(Parameters, Options, Relations)
print(dense_adj.shape)
print(edge_index.shape)
print(dense_adj)
for i in range(dense_adj.shape[0]):
	print(dense_adj[i])
print(edge_index)
print()
print(node_index_to_type_value)
print()
print(key_type_value_index)
print()
print(edge_index_dict)
'''
