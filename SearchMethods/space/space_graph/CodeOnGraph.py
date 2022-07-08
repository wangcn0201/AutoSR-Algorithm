import torch
#from ModuleRelation import *


class CodeOnGraph(object):
	'''
	Function: Transform the model_code in the list into the index mode in the knowledge graph, where each index denotes the value choice of each parameter
	Input: code_list (a set of model_code: (key_name)->[key_type, key_value])
		   edge_index_dict((node1_index,node2_index)->edge_index)
		   key_type_value_index((key_type,key_value)->index)
	Output: code_node_index_list (index representation of nodes in code_list: (key_name->[key_type, key_value])->node_index in graph)
		    code_edge_index_list (index representation of edges in code_list: (node_index1, node_index2)->edge index in graph)
	'''
	def __init__(self, edge_index_dict, key_type_value_index):
		super(CodeOnGraph, self).__init__()
		self.edge_index_dict = edge_index_dict
		self.key_type_value_index = key_type_value_index
		return

	def main(self, code_list):
		code_node_index_list = []
		code_edge_index_list = []
		for i in range(len(code_list)):
			try:
				model_path = code_list[i][1]
				key_name_list = list(model_path.keys())
				code_node_index = []
				for j in range(len(key_name_list)):
					key_name = key_name_list[j]
					key_type, key_value = model_path[key_name]
					key_value_index = self.key_type_value_index[str(key_type)+","+str(key_value)]
					code_node_index.append(key_value_index)

				code_edge_index = [self.edge_index_dict[str(code_node_index[t+1])+","+str(code_node_index[t])] for t in range(len(code_node_index)-1)]
				code_node_index_list.append(list(code_node_index))
				code_edge_index_list.append(list(code_edge_index))
			except:
				print("Wrong model_code (code: %d) in code_list"% (i+1))
				return None, None
		code_node_index_list = code_node_index_list
		code_edge_index_list = code_edge_index_list
		return code_node_index_list, code_edge_index_list
'''
from BuildGraph import *
from RandomCode import *
dense_adj, edge_index, edge_index_dict, key_type_value_index, node_index_to_type_value, key_name_list = BuildReverseGraph()
print()
print(key_name_list)
obj = CodeOnGraph(edge_index_dict, key_type_value_index, key_name_list)
random_code_list = RandomCode(2)
print()
print(random_code_list[0][1])
print()
print(random_code_list[1][1])
code_node_index_list, code_edge_index_list = obj.main(random_code_list)
print()
print(code_node_index_list)
print(code_edge_index_list)
print()
'''
