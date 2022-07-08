import argparse
from NAS_AutoSR import AutoSR
from NAS_Evolution import Evolution
from NAS_Random import Random
from NAS_RL import RL
from CodeEvaluation import *
from Ablation_AutoSR_EdgeWeight import AutoSR_EdgeWeight
from Ablation_AutoSR_GCN import AutoSR_GCN
from Ablation_AutoSR_Graph import AutoSR_Graph


# Parser arguments
parser = argparse.ArgumentParser(description='NAS Method for Sequential Recommendation Model Design.')
parser.add_argument('--config_path', '--cp', type=str, default="SConfig.json", help='the file path for the config information')
parser.add_argument('--data_name', '--dm', type=str, default="Movies_and_TV", help='the SR dataset name')
parser.add_argument('--nas_method', '--nm', type=str, default="AutoSR", help='the file path for the config information')
parser.add_argument('--gpu_device', '--gpu', type=int, default=3, help='gpu device number')
args = parser.parse_args()

if __name__ == '__main__':
	if args.nas_method == "Random":
		Random(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
	elif args.nas_method == "AutoSR":
		AutoSR(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
	elif args.nas_method == "Evolution":
		Evolution(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
	elif args.nas_method == "RL":
		RL(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
	elif args.nas_method == "AutoSR_EdgeWeight":
		AutoSR_EdgeWeight(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
	elif args.nas_method == "AutoSR_GCN":
		AutoSR_GCN(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
	elif args.nas_method == "AutoSR_Graph":
		AutoSR_Graph(args.config_path, args.data_name, args.gpu_device, CodeEvaluation).main()
