import argparse
from CodeEvaluation import *


# Parser arguments
parser = argparse.ArgumentParser(description='Models for Sequential Recommendation.')
parser.add_argument('--config_path', '--cp', type=str, default="EConfig.json", help='the file path for the config information')
parser.add_argument('--data_name', '--dm', type=str, default="Movies_and_TV", help='the SR dataset name')
parser.add_argument('--model_name', '--mn', type=str, default="P1-MARank", help='the file path for the config information')
parser.add_argument('--model_code', '--mc', type=str, default="None", help='the file path for the config information')
parser.add_argument('--gpu_device', '--gpu', type=int, default=3, help='gpu device number')
parser.add_argument('--mode', '--m', type=str, default="train", help='train a SR model or load a SR model from model_load_dir')
parser.add_argument('--model_load_dir', '--mld', type=str, default="", help='the file path of weights for a SR model')
args = parser.parse_args()

if __name__ == '__main__':
	config_path = args.config_path
	data_name = args.data_name
	model_name = args.model_name
	gpu_device = args.gpu_device
	model_code = eval(args.model_code)
	mode = args.mode
	model_load_dir = args.model_load_dir

	if args.model_name == "P1-MARank":
		model_code = Parameters = {
			'Start': None,
			'HIPM': 'HIPM_1', 'HIPM-EmbSize': 128, 'HIPM-WinSize': 5,
			'UVPM': 'UVPM_1',
			'1stFExS': 'FExS_1', '1stFExS-k': 2, '1stFExS-lhid': 10, '1stFExS-Agg': 'Max', '1stFExS-None': None,
			'1stFEnS': 'FEnS_1', '1stFEnS-L': 2, '1stFEnS-None': None,
			'2ndFExS': 'FExS_5', '2ndFExS-k': 2, '2ndFExS-lhid': 10, '2ndFExS-Agg': 'Max', '2ndFExS-None': None,
			'2ndFEnS': 'FEnS_2', '2ndFEnS-L': 2, '2ndFEnS-None': None,
			'IRM': 'IRM_1',
			'PS': 'PS_1', 'PS-None1': None, 'PS-K': 15, 'PS-None2': None,
			'1stLF': 'LF_1',
			'2ndLF': 'LF_1',
			'OF': 'Adam',
			'LR': 1e-3,
			'End': None
		}
	elif args.model_name == "P2-GRU-DIB-PEB":
		model_code = Parameters = {
			'Start': None,
			'HIPM': 'HIPM_2', 'HIPM-EmbSize': 128, 'HIPM-WinSize': 5,
			'UVPM': 'UVPM_2',
			'1stFExS': 'FExS_2', '1stFExS-k': 2, '1stFExS-lhid': 10, '1stFExS-Agg': 'Max', '1stFExS-None': None,
			'1stFEnS': 'FEnS_2', '1stFEnS-L': 2, '1stFEnS-None': None,
			'2ndFExS': 'FExS_5', '2ndFExS-k': 2, '2ndFExS-lhid': 10, '2ndFExS-Agg': 'Max', '2ndFExS-None': None,
			'2ndFEnS': 'FEnS_2', '2ndFEnS-L': 2, '2ndFEnS-None': None,
			'IRM': 'IRM_1',
			'PS': 'PS_2', 'PS-None1': None, 'PS-K': 15, 'PS-None2': None,
			'1stLF': 'LF_2',
			'2ndLF': 'LF_2',
			'OF': 'Adam',
			'LR': 1e-3,
			'End': None
		}
	elif args.model_name == "P3-S-DIV+CB":
		model_code = Parameters = {
			'Start': None,
			'HIPM': 'HIPM_1', 'HIPM-EmbSize': 128, 'HIPM-WinSize': 5,
			'UVPM': 'UVPM_2',
			'1stFExS': 'FExS_2', '1stFExS-k': 2, '1stFExS-lhid': 10, '1stFExS-Agg': 'Max', '1stFExS-None': None,
			'1stFEnS': 'FEnS_2', '1stFEnS-L': 2, '1stFEnS-None': None,
			'2ndFExS': 'FExS_5', '2ndFExS-k': 2, '2ndFExS-lhid': 10, '2ndFExS-Agg': 'Max', '2ndFExS-None': None,
			'2ndFEnS': 'FEnS_2', '2ndFEnS-L': 2, '2ndFEnS-None': None,
			'IRM': 'IRM_1',
			'PS': 'PS_3', 'PS-None1': None, 'PS-K': 15, 'PS-None2': None,
			'1stLF': 'LF_4',
			'2ndLF': 'LF_4',
			'OF': 'Adagrad',
			'LR': 1e-2,
			'End': None
		}
	elif args.model_name == "P4-HGN":
		model_code = Parameters = {
			'Start': None,
			'HIPM': 'HIPM_3', 'HIPM-EmbSize': 128, 'HIPM-WinSize': 5,
			'UVPM': 'UVPM_2',
			'1stFExS': 'FExS_4', '1stFExS-k': 2, '1stFExS-lhid': 10, '1stFExS-Agg': 'Max', '1stFExS-None': None,
			'1stFEnS': 'FEnS_2', '1stFEnS-L': 2, '1stFEnS-None': None,
			'2ndFExS': 'FExS_5', '2ndFExS-k': 2, '2ndFExS-lhid': 10, '2ndFExS-Agg': 'Max', '2ndFExS-None': None,
			'2ndFEnS': 'FEnS_2', '2ndFEnS-L': 2, '2ndFEnS-None': None,
			'IRM': 'IRM_3',
			'PS': 'PS_4', 'PS-None1': None, 'PS-K': 15, 'PS-None2': None,
			'1stLF': 'LF_5',
			'2ndLF': 'LF_5',
			'OF': 'Adam',
			'LR': 1e-3,
			'End': None
		}
	elif args.model_name == "P5-Chorus":
		model_code = Parameters = {
			'Start': None,
			'HIPM': 'HIPM_1', 'HIPM-EmbSize': 128, 'HIPM-WinSize': 5,
			'UVPM': 'UVPM_2',
			'1stFExS': 'FExS_5', '1stFExS-k': 2, '1stFExS-lhid': 10, '1stFExS-Agg': 'Max', '1stFExS-None': None,
			'1stFEnS': 'FEnS_2', '1stFEnS-L': 2, '1stFEnS-None': None,
			'2ndFExS': 'FExS_5', '2ndFExS-k': 2, '2ndFExS-lhid': 10, '2ndFExS-Agg': 'Max', '2ndFExS-None': None,
			'2ndFEnS': 'FEnS_2', '2ndFEnS-L': 2, '2ndFEnS-None': None,
			'IRM': 'IRM_2',
			'PS': 'PS_5', 'PS-None1': None, 'PS-K': 15, 'PS-None2': None,
			'1stLF': 'LF_5',
			'2ndLF': 'LF_5',
			'OF': 'Adam',
			'LR': 1e-3,
			'End': None
		}
	elif args.model_name == "P5-Chorus-GMF":
		model_code = Parameters = {
			'Start': None,
			'HIPM': 'HIPM_1', 'HIPM-EmbSize': 128, 'HIPM-WinSize': 5,
			'UVPM': 'UVPM_2',
			'1stFExS': 'FExS_5', '1stFExS-k': 2, '1stFExS-lhid': 10, '1stFExS-Agg': 'Max', '1stFExS-None': None,
			'1stFEnS': 'FEnS_2', '1stFEnS-L': 2, '1stFEnS-None': None,
			'2ndFExS': 'FExS_5', '2ndFExS-k': 2, '2ndFExS-lhid': 10, '2ndFExS-Agg': 'Max', '2ndFExS-None': None,
			'2ndFEnS': 'FEnS_2', '2ndFEnS-L': 2, '2ndFEnS-None': None,
			'IRM': 'IRM_2',
			'PS': 'PS_6', 'PS-None1': None, 'PS-K': 15, 'PS-None2': None,
			'1stLF': 'LF_5',
			'2ndLF': 'LF_5',
			'OF': 'Adam',
			'LR': 1e-3,
			'End': None
		}

	CodeEvaluation(config_path, data_name, model_name, gpu_device, model_code, mode, model_load_dir).main(model_code)
