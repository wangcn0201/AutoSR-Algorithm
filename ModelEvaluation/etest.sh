#!/bin/sh

python Main_Model_Comparison.py \
--data_name "Movies_and_TV" \
--model_name "New+AutoSR+Movie" \
--model_code "{'Start': None, 'HIPM': 'HIPM_2', 'HIPM-EmbSize': 64, 'HIPM-WinSize': 10, 'UVPM': 'UVPM_1', '1stFExS': 'FExS_2', '1stFExS-k': None, '1stFExS-lhid': 50, '1stFExS-Agg': None, '1stFExS-None': None, '1stFEnS': 'FEnS_2', '1stFEnS-L': None, '1stFEnS-None': None, '2ndFExS': 'FExS_2', '2ndFExS-k': None, '2ndFExS-lhid': 550, '2ndFExS-Agg': None, '2ndFExS-None': None, '2ndFEnS': 'FEnS_1', '2ndFEnS-L': 2, '2ndFEnS-None': None, 'IRM': 'IRM_2', 'PS': 'PS_3', 'PS-None1': None, 'PS-K': None, 'PS-None2': None, '1stLF': 'LF_3', '2ndLF': 'LF_4', 'OF': 'Adam', 'LR': 0.005, 'End': None}" \
--mode "train" \
--gpu_device 0 


