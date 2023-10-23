import os
import json

'''
Explanation:
This script selects the best model for process_unit and memory_mode.
According to the results, exp1 has the best result, and we choose exp1 as the optimal model parameter.

The best parameter:
                    process_unit: transformer
                    memory_mode: TL-MHA
'''

####exp1
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp1_MHA_transformer"
data['model']['process_unit'] = "transformer"
data['model']['memory_mode'] = "TL-MHA"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")

#exp2
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp2_AddErase_transformer"
data['model']['process_unit'] = "transformer"
data['model']['memory_mode'] = "TL-AddErase"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")

#exp3
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp3_TL_transformer"
data['model']['process_unit'] = "transformer"
data['model']['memory_mode'] = "TL"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")

#exp4
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp4_MHA_mlp"
data['model']['process_unit'] = "mlp"
data['model']['memory_mode'] = "TL-MHA"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")

#exp5
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp5_AddErase_mlp"
data['model']['process_unit'] = "mlp"
data['model']['memory_mode'] = "TL-AddErase"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")

#exp6
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp6_TL_mlp"
data['model']['process_unit'] = "mlp"
data['model']['memory_mode'] = "TL"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")

#exp7
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp7_MHA_mixer"
data['model']['process_unit'] = "mixer"
data['model']['memory_mode'] = "TL-MHA"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")
#exp8
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp8_AddErase_mixer"
data['model']['process_unit'] = "mixer"
data['model']['memory_mode'] = "TL"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")
#exp9
with open('./config/best_process_unit_and_memory_mode.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp9_TL_mixer"
data['model']['process_unit'] = "mixer"
data['model']['memory_mode'] = "TL-AddErase"
with open('./config/best_process_unit_and_memory_mode.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")