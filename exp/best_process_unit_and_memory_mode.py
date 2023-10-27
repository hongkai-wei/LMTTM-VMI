import os
import json

'''
Explanation:
This script selects the best model for process_unit and memory_mode.
According to the results, exp3 has the best result, and we choose exp3 as the optimal model parameter.

The best parameter:
                    process_unit: transformer
                    memory_mode: TL
'''

exp_json = "best_process_unit_and_memory_mode.json"

def run_exp(exp_json):
    os.system("python exp\\train_continual.py " + exp_json)
    os.system("python exp\\predict_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp001_MHA_transformer_pro","exp002_AddErase_transformer_pro","exp003_TL_transformer_pro-4_tken8",
             "exp4_MHA_mlp",        "exp5_AddErase_mlp",        "exp6_TL_mlp",
             "exp7_MHA_mixer",      "exp8_AddErase_mixer",      "exp9_TL_mixer"],

    "process_unit":["transformer","transformer","transformer",
                    "mlp","mlp","mlp",
                    "mixer","mixer","mixer"],

    "memory_mode":["TL-MHA","TL-AddErase","TL",
                   "TL-MHA","TL-AddErase","TL",
                   "TL-MHA","TL-AddErase","TL",],

}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
        if i == 2:
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['process_unit'] = train_config["process_unit"][i]
            data['model']['memory_mode'] = train_config["memory_mode"][i]
            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)
