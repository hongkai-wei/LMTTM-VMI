import os
import json

'''
Explanation:
This script selects the best model for variant.
According to the results, exp3 has the best result, and we choose exp3 as the optimal model parameter.

The best parameter:
                    load_memory_add_noise_mode: laplace
'''
exp_json = "best_load_memory_add_noise_mode.json"

def run_exp(exp_json):
    os.system("python exp\\train_continual.py " + exp_json)
    os.system("python exp\\predict_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp1_load_memory_add_noise_None", "exp1_load_memory_add_noise_uniform",
             "exp1_load_memory_add_noise_laplace", "exp1_load_memory_add_noise_normal",
             "exp1_load_memory_add_noise_exp", "exp1_load_memory_add_noise_gamma",
             "exp1_load_memory_add_noise_poisson"],

    "load_memory_add_noise":[False, True, True, True, True, True, True],
    "load_memory_add_noise_mode":["None", "uniform", "laplace", "normal", "exp", "gamma", "poisson"]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
        with open(f'./config/{exp_json}', 'r') as file:
            data = json.load(file)

        data['train']['name'] = train_config["name"][i]
        data['model']['load_memory_add_noise'] = train_config["load_memory_add_noise"][i]
        data['model']['load_memory_add_noise_mode'] = train_config["load_memory_add_noise_mode"][i]

        with open(f'./config/{exp_json}', 'w') as file:
            json.dump(data, file, indent=4)
        
        run_exp(exp_json)