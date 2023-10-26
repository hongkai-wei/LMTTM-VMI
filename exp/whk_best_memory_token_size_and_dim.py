import os
import json

'''
Explanation:
This script selects the best model for variant.
According to the results, exp3 has the best result, and we choose exp3 as the optimal model parameter.

The best parameter:
                    memory_token_size: 
                    dim: 

'''
exp_json = "best_memory_token_size_and_dim.json"

def run_exp(exp_json):
    os.system("python exp\\train_continual.py " + exp_json)
    os.system("python exp\\predict_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp2_memory128_and_dim128",
             "exp4_memory256_and_dim128",
             "exp6_memory512_and_dim128",
             "exp8_memory1024_and_dim128"],

    "memory_tokens_size":[128, 256, 512, 1024],
    "dim":[128, 128, 128, 128],
    "batch_size":[96, 48, 24, 12],
    "epoch":[600, 1200, 2400, 4800]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):

            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['memory_tokens_size'] = train_config["memory_tokens_size"][i]
            data['model']['dim'] = train_config["dim"][i]
            data['batch_size'] = train_config["batch_size"][i]
            data['train']['epoch'] = train_config["epoch"][i]

            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)