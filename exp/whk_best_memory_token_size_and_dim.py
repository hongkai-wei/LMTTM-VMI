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
    "name": ["exp3_memory128_and_dim192", "exp4_memory128_and_dim256",
             "exp7_memory256_and_dim192", "exp8_memory256_and_dim256",
             "exp11_memory512_and_dim192", "exp12_memory512_and_dim256",
             "exp15_memory1024_and_dim192", "exp16_memory1024_and_dim256"],

    "memory_tokens_size":[128, 128, 256, 256, 512, 512, 1024, 1024],
    "dim":[192, 256, 192, 256, 192, 256, 192, 256],
    "batch_size":[36, 36, 36, 36, 36, 36, 36, 36],
    "epoch":[600, 600, 900, 900, 1500, 1500, 2400, 2400]
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
