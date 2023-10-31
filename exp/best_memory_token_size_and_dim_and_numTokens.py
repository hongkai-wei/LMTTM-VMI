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
exp_json = "best_memory_token_size_and_dim_and_numTokens.json"

def run_exp(exp_json):
    # os.system("python exp\\train_continual.py " + exp_json)
    os.system("python exp\\predict_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp1_memory512_and_dim128_and_numTokens8",
             "exp2_memory512_and_dim128_and_numTokens32",
             "exp3_memory512_and_dim128_and_numTokens64",
             "exp4_memory512_and_dim512_and_numTokens8",
             "exp5_memory512_and_dim512_and_numTokens16",
             "exp6_memory512_and_dim512_and_numTokens32",
             "exp7_memory512_and_dim512_and_numTokens64"],

    "memory_tokens_size":[512, 512, 512, 512, 512, 512, 512],
    "dim":[128, 128, 128, 512, 512, 512, 512],
    "batch_size":[24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
    "epoch":[800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800],
    "num_tokens":[8, 32, 64, 8, 16, 32, 64]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['memory_tokens_size'] = train_config["memory_tokens_size"][i]
            data['model']['dim'] = train_config["dim"][i]
            data['model']['num_tokens'] = train_config["num_tokens"][i]
            data['batch_size'] = train_config["batch_size"][i]
            data['train']['epoch'] = train_config["epoch"][i]

            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)
