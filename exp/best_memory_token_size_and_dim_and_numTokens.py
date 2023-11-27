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
    os.system("python exp\\train_hmdb_continual.py " + exp_json)
    os.system("python exp\\predict_hmdb_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp0_memory8_and_dim32_and_numTokens8",
             "exp1_memory16_and_dim32_and_numTokens8",
             "exp2_memory32_and_dim32_and_numTokens8",
             "exp3_memory64_and_dim32_and_numTokens8",
             "exp4_memory96_and_dim32_and_numTokens8",
             "exp0_memory8_and_dim64_and_numTokens8",
             "exp1_memory16_and_dim64_and_numTokens8",
             "exp2_memory32_and_dim64_and_numTokens8",
             "exp3_memory64_and_dim64_and_numTokens8",
             "exp4_memory96_and_dim64_and_numTokens8",
             "exp0_memory8_and_dim32_and_numTokens16",
             "exp1_memory16_and_dim32_and_numTokens16",
             "exp2_memory32_and_dim32_and_numTokens16",
             "exp3_memory64_and_dim32_and_numTokens16",
             "exp4_memory96_and_dim32_and_numTokens16",
             "exp0_memory8_and_dim64_and_numTokens16",
             "exp1_memory16_and_dim64_and_numTokens16",
             "exp2_memory32_and_dim64_and_numTokens16",
             "exp3_memory64_and_dim64_and_numTokens16",
             "exp4_memory96_and_dim64_and_numTokens16"],

    "memory_tokens_size":[8, 16, 32, 64, 96,
                          8, 16, 32, 64, 96,
                          8, 16, 32, 64, 96,
                          8, 16, 32, 64, 96],
    "dim":[32, 32, 32, 32, 32, 
           64, 64, 64, 64, 64,
           32, 32, 32, 32, 32, 
           64, 64, 64, 64, 64],
    "summerize_num_tokens":[8, 8, 8, 8, 8,
                  8, 8, 8, 8, 8,
                  16, 16, 16, 16, 16,
                  16, 16, 16, 16, 16]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['memory_tokens_size'] = train_config["memory_tokens_size"][i]
            data['model']['dim'] = train_config["dim"][i]
            data['model']['summerize_num_tokens'] = train_config["summerize_num_tokens"][i]

            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)
