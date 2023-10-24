import os
import json

'''
Explanation:
This script selects the best model for positional_embedding.
According to the results, exp1 has the best result, and we choose exp1 as the optimal model parameter.

The best parameter:
                    positional_embedding: None_positional_embedding
'''
exp_json = "best_positional_embedding.json"

def run_exp(exp_json):
    # os.system("python exp\\train_continual.py " + exp_json)
    # os.system("python exp\\predict_continual.py " + exp_json)
    os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp1_None_positional_embedding", "exp2_Read_positional_embedding", "exp3_Write_positional_embedding", "exp4_ReadWrite_positional_embedding"],
    "read_use_positional_embedding": [False, True, False, True],
    "write_use_positional_embedding": [False, False, True, True]
}

for i in range(len(train_config["name"])):

    with open(f'./config/{exp_json}', 'r') as file:
        data = json.load(file)

    data['train']['name'] = train_config["name"][i]
    data['model']['Read_use_positional_embedding'] = train_config["read_use_positional_embedding"][i]
    data['model']['Write_use_positional_embedding'] = train_config["write_use_positional_embedding"][i]

    with open(f'./config/{exp_json}', 'w') as file:
        json.dump(data, file, indent=4)

    run_exp(exp_json)