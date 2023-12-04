import os
import json

'''
Explanation:
This script selects the best model for variant.
According to the results, exp has the best result, and we choose exp as the optimal model parameter.

The best parameter:
                    variant: variant1
'''
exp_json = "base_hmdb.json"

def run_exp(exp_json):
    os.system("python exp\\train_hmdb_continual.py " + exp_json)
    os.system("python exp\\predict_hmdb_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["ex06_256_numb16_hmdb", "ex07_512_numb16_hmdb", "ex03_512_numb64_hmdb", "ex04_1024_numb128_hmdb", "ex05_2048_numb256_hmdb"],

    "memory_tokens_size":[256, 512, 512, 1024, 2048],

    "num_blocks":[16, 16, 64, 128, 256]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
        with open(f'./config/{exp_json}', 'r') as file:
            data = json.load(file)

        data['train']['name'] = train_config["name"][i]
        data['model']['num_blocks'] = train_config["num_blocks"][i]
        data['model']['memory_tokens_size'] = train_config["memory_tokens_size"][i]

        with open(f'./config/{exp_json}', 'w') as file:
            json.dump(data, file, indent=4)
        
        run_exp(exp_json)