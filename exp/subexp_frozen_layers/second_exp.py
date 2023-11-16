import os
import json

'''
Explanation:
This script selects the best model for variant.
According to the results, exp has the best result, and we choose exp as the optimal model parameter.

The best parameter:
                    variant: variant1
'''
exp_json = "base.json"

def run_exp(exp_json):
    os.system("python exp\\second_train.py " + exp_json)
    os.system("python exp\\predict_continual.py " + exp_json)
    # os.system("python exp\\tesorboard2excel.py " + exp_json)

train_config = {
    "name": ["exp2"]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
        with open(f'./config/{exp_json}', 'r') as file:
            data = json.load(file)

        data['train']['name'] = train_config["name"][i]
        data['model']['variant'] = train_config["variant"][i]

        with open(f'./config/{exp_json}', 'w') as file:
            json.dump(data, file, indent=4)
        
        run_exp(exp_json)