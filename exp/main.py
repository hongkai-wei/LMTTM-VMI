import os
import json

exp_json = "base.json" 

def run_exp(exp_json):
    # os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\evaluate.py " + exp_json)

train_config = {
    "checkpoint_dir":["./checkpoints"]
}

if __name__ == "__main__":
    for i in range(len(train_config["checkpoint_dir"])):
            
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['checkpoint_dir'] = train_config["checkpoint_dir"][i]


            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)
