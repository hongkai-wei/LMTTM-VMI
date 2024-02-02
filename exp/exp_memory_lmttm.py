import os
import json

exp_json = "exp_memory_lmttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\evaluate.py " + exp_json)

train_config = {

    "name":["exp0_mem64_dim64", "exp1_mem64_dim160", "exp2_mem64_dim256", "exp3_mem64_dim352", "exp4_mem64_dim448",
            "exp5_mem160_dim64", "exp6_mem160_dim160", "exp7_mem160_dim256", "exp8_mem160_dim352", "exp9_mem160_dim448",
            "exp10_mem256_dim64", "exp11_mem256_dim160", "exp12_mem256_dim256", "exp13_mem256_dim352", "exp14_mem256_dim448",
            "exp15_mem352_dim64", "exp16_mem352_dim160", "exp17_mem352_dim256", "exp18_mem352_dim352", "exp19_mem352_dim448",
            "exp20_mem448_dim64", "exp21_mem448_dim160", "exp22_mem448_dim256", "exp23_mem448_dim352", "exp24_mem448_dim448"],    

    "memory_tokens_size":[64, 64, 64, 64, 64,
                          160, 160, 160, 160, 160,
                          256, 256, 256, 256, 256,
                          352, 352, 352, 352, 352,
                          448, 448, 448, 448, 448],

    "dim":[64, 160, 256, 352, 448,
           64, 160, 256, 352, 448,
           64, 160, 256, 352, 448,
           64, 160, 256, 352, 448,
           64, 160, 256, 352, 448],
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
       
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['memory_tokens_size'] = train_config["memory_tokens_size"][i]
            data['model']['dim'] = train_config["dim"][i]

            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)