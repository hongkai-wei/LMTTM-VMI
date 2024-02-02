import os
import json

exp_json = "exp_preprocess_noise.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\evaluate.py " + exp_json)

train_config = {
    "name":["exp0_3d_None", "exp1_3d_uniform", "exp2_3d_laplace", "exp3_3d_normal", "exp4_3d_exp", "exp5_3d_gamma", "exp6_3d_poisson",
            "exp7_3dBN_None", "exp8_3dBN_uniform", "exp9_3dBN_laplace", "exp10_3dBN_normal", "exp11_3dBN_exp", "exp12_3dBN_gamma", "exp13_3dBN_poisson"],    

    "preprocess_mode":["3d", "3d", "3d", "3d", "3d", "3d", "3d",
                       "3dBN", "3dBN", "3dBN", "3dBN", "3dBN", "3dBN", "3dBN"],

    "load_memory_add_noise":[False, True, True, True, True, True, True,
                             False, True, True, True, True, True, True,],

    "load_memory_add_noise_mode":["None", "uniform", "laplace", "normal", "exp", "gamma", "poisson",
                                  "None", "uniform", "laplace", "normal", "exp", "gamma", "poisson",]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
       
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['preprocess_mode'] = train_config["preprocess_mode"][i]
            data['model']['load_memory_add_noise'] = train_config["load_memory_add_noise"][i]
            data['model']['load_memory_add_noise_mode'] = train_config["load_memory_add_noise_mode"][i]

            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)