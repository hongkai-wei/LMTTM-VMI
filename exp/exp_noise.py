import os
import json

exp_json = "exp_noise.json" 

def run_exp(exp_json):
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {
    "model":["ttm", "ttm","ttm","ttm","ttm","ttm","ttm",
             "lmttm", "lmttm", "lmttm", "lmttm", "lmttm", "lmttm", "lmttm", 
             "lmttm", "lmttm", "lmttm", "lmttm", "lmttm", "lmttm", "lmttm", 
             "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2","lmttm_v2",
             "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2", "lmttm_v2"],

    "load_memory_add_noise":[False,True,True,True,True,True,True,
                             False,True,True,True,True,True,True,
                             False,True,True,True,True,True,True,
                             False,True,True,True,True,True,True,
                             False,True,True,True,True,True,True,],

    "load_memory_add_noise_mode":["None","uniform","laplace","normal","exp","gamma","poisson",
                                  "None","uniform","laplace","normal","exp","gamma","poisson",
                                  "None","uniform","laplace","normal","exp","gamma","poisson",
                                  "None","uniform","laplace","normal","exp","gamma","poisson",
                                  "None","uniform","laplace","normal","exp","gamma","poisson"],

    "num_blocks":[0,0,0,0,0,0,0,
                  16,16,16,16,16,16,16,
                  96,96,96,96,96,96,96,
                  16,16,16,16,16,16,16,
                  96,96,96,96,96,96,96],

    "name":["exp0_TTM_None","exp1_TTM_uniform","exp2_TTM_laplace","exp3_TTM_normal","exp4_TTM_exp","exp5_TTM_gamma","exp6_TTM_poisson",
            "exp7_LMTTM_1_None","exp8_LMTTM_1_uniform","exp9_LMTTM_1_laplace","exp10_LMTTM_1_normal","exp11_LMTTM_1_exp","exp12_LMTTM_1_gamma","exp13_LMTTM_1_poisson",
            "exp14_LMTTM_2_None" ,"exp15_LMTTM_2_uniform","exp16_LMTTM_2_laplace","exp17_LMTTM_2_normal","exp18_LMTTM_2_exp","exp19_LMTTM_2_gamma","exp20_LMTTM_2_poisson",
            "exp21_LMTTMv2_1_None","exp22_LMTTMv2_1_uniform","exp23_LMTTMv2_1_laplace","exp24_LMTTMv2_1_normal","exp25_LMTTMv2_1_exp","exp26_LMTTMv2_1_gamma","exp27_LMTTMv2_1_poisson",
            "exp28_LMTTMv2_2_None","exp29_LMTTMv2_2_uniform","exp30_LMTTMv2_2_laplace","exp31_LMTTMv2_2_normal","exp32_LMTTMv2_2_exp","exp33_LMTTMv2_2_gamma","exp34_LMTTMv2_2_poisson"
            ]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
            
            with open(f'./config/{exp_json}', 'r') as file:
                data = json.load(file)

            data['train']['name'] = train_config["name"][i]
            data['model']['load_memory_add_noise_mode'] = train_config["load_memory_add_noise_mode"][i]
            data['model']['load_memory_add_noise'] = train_config["load_memory_add_noise"][i]
            data['model']['num_blocks'] = train_config["num_blocks"][i]
            data['model']['model'] = train_config["model"][i]


            with open(f'./config/{exp_json}', 'w') as file:
                json.dump(data, file, indent=4)
            
            run_exp(exp_json)
