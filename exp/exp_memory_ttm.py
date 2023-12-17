import os
import json

exp_json = "exp_memory_ttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {

    "name":["exp0_ttm_mem64_dim64", "exp1_ttm_mem64_dim128", "exp2_ttm_mem64_dim192", "exp3_ttm_mem64_dim256", "exp4_ttm_mem64_dim320", "exp5_ttm_mem64_dim384",
            "exp6_ttm_mem128_dim64", "exp7_ttm_mem128_dim128", "exp8_ttm_mem128_dim192", "exp9_ttm_mem128_dim256", "exp10_ttm_mem128_dim320", "exp11_ttm_mem128_dim384",
            "exp12_ttm_mem192_dim64", "exp13_ttm_mem192_dim128", "exp14_ttm_mem192_dim192", "exp15_ttm_mem192_dim256", "exp16_ttm_mem192_dim320", "exp17_ttm_mem192_dim384",
            "exp18_ttm_mem256_dim64", "exp19_ttm_mem256_dim128", "exp20_ttm_mem256_dim192", "exp21_ttm_mem256_dim256", "exp22_ttm_mem256_dim320", "exp23_ttm_mem256_dim384",
            "exp24_ttm_mem320_dim64", "exp25_ttm_mem320_dim128", "exp26_ttm_mem320_dim192", "exp27_ttm_mem320_dim256", "exp28_ttm_mem320_dim320", "exp29_ttm_mem320_dim384",
            "exp30_ttm_mem384_dim64", "exp31_ttm_mem384_dim128", "exp32_ttm_mem384_dim192", "exp33_ttm_mem384_dim256", "exp34_ttm_mem384_dim320", "exp35_ttm_mem384_dim384"],

    "memory_tokens_size":[64, 64, 64, 64, 64, 64,
                          128, 128, 128, 128, 128, 128,
                          192, 192, 192, 192, 192, 192,
                          256, 256, 256, 256, 256, 256,
                          320, 320, 320, 320, 320, 320,
                          384, 384, 384, 384, 384, 384],

    "dim":[64, 128, 192, 256, 320, 384,
           64, 128, 192, 256, 320, 384,
           64, 128, 192, 256, 320, 384,
           64, 128, 192, 256, 320, 384,
           64, 128, 192, 256, 320, 384,
           64, 128, 192, 256, 320, 384],

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