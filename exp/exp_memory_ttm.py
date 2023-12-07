import os
import json

exp_json = "exp_memory_ttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {

    "name":["exp0_ttm_mem32_dim32", "exp1_ttm_mem32_dim64", "exp2_ttm_mem32_dim96", "exp3_ttm_mem32_dim128", "exp4_ttm_mem32_dim160", "exp5_ttm_mem32_dim192",
            "exp6_ttm_mem64_dim32", "exp7_ttm_mem64_dim64", "exp8_ttm_mem64_dim96", "exp9_ttm_mem64_dim128", "exp10_ttm_mem64_dim160", "exp11_ttm_mem64_dim192",
            "exp12_ttm_mem96_dim32", "exp13_ttm_mem96_dim64", "exp14_ttm_mem96_dim96", "exp15_ttm_mem96_dim128", "exp16_ttm_mem96_dim160", "exp17_ttm_mem96_dim192",
            "exp18_ttm_mem128_dim32", "exp19_ttm_mem128_dim64", "exp20_ttm_mem128_dim96", "exp21_ttm_mem128_dim128", "exp22_ttm_mem128_dim160", "exp23_ttm_mem128_dim192",
            "exp24_ttm_mem160_dim32", "exp25_ttm_mem160_dim64", "exp26_ttm_mem160_dim96", "exp27_ttm_mem160_dim128", "exp28_ttm_mem160_dim160", "exp29_ttm_mem160_dim192",
            "exp30_ttm_mem192_dim32", "exp31_ttm_mem192_dim64", "exp32_ttm_mem192_dim96", "exp33_ttm_mem192_dim128", "exp34_ttm_mem192_dim160", "exp35_ttm_mem192_dim192"],

    "memory_tokens_size":[32, 32, 32, 32, 32, 32,
                          64, 64, 64, 64, 64, 64,
                          96, 96, 96, 96, 96, 96,
                          128, 128, 128, 128, 128, 128,
                          160, 160, 160, 160, 160, 160,
                          192, 192, 192, 192, 192, 192],

    "dim":[32, 64, 96, 128, 160, 192,
           32, 64, 96, 128, 160, 192,
           32, 64, 96, 128, 160, 192,
           32, 64, 96, 128, 160, 192,
           32, 64, 96, 128, 160, 192,
           32, 64, 96, 128, 160, 192],

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