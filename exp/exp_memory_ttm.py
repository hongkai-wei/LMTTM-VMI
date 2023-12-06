import os
import json

exp_json = "exp_memory_ttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {

    "name":["exp0_lmttm_mem32_dim32_block16", "exp1_lmttm_mem32_dim64_block16", "exp2_lmttm_mem32_dim96_block16", "exp3_lmttm_mem32_dim128_block16", "exp4_lmttm_mem32_dim160_block16", "exp5_lmttm_mem32_dim192_block16",
            "exp6_lmttm_mem64_dim32_block16", "exp7_lmttm_mem64_dim64_block16", "exp8_lmttm_mem64_dim96_block16", "exp9_lmttm_mem64_dim128_block16", "exp10_lmttm_mem64_dim160_block16", "exp11_lmttm_mem64_dim192_block16",
            "exp12_lmttm_mem96_dim32_block16", "exp13_lmttm_mem96_dim64_block16", "exp14_lmttm_mem96_dim96_block16", "exp15_lmttm_mem96_dim128_block16", "exp16_lmttm_mem96_dim160_block16", "exp17_lmttm_mem96_dim192_block16",
            "exp18_lmttm_mem128_dim32_block16", "exp19_lmttm_mem128_dim64_block16", "exp20_lmttm_mem128_dim96_block16", "exp21_lmttm_mem128_dim128_block16", "exp22_lmttm_mem128_dim160_block16", "exp23_lmttm_mem128_dim192_block16",
            "exp24_lmttm_mem160_dim32_block16", "exp25_lmttm_mem160_dim64_block16", "exp26_lmttm_mem160_dim96_block16", "exp27_lmttm_mem160_dim128_block16", "exp28_lmttm_mem160_dim160_block16", "exp29_lmttm_mem160_dim192_block16",
            "exp30_lmttm_mem192_dim32_block16", "exp31_lmttm_mem192_dim64_block16", "exp32_lmttm_mem192_dim96_block16", "exp33_lmttm_mem192_dim128_block16", "exp34_lmttm_mem192_dim160_block16", "exp35_lmttm_mem192_dim192_block16"],

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