import os
import json

exp_json = "exp_memory_lmttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {

    "name":["exp0_lmttm1_mem32_dim32_block16", "exp1_lmttm1_mem32_dim64_block16", "exp2_lmttm1_mem32_dim96_block16", "exp3_lmttm1_mem32_dim128_block16", "exp4_lmttm1_mem32_dim160_block16", "exp5_lmttm1_mem32_dim192_block16",
            "exp6_lmttm1_mem64_dim32_block16", "exp7_lmttm1_mem64_dim64_block16", "exp8_lmttm1_mem64_dim96_block16", "exp9_lmttm1_mem64_dim128_block16", "exp10_lmttm1_mem64_dim160_block16", "exp11_lmttm1_mem64_dim192_block16",
            "exp12_lmttm1_mem96_dim32_block16", "exp13_lmttm1_mem96_dim64_block16", "exp14_lmttm1_mem96_dim96_block16", "exp15_lmttm1_mem96_dim128_block16", "exp16_lmttm1_mem96_dim160_block16", "exp17_lmttm1_mem96_dim192_block16",
            "exp18_lmttm1_mem128_dim32_block16", "exp19_lmttm1_mem128_dim64_block16", "exp20_lmttm1_mem128_dim96_block16", "exp21_lmttm1_mem128_dim128_block16", "exp22_lmttm1_mem128_dim160_block16", "exp23_lmttm1_mem128_dim192_block16",
            "exp24_lmttm1_mem160_dim32_block16", "exp25_lmttm1_mem160_dim64_block16", "exp26_lmttm1_mem160_dim96_block16", "exp27_lmttm1_mem160_dim128_block16", "exp28_lmttm1_mem160_dim160_block16", "exp29_lmttm1_mem160_dim192_block16",
            "exp30_lmttm1_mem192_dim32_block16", "exp31_lmttm1_mem192_dim64_block16", "exp32_lmttm1_mem192_dim96_block16", "exp33_lmttm1_mem192_dim128_block16", "exp34_lmttm1_mem192_dim160_block16", "exp35_lmttm1_mem192_dim192_block16"],

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

    "num_blocks":[16, 16, 16, 16, 16, 16,
                  16, 16, 16, 16, 16, 16,
                  16, 16, 16, 16, 16, 16,
                  16, 16, 16, 16, 16, 16,
                  16, 16, 16, 16, 16, 16,
                  16, 16, 16, 16, 16, 16]
}

if __name__ == "__main__":
    for i in range(len(train_config["name"])):
        with open(f'./config/{exp_json}', 'r') as file:
            data = json.load(file)

        data['train']['name'] = train_config["name"][i]
        data['model']['memory_tokens_size'] = train_config["memory_tokens_size"][i]
        data['model']['dim'] = train_config["dim"][i]
        data['model']['num_blocks'] = train_config["num_blocks"][i]

        with open(f'./config/{exp_json}', 'w') as file:
            json.dump(data, file, indent=4)
        
        run_exp(exp_json)