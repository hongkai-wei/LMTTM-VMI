import os
import json

exp_json = "exp_memory_lmttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {

    "name":["exp0_lmttm2_mem32_dim32_block32", "exp1_lmttm2_mem32_dim64_block32", "exp2_lmttm2_mem32_dim96_block32", "exp3_lmttm2_mem32_dim128_block32", "exp4_lmttm2_mem32_dim160_block32", "exp5_lmttm2_mem32_dim192_block32",
            "exp6_lmttm2_mem64_dim32_block64", "exp7_lmttm2_mem64_dim64_block64", "exp8_lmttm2_mem64_dim96_block64", "exp9_lmttm2_mem64_dim128_block64", "exp10_lmttm2_mem64_dim160_block64", "exp11_lmttm2_mem64_dim192_block64",
            "exp12_lmttm2_mem96_dim32_block96", "exp13_lmttm2_mem96_dim64_block96", "exp14_lmttm2_mem96_dim96_block96", "exp15_lmttm2_mem96_dim128_block96", "exp16_lmttm2_mem96_dim160_block96", "exp17_lmttm2_mem96_dim192_block96",
            "exp18_lmttm2_mem128_dim32_block128", "exp19_lmttm2_mem128_dim64_block128", "exp20_lmttm2_mem128_dim96_block128", "exp21_lmttm2_mem128_dim128_block128", "exp22_lmttm2_mem128_dim160_block128", "exp23_lmttm2_mem128_dim192_block128",
            "exp24_lmttm2_mem160_dim32_block160", "exp25_lmttm2_mem160_dim64_block160", "exp26_lmttm2_mem160_dim96_block160", "exp27_lmttm2_mem160_dim128_block160", "exp28_lmttm2_mem160_dim160_block160", "exp29_lmttm2_mem160_dim192_block160",
            "exp30_lmttm2_mem192_dim32_block192", "exp31_lmttm2_mem192_dim64_block192", "exp32_lmttm2_mem192_dim96_block192", "exp33_lmttm2_mem192_dim128_block192", "exp34_lmttm2_mem192_dim160_block192", "exp35_lmttm2_mem192_dim192_block192"],

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

    "num_blocks":[32, 32, 32, 32, 32, 32,
                  64, 64, 64, 64, 64, 64,
                  96, 96, 96, 96, 96, 96,
                  128, 128, 128, 128, 128, 128,
                  160, 160, 160, 160, 160, 160,
                  192, 192, 192, 192, 192, 192]
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