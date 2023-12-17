import os
import json

exp_json = "exp_memory_lmttm.json" 

def run_exp(exp_json): 
    os.system("python exp\\train.py " + exp_json)
    os.system("python exp\\predict.py " + exp_json)

train_config = {

    "name":["exp0_lmttm1_mem64_dim64_block4", "exp1_lmttm1_mem64_dim128_block4", "exp2_lmttm1_mem64_dim192_block4", "exp3_lmttm1_mem64_dim256_block4", "exp4_lmttm1_mem64_dim320_block4", "exp5_lmttm1_mem64_dim384_block4",
            "exp6_lmttm1_mem128_dim64_block4", "exp7_lmttm1_mem128_dim128_block4", "exp8_lmttm1_mem128_dim192_block4", "exp9_lmttm1_mem128_dim256_block4", "exp10_lmttm1_mem128_dim320_block4", "exp11_lmttm1_mem128_dim384_block4",
            "exp12_lmttm1_mem192_dim64_block4", "exp13_lmttm1_mem192_dim128_block4", "exp14_lmttm1_mem192_dim192_block4", "exp15_lmttm1_mem192_dim256_block4", "exp16_lmttm1_mem192_dim320_block4", "exp17_lmttm1_mem192_dim384_block4",
            "exp18_lmttm1_mem256_dim64_block4", "exp19_lmttm1_mem256_dim128_block4", "exp20_lmttm1_mem256_dim192_block4", "exp21_lmttm1_mem256_dim256_block4", "exp22_lmttm1_mem256_dim320_block4", "exp23_lmttm1_mem256_dim384_block4",
            "exp24_lmttm1_mem320_dim64_block4", "exp25_lmttm1_mem320_dim128_block4", "exp26_lmttm1_mem320_dim192_block4", "exp27_lmttm1_mem320_dim256_block4", "exp28_lmttm1_mem320_dim320_block4", "exp29_lmttm1_mem320_dim384_block4",
            "exp30_lmttm1_mem384_dim64_block4", "exp31_lmttm1_mem384_dim128_block4", "exp32_lmttm1_mem384_dim192_block4", "exp33_lmttm1_mem384_dim256_block4", "exp34_lmttm1_mem384_dim320_block4", "exp35_lmttm1_mem384_dim384_block4"],    "memory_tokens_size":[64, 64, 64, 64, 64, 64,
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

    "num_blocks":[4, 4, 4, 4, 4, 4,
                  4, 4, 4, 4, 4, 4,
                  4, 4, 4, 4, 4, 4,
                  4, 4, 4, 4, 4, 4,
                  4, 4, 4, 4, 4, 4,
                  4, 4, 4, 4, 4, 4]
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