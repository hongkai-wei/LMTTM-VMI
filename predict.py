import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.get_data_iter import get_dataloader
import time
from utils.log import logger
from config import Config
import torch
import tqdm
import torchvision.transforms as transforms
import torch.nn as nn 
import os
from torch.utils.data import Dataset,DataLoader
import sys

# json_path = sys.argv[1]
json_path = "base.json"
config = Config.getInstance(json_path)
if config["model"]["model"] == "ttm":
    from model.TTM import TokenTuringMachineEncoder
elif config["model"]["model"] == "lmttm":
    from model.LMTTM import TokenTuringMachineEncoder
elif config["model"]["model"] == "lmttmV2":
    from model.LMTTMv2 import TokenTuringMachineEncoder

log_writer = logger(config["train"]["name"] + "_test")()
test_loader = get_dataloader("test", config=config, download=False, transform=None)
pth = f".\\check_point\\{config['train']['name']}\\"
pth_files = [f"{pth}{config['train']['name']}_epoch_{i}.pth" for i in range(1, 11)] 

def predict():
    avg_acc = 0
    for i in tqdm.tqdm(range(len(pth_files)),leave=True):
        checkpoint = torch.load(pth_files[i])
        load_state = checkpoint["model"]
        load_memory_tokens = checkpoint["memory_tokens"]
        memory_tokens = load_memory_tokens
        model = TokenTuringMachineEncoder(config).cuda()
        model.eval()
        model.load_state_dict(load_state)
        all_y = 0
        all_real = 0
        for x,y in tqdm.tqdm(test_loader,leave=False):
            x = x.to("cuda", dtype = torch.float32)
            y = y.to("cuda", dtype = torch.long)
            if config["train"]["load_memory_tokens"]:
                out, memory_tokens = model(x, memory_tokens)
            else:
                out, memory_tokens = model(x, memory_tokens = None)

            out = torch.argmax(out, dim=1)
            if config["dataset_name"] == "organmnist3d":
                y = y.squeeze(1)
            all = y.size(0)
            result = (out == y).sum().item()

            all_y += all

            all_real += result
            ###   B,C,STEP,H,W
        print("\n Total sample size:",all_y,"Predicting the right amount:",all_real)
        print("acc is {}%".format((all_real/all_y)*100))

        acc = (all_real/all_y)*100
        log_writer.add_scalar("acc per num weight ", acc, i)
        avg_acc += acc
        all_real = 0
        all_y = 0

    test_acc = avg_acc/(i+1)
    test_acc = round(test_acc, 1)

    print("test acc is {}%".format(test_acc))

    if os.path.exists("./experiment"):
        pass
    else:
        os.mkdir("./experiment")
    experiment_path = "./experiment/experiment.txt"
    with open(experiment_path, "a") as file:
        # Redirecting data from print to file
        print(f"{config['train']['name']} test_acc: {test_acc}%", file=file)
        print(" ", file=file)

    log_writer.close()

if __name__ == "__main__":
    predict()