import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_data_iter import get_dataloader
from model.ttm_basic_network import TokenTuringMachineEncoder
from utils.log import logger
from config import Config
from utils.get_data_iter import get_dataloader
from einops.layers.torch import Rearrange
import torch
import tqdm
import os

config = Config.getInstance("best_dim.json")
batch_size = config["batch_size"]
config = config["train"]
log_writer = logger(config["name"] + "_test")()
name = config["name"]
data_test = get_dataloader("test")

pth = f".\\check_point\\{name}\\"
pth_files = [f"{pth}{name}_epoch_{i}.pth" for i in range(1, 51)] 

avg_acc = 0

for i in range(len(pth_files)):
    checkpoint = torch.load(pth_files[i])
    load_state = checkpoint["model"]
    load_memory_tokens = checkpoint["memory_tokens"]
    model = TokenTuringMachineEncoder().cuda()
    model.load_state_dict(load_state)

    all_y = 0
    all_real = 0

    for x,y in tqdm.tqdm(data_test):
        x = x.to("cuda", dtype = torch.float32)
        y = y.to("cuda", dtype = torch.long)

        if (config["load_memory_tokens"] == "True"):
            out, memory_tokens = model(x, load_memory_tokens)
        else:
            out, memory_tokens = model(x, memory_tokens = None)

        out = torch.argmax(out, dim=1)
        y = y.squeeze(1)
        all = y.size(0)
        result = (out == y).sum().item()

        all_y += all

        all_real += result
         ###   B,C,STEP,H,W
        print("总样本数：",all_y,"预测对的数目:",all_real)
        print("acc is {}%".format((all_real/all_y)*100))
        acc = (all_real/all_y)*100
        log_writer.add_scalar("acc per 100 step ", acc, i)
        
    avg_acc += acc

test_acc = avg_acc/(i+1)
test_acc = round(test_acc, 1)

print("test acc is {}%".format(test_acc))

if os.path.exists("./experiment"):
    pass
else:
    os.mkdir("./experiment")

experiment_path = "./experiment/experiment_record.txt"

# 打开文件，以追加模式写入数据
with open(experiment_path, "a") as file:
    # 将print的数据重定向到文件中
    print(f"{name} test_acc: {test_acc}%", file=file)
    print(" ", file=file)

log_writer.close()