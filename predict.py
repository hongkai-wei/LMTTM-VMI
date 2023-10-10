from utils.get_data_iter import get_iter
from model.ttm_basic_network import TokenTuringMachineEncoder
from utils.log import logger
from config import Config
from utils.get_data_iter import get_iter
from einops.layers.torch import Rearrange
import torch
import tqdm

config = Config.getInstance()
batch_size = config["batch_size"]
log_writer = logger(config["test"]["name"])
log_writer = log_writer.get()

name = config["train"]["name"]

data_test = get_iter("test")

pth = ".\\check_point\\"
pth_files = [f"{pth}{name}_epoch_{i}.pth" for i in range(1, 10)] 

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
        out, mem = model(x)
        # out, mem = model(x, load_mem)
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