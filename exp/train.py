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
json_path = sys.argv[1]
# json_path = "exp_noise.json"
config = Config.getInstance(json_path)
if config["model"]["model"] == "ttm":
    from model.TTM import TokenTuringMachineEncoder
elif config["model"]["model"] == "lmttm":
    from model.LMTTM import TokenTuringMachineEncoder
elif config["model"]["model"] == "lmttm_v2":
    from model.LMTTMv2 import TokenTuringMachineEncoder

log_writer = logger(config['train']["name"] + "_train")()
if not os.path.exists("./check_point"):
    os.mkdir("./check_point")
checkpoint_path = f"./check_point/{config['train']['name']}"
if os.path.exists(checkpoint_path):
    pass
else:
    os.mkdir(checkpoint_path)

data_loader = get_dataloader("train", config=config, download=False, transform=None)
val_loader = get_dataloader("val", config=config, download=False, transform=None)


torch.manual_seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train():
    memory_tokens = None
    model = TokenTuringMachineEncoder(config).cuda() 
    out_name = f'{config["model"]["model"]:^{10}}'  
    print("-"*35,out_name,"Model Info","-"*35)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters,"\n","-"*90)
    model.apply(init_weights)##init weight
    if config['train']["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=config['train']["lr"], weight_decay=config['train']["weight_decay"])
    elif config['train']["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['train']["lr"], weight_decay=config['train']["weight_decay"])
    citizer = torch.nn.CrossEntropyLoss()
    epoch_bar = tqdm.tqdm(range(config['train']["epoch"]))
    train_nums = 0
    val_acc_nums = 0
    val_acc = 0
    save_loss = []
    convergence_batch = -1
    convergence_flag = -1
    avg_loss = 0
    reals_out  =  0
    reals_all = 0
    convergence_batch2 = -1
    acc_lis=[]
    for _ in epoch_bar:
        epoch_bar.set_description(
            f"train epoch is {format(_+1)} of {config['train']['epoch']}")
        bar = tqdm.tqdm(data_loader, leave=False)
        losses = []
        time_ = 0 
        for input, target in bar:
            time1 = time.time()
            input = input.to("cuda", dtype=torch.float32)  # B C T H W
            # input = input.transpose(1,2)# for medmnist ,if the input format is  B,T,C,H,W,please delete this lin
            target = target.to("cuda", dtype=torch.long)  # B w

            if config["dataset_name"] == "organmnist3d":
                target = target.squeeze(1)

            model.train()
            if (config['train']["load_memory_tokens"]):
                output, memory_tokens = model(input, memory_tokens)
            else:
                output, memory_tokens = model(input, memory_tokens = None)
            train_nums += 1
            loss = citizer(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            bar.set_postfix(loss=loss.item(), val_acc=val_acc,batch_time=time_)
            log_writer.add_scalar("loss per step", loss.item(), train_nums)
            time2 = time.time()
            time_ = time2-time1
            if train_nums % config['train']["val_gap"] == 0:
                avg_loss = sum(losses)/len(losses)          
                if avg_loss <= 0.2 and convergence_flag == -1:
                    convergence_batch = (train_nums * config["batch_size"])
                    convergence_batch2 = _ + 1
                    convergence_flag = 1
                log_writer.add_scalar("loss per 100 step", avg_loss, train_nums)
                losses = []
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        model.eval()
                        val_x = val_x.to("cuda", dtype=torch.float32)
                        val_y = val_y.to("cuda", dtype=torch.long)

                        if config["dataset_name"] == "organmnist3d":
                            val_y = val_y.squeeze(1)

                        if (config['train']["load_memory_tokens"]):
                            out, memory_tokens = model(val_x, memory_tokens)
                        else:
                            out, memory_tokens = model(val_x, memory_tokens = None)
                        out = torch.argmax(out, dim=1)

                        all = val_y.size(0)
                        result = (out == val_y).sum().item()
                        reals_out += result
                        reals_all += all
                    val_acc = (reals_out/reals_all)*100
                    log_writer.add_scalar("val acc", val_acc, val_acc_nums)
                    val_acc_nums += 1 
        if _ >= (config['train']["epoch"]-10):
            save_name = f"./check_point/{config['train']['name']}/{config['train']['name']}_epoch_{_ -config['train']['epoch'] + 11}.pth"
            torch.save({"model": model.state_dict(), "memory_tokens": memory_tokens}, save_name)
        if _ >= (config['train']["epoch"]-10):
            save_loss.append(avg_loss)
            acc_lis.append(val_acc)
    final_save_loss = sum(save_loss)/(len(save_loss))
    final_save_loss = round(final_save_loss, 2)
    out_acc=sum(acc_lis)/len(acc_lis)
    out_accs=round(out_acc,4)
    print(f"train loss is {final_save_loss},and convergence batch is {convergence_batch},and convergence _epoch is {convergence_batch2},acc is{out_accs}")

    if os.path.exists("./experiment"):
        pass
    else:
        os.mkdir("./experiment")
    experiment_path = "./experiment/experiment.txt"
    with open(experiment_path, "a") as file:
        print(f"{config['train']['name']} convergence_batch: {convergence_batch} , train_loss: {final_save_loss},and sL_batch={convergence_batch2},acc is{out_accs}", file=file)
if __name__ == "__main__":
    time_1 = time.time()
    train()
    time_2 = time.time()
    print("All Epoch Train Time Is ",time_2-time_1)