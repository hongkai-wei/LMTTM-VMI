import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
from utils.get_data_iter import get_dataloader
from model.ttm_basic_network import TokenTuringMachineEncoder
from utils.log import logger
from config import Config
import torch
import tqdm
from utils.video_transforms import *
import torch.nn as nn 
'''
1 无冻结模块 看mem无影响
2  有冻结模块 TL1
3  有冻结模块 TL2
4 有冻结模块 TL1+2


'''
config = Config.getInstance()
log_writer = logger(config['train']["name"] + "second_train")()
if not os.path.exists("./check_point"):
    os.mkdir("./check_point")
checkpoint_path = f"./check_point/{config['train']['name']}"
if os.path.exists(checkpoint_path):
    pass
else:
    os.mkdir(checkpoint_path)

transform_train = Compose([
    ShuffleTransforms(mode="CWH")
      ])
transform_val = Compose([
     ShuffleTransforms(mode="CWH")
])

data_train = get_dataloader("train",config=config ,download=False,transform=transform_train)
data_val = get_dataloader("val",config=config,download=False, transform=transform_val)
torch.manual_seed(0)
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def frozen_token_summary_layer(pth_file:str):
    pth_load = torch.load(pth_file)
    memory_tokens = pth_load["memory_tokens"]
    weight = pth_load["model"]
    new_dict = {}
    for i in weight.keys():
        if "tokenLearnerMHA1" in i or "tokenLearner1" in i:
            new_dict[i] = weight[i]
    
    return new_dict,memory_tokens

def train(pth:str):
    frozen_dict, memory_tokens = frozen_token_summary_layer(pth)
    model = TokenTuringMachineEncoder(config).cuda()
    model.apply(init_weights)
    model.load_state_dict(frozen_dict,strict=False) 
    

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
    for _ in epoch_bar:
        epoch_bar.set_description(
            f"train epoch is {format(_+1)} of {config['train']['epoch']}")
        bar = tqdm.tqdm(data_train, leave=False)
        losses = []
        for input, target in bar:
            input = input.to("cuda", dtype=torch.float32)  # B C T H W
            target = target.to("cuda", dtype=torch.long)  # B 1
            target = target.squeeze(1)  # B 1
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
            bar.set_postfix(loss=loss.item(), val_acc=val_acc)
            log_writer.add_scalar("loss per step", loss.item(), train_nums)
            if train_nums % config['train']["val_gap"] == 0:
                avg_loss = sum(losses)/len(losses)
                if avg_loss <= 0.2 and convergence_flag == -1:
                    convergence_batch = (train_nums * config["batch_size"])
                    convergence_flag = 1      
                log_writer.add_scalar("loss per 100 step", avg_loss, train_nums)
                losses = []
                result_all = 0 
                all_all = 0
                for val_x, val_y in data_val:
                    model.eval()
                    val_x = val_x.to("cuda", dtype=torch.float32)
                    val_y = val_y.to("cuda", dtype=torch.long)
                    if (config['train']["load_memory_tokens"]):
                        out, memory_tokens = model(val_x, memory_tokens)
                    else:
                        out, memory_tokens = model(val_x, memory_tokens = None)
                    out = torch.argmax(out, dim=1)
                    val_y = val_y.squeeze(1)
                    all = val_y.size(0)
                    result = (out == val_y).sum().item()
                    result_all += result
                    all_all += all
                val_acc = (result_all/all_all)*100
                result_all = 0 
                all_all = 0
                log_writer.add_scalar("val acc", val_acc, val_acc_nums)
                val_acc_nums += 1
        if _ >= (config['train']["epoch"]-50):
            save_name = f"./check_point/{config['train']['name']}/{config['train']['name']}_epoch_{_ -config['train']['epoch'] + 51}.pth"
            torch.save({"model": model.state_dict(), "memory_tokens": memory_tokens}, save_name)
        if _ >= (config['train']["epoch"]-50):
            save_loss.append(avg_loss)
    final_save_loss = sum(save_loss)/(len(save_loss))
    final_save_loss = round(final_save_loss, 2)
    print(f"train loss is {final_save_loss},and convergence batch is {convergence_batch}")

    if os.path.exists("./experiment"):
        pass
    else:
        os.mkdir("./experiment")
    experiment_path = "./experiment/experiment_record.txt"
    with open(experiment_path, "a") as file:
        print(f"{config['train']['name']} convergence_batch: {convergence_batch} , train_loss: {final_save_loss}", file=file)
if __name__ == "__main__":
    pth=r"your_corresponding_pth_file"
    train(pth)
