import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.get_data_iter import get_dataloader
from model.TTM import TokenTuringMachineEncoder
from utils.log import logger
from config import Config
import torch
import tqdm
import torchvision.transforms as transforms
import torch.nn as nn 
import os
from torch.utils.data import Dataset,DataLoader
import sys
config = Config.getInstance()
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

    # pth = f".\\check_point\\exp0_memory8_and_dim32_and_numTokens8\\exp0_memory8_and_dim32_and_numTokens8_epoch_5.pth"
    # checkpoint = torch.load(pth)
    # load_memory_tokens = checkpoint["memory_tokens"]
    # memory_tokens = load_memory_tokens
    memory_tokens = None
    model = TokenTuringMachineEncoder(config).cuda()
    model.apply(init_weights)
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
        for input, target in bar:
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
                    convergence_batch2 = _ + 1
                    convergence_flag = 1
            
                log_writer.add_scalar("loss per 100 step", avg_loss, train_nums)
                losses = []
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        model.eval()
                        val_x = val_x.to("cuda", dtype=torch.float32)
                        val_y = val_y.to("cuda", dtype=torch.long)
                        if (config['train']["load_memory_tokens"]):
                            out, memory_tokens = model(val_x, memory_tokens)
                        else:
                            out, memory_tokens = model(val_x, memory_tokens = None)
                        out = torch.argmax(out, dim=1)
                        # val_y = val_y.squeeze(1)
                        all = val_y.size(0)
                        result = (out == val_y).sum().item()
                    reals_out += result
                    reals_all += all
                    val_acc = (reals_out/reals_all)*100
                    log_writer.add_scalar("val acc", val_acc, val_acc_nums)
                    val_acc_nums += 1
            # Save the model for the next 50 epochs

        if _ >= (config['train']["epoch"]-50):
            save_name = f"./check_point/{config['train']['name']}/{config['train']['name']}_epoch_{_ -config['train']['epoch'] + 6}.pth"
            torch.save({"model": model.state_dict(), "memory_tokens": memory_tokens}, save_name)


        if _ >= (config['train']["epoch"]-50):
            save_loss.append(avg_loss)
            acc_lis.append(val_acc)

    final_save_loss = sum(save_loss)/(len(save_loss))
    final_save_loss = round(final_save_loss, 2)
    out_acc=sum(acc_lis)/len(acc_lis)
    out_accs=round(out_acc,4)

    print(f"train loss is {final_save_loss},and convergence batch is {convergence_batch},and sl_batch is {convergence_batch2},acc is{out_accs}")

    if os.path.exists("./experiment"):
        pass
    else:
        os.mkdir("./experiment")

    experiment_path = "./experiment/experiment.txt"

    # Open a file and write data in append mode
    with open(experiment_path, "a") as file:
        # Redirecting data from print to file
        print(f"{config['train']['name']} convergence_batch: {convergence_batch} , train_loss: {final_save_loss},and sL_batch={convergence_batch2},acc is{out_accs}", file=file)

if __name__ == "__main__":
    train()