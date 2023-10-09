from model.ttm_basic_network import ttm
from utils.log import logger
from config import Config
from model.ttm_basic_network import ttm
import torch
import tqdm
import os
configs = Config.getInstance()["train"]
log_writer = logger(configs["name"])
if os.path.exists("../check_point"):
    pass
else:
    os.mkdir("../check_point")

# data
data = zip(1, 2)
####待修改
model = ttm().cuda()
if configs["optimizer"] != "Adam":
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"])
else:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"])

citizer = torch.nn.CrossEntropyLoss()
epoch_bar = tqdm.tqdm(range(configs["epoch"]))
train_nums = 0
val_acc_nums = 0
for _ in epoch_bar:
    epoch_bar.set_description(
        f"train epoch is {format(_+1)} of {configs['epoch']}")
    bar = tqdm.tqdm(data)
    losses = []
    for input, target in bar:
        model.train()
        input = input.to("cuda", dtype=torch.float32)
        target = target.to("cuda", dtype=torch.long)
        if (configs["if_mem"] == "True"):
            output, mem = model(input, mem)
        else:
            output, mem = model(input)
        train_nums += 1
        loss = citizer(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        bar.set_postfix(loss=loss.item())
        log_writer.add_scalar("loss per step", loss.item(), train_nums)
        if train_nums % configs["val_gap"] == 0:
            avg_loss = sum(losses)/len(losses)
            log_writer.add_scalar("loss per 100 step", avg_loss, train_nums)
            if configs["save_check_point_only_once"] == "True":
                if avg_loss <= 0.042:
                    torch.save({"model": model.state_dict(), "mem": mem},
                               f"../check_point/{configs['name']}.pth")
            for val_input, val_target in data:
                model.eval()
                val_input = val_input.to("cuda", dtype=torch.float32)
                val_target = val_target.to("cuda", dtype=torch.long)
                if (configs["if_mem"] == "True"):
                    output, _mem = model(input, mem)
                else:
                    output, _mem = model(input)
                out = torch.argmax(output, dim=1)
                val_target = val_target.squeeze(1)
                all_nums = val_target.size(0)
                result = (out == val_target).sum().item()
                acc = (result/all_nums)
                log_writer.add_scalar("acc per {} step ".format(
                    configs["val_gap"]), acc, val_acc_nums)
                val_acc_nums += 1
