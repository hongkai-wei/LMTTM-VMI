from utils.get_data_iter import get_dataloader
from model.ttm_basic_network import TokenTuringMachineEncoder
from utils.log import logger
from config import Config
import torch
import tqdm
import os

from utils.video_transforms import Compose, CutMix

config = Config.getInstance()
batch_size = config["batch_size"]
config = config["train"]
name = config["name"]
log_writer = logger(config["name"] + "_train")
log_writer = log_writer.get()

if os.path.exists("./check_point"):
    pass
else:
    os.mkdir("./check_point")

checkpoint_path = f"./check_point/{config['name']}"
if os.path.exists(checkpoint_path):
    pass
else:
    os.mkdir(checkpoint_path)

transform_train = None
transform_val = None
data_train = get_dataloader("train", download=True, transform=transform_train)
data_val = get_dataloader("val", download=transform_val)

memory_tokens = None
model = TokenTuringMachineEncoder().cuda()

if config["optimizer"] == "RMSprop":
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
elif config["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

citizer = torch.nn.CrossEntropyLoss()
epoch_bar = tqdm.tqdm(range(config["epoch"]))
train_nums = 0
val_acc_nums = 0
val_acc = 0
save_loss = []

convergence_batch = -1
convergence_flag = -1
avg_loss = 0

model.train()

for _ in epoch_bar:
    epoch_bar.set_description(
        f"train epoch is {format(_+1)} of {config['epoch']}")
    bar = tqdm.tqdm(data_train, leave=False)
    losses = []
    for input, target in bar:
        input = input.to("cuda", dtype=torch.float32)  # B C T H W
        target = target.to("cuda", dtype=torch.long)  # B 1
        target = target.squeeze(1)  # B 1
        if (config["load_memory_tokens"] == "True"):
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

        if train_nums % config["val_gap"] == 0:
            avg_loss = sum(losses)/len(losses)

            if avg_loss <= 0.1 and convergence_flag == -1:
                convergence_batch = (train_nums * batch_size)
                convergence_flag = 1
            
            log_writer.add_scalar("loss per 100 step", avg_loss, train_nums)
            losses = []

            for val_x, val_y in data_val:
                model.eval()
                val_x = val_x.to("cuda", dtype=torch.float32)
                val_y = val_y.to("cuda", dtype=torch.long)
                out, memory_tokens = model(val_x, memory_tokens)
                out = torch.argmax(out, dim=1)
                val_y = val_y.squeeze(1)
                all = val_y.size(0)
                result = (out == val_y).sum().item()
                val_acc = (result/all)*100
                log_writer.add_scalar("val acc", val_acc, val_acc_nums)
                val_acc_nums += 1
        # 保存后面50个epoch的模型

    if _ >= (config["epoch"]-50):
        save_name = f"./check_point/{config['name']}/{config['name']}_epoch_{_ -config['epoch'] + 51}.pth"
        torch.save({"model": model.state_dict(), "memory_tokens": memory_tokens}, save_name)


    if _ >= (config["epoch"]-50):
        save_loss.append(avg_loss)

final_save_loss = sum(save_loss)/(len(save_loss))
final_save_loss = round(final_save_loss, 2)
print(f"train loss is {final_save_loss},and convergence batch is {convergence_batch}")

if os.path.exists("./experiment"):
    pass
else:
    os.mkdir("./experiment")

experiment_path = "./experiment/experiment_record.txt"

# 打开文件，以追加模式写入数据
with open(experiment_path, "a") as file:
    # 将print的数据重定向到文件中
    print(f"{name} convergence_batch: {convergence_batch} , train_loss: {final_save_loss}", file=file)

