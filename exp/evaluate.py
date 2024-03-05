import numpy as np 
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO # Evaluator
import torch
# from sklearn.metrics import roc_curve, auc
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.get_data_iter import get_dataloader
import time
from utils.log import logger
from config import Config
import tqdm
import torchvision.transforms as transforms
import torch.nn as nn 
import os
from utils.video_transforms import *
from torch.utils.data import Dataset,DataLoader

json_path = sys.argv[1]
# json_path = "base.json"
config = Config.getInstance(json_path)
os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["gpu"]

if config["model"]["model"] == "ttm":
    from model.TTM import TokenTuringMachineEncoder
elif config["model"]["model"] == "lmttm":
    from model.LMTTM import TokenTuringMachineEncoder

transform_test = Compose([
    ShuffleTransforms(mode="CWH")
])

test_loader = get_dataloader("test", config=config, download=False, transform=None)
test_evaluator = medmnist.Evaluator(config["dataset_name"], 'test',config["root"])

log_writer = logger(config["train"]["name"] + "_evaluate")()

def test(model, evaluator, data_loader, criterion, device, run, save_folder=None,memory_tokens=None):
        model.eval()
        total_loss = []
        y_score = torch.tensor([]).to(device)
        y_truth=torch.tensor([]).to(device)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inoput = inputs.to(device, dtype=torch.float32)
                outputs,memory_tokens = model(inoput,memory_tokens)
                out = torch.argmax(outputs, dim=1)
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.long().resize_(len(targets), 1)

                total_loss.append(loss.item())

                y_score = torch.cat((y_score, outputs), 0)
                y_truth = torch.cat((y_truth, targets), 0)
            y_truth = y_truth.detach().cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            # # truthss =( y_truth == (torch.argmax(y_score, dim=1) ))
            # print(truthss/y_truth.shape[0])
            # print(y_truth.shape)
            auc, acc = evaluator.evaluate(y_score = y_score, kx = y_truth, save_folder = None, run = None)  #You have to change the remote code here.
            # auc, acc = evaluator.evaluate(y_score, save_folder, run)
            test_loss = sum(total_loss) / len(total_loss)

            return [test_loss, auc, acc]

if __name__ == "__main__":
    avg_auc = 0
    avg_acc = 0

    pth = f".\\check_point\\{config['train']['name']}\\"
    pth_files = [f"{pth}{config['train']['name']}_epoch_{i}.pth" for i in range(1, 21)] 
    for i in tqdm.tqdm(range(len(pth_files)),leave=True):
        checkpoint = torch.load(pth_files[i])
        load_state = checkpoint["model"]
        load_memory_tokens = checkpoint["memory_tokens"]
        memory_tokens = load_memory_tokens
        model = TokenTuringMachineEncoder(config).cuda()
        model.load_state_dict(load_state)
        criterion = nn.CrossEntropyLoss()
        evaluate_loss, evaluate_auc, evaluate_acc = test(model, test_evaluator, test_loader, criterion, "cuda", "run", save_folder = None, memory_tokens = memory_tokens)
        
        avg_auc += evaluate_auc
        avg_acc += evaluate_acc

        evaluate_auc = round(evaluate_auc, 3)
        evaluate_acc = round(evaluate_acc, 3)
        
        log_writer.add_scalar("loss per pth", evaluate_loss, i)
        log_writer.add_scalar("auc per pth", evaluate_auc, i)
        log_writer.add_scalar("acc per pth", evaluate_acc, i)

        if os.path.exists("./experiment"):
            pass
        else:
            os.mkdir("./experiment")
        experiment_path = f".\\experiment\\" + config["dataset_name"] + "_exp.txt"
        with open(experiment_path, "a") as file:
            # Redirecting data from print to file
            print(f"{config['train']['name']} pth{i} evaluate_auc: {evaluate_auc}", file=file)
        with open(experiment_path, "a") as file:
            # Redirecting data from print to file
            print(f"{config['train']['name']} pth{i} evaluate_acc: {evaluate_acc}", file=file)

    avg_auc = round(avg_auc/i, 3)
    avg_acc = round(avg_acc/i, 3)
    with open(experiment_path, "a") as file:
        # Redirecting data from print to file
        print(f"{config['train']['name']} avg_auc: {avg_auc}", file=file)
    print(f"{config['train']['name']} avg_auc: {avg_auc}")

    with open(experiment_path, "a") as file:
        # Redirecting data from print to file
        print(f"{config['train']['name']} avg_acc: {avg_acc}", file=file)
    print(f"{config['train']['name']} avg_acc: {avg_acc}")

    with open(experiment_path, "a") as file:
        print(" ", file=file)

    log_writer.close()