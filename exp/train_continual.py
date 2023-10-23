import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_data_iter import get_dataloader
from model.ttm_basic_network import TokenTuringMachineEncoder
from utils.log import logger
from config import Config
import torch
import tqdm
from utils.video_transforms import *

# get the config
config = Config.getInstance()
batch_size = config["batch_size"]
config = config["train"]
name = config["name"]
log_writer = logger(config["name"] + "_train")
log_writer = log_writer.get()

    