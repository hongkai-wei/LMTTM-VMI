from torch.utils.tensorboard import SummaryWriter
import os
from config import Config

configs = Config.getInstance()

if os.path.exists(configs["log_dir"]) == False:#if log dir not exist, create it
    os.mkdir(configs["log_dir"])

class logger():
    def __init__(self, name) -> None:

        log_dir = os.path.join(configs["log_dir"], name)
        self.logger = SummaryWriter(log_dir=log_dir)

    def __call__(self):
        return self.logger
