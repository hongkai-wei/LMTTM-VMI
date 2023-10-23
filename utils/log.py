from torch.utils.tensorboard import SummaryWriter
import os
from config import Config

config = Config.getInstance()

########################################### for use tensorboard log, if upload this code, please delete this block code
if os.path.exists(config["log_dir"]) == False:#if log dir not exist, create it
    os.mkdir(config["log_dir"])
###########################################

class logger():
    def __init__(self, name) -> None:

        log_dir = os.path.join(config["log_dir"], name)
        self.logger = SummaryWriter(log_dir=log_dir)

    def __call__(self):
        return self.logger
