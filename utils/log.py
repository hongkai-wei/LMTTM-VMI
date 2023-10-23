from torch.utils.tensorboard import SummaryWriter
import os
from config import Config
configs = Config.getInstance()
log =configs["log_dir"]
if os.path.exists(log) == False:#if log dir not exist, create it
    os.mkdir(log)

class logger():
    def __init__(self, name) -> None:
        log_dir = os.path.join(log, name)
        self.logger = SummaryWriter(log_dir=log_dir)

    def __call__(self):
        return self.logger
