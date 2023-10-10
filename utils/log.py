from torch.utils.tensorboard import SummaryWriter
import os

log = "./log"
if os.path.exists(log) == False:
    os.mkdir(log)


class logger():
    def __init__(self, name) -> None:
        log_dir = os.path.join(log, name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def get(self):
        return self.writer
