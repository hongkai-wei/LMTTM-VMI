from torch.utils.tensorboard import SummaryWriter
import os

log_dir = "../log"
if os.path.exists(log_dir) == False:
    os.mkdir(log_dir)


class logger():
    def __init__(self, name) -> None:
        log_dir_ = os.path.join(log_dir, name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def get(self):
        return self.writer
