from model.mem_basic_network import ttm
from utils.log import logger
from config import Config
configs = Config.getInstance()["train"]
log_writer=logger(configs["name"])

#data
data=zip(1,2)

