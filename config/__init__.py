import os
from .configure import Config


config_file = "base.json"
config_file = os.path.join(os.path.dirname(__file__), config_file)

Config(config_file)
