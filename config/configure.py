import json
import os

# create singleton config to load the json configure file
class Config:
    __instance = None
    __config = None

    @staticmethod
    def getInstance():
        if Config.__instance == None:
            Config()
        return Config.__instance

    def __init__(self, config_file="base.json"):
        if Config.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Config.__instance = self
            self.__loadConfig(os.path.join(os.path.dirname(__file__), "base.json"))
            if config_file != "base.json":
                self.__loadConfig(os.path.join(os.path.dirname(__file__), config_file))

    def __loadConfig(self, config_file):
        assert os.path.exists(config_file), "The configure file does not exist!"
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.__config == None:
            self.__config = config
        else:
            self.__update(self.__config, config)
            
    def __update(self, config, config_file):
        for key in config_file:
            if key in config:
                if isinstance(config[key], dict):
                    self.__update(config[key], config_file[key])
                else:
                    config[key] = config_file[key]
            else:
                config[key] = config_file[key]

    def get(self, key):
        return self.__config[key]
    
    def __getitem__(self, key):
        return self.__config[key]

    def set(self, key, value):
        self.__config[key] = value

    def __str__(self):
        return str(self.__config)
    