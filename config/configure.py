import json
import os

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
                
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
                
def dict_to_object_recursive(input_dict):
    return DictToObject(input_dict)

# create singleton config to load the json configure file
class Config:
    __instance = None
    __config = None
    config_file = None

    @staticmethod
    def getInstance(config_file="base.json"):
        if Config.__instance == None or Config.config_file != config_file:
            Config(config_file)
        return Config.__instance.__config

    def __init__(self, config_file="base.json"):
        if Config.__instance != None and Config.config_file == config_file:
            raise Exception("This class is a singleton!")
        else:
            Config.config_file = config_file
            Config.__instance = self
            self.__loadConfig(os.path.join(os.path.dirname(__file__), "base.json"))#__file__在Python里面是指当前文件的文件名
            if config_file != "base.json":
                self.__loadConfig(os.path.join(os.path.dirname(__file__), config_file))
            
            self.__config = dict_to_object_recursive(self._Config__config)


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



if __name__=="__main__":
    config_file = "base.json"
    config_file = os.path.join(os.path.dirname(__file__), config_file)

    Config(config_file)
    shili=Config.getInstance()
    pass
    pass