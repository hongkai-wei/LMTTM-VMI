import json
import os

# create singleton config to load the json configure file
class Config:
    __instance = None
    __config = None
    config_file = None

    @staticmethod
    def getInstance(config_file="base.json"):
        if Config.__instance == None or Config.config_file != config_file:
            Config(config_file)
        return Config.__instance

    def __init__(self, config_file="base.json"):
        if Config.__instance != None and Config.config_file == config_file:
            raise Exception("This class is a singleton!")
        else:
            Config.config_file = config_file
            Config.__instance = self
            self.__loadConfig(os.path.join(os.path.dirname(__file__), "base.json"))#__file__在Python里面是指当前文件的文件名
            if config_file != "base.json":
                self.__loadConfig(os.path.join(os.path.dirname(__file__), config_file))
            
            self.__convert_key()
            
    
    def __convert_key(self, key, value):
        if isinstance(value, dict):
            for sub_key in value:
                self.__convert_key(key + "." + sub_key, value[sub_key])
        else:
            self.set(key, value)

    def convert_key(self):
        for key in self.__config:
            self.__convert_key(key, self.__config[key])

        
         

    def __loadConfig(self, config_file):
        assert os.path.exists(config_file), "The configure file does not exist!"
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.__config == None:
            self.__config = config
        else:
            self.__update(self.__config, config)
            
    def __update(self, config, config_file):#参数是两个字典 源 现字典
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