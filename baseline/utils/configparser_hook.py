import functools
import configparser

class ConfigParserHook(object):
    def __init__(self):
        self.config = configparser.RawConfigParser()

    def read(self, config_file):
        self.config.read(config_file, encoding="utf-8")

def set_hook(func_name):
    @functools.wraps(getattr(configparser.RawConfigParser, func_name))
    def wrapper(self, *args, **kwargs):
        return getattr(self.config, func_name)(*args, **kwargs)

    return wrapper

def get_config(config_file):
    # 去掉所有魔术方法
    # func_list = [func_name for func_name in dir(configparser.RawConfigParser) if not func_name.startswith("_") and func_name != "read"]
    for func_name in dir(configparser.RawConfigParser):
        if not func_name.startswith("_") and func_name != "read":
            setattr(ConfigParserHook, func_name, set_hook(func_name))
    setattr(ConfigParserHook, "__getitem__", set_hook("__getitem__"))
    # 将RawConfigParser的方法都给ConfigParserHook
    config = ConfigParserHook()
    config.read(config_file)

    # print(config.config['train']['epoch'])
    return config