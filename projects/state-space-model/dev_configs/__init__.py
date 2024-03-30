from model_config import ModelConfig
from optimizer_config import OptimizerConfig
from platform_config import PlatformConfig
from task_config import TaskConfig


def merge_configs(default, custom):
    final_config = default.copy()
    for key, value in custom.items():
        if isinstance(value, dict):
            node = final_config.setdefault(key, {})
            final_config[key] = merge_configs(node, value)
        else:
            final_config[key] = value
    return final_config


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


class WrapConfigs:

    def __init__(self, model_name_or_path, opt_name, platform_name, data_name) -> None:
        self.model_name_or_path = model_name_or_path
        self.opt_name = opt_name
        self.platform_name = platform_name
        self.data_name = data_name
        self.all_configs = self.set_all_configs()

    def all_configs(self):
        model_config = ModelConfig(self.model_name_or_path)
        optimizer_config = OptimizerConfig(self.opt_name)
        platform_config = PlatformConfig(self.platform_name)
        task_config = TaskConfig(self.data_name)

        default_config = {
            "model": model_config,
            "optimizer": optimizer_config,
            "platform": platform_config,
            "task": task_config,
        }

        return DotDict(default_config)
