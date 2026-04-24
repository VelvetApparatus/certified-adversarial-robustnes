from src.config.conf import ModelConfig
from resnet import build_resnet_18_10

def get_model(cfg: ModelConfig):
    if cfg.name == 'resnet18_10':
        return build_resnet_18_10(cfg.weights_path)
    else:
        raise Exception("Unknown dataset: {}".format(cfg.name))