from src.config._parsers import ModelConfig
from .resnet import build_resnet_18_10


def get_model(cfg: ModelConfig, device: str):
    if cfg.name == "resnet18_10":
        return build_resnet_18_10(
            weights_path=cfg.weights_path,
            device=device,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
        )
    else:
        raise Exception("Unknown model: {}".format(cfg.name))
