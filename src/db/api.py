from torchvision import datasets
from src.config.conf import DatasetConfig
from torch.utils.data import Dataset

def get_dataset(
        cfg: DatasetConfig
) -> Dataset:
    if cfg.download and not cfg.root_dir:
        raise Exception("root_dir should be provided with download=True")

    if cfg.name == "cifar10":
        return datasets.CIFAR10(
            root=cfg.root_dir,
            train=cfg.train,
            transform=cfg.transform,
            download=cfg.download,
        )
    else:
        raise Exception("Unknown dataset: {}".format(cfg.name))
