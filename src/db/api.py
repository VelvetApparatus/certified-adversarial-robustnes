from torchvision import datasets, transforms
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
            # TODO: add later
            # transform=cfg.transform
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]),
            download=cfg.download,
        )
    else:
        raise Exception("Unknown dataset: {}".format(cfg.name))
