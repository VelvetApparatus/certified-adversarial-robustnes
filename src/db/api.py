from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, Subset, random_split
from src.config._parsers import DatasetConfig
from src.config.common import DatasetSplitConfig
from torch.utils.data import DataLoader


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


def split_train_eval_dataset(
        dataset: Dataset,
        split_cfg: DatasetSplitConfig,
):
    dataset_size = len(dataset)

    if split_cfg.eval_size is not None:
        eval_size = split_cfg.eval_size
    else:
        eval_size = int(dataset_size * split_cfg.eval_ratio)

    if eval_size <= 0:
        raise ValueError(f"eval_size must be positive, got {eval_size}")

    if eval_size >= dataset_size:
        raise ValueError(
            f"eval_size must be smaller than dataset size: "
            f"eval_size={eval_size}, dataset_size={dataset_size}"
        )

    train_size = dataset_size - eval_size

    if split_cfg.shuffle:
        generator = torch.Generator().manual_seed(split_cfg.seed)

        return random_split(
            dataset,
            [train_size, eval_size],
            generator=generator,
        )

    train_indices = list(range(train_size))
    eval_indices = list(range(train_size, dataset_size))

    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def build_train_eval_loaders(dataset_cfg, split_cfg):
    full_train_dataset = get_dataset(dataset_cfg)

    if split_cfg.enabled:
        train_dataset, eval_dataset = split_train_eval_dataset(
            dataset=full_train_dataset,
            split_cfg=split_cfg,
        )
    else:
        train_dataset = full_train_dataset
        eval_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=True,
        num_workers=dataset_cfg.num_workers,
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=False,
            num_workers=dataset_cfg.num_workers,
        )

    return train_loader, eval_loader, train_dataset, eval_dataset
