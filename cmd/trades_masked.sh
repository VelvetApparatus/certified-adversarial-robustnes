CONFIG_PGD_CLEAN=configs/train/trades-masked-pgdclean-resnet18-cifar10.yaml
CONFIG_PGD_MASKED=configs/train/trades-masked-pgdmasked-resnet18-cifar10.yaml


python -m src.exp.trades_masked --config {CONFIG_PGD_CLEAN}
python -m src.exp.trades_masked --config {CONFIG_PGD_MASKED}