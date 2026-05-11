CONFIG_PATH=configs/train/smoothed-AWP-resnet18-cifar10.yaml
python -m src.exp.smooth_adv --config "${CONFIG_PATH}"