CONFIG_PATH=${CONFIG_PATH:-configs/train/smoothed-AWP-resnet18-cifar10.yaml}
python -m src.exp.smoothed_awp --config "${CONFIG_PATH}"