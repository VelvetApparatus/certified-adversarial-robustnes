# export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/train/trades-resnet18-cifar10.yaml

python -m src.exp.trades --config ${CONFIG_PATH}