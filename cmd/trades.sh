# export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/train/training-resnet18-cifar-10.yaml

python -m src.exp.trades --config ${CONFIG_PATH}