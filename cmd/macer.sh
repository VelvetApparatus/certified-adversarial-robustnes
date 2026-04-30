export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/macer-resnet18-cifar10.yaml

python -m src.exp.macer --config ${CONFIG_PATH}