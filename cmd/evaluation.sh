export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/evaluation-resnet-18-10-cifar-10.yaml

python -m src.exp.evaluate --config ${CONFIG_PATH}