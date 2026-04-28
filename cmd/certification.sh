export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/certification-resnet-18-10-cifar-10.yaml

python -m src.exp.certification --config ${CONFIG_PATH}