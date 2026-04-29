export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/training-resnet18-cifar-10.yaml

python -m src.exp.adversarial_training --config ${CONFIG_PATH}