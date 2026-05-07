# export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/adversarial_training-resnet18-cifar10.yaml

python -m src.exp.adversarial_training_pgd --config ${CONFIG_PATH}