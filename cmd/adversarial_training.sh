# export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/train/adv-PGD-resnet18-cifar10.yaml

python -m src.exp.adversarial_training_pgd --config ${CONFIG_PATH}