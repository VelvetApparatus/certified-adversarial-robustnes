# export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/train/adv-FGSM-resnet18-cifar10.yaml

python -m src.exp.adversarial_training_fgsm --config ${CONFIG_PATH}