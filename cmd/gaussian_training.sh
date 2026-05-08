# export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/train/gaussian_noise-resnet8-cifar10.yaml

python -m src.exp.gaussian_noise --config ${CONFIG_PATH}