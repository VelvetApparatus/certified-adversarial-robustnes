#export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/smooth-adv-resnet18-cifar10.yaml

python -m src.exp.smooth_adv --config ${CONFIG_PATH}