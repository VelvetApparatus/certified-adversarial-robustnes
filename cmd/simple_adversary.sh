export PYTORCH_ENABLE_MPS_FALLBACK=1
CONFIG_PATH=configs/first.yaml

python -m src.exp.simple_adversary --config ${CONFIG_PATH}