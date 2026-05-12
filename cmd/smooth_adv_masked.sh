

PGD_CLEAN_CONFIG=configs/train/smooth-adv-masked-pgdclean-resnet18-cifar10.yaml
PGD_MASKED_CONFIG=configs/train/smooth-adv-masked-pgdmasked-resnet18-cifar10.yaml

python -m src.exp.smooth_adv_masked --config ${PGD_CLEAN_CONFIG}
python -m src.exp.smooth_adv_masked --config ${PGD_MASKED_CONFIG}