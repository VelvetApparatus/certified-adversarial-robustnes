

# #SYNC DIR
# python -m src.cli.gdrive_artifacts_sync sync-dir ./outputs \
#   --recursive \
#   --extensions .pt .pth .ckpt .safetensors .onnx


# DOWNLOAD FILES
# python -m src.cli.gdrive_artifacts_sync \
  # --manifest artifacts/gdrive_manifest.json \
  # download-all \
  # --output-dir outputs


# python -m src.cli.gdrive_artifacts_sync \
#   --manifest artifacts/gdrive_manifest.json \
#   download \
#   --output-dir /root \
#   --relative-path \
#     outputs/train/trades_awp_masked_pgd_masked/cifar10_resnet18/ResNet_cifar10_20260513_165019/checkpoints/best.pth \
#     outputs/train/trades_awp_masked_pgd_masked/cifar10_resnet18/ResNet_cifar10_20260513_165019/checkpoints/last.pth \
#     outputs/train/trades_awp_masked_pgd_masked/cifar10_resnet18/ResNet_cifar10_20260513_165019/config.yaml \
#     outputs/train/trades_smoothadv_consistency/cifar10_resnet18/ResNet_cifar10_20260513_190108/checkpoints/best.pth \
#     outputs/train/trades_smoothadv_consistency/cifar10_resnet18/ResNet_cifar10_20260513_190108/checkpoints/last.pth \
#     outputs/train/trades_smoothadv_consistency/cifar10_resnet18/ResNet_cifar10_20260513_190108/config.yaml


python -m src.cli.gdrive_artifacts_sync \
  --auth oauth \
  --manifest artifacts/gdrive_manifest.json \
  sync-files \
    outputs/train/trades_awp_masked_pgd_masked/cifar10_resnet18/ResNet_cifar10_20260513_165019/checkpoints/best.pth \
    outputs/train/trades_awp_masked_pgd_masked/cifar10_resnet18/ResNet_cifar10_20260513_165019/checkpoints/last.pth \
    outputs/train/trades_awp_masked_pgd_masked/cifar10_resnet18/ResNet_cifar10_20260513_165019/config.yaml \
    outputs/train/trades_smoothadv_consistency/cifar10_resnet18/ResNet_cifar10_20260513_190108/checkpoints/best.pth \
    outputs/train/trades_smoothadv_consistency/cifar10_resnet18/ResNet_cifar10_20260513_190108/checkpoints/last.pth \
    outputs/train/trades_smoothadv_consistency/cifar10_resnet18/ResNet_cifar10_20260513_190108/config.yaml \
  --base-dir outputs \
  # --folder-id "$GDRIVE_FOLDER_ID"

# # BUILD MANIFEST
# python -m src.cli.gdrive_artifacts_sync \
#   --auth oauth \
#   --manifest artifacts/gdrive_manifest.json \
#   build-manifest \
#   --recursive \
#   --replace
