python -m src.cli.gdrive_artifacts_sync sync-dir ./outputs \
  --recursive \
  --extensions .pt .pth .ckpt .safetensors .onnx



#python -m src.cli.gdrive_artifacts_sync sync-files \
 #  outputs/exp_001/checkpoints/epoch_10.safetensors \
 #  outputs/exp_001/config.yaml \
 #  outputs/exp_001/metrics.json