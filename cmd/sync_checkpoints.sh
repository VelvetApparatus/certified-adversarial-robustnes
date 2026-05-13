

#SYNC DIR
python -m src.cli.gdrive_artifacts_sync sync-dir ./outputs \
  --recursive \
  --extensions .pt .pth .ckpt .safetensors .onnx


# DOWNLOAD FILES
python -m src.cli.gdrive_artifacts_sync \
  --manifest artifacts/gdrive_manifest.json \
  download-all \
  --output-dir outputs/train



# BUILD MANIFEST
python -m src.cli.gdrive_artifacts_sync \
  --auth oauth \
  --manifest artifacts/gdrive_manifest.json \
  build-manifest \
  --recursive \
  --replace
