#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-./data/dataset/cifar_10}"
ARCHIVE_NAME="cifar-10-python.tar.gz"
ARCHIVE_PATH="${ROOT_DIR}/${ARCHIVE_NAME}"
EXTRACTED_DIR="${ROOT_DIR}/cifar-10-batches-py"

URLS=(
  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
)

mkdir -p "${ROOT_DIR}"

if [ -d "${EXTRACTED_DIR}" ]; then
  echo "CIFAR-10 already exists: ${EXTRACTED_DIR}"
  exit 0
fi

echo "Downloading CIFAR-10 into: ${ROOT_DIR}"

success=0

for url in "${URLS[@]}"; do
  echo "Trying mirror: ${url}"

  rm -f "${ARCHIVE_PATH}"

  if command -v wget >/dev/null 2>&1; then
    if wget -O "${ARCHIVE_PATH}" "${url}"; then
      success=1
      break
    fi
  elif command -v curl >/dev/null 2>&1; then
    if curl -L -o "${ARCHIVE_PATH}" "${url}"; then
      success=1
      break
    fi
  else
    echo "Neither wget nor curl is installed."
    exit 1
  fi

  echo "Mirror failed: ${url}"
done

if [ "${success}" -ne 1 ]; then
  echo "Failed to download CIFAR-10 from all mirrors."
  exit 1
fi

echo "Extracting ${ARCHIVE_PATH}"
tar -xzf "${ARCHIVE_PATH}" -C "${ROOT_DIR}"

if [ ! -d "${EXTRACTED_DIR}" ]; then
  echo "Extraction finished, but expected directory was not found:"
  echo "${EXTRACTED_DIR}"
  exit 1
fi

echo "CIFAR-10 is ready:"
echo "${EXTRACTED_DIR}"