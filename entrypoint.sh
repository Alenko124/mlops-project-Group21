#!/usr/bin/env bash
set -e

echo "▶ Checking dataset..."

if [ ! -d "data/raw" ]; then
  echo "⬇ Downloading data from GCS..."
  mkdir -p data
  gsutil -m cp -r gs://mlops-group21/data/* data/
else
  echo "✅ Data already exists, skipping download."
fi

echo "🚀 Starting training..."
exec "$@"
