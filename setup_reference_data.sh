#!/bin/bash
# Reference Baseline Generation Examples

# ============================================================================
# SETUP: Generate Reference Baseline from Training Data
# ============================================================================
# Run this ONCE to create your reference/baseline from training images

# Option 1: Process all training images
# (This may take a while if you have many images)
echo "Generating reference baseline from all training images..."
uv run python -m eurosat.reference_generator \
  --data-dir data/raw/train \
  --bucket-name mlops-group21 \
  --reference-folder predictions/reference_data

# Option 2: Sample every 5th image (faster)
# Good for testing or large datasets
echo "Generating reference baseline with sampling (every 5th image)..."
uv run python -m eurosat.reference_generator \
  --data-dir data/raw/train \
  --sample-every 5 \
  --bucket-name mlops-group21 \
  --reference-folder predictions/reference_data

# Option 3: With custom model path
# If you have a local model file instead of loading from GCS
echo "Generating reference with local model..."
uv run python -m eurosat.reference_generator \
  --data-dir data/raw/train \
  --model-path models/resnet18_eurosat_latest.pt \
  --sample-every 1 \
  --bucket-name mlops-group21 \
  --reference-folder predictions/reference_data

# ============================================================================
# VERIFY: Check what was generated
# ============================================================================

# List reference data in GCS
gsutil ls gs://mlops-group21/predictions/reference_data/

# Count records in reference baseline
gsutil cat gs://mlops-group21/predictions/reference_data/reference_baseline.jsonl | wc -l

# ============================================================================
# PRODUCTION: Run API to collect current data
# ============================================================================

# Start API (in another terminal)
uvicorn eurosat.api:app --reload --host 0.0.0.0 --port 8080

# In another terminal, test predictions
for i in {1..10}; do
  curl -X POST http://localhost:8080/predict \
    -F "file=@data/raw/test/0/image_sample_1.jpg"
done

# ============================================================================
# MONITOR: Check accumulated production data
# ============================================================================

# List recent predictions in GCS
gsutil ls gs://mlops-group21/predictions/data_logs/

# Count records collected today
gsutil cat gs://mlops-group21/predictions/data_logs/*.jsonl | wc -l

# ============================================================================
# EXAMPLES: What gets saved
# ============================================================================

# View one record from reference data
echo "Reference Data Record:"
gsutil cat gs://mlops-group21/predictions/reference_data/reference_baseline.jsonl | head -1 | python -m json.tool

# View one record from production data
echo "Production Data Record:"
gsutil cat gs://mlops-group21/predictions/data_logs/*.jsonl | head -1 | python -m json.tool

# ============================================================================
# ANALYSIS: Quick statistics
# ============================================================================

# Extract and analyze brightness from reference data
echo "Reference Brightness Statistics:"
gsutil cat gs://mlops-group21/predictions/reference_data/reference_baseline.jsonl | \
  python -c "import sys, json; records=[json.loads(line) for line in sys.stdin]; import statistics; bright=[r['image_features']['brightness'] for r in records]; print(f'Mean: {statistics.mean(bright):.2f}, Std: {statistics.stdev(bright):.2f}')"

# Extract and analyze brightness from recent production data
echo "Production Brightness Statistics:"
gsutil cat gs://mlops-group21/predictions/data_logs/*.jsonl | \
  python -c "import sys, json; records=[json.loads(line) for line in sys.stdin]; import statistics; bright=[r['image_features']['brightness'] for r in records]; print(f'Mean: {statistics.mean(bright):.2f}, Std: {statistics.stdev(bright):.2f}')"

# ============================================================================
# NEXT STEPS
# ============================================================================

# After this, create drift detector to:
# 1. Load reference baseline
# 2. Load recent production data
# 3. Compare distributions
# 4. Detect statistical drift
# 5. Generate alerts

echo "To check reference data was created:"
echo "  gsutil ls gs://mlops-group21/predictions/reference_data/"
echo ""
echo "To make test predictions:"
echo "  uvicorn eurosat.api:app --reload"
echo "  curl -X POST http://localhost:8080/predict -F 'file=@data/raw/test/0/image.jpg'"
