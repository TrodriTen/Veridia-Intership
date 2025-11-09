#!/bin/bash
echo "=========================================="
echo "  Training Resume Classifier Model"
echo "=========================================="
echo ""

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Running training pipeline..."
python -m app.train_pipeline

echo ""
echo "=========================================="
echo "  Training completed"
echo "=========================================="
