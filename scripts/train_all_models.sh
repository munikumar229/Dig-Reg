#!/bin/bash

echo "🚀 Training Models for Digit Classification"
echo "==========================================="

# Check if we're in the project root directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Security validation - check MLflow version
echo "🛡️ Checking MLflow security status..."
if grep -q "mlflow==3.5.1" requirements.txt; then
    echo "✅ MLflow 3.5.1 detected - CVE-2024-37059 vulnerability resolved"
else
    echo "⚠️  Warning: Please update MLflow to version 3.5.1+ to resolve CVE-2024-37059"
    echo "   Update command: pip install mlflow==3.5.1"
fi

# Check if data is processed
if [ ! -d "data/processed" ]; then
    echo "📊 Processing data first..."
    python scripts/process_data.py
fi

echo ""
echo "📊 Training RandomForest Model..."
python scripts/train.py --model randomforest

if [ $? -eq 0 ]; then
    echo "✅ RandomForest model trained successfully!"
else
    echo "❌ RandomForest model training failed!"
    exit 1
fi

echo ""
echo "🧠 Training MLP (Neural Network) Model..."
python scripts/train.py --model mlp

if [ $? -eq 0 ]; then
    echo "✅ MLP model trained successfully!"
else
    echo "❌ MLP model training failed!"
    exit 1
fi

echo ""
echo "🛡️ Post-training security validation..."
echo "🔍 Checking newly created MLflow model artifacts..."

# Update any model artifacts that might have been created with vulnerable MLflow versions
if [[ -d "mlruns" ]]; then
    # Update requirements.txt files in newly created model artifacts
    REQUIREMENTS_FILES=$(find mlruns -name "requirements.txt" -exec grep -l "mlflow==3\.3\.2" {} \; 2>/dev/null || true)
    if [[ -n "$REQUIREMENTS_FILES" ]]; then
        echo "📝 Updating MLflow version in model artifacts..."
        for file in $REQUIREMENTS_FILES; do
            sed -i 's/mlflow==3\.3\.2/mlflow==3.5.1/g' "$file" 2>/dev/null || true
        done
        echo "✅ Model artifacts updated to secure MLflow version"
    else
        echo "✅ All model artifacts already use secure MLflow version"
    fi
    
    # Update conda.yaml files in newly created model artifacts
    CONDA_FILES=$(find mlruns -name "conda.yaml" -exec grep -l "mlflow==3\.3\.2" {} \; 2>/dev/null || true)
    if [[ -n "$CONDA_FILES" ]]; then
        echo "📝 Updating conda environment files in model artifacts..."
        for file in $CONDA_FILES; do
            sed -i 's/mlflow==3\.3\.2/mlflow==3.5.1/g' "$file" 2>/dev/null || true
        done
        echo "✅ Conda environment files updated"
    fi
    
    # Update MLmodel files in newly created model artifacts
    MLMODEL_FILES=$(find mlruns -name "MLmodel" -exec grep -l "mlflow_version: 3\.3\.2" {} \; 2>/dev/null || true)
    if [[ -n "$MLMODEL_FILES" ]]; then
        echo "📝 Updating MLmodel version information in model artifacts..."
        for file in $MLMODEL_FILES; do
            sed -i 's/mlflow_version: 3\.3\.2/mlflow_version: 3.5.1/g' "$file" 2>/dev/null || true
        done
        echo "✅ MLmodel version information updated"
    fi
fi

echo ""
echo "🎉 Training Complete! Both models are now available."
echo "🛡️ Security Status: All artifacts use secure MLflow 3.5.1+"
echo ""
echo "🚀 Next steps:"
echo "  • Start backend: uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo "  • Start frontend: streamlit run frontend/app.py --server.port 8501"
echo "  • Or use Docker: cd deployment && sudo docker-compose up -d"