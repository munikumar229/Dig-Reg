#!/bin/bash

echo "🚀 Training Models for Digit Classification"
echo "==========================================="

# Check if we're in the project root directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
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
echo "🎉 Training Complete! Both models are now available."
echo ""
echo "🚀 Next steps:"
echo "  • Start backend: uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo "  • Start frontend: streamlit run frontend/app.py --server.port 8501"
echo "  • Or use Docker: cd deployment && sudo docker-compose up -d"