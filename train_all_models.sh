#!/bin/bash

echo "🚀 Training Models for Digit Classification"
echo "==========================================="

echo ""
echo "📊 Training RandomForest Model..."
python src/train.py --model randomforest

echo ""
echo "🧠 Training MLP (Neural Network) Model..."
python src/train.py --model mlp

echo ""
echo " Training Complete! Both models are now available in the Streamlit app."
echo "  Run 'streamlit run app.py' to test both models!"