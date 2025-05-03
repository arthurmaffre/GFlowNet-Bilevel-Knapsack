#!/bin/bash

echo "=========================================="
echo "🚀 Installing Python dependencies from requirements.txt"
echo "=========================================="

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "❌ requirements.txt not found in the current directory!"
    exit 1
fi

# Install dependencies
echo "📦 Running: pip install -r requirements.txt"
pip install -r requirements.txt

# Final message
echo "=========================================="
echo "✅ Dependencies installed successfully!"
echo "=========================================="