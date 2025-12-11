#!/bin/bash

# NutriSnap UI Launcher Script
# This script helps you quickly launch the frontend

echo "🥗 NutriSnap - Starting Frontend..."
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "frontend/index.html" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

cd frontend

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found! Starting HTTP server..."
    echo ""
    echo "🌐 Open your browser to: http://localhost:3000"
    echo "📍 Press Ctrl+C to stop the server"
    echo ""
    python3 -m http.server 3000
elif command -v python &> /dev/null; then
    echo "✅ Python found! Starting HTTP server..."
    echo ""
    echo "🌐 Open your browser to: http://localhost:3000"
    echo "📍 Press Ctrl+C to stop the server"
    echo ""
    python -m http.server 3000
else
    echo "❌ Python not found!"
    echo ""
    echo "Please install Python or open frontend/index.html directly in your browser"
    exit 1
fi
