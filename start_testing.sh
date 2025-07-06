#!/bin/bash

# MedBillGuardAgent Testing Startup Script
# This script starts both the API server and frontend dashboard

set -e

echo "🚀 Starting MedBillGuardAgent Testing Environment"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "medbillguardagent.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Python dependencies are installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry not found. Please install Poetry first."
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
poetry install --quiet

# Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis not running - some features may be limited"
    fi
else
    echo "⚠️  Redis not installed - some features may be limited"
fi

# Create frontend directory if it doesn't exist
mkdir -p frontend

echo ""
echo "🎯 Starting services..."
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    jobs -p | xargs -r kill
    exit 0
}
trap cleanup EXIT

# Start API server in background
echo "🔧 Starting API server on http://localhost:8000"
poetry run uvicorn medbillguardagent:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start frontend dashboard in background
echo "🌐 Starting dashboard on http://localhost:3000"
cd frontend
python serve.py &
FRONTEND_PID=$!
cd ..

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 5

# Test API health
if curl -s http://localhost:8000/healthz > /dev/null; then
    echo "✅ API server is healthy"
else
    echo "❌ API server failed to start"
fi

# Test frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend dashboard is running"
else
    echo "❌ Frontend dashboard failed to start"
fi

echo ""
echo "🎉 MedBillGuardAgent Testing Environment Ready!"
echo "================================================"
echo ""
echo "📋 Quick Access Links:"
echo "   🌐 Dashboard:     http://localhost:3000"
echo "   🔧 API Health:    http://localhost:8000/healthz"
echo "   📚 API Docs:      http://localhost:8000/docs"
echo "   🧪 Quick Test:    http://localhost:8000/debug/example"
echo ""
echo "📁 Test Files:"
echo "   📄 Sample Bill:   fixtures/sample_bill.txt"
echo ""
echo "🧪 Testing Commands:"
echo "   curl http://localhost:8000/healthz"
echo "   curl http://localhost:8000/debug/example"
echo "   python test_api.py"
echo ""
echo "⏹️  Press Ctrl+C to stop all services"
echo ""

# Keep script running and show logs
wait 