#!/bin/bash

# MedBillGuardAgent Dashboard Startup Script
echo "🚀 Starting MedBillGuardAgent Dashboard"
echo "======================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "💡 To get started:"
    echo "   1. Copy the template: cp env.example .env"
    echo "   2. Edit .env and add your OpenAI API key"
    echo "   3. Run this script again"
    exit 1
fi

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Verify API key is set
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Error: OPENAI_API_KEY not configured in .env file"
    echo "Please edit .env and set your real OpenAI API key"
    exit 1
fi

# Kill any existing processes on ports 8000 and 8001
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8001 | xargs kill -9 2>/dev/null || true

sleep 2

echo "🔧 Starting Agent Server (Port 8001)..."
# Start the agent server in background
PYTHONPATH=. python3 agents/server.py > agent_server.log 2>&1 &
AGENT_PID=$!
echo "   Agent Server PID: $AGENT_PID"

sleep 3

echo "🌐 Starting Frontend Server (Port 8000)..."
# Start the frontend server in background
cd frontend && python3 serve.py > ../frontend_server.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   Frontend Server PID: $FRONTEND_PID"

sleep 3

# Check if servers are running
echo ""
echo "🔍 Checking server status..."

if curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "✅ Agent Server is running on http://localhost:8001"
else
    echo "❌ Agent Server failed to start (check agent_server.log)"
fi

if curl -s http://localhost:8000 >/dev/null 2>&1; then
    echo "✅ Frontend Server is running on http://localhost:8000"
else
    echo "❌ Frontend Server failed to start (check frontend_server.log)"
fi

echo ""
echo "🎉 Dashboard Ready!"
echo "📱 Open your browser and go to: http://localhost:8000/dashboard.html"
echo ""
echo "📊 Sample test files created in: frontend/sample_bills/"
echo "   - apollo_high_overcharge.txt (High overcharge scenario)"
echo "   - government_normal.txt (Normal government hospital bill)"
echo ""
echo "⚠️  To stop the servers later:"
echo "   kill $AGENT_PID $FRONTEND_PID"
echo ""
echo "📋 Server Logs:"
echo "   Agent Server: agent_server.log"
echo "   Frontend: frontend_server.log" 