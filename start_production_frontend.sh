#!/bin/bash

# MedBillGuard Production Frontend Launcher
# This script starts the production frontend for testing the Railway deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${2}${1}${NC}"
}

# Header
print_color "🚀 MedBillGuard Production Frontend Launcher" "$BLUE"
print_color "=============================================" "$BLUE"

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    print_color "❌ Error: 'frontend' directory not found!" "$RED"
    print_color "Please run this script from the VivaranAI root directory." "$RED"
    exit 1
fi

# Check if production files exist
if [ ! -f "frontend/index-production.html" ]; then
    print_color "❌ Error: Production frontend files not found!" "$RED"
    print_color "Please ensure frontend/index-production.html exists." "$RED"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_color "❌ Error: Python not found!" "$RED"
        print_color "Please install Python 3.x to run the frontend server." "$RED"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check Railway deployment status
print_color "🔍 Checking Railway deployment status..." "$YELLOW"
RAILWAY_URL="https://endearing-prosperity-production.up.railway.app"

if curl -s --max-time 10 "$RAILWAY_URL/health" > /dev/null 2>&1; then
    print_color "✅ Railway deployment is online!" "$GREEN"
else
    print_color "⚠️  Railway deployment may be offline or starting up..." "$YELLOW"
    print_color "   This is normal for Railway cold starts (30-60 seconds)" "$YELLOW"
fi

# Default port
PORT=${1:-3000}

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_color "⚠️  Port $PORT is already in use!" "$YELLOW"
    print_color "   Trying next available port..." "$YELLOW"
    PORT=$((PORT + 1))
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the frontend server
print_color "🌐 Starting production frontend server..." "$BLUE"
print_color "📍 URL: http://localhost:$PORT" "$GREEN"
print_color "🔗 Railway Backend: $RAILWAY_URL" "$GREEN"
print_color "📁 Serving from: $(pwd)/frontend" "$BLUE"
print_color "📝 Logs: logs/frontend.log" "$BLUE"
print_color "" "$NC"
print_color "Features available:" "$BLUE"
print_color "  • Environment toggle (Local/Production)" "$BLUE"
print_color "  • Real-time system monitoring" "$BLUE"
print_color "  • Quick endpoint testing" "$BLUE"
print_color "  • File upload and analysis" "$BLUE"
print_color "  • Comprehensive error handling" "$BLUE"
print_color "" "$NC"
print_color "⏹️  Press Ctrl+C to stop the server" "$YELLOW"
print_color "=============================================" "$BLUE"

# Change to frontend directory and start server
cd frontend

# Start the server and log output
$PYTHON_CMD serve-production.py $PORT 2>&1 | tee ../logs/frontend.log

print_color "👋 Frontend server stopped" "$GREEN" 