#!/bin/bash

# Start OAuth2 Frontend for VivaranAI
# This script starts the OAuth2-enabled frontend on port 3000

echo "🚀 Starting VivaranAI OAuth2 Frontend..."
echo "   Frontend URL: http://localhost:3000"
echo "   OAuth2 File: frontend/index-oauth.html"
echo ""
echo "📋 Setup checklist:"
echo "   ✅ Backend server running on port 8001"
echo "   ✅ OAuth2 credentials configured in .env"
echo "   ✅ Database migrations completed"
echo ""
echo "🔐 OAuth2 Providers:"
echo "   - Google OAuth2 (if configured)"
echo "   - GitHub OAuth2 (if configured)"
echo ""
echo "Press Ctrl+C to stop the frontend server"
echo ""

# Navigate to frontend directory
cd frontend

# Start simple HTTP server
python3 -m http.server 3000 2>/dev/null || python -m http.server 3000 