#!/bin/bash

# Railway Agent Registration Fix Deployment Script
# This script deploys the agent registration fix and monitors the results

set -e

echo "🚀 Deploying Railway Agent Registration Fix"
echo "=========================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Please install it first:"
    echo "   npm install -g @railway/cli"
    exit 1
fi

# Check if logged in to Railway
if ! railway status &> /dev/null; then
    echo "❌ Not logged in to Railway. Please login first:"
    echo "   railway login"
    exit 1
fi

echo "✅ Railway CLI is ready"

# Deploy the changes
echo "📦 Deploying changes to Railway..."
railway up --detach

echo "⏳ Waiting for deployment to complete..."
sleep 30

# Check deployment status
echo "🔍 Checking deployment status..."
railway status

# Wait for the service to be ready
echo "⏳ Waiting for service to be ready (this may take a few minutes)..."
sleep 60

# Test the agent registration
echo "🧪 Testing agent registration..."
python3 test_agent_registration.py --railway

if [ $? -eq 0 ]; then
    echo "✅ Agent registration test passed!"
else
    echo "❌ Agent registration test failed!"
    echo "📋 Checking Railway logs for debugging..."
    railway logs
    exit 1
fi

# Monitor for stability
echo "📊 Monitoring agent registration for 5 minutes..."
python3 test_agent_registration.py --railway --monitor 300

echo "🎉 Railway Agent Registration Fix deployment completed!"
echo ""
echo "📝 Summary:"
echo "- ✅ Deployed railway_startup.py for better initialization"
echo "- ✅ Added persistent agent registration with retries"
echo "- ✅ Implemented background registration monitoring"
echo "- ✅ Extended heartbeat timeouts for Railway (10 minutes)"
echo "- ✅ Added health check re-registration triggers"
echo ""
echo "🔗 Your Railway app: https://endearing-prosperity-production.up.railway.app"
echo "🔗 Health check: https://endearing-prosperity-production.up.railway.app/health"
echo "🔗 Agent list: https://endearing-prosperity-production.up.railway.app/agents"
echo ""
echo "The agents should now stay registered and automatically re-register after Railway cold starts!" 