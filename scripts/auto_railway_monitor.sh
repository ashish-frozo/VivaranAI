#!/bin/bash

# Auto Railway Monitor - Continuously monitor Railway deployment
# This script automatically checks Railway logs, health status, and deployment progress

echo "🚀 Starting Auto Railway Monitor for VivaranAI..."
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HEALTH_URL="https://endearing-prosperity-production.up.railway.app/health"
CHECK_INTERVAL=30  # seconds
LOG_FILE="railway_monitor.log"

# Function to check health
check_health() {
    echo -e "${BLUE}[$(date)] Checking health...${NC}"
    
    response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$HEALTH_URL" 2>/dev/null)
    http_code="${response: -3}"
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ Health check PASSED${NC}"
        cat /tmp/health_response.json | jq '.' 2>/dev/null || cat /tmp/health_response.json
        return 0
    else
        echo -e "${RED}❌ Health check FAILED (HTTP $http_code)${NC}"
        cat /tmp/health_response.json 2>/dev/null
        return 1
    fi
}

# Function to check Railway logs
check_logs() {
    echo -e "${BLUE}[$(date)] Checking Railway logs for errors...${NC}"
    
    # Get recent logs and check for errors
    railway logs 2>&1 | tail -20 | tee -a "$LOG_FILE"
    
    # Check for specific error patterns
    recent_errors=$(railway logs 2>&1 | tail -50 | grep -i "error\|traceback\|exception\|failed" | tail -5)
    
    if [ ! -z "$recent_errors" ]; then
        echo -e "${RED}🚨 Recent errors found:${NC}"
        echo "$recent_errors"
        return 1
    else
        echo -e "${GREEN}✅ No recent errors in logs${NC}"
        return 0
    fi
}

# Function to check deployment status
check_deployment() {
    echo -e "${BLUE}[$(date)] Checking deployment status...${NC}"
    
    # Get Railway status
    railway status 2>/dev/null | tee -a "$LOG_FILE"
}

# Function to analyze error patterns
analyze_errors() {
    echo -e "${YELLOW}🔍 Analyzing error patterns...${NC}"
    
    # Common error patterns to check for
    declare -A error_patterns=(
        ["ModuleNotFoundError"]="Missing Python dependency"
        ["ImportError"]="Import/dependency issue"
        ["AttributeError"]="Method/attribute not found"
        ["TypeError"]="Parameter type mismatch"
        ["ConnectionError"]="Network/database connection issue"
        ["redis.exceptions"]="Redis connection problem"
        ["AgentRegistry"]="Agent registration issue"
        ["UnboundLocalError"]="Variable scope issue"
    )
    
    for pattern in "${!error_patterns[@]}"; do
        if railway logs 2>&1 | tail -100 | grep -q "$pattern"; then
            echo -e "${RED}⚠️  Found: $pattern - ${error_patterns[$pattern]}${NC}"
        fi
    done
}

# Main monitoring loop
monitor_loop() {
    echo -e "${GREEN}🔄 Starting continuous monitoring (every ${CHECK_INTERVAL}s)${NC}"
    echo "Press Ctrl+C to stop monitoring"
    echo "Logs saved to: $LOG_FILE"
    echo ""
    
    while true; do
        echo "=========================================="
        echo -e "${BLUE}[$(date)] Monitoring Cycle${NC}"
        
        # Check deployment status
        check_deployment
        echo ""
        
        # Check application health
        if check_health; then
            echo -e "${GREEN}🎉 Application is healthy!${NC}"
        else
            echo -e "${YELLOW}⚠️  Application unhealthy, checking logs...${NC}"
            check_logs
            analyze_errors
        fi
        
        echo ""
        echo -e "${BLUE}Waiting ${CHECK_INTERVAL} seconds before next check...${NC}"
        sleep "$CHECK_INTERVAL"
    done
}

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Monitoring stopped by user${NC}"; exit 0' INT

# Start monitoring
echo "Starting Railway monitoring at $(date)" > "$LOG_FILE"
monitor_loop 