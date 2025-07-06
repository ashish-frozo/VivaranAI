#!/bin/bash
# MedBillGuardAgent - End-to-End Testing Startup Script
# This script sets up and starts the complete system for testing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    log_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service_name failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command_exists docker; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check Python
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check OpenAI API Key
    if [ -z "$OPENAI_API_KEY" ]; then
        log_warning "OPENAI_API_KEY environment variable is not set."
        log_info "Some tests may fail without a valid OpenAI API key."
        log_info "Set it with: export OPENAI_API_KEY='sk-your-key-here'"
    fi
    
    log_success "Prerequisites check completed"
}

# Check and free ports if needed
check_ports() {
    log_info "Checking required ports..."
    
    local ports=(8001 6379 5432 9090 3000 16686)
    local ports_in_use=()
    
    for port in "${ports[@]}"; do
        if port_in_use $port; then
            ports_in_use+=($port)
        fi
    done
    
    if [ ${#ports_in_use[@]} -gt 0 ]; then
        log_warning "The following ports are in use: ${ports_in_use[*]}"
        log_info "This may cause conflicts. Consider stopping other services or changing ports."
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Port check completed"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install requirements
    pip install -q --upgrade pip
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -q -r requirements-dev.txt
    elif [ -f "requirements.txt" ]; then
        pip install -q -r requirements.txt
    fi
    
    # Install additional test dependencies
    pip install -q aiohttp redis psycopg2-binary
    
    log_success "Dependencies installed"
}

# Start infrastructure services
start_infrastructure() {
    log_info "Starting infrastructure services..."
    
    # Stop any existing containers
    log_info "Stopping existing containers..."
    docker-compose down >/dev/null 2>&1 || true
    
    # Start services
    log_info "Starting Docker Compose services..."
    docker-compose up -d
    
    log_success "Infrastructure services started"
}

# Wait for all services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for Redis
    wait_for_service "http://localhost:6379" "Redis" || return 1
    
    # Wait for Agent Server
    wait_for_service "http://localhost:8001/health" "Agent Server" || return 1
    
    # Wait for Prometheus (optional)
    wait_for_service "http://localhost:9090" "Prometheus" || log_warning "Prometheus not ready"
    
    log_success "All critical services are ready"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check agent server
    if curl -s http://localhost:8001/health | grep -q "status"; then
        log_success "Agent server health check passed"
    else
        log_error "Agent server health check failed"
        return 1
    fi
    
    # Check Redis
    if docker exec medbillguard-redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis health check passed"
    else
        log_warning "Redis health check failed"
    fi
    
    # Check PostgreSQL
    if docker exec medbillguard-postgres pg_isready -U medbillguard | grep -q "accepting connections"; then
        log_success "PostgreSQL health check passed"
    else
        log_warning "PostgreSQL health check failed"
    fi
    
    log_success "Health checks completed"
}

# Run tests
run_tests() {
    local test_type=$1
    
    log_info "Running $test_type tests..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    case $test_type in
        "quick")
            python test_e2e_runner.py --quick
            ;;
        "full")
            python test_e2e_runner.py --full
            ;;
        "load")
            python test_e2e_runner.py --load
            ;;
        *)
            python test_e2e_runner.py --quick
            ;;
    esac
}

# Show service URLs
show_service_urls() {
    log_info "Service URLs:"
    echo "  🏥 Agent Server:      http://localhost:8001"
    echo "  📊 Grafana:          http://localhost:3000 (admin/admin)"
    echo "  📈 Prometheus:       http://localhost:9090"
    echo "  🔍 Jaeger:           http://localhost:16686"
    echo "  🗄️  Redis Insight:    http://localhost:8080"
    echo "  🐘 pgAdmin:          http://localhost:8082 (admin@medbillguard.com/admin)"
    echo ""
    echo "  📋 Health Check:     curl http://localhost:8001/health"
    echo "  📊 Metrics:          curl http://localhost:8001/metrics"
    echo ""
}

# Main function
main() {
    local command=${1:-"quick"}
    
    echo "🚀 MedBillGuardAgent End-to-End Testing Setup"
    echo "=============================================="
    echo ""
    
    case $command in
        "start")
            check_prerequisites
            check_ports
            install_dependencies
            start_infrastructure
            wait_for_services
            run_health_checks
            show_service_urls
            log_success "System is ready for testing!"
            echo ""
            log_info "Run tests with:"
            echo "  ./start_e2e_testing.sh quick    # Quick smoke tests"
            echo "  ./start_e2e_testing.sh full     # Full test suite"
            echo "  ./start_e2e_testing.sh load     # Performance tests"
            ;;
        "stop")
            log_info "Stopping all services..."
            docker-compose down
            log_success "All services stopped"
            ;;
        "restart")
            log_info "Restarting all services..."
            docker-compose restart
            wait_for_services
            run_health_checks
            log_success "All services restarted"
            ;;
        "status")
            log_info "Checking service status..."
            docker-compose ps
            run_health_checks
            ;;
        "logs")
            log_info "Showing service logs..."
            docker-compose logs -f medbillguard-agent
            ;;
        "quick"|"full"|"load")
            run_tests $command
            ;;
        "help"|"--help"|"-h")
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  start     Start all services and prepare for testing"
            echo "  stop      Stop all services"
            echo "  restart   Restart all services"
            echo "  status    Check service status"
            echo "  logs      Show agent logs"
            echo "  quick     Run quick smoke tests"
            echo "  full      Run full test suite"
            echo "  load      Run performance tests"
            echo "  help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 start     # Start everything"
            echo "  $0 quick     # Run quick tests"
            echo "  $0 full      # Run all tests"
            echo "  $0 stop      # Stop everything"
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 