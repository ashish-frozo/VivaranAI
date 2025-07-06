#!/bin/bash

# MedBillGuard Agent System Deployment Script
# Deploys the complete production-ready infrastructure to Kubernetes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="medbillguard"
DOCKER_IMAGE="medbillguard/agent:latest"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
DATABASE_PASSWORD="${DATABASE_PASSWORD:-$(openssl rand -base64 32)}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:-$(openssl rand -base64 64)}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-$(openssl rand -base64 32)}"

# Functions
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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if OpenAI API key is provided
    if [[ -z "$OPENAI_API_KEY" ]]; then
        log_error "OPENAI_API_KEY environment variable is not set."
        log_info "Please set it: export OPENAI_API_KEY='your-api-key'"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    if [[ "${SKIP_BUILD:-false}" == "true" ]]; then
        log_warning "Skipping Docker build (SKIP_BUILD=true)"
        return
    fi
    
    docker build -t "${DOCKER_IMAGE}" .
    
    if [[ "${PUSH_IMAGE:-true}" == "true" ]]; then
        log_info "Pushing Docker image to registry..."
        docker push "${DOCKER_IMAGE}"
    fi
    
    log_success "Docker image built and pushed"
}

create_namespace() {
    log_info "Creating namespace..."
    kubectl apply -f k8s/namespace.yaml
    log_success "Namespace created"
}

create_secrets() {
    log_info "Creating secrets..."
    
    # Create main secrets
    kubectl create secret generic medbillguard-secrets \
        --namespace="${NAMESPACE}" \
        --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --from-literal=DATABASE_PASSWORD="${DATABASE_PASSWORD}" \
        --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
        --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
        --from-literal=ENCRYPTION_KEY="${ENCRYPTION_KEY}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create PostgreSQL secrets
    kubectl create secret generic medbillguard-postgres-secret \
        --namespace="${NAMESPACE}" \
        --from-literal=POSTGRES_DB="medbillguard" \
        --from-literal=POSTGRES_USER="medbillguard" \
        --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create Redis secrets (optional, if auth enabled)
    kubectl create secret generic medbillguard-redis-secret \
        --namespace="${NAMESPACE}" \
        --from-literal=REDIS_PASSWORD="" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets created"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Apply ConfigMaps
    kubectl apply -f k8s/configmap.yaml
    
    # Apply ServiceAccount and RBAC
    kubectl apply -f k8s/serviceaccount.yaml
    
    # Apply PersistentVolumeClaims
    kubectl apply -f k8s/pvc.yaml
    
    # Apply Services
    kubectl apply -f k8s/service.yaml
    
    log_success "Infrastructure components deployed"
}

deploy_applications() {
    log_info "Deploying applications..."
    
    # Apply Deployments
    kubectl apply -f k8s/deployment.yaml
    
    # Apply HPA
    kubectl apply -f k8s/hpa.yaml
    
    log_success "Applications deployed"
}

wait_for_pods() {
    log_info "Waiting for pods to be ready..."
    
    # Wait for agent pods
    kubectl wait --for=condition=ready pod \
        --selector=app.kubernetes.io/name=medbillguard-agent \
        --namespace="${NAMESPACE}" \
        --timeout=300s
    
    # Wait for Redis
    kubectl wait --for=condition=ready pod \
        --selector=app.kubernetes.io/name=medbillguard-redis \
        --namespace="${NAMESPACE}" \
        --timeout=300s
    
    # Wait for PostgreSQL
    kubectl wait --for=condition=ready pod \
        --selector=app.kubernetes.io/name=medbillguard-postgres \
        --namespace="${NAMESPACE}" \
        --timeout=300s
    
    log_success "All pods are ready"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n "${NAMESPACE}"
    
    # Check service status
    log_info "Service status:"
    kubectl get services -n "${NAMESPACE}"
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    AGENT_POD=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/name=medbillguard-agent -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "${NAMESPACE}" "${AGENT_POD}" -- curl -f http://localhost:8001/health/liveness > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_warning "Health check failed - service may still be starting"
    fi
    
    # Display access information
    log_info "Deployment completed successfully!"
    echo ""
    echo "Access Information:"
    echo "=================="
    
    # Get LoadBalancer IP if available
    EXTERNAL_IP=$(kubectl get service medbillguard-agent-external -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [[ -n "$EXTERNAL_IP" ]]; then
        echo "External API URL: http://${EXTERNAL_IP}/health"
        echo "External API Docs: http://${EXTERNAL_IP}/docs"
    else
        echo "Use port-forward to access the service:"
        echo "kubectl port-forward -n ${NAMESPACE} service/medbillguard-agent 8001:8001"
        echo "Then access: http://localhost:8001/health"
    fi
    
    echo ""
    echo "Monitoring URLs (use port-forward):"
    echo "kubectl port-forward -n ${NAMESPACE} service/prometheus 9090:9090"
    echo "kubectl port-forward -n ${NAMESPACE} service/grafana 3000:3000"
    echo ""
    echo "Generated Passwords:"
    echo "Database Password: ${DATABASE_PASSWORD}"
    echo "JWT Secret: ${JWT_SECRET_KEY}"
    echo ""
    echo "Save these credentials securely!"
}

cleanup() {
    log_warning "Cleaning up previous deployment..."
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true --wait=true
    log_success "Cleanup completed"
}

show_help() {
    echo "MedBillGuard Agent System Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -c, --cleanup           Clean up existing deployment"
    echo "  --skip-build            Skip Docker image build"
    echo "  --no-push               Don't push Docker image to registry"
    echo "  --namespace NAME        Use custom namespace (default: medbillguard)"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY          OpenAI API key (required)"
    echo "  DATABASE_PASSWORD       Custom database password (auto-generated if not set)"
    echo "  SKIP_BUILD             Skip Docker build (default: false)"
    echo "  PUSH_IMAGE             Push image to registry (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full deployment"
    echo "  $0 --cleanup            # Clean up existing deployment"
    echo "  $0 --skip-build         # Deploy without building image"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--cleanup)
            cleanup
            exit 0
            ;;
        --skip-build)
            export SKIP_BUILD=true
            shift
            ;;
        --no-push)
            export PUSH_IMAGE=false
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main deployment flow
main() {
    log_info "Starting MedBillGuard Agent System deployment..."
    
    check_prerequisites
    build_docker_image
    create_namespace
    create_secrets
    deploy_infrastructure
    deploy_applications
    wait_for_pods
    verify_deployment
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@" 