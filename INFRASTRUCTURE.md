# MedBillGuard Agent System - Infrastructure Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the **MedBillGuard Agent System** - a production-ready, multi-agent microservice for detecting medical bill overcharging in India. The system leverages OpenAI Agent SDK for intelligent coordination and provides comprehensive monitoring and observability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ MedBillGuard │  │    Router    │  │   Registry   │        │
│  │    Agent     │  │    Agent     │  │    Service   │        │
│  │  (FastAPI)   │  │ (Workflow)   │  │ (Discovery)  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         │                  │                  │              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │    Redis     │  │ PostgreSQL   │  │  Prometheus  │        │
│  │  (Cache)     │  │ (Database)   │  │ (Metrics)    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                               │
├─────────────────────────────────────────────────────────────────┤
│                   Monitoring Stack                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Grafana    │  │    Jaeger    │  │ Alertmanager │        │
│  │ (Dashboard)  │  │  (Tracing)   │  │  (Alerts)    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities
- **Multi-Agent Architecture**: Intelligent agent coordination using OpenAI Agent SDK
- **Document Processing**: OCR pipeline with multi-language support (English, Hindi, Bengali, Tamil)
- **Rate Validation**: Real-time validation against CGHS, ESI, and NPPA rates
- **Duplicate Detection**: Intelligent duplicate item detection with confidence scoring
- **Prohibited Item Detection**: Insurance-specific prohibited item identification
- **Confidence Scoring**: ML-powered confidence assessment and recommendations

### Production Features
- **Horizontal Pod Autoscaling**: Automatic scaling based on CPU, memory, and custom metrics
- **Health Checks**: Kubernetes-native liveness, readiness, and startup probes
- **Observability**: Comprehensive monitoring with Prometheus, Grafana, and Jaeger
- **Security**: RBAC, non-root containers, read-only filesystems, secret management
- **High Availability**: Multi-replica deployments with anti-affinity rules

## Prerequisites

### Required Tools
- `kubectl` (v1.25+)
- `docker` (v20.0+)
- Kubernetes cluster (v1.25+)
- OpenAI API key

### Cluster Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, 100GB storage
- **Recommended**: 8 CPU cores, 16GB RAM, 500GB storage
- **Storage Classes**: `fast-ssd` for databases, `standard` for monitoring

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DATABASE_PASSWORD="your-secure-password"  # Optional: auto-generated
```

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd VivaranAI
```

### 2. Deploy Complete System
```bash
# Full deployment with all components
./scripts/deploy.sh

# Or with custom options
./scripts/deploy.sh --namespace my-namespace --skip-build
```

### 3. Verify Deployment
```bash
# Check pod status
kubectl get pods -n medbillguard

# Test health endpoint
kubectl port-forward -n medbillguard service/medbillguard-agent 8001:8001
curl http://localhost:8001/health
```

## Component Details

### MedBillGuard Agent
- **Image**: `medbillguard/agent:latest`
- **Port**: 8001 (HTTP), 8002 (Metrics)
- **Replicas**: 3 (auto-scaling 2-20)
- **Resources**: 250m-1000m CPU, 512Mi-2Gi RAM

### Redis Cache
- **Image**: `redis:7.2-alpine`
- **Port**: 6379
- **Storage**: 10Gi persistent volume
- **Configuration**: AOF persistence, LRU eviction

### PostgreSQL Database
- **Image**: `postgres:15-alpine`
- **Port**: 5432
- **Storage**: 50Gi persistent volume
- **Database**: `medbillguard`

### Monitoring Stack
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Visualization dashboard (Port 3000)
- **Jaeger**: Distributed tracing (Port 16686)

## Configuration

### ConfigMap Settings
Key configurations in `k8s/configmap.yaml`:

```yaml
# Server settings
HOST: "0.0.0.0"
PORT: "8001"
LOG_LEVEL: "INFO"
WORKERS: "2"

# Performance tuning
MAX_CONCURRENT_REQUESTS: "10"
PROCESSING_TIMEOUT_SECONDS: "300"
MAX_FILE_SIZE_MB: "50"

# Feature flags
ENABLE_RATE_LIMITING: "true"
ENABLE_CACHING: "true"
CACHE_TTL_SECONDS: "3600"
```

### Secrets Management
Sensitive data in Kubernetes secrets:

```yaml
# Required secrets
OPENAI_API_KEY: "<your-api-key>"
DATABASE_PASSWORD: "<auto-generated>"
JWT_SECRET_KEY: "<auto-generated>"
ENCRYPTION_KEY: "<auto-generated>"
```

## Monitoring and Observability

### Prometheus Metrics
Custom metrics exposed by the application:

- `medbillguard_requests_total`: HTTP request counter
- `medbillguard_request_duration_seconds`: Request duration histogram
- `medbillguard_analysis_total`: Medical bill analysis counter
- `medbillguard_analysis_duration_seconds`: Analysis duration histogram
- `medbillguard_active_agents`: Number of active agents

### Grafana Dashboards
Access Grafana dashboards:

```bash
kubectl port-forward -n medbillguard service/grafana 3000:3000
# Access: http://localhost:3000 (admin/admin)
```

### Alerts Configuration
Production-ready alerts in `monitoring/alerts.yml`:

- **Critical**: Agent down, Database down, High error rate
- **Warning**: High response time, Resource exhaustion
- **Info**: Business metrics, Security events

## Scaling and Performance

### Horizontal Pod Autoscaler
Automatic scaling based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Analysis duration (target: 5s average)

### Vertical Pod Autoscaler
Resource optimization:
- **Min**: 100m CPU, 256Mi RAM
- **Max**: 2000m CPU, 4Gi RAM
- **Mode**: Auto (recommendations applied automatically)

### Performance Tuning
Optimize for your workload:

```yaml
# High throughput
MAX_CONCURRENT_REQUESTS: "20"
WORKERS: "4"

# Low latency
PROCESSING_TIMEOUT_SECONDS: "60"
CACHE_TTL_SECONDS: "1800"
```

## Security

### RBAC Permissions
Minimal required permissions:
- Pod management for service discovery
- ConfigMap/Secret access for configuration
- Metrics collection for monitoring

### Container Security
- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Capabilities dropped

### Network Security
- Internal ClusterIP services
- LoadBalancer for external access
- Network policies (optional)

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting
```bash
# Check events
kubectl describe pod <pod-name> -n medbillguard

# Check logs
kubectl logs <pod-name> -n medbillguard

# Common causes:
# - Missing secrets (OPENAI_API_KEY)
# - Storage class not available
# - Resource constraints
```

#### 2. Health Check Failures
```bash
# Test directly
kubectl exec -n medbillguard <pod-name> -- curl http://localhost:8001/health

# Check dependencies
kubectl get pods -n medbillguard | grep redis
kubectl get pods -n medbillguard | grep postgres
```

#### 3. High Resource Usage
```bash
# Check resource usage
kubectl top pods -n medbillguard

# Adjust resources in deployment.yaml
resources:
  requests:
    memory: "1Gi"  # Increase if needed
    cpu: "500m"
```

### Debug Commands
```bash
# Get all resources
kubectl get all -n medbillguard

# Check resource usage
kubectl top pods -n medbillguard
kubectl top nodes

# Port forward for local access
kubectl port-forward -n medbillguard service/medbillguard-agent 8001:8001
kubectl port-forward -n medbillguard service/prometheus 9090:9090
kubectl port-forward -n medbillguard service/grafana 3000:3000

# View logs
kubectl logs -f deployment/medbillguard-agent -n medbillguard
```

## Maintenance

### Updates and Rollouts
```bash
# Update image
kubectl set image deployment/medbillguard-agent \
  medbillguard-agent=medbillguard/agent:v1.1.0 \
  -n medbillguard

# Check rollout status
kubectl rollout status deployment/medbillguard-agent -n medbillguard

# Rollback if needed
kubectl rollout undo deployment/medbillguard-agent -n medbillguard
```

### Backup and Recovery
```bash
# Backup PostgreSQL
kubectl exec -n medbillguard deployment/medbillguard-postgres -- \
  pg_dump -U medbillguard medbillguard > backup.sql

# Backup Redis
kubectl exec -n medbillguard deployment/medbillguard-redis -- \
  redis-cli BGSAVE
```

### Cleanup
```bash
# Clean up deployment
./scripts/deploy.sh --cleanup

# Or manually
kubectl delete namespace medbillguard
```

## API Usage

### Health Endpoints
```bash
# Basic health check
curl http://localhost:8001/health

# Liveness probe
curl http://localhost:8001/health/liveness

# Readiness probe
curl http://localhost:8001/health/readiness
```

### Analysis API
```bash
# Analyze medical bill
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "<base64-encoded-pdf>",
    "doc_id": "doc123",
    "user_id": "user456",
    "language": "english",
    "insurance_type": "cghs"
  }'
```

### Metrics
```bash
# Prometheus metrics
curl http://localhost:8001/metrics

# Application metrics summary
curl http://localhost:8001/metrics/summary
```

## Production Checklist

- [ ] OpenAI API key configured
- [ ] Database passwords secured
- [ ] Resource limits set appropriately
- [ ] Monitoring alerts configured
- [ ] Backup strategy implemented
- [ ] Network policies applied (if required)
- [ ] SSL/TLS certificates configured
- [ ] Log aggregation setup
- [ ] Security scanning completed

## Support

### Documentation
- API Documentation: `/docs` endpoint
- Prometheus Metrics: `/metrics` endpoint
- Health Status: `/health` endpoint

### Contact
- Team: Platform Engineering
- Email: platform@medbillguard.com
- Slack: #medbillguard-support

---

**Note**: This infrastructure supports production workloads with proper monitoring, scaling, and security. Customize configurations based on your specific requirements and compliance needs. 