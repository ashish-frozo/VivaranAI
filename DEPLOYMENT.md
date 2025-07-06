# Deployment Guide 🚀

This guide covers deploying VivaranAI MedBillGuardAgent to production environments using Docker, Kubernetes, and cloud platforms.

## Quick Start Deployment

### Docker Compose (Recommended for Development)

```bash
# Clone and setup
git clone https://github.com/ashish-frozo/VivaranAI.git
cd VivaranAI

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Deploy with Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8001/health
```

### Simple Server (Testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key-here"

# Run simple server
python simple_server.py

# Access at http://localhost:8001
```

## Production Deployment

### Prerequisites

- Kubernetes cluster (v1.20+)
- Docker registry access
- SSL certificates
- Domain name
- Monitoring stack (Prometheus/Grafana)

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key
LOG_LEVEL=INFO
ENVIRONMENT=production

# Optional
REDIS_URL=redis://redis-service:6379
CACHE_TTL=21600
MAX_FILE_SIZE=10485760
RATE_LIMIT=100
```

### Kubernetes Deployment

#### 1. Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

#### 2. Configure Secrets

```bash
# Create secret for API keys
kubectl create secret generic api-keys \
  --from-literal=openai-api-key="your-openai-key" \
  -n vivaranai

# Create TLS secret for HTTPS
kubectl create secret tls vivaranai-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n vivaranai
```

#### 3. Deploy Services

```bash
# Deploy ConfigMap
kubectl apply -f k8s/configmap.yaml

# Deploy PVC for persistent storage
kubectl apply -f k8s/pvc.yaml

# Deploy main application
kubectl apply -f k8s/deployment.yaml

# Deploy service
kubectl apply -f k8s/service.yaml

# Deploy HPA for auto-scaling
kubectl apply -f k8s/hpa.yaml
```

#### 4. Configure Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vivaranai-ingress
  namespace: vivaranai
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.vivaranai.com
    secretName: vivaranai-tls
  rules:
  - host: api.vivaranai.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vivaranai-service
            port:
              number: 80
```

### Cloud Platform Deployments

#### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster --name vivaranai-cluster --region us-east-1

# Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name vivaranai-cluster

# Deploy application
kubectl apply -f k8s/
```

#### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create vivaranai-cluster \
  --zone us-central1-a \
  --num-nodes 3

# Get credentials
gcloud container clusters get-credentials vivaranai-cluster --zone us-central1-a

# Deploy application
kubectl apply -f k8s/
```

#### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name vivaranai-cluster \
  --node-count 3 \
  --enable-addons monitoring

# Get credentials
az aks get-credentials --resource-group myResourceGroup --name vivaranai-cluster

# Deploy application
kubectl apply -f k8s/
```

## Monitoring & Observability

### Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vivaranai'
    static_configs:
      - targets: ['vivaranai-service:8001']
    metrics_path: /metrics
    scrape_interval: 30s
```

### Grafana Dashboard

Key metrics to monitor:
- Request rate and latency
- Error rates
- Processing time by document type
- Cache hit rate
- Memory and CPU usage
- Queue length

### Alerts

```yaml
# monitoring/alerts.yml
groups:
- name: vivaranai
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High latency detected
```

## Performance Optimization

### Horizontal Pod Autoscaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vivaranai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vivaranai-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Resource Limits

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Caching Strategy

```yaml
# Redis configuration
redis:
  enabled: true
  host: redis-service
  port: 6379
  ttl: 21600  # 6 hours
  maxmemory: 1gb
  maxmemory-policy: allkeys-lru
```

## Security Configuration

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vivaranai-netpol
spec:
  podSelector:
    matchLabels:
      app: vivaranai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8001
```

### Pod Security Standards

```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: vivaranai
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

## Backup & Recovery

### Database Backup

```bash
# Backup Redis data
kubectl exec redis-pod -- redis-cli BGSAVE

# Copy backup file
kubectl cp redis-pod:/data/dump.rdb ./backup/dump.rdb
```

### Configuration Backup

```bash
# Backup all configurations
kubectl get all,secrets,configmaps -n vivaranai -o yaml > backup/vivaranai-backup.yaml
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod status
kubectl get pods -n vivaranai

# Check pod logs
kubectl logs pod-name -n vivaranai

# Describe pod for events
kubectl describe pod pod-name -n vivaranai
```

#### 2. API Key Issues

```bash
# Verify secret exists
kubectl get secrets -n vivaranai

# Check secret content
kubectl get secret api-keys -n vivaranai -o yaml
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top pods -n vivaranai

# Check HPA status
kubectl get hpa -n vivaranai

# Check cluster events
kubectl get events -n vivaranai --sort-by='.lastTimestamp'
```

### Health Checks

```bash
# Application health
curl http://your-domain/health

# Kubernetes health
kubectl get componentstatuses

# Node health
kubectl get nodes

# Pod readiness
kubectl get pods -n vivaranai -o wide
```

## Scaling Guidelines

### Vertical Scaling

```yaml
# Increase resources for better performance
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment vivaranai-deployment --replicas=5 -n vivaranai

# Auto-scaling configuration
kubectl apply -f k8s/hpa.yaml
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f locustfile.py --host=http://your-domain
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t vivaranai:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        docker tag vivaranai:${{ github.sha }} your-registry/vivaranai:${{ github.sha }}
        docker push your-registry/vivaranai:${{ github.sha }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/vivaranai-deployment \
          vivaranai=your-registry/vivaranai:${{ github.sha }} \
          -n vivaranai
```

## Maintenance

### Updates

```bash
# Rolling update
kubectl set image deployment/vivaranai-deployment \
  vivaranai=new-image:tag -n vivaranai

# Check rollout status
kubectl rollout status deployment/vivaranai-deployment -n vivaranai

# Rollback if needed
kubectl rollout undo deployment/vivaranai-deployment -n vivaranai
```

### Cleanup

```bash
# Remove old ReplicaSets
kubectl delete rs $(kubectl get rs -n vivaranai -o jsonpath='{.items[?(@.spec.replicas==0)].metadata.name}') -n vivaranai

# Clean up unused images
docker system prune -a
```

---

## Support

For deployment issues:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [GitHub Issues](https://github.com/ashish-frozo/VivaranAI/issues)
- Contact support team

**Happy Deploying!** 🚀 