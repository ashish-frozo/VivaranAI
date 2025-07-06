# Environment Setup Guide

This guide explains how to configure VivaranAI using environment variables and `.env` files for secure and flexible deployment.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp env.example .env
   ```

2. **Edit the .env file with your settings:**
   ```bash
   nano .env
   ```

3. **Add your OpenAI API key:**
   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

4. **Run the server:**
   ```bash
   python simple_server.py
   ```

## Environment Configuration

### 📁 File Structure

```
VivaranAI/
├── env.example          # Template file with all options
├── .env                 # Your actual configuration (git-ignored)
├── config/
│   └── env_config.py    # Configuration loading module
└── simple_server.py     # Main server using config
```

### 🔧 Configuration Options

#### Required Settings

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key for GPT-4 | ✅ Yes |

#### Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8001` | Server port number |
| `DEBUG` | `false` | Enable debug mode |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `MAX_WORKERS` | `4` | Maximum worker threads |
| `TIMEOUT_SECONDS` | `30` | Request timeout |
| `API_RATE_LIMIT` | `100` | API rate limit per minute |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `agent_server.log` | Log file path |

### 🔐 Security Features

#### Automatic Secret Detection
- **GitHub Push Protection**: Prevents hardcoded API keys in commits
- **Environment Isolation**: API keys only in `.env` files (git-ignored)
- **Validation**: Automatic checks for missing or placeholder values

#### Configuration Validation
```python
from config.env_config import check_required_config

if not check_required_config():
    print("Configuration error!")
    exit(1)
```

## Usage Examples

### 🚀 Server Startup

**Simple Server:**
```bash
# Uses .env file automatically
python simple_server.py
```

**With Custom Environment:**
```bash
# Load from custom file
ENVIRONMENT_FILE=.env.production python simple_server.py
```

### 🧪 Demo Scripts

**Smart Data Agent Demo:**
```bash
python demo_smart_data_agent.py
```

**Dashboard Startup:**
```bash
./start_dashboard.sh
```

### 🐍 Python Code Usage

```python
from config.env_config import config

# Access configuration values
api_key = config.openai_api_key
port = config.port
debug_mode = config.debug

# Create agent with config
agent = MedicalBillAgent(openai_api_key=config.openai_api_key)
```

## Deployment Scenarios

### 🏠 Local Development

1. Copy template: `cp env.example .env`
2. Add your API key to `.env`
3. Run: `python simple_server.py`

### 🐳 Docker Deployment

```dockerfile
# In Dockerfile
COPY env.example /app/.env.example

# At runtime
docker run -e OPENAI_API_KEY=your-key vivaranai
```

### ☁️ Cloud Deployment

**Environment Variables (Recommended):**
```bash
export OPENAI_API_KEY="your-key"
export PORT="8080"
export DEBUG="false"
```

**Cloud Platform Examples:**

**Heroku:**
```bash
heroku config:set OPENAI_API_KEY=your-key
```

**AWS ECS:**
```json
{
  "environment": [
    {"name": "OPENAI_API_KEY", "value": "your-key"},
    {"name": "PORT", "value": "8080"}
  ]
}
```

**Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: vivaranai-secrets
data:
  openai-api-key: <base64-encoded-key>
```

### 🔧 Production Configuration

**Production .env example:**
```env
# Production settings
OPENAI_API_KEY=sk-your-production-key
HOST=0.0.0.0
PORT=8080
DEBUG=false
MAX_WORKERS=8
TIMEOUT_SECONDS=60
API_RATE_LIMIT=200
LOG_LEVEL=WARNING
REDIS_URL=redis://redis-cluster:6379
```

## Troubleshooting

### ❌ Common Errors

**"OPENAI_API_KEY is required"**
- Solution: Copy `env.example` to `.env` and add your API key

**"Configuration error"**
- Check: `.env` file exists and contains valid settings
- Verify: API key is not the placeholder value

**"python-dotenv not installed"**
- Install: `pip install python-dotenv`

### 🔍 Debug Configuration

**Test your configuration:**
```bash
python config/env_config.py
```

**Expected output:**
```
✅ Loaded environment from: /path/to/.env
✅ Configuration validated successfully
🌐 Server will run on 0.0.0.0:8001
📊 Log level: INFO
🔗 Redis URL: redis://localhost:6379
```

### 📝 Configuration Hierarchy

1. **Environment variables** (highest priority)
2. **`.env` file** 
3. **Default values** (lowest priority)

This allows flexible overrides in different deployment scenarios.

## Migration from Hardcoded Values

If upgrading from an older version:

1. **Remove hardcoded API keys** from code
2. **Create `.env` file** from template
3. **Update deployment scripts** to use environment variables
4. **Test configuration loading** before deployment

## Best Practices

### ✅ Do
- Use `.env` files for local development
- Use environment variables in production
- Keep API keys out of source code
- Use different configurations for different environments

### ❌ Don't
- Commit `.env` files to git
- Hardcode API keys in source code
- Use development keys in production
- Share `.env` files between developers

---

**Need help?** Check the [main README](../README.md) or create an issue on GitHub. 