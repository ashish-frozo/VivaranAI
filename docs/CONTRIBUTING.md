# Contributing to VivaranAI MedBillGuardAgent 🤝

We love your input! We want to make contributing to VivaranAI as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, track issues and feature requests, and accept pull requests.

### Code Changes via Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- OpenAI API Key
- Docker (optional)

### Local Development

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/VivaranAI.git
cd VivaranAI

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export LOG_LEVEL="DEBUG"

# 6. Run tests to verify setup
python -m pytest tests/ -v
```

## Code Quality Standards

### Linting and Formatting

We use several tools to maintain code quality:

```bash
# Ruff for linting and formatting (replaces flake8, black, isort)
ruff check .
ruff format .

# Type checking with MyPy
mypy .

# Security scanning with Bandit
bandit -r .

# Run all quality checks
pre-commit run --all-files
```

### Testing Requirements

- **Minimum Coverage**: 90% test coverage required
- **Test Types**: Unit tests, integration tests, E2E tests
- **Test Structure**: Follow AAA pattern (Arrange, Act, Assert)

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=medbillguardagent --cov-report=html

# Run specific test categories
python -m pytest tests/test_medical_bill_agent.py -v
python -m pytest tests/test_duplicate_detector.py -v

# Performance testing
python -m pytest tests/test_performance.py -v
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Type Hints**: Full type annotation required
- **API Documentation**: Keep OpenAPI docs updated
- **README Updates**: Update relevant documentation for new features

Example docstring:
```python
def analyze_medical_bill(
    self, 
    file_content: bytes, 
    doc_id: str, 
    language: str = "english"
) -> Dict[str, Any]:
    """Analyze a medical bill for overcharges and issues.
    
    Args:
        file_content: Binary content of the medical bill file
        doc_id: Unique identifier for the document
        language: Language for OCR processing ("english", "hindi", etc.)
        
    Returns:
        Dictionary containing analysis results with keys:
        - success: Boolean indicating if analysis succeeded
        - verdict: String verdict ("normal", "overcharge_detected", etc.)
        - total_overcharge: Float amount of detected overcharge
        - confidence_score: Float confidence score (0-100)
        - red_flags: List of detected issues
        
    Raises:
        ValueError: If file_content is empty or invalid
        DocumentProcessingError: If OCR processing fails
        
    Example:
        >>> agent = MedicalBillAgent(api_key="sk-...")
        >>> with open("bill.pdf", "rb") as f:
        ...     result = await agent.analyze_medical_bill(
        ...         file_content=f.read(),
        ...         doc_id="bill_001"
        ...     )
        >>> print(result["verdict"])
        "overcharge_detected"
    """
```

## Project Architecture

### Directory Structure

```
VivaranAI/
├── agents/                     # Multi-agent system
│   ├── medical_bill_agent.py   # Medical bill analysis
│   ├── smart_data_agent.py     # AI web scraping
│   ├── router_agent.py         # Request routing
│   └── tools/                  # Specialized tools
├── medbillguardagent/          # Core analysis engine
├── tests/                      # Test suite
├── frontend/                   # Web interface
├── data/                       # Reference data
├── config/                     # Configuration files
├── docs/                       # Documentation
└── k8s/                        # Kubernetes manifests
```

### Key Design Principles

1. **Single Responsibility**: Each module has a clear, focused purpose
2. **Dependency Injection**: Use dependency injection for testability
3. **Error Handling**: Comprehensive error handling with custom exceptions
4. **Async/Await**: Use async patterns for I/O operations
5. **Configuration**: Environment-based configuration management
6. **Logging**: Structured logging with correlation IDs

### Adding New Features

#### 1. New Agent Development

When adding a new agent:

```python
from agents.base_agent import BaseAgent
from typing import Dict, Any

class NewAnalysisAgent(BaseAgent):
    """Agent for analyzing new document type."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize specific dependencies
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process task according to BaseAgent interface."""
        # Implementation here
        pass
    
    async def analyze_document(self, content: bytes) -> Dict[str, Any]:
        """Analyze document specific to this agent's domain."""
        # Implementation here
        pass
```

#### 2. New Tool Development

When adding a new tool:

```python
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class NewAnalysisTool:
    """Tool for specific analysis functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """Main tool execution method."""
        try:
            # Tool logic here
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}
```

#### 3. API Endpoint Development

When adding new endpoints:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

class NewAnalysisRequest(BaseModel):
    """Request model for new analysis endpoint."""
    document_id: str
    analysis_type: str
    options: Dict[str, Any] = {}

class NewAnalysisResponse(BaseModel):
    """Response model for new analysis endpoint."""
    success: bool
    analysis_id: str
    results: Dict[str, Any]

@router.post("/analyze-new", response_model=NewAnalysisResponse)
async def analyze_new_document(request: NewAnalysisRequest):
    """Analyze document with new analysis type."""
    try:
        # Implementation here
        return NewAnalysisResponse(
            success=True,
            analysis_id="analysis_123",
            results={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                       # Unit tests for individual components
├── integration/                # Integration tests for component interaction
├── e2e/                       # End-to-end tests for full workflows
├── performance/               # Performance and load tests
├── fixtures/                  # Test data and fixtures
└── conftest.py               # Pytest configuration and fixtures
```

### Writing Tests

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
from medbillguardagent.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create DocumentProcessor instance for testing."""
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Load sample PDF content for testing."""
        with open("tests/fixtures/sample_bill.pdf", "rb") as f:
            return f.read()
    
    async def test_process_pdf_success(self, processor, sample_pdf_content):
        """Test successful PDF processing."""
        # Arrange
        expected_text = "Sample medical bill content"
        
        # Act
        result = await processor.process_document(
            content=sample_pdf_content,
            file_format="pdf"
        )
        
        # Assert
        assert result["success"] is True
        assert "raw_text" in result
        assert len(result["raw_text"]) > 0
    
    async def test_process_invalid_format(self, processor):
        """Test processing with invalid file format."""
        # Arrange
        invalid_content = b"invalid content"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file format"):
            await processor.process_document(
                content=invalid_content,
                file_format="invalid"
            )
    
    @patch('medbillguardagent.document_processor.tesseract_ocr')
    async def test_ocr_failure_handling(self, mock_tesseract, processor):
        """Test graceful handling of OCR failures."""
        # Arrange
        mock_tesseract.side_effect = Exception("OCR failed")
        content = b"fake image content"
        
        # Act
        result = await processor.process_document(content, "jpg")
        
        # Assert
        assert result["success"] is False
        assert "error" in result
```

#### Integration Tests

```python
import pytest
from agents.medical_bill_agent import MedicalBillAgent

class TestMedicalBillAgentIntegration:
    """Integration tests for MedicalBillAgent."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        return MedicalBillAgent(
            openai_api_key="test-key",
            cache_manager=Mock(),
            document_processor=Mock()
        )
    
    async def test_full_analysis_workflow(self, agent):
        """Test complete bill analysis workflow."""
        # This would test the integration between components
        pass
```

#### E2E Tests

```python
import pytest
import httpx
from fastapi.testclient import TestClient
from agents.server import app

class TestE2EWorkflow:
    """End-to-end tests for complete workflows."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_complete_bill_analysis(self, client):
        """Test complete bill analysis from API call to response."""
        # Prepare test data
        with open("tests/fixtures/sample_bill.pdf", "rb") as f:
            file_content = f.read()
        
        # Make API call
        response = client.post("/analyze", json={
            "file_content": base64.b64encode(file_content).decode(),
            "filename": "test_bill.pdf",
            "language": "english",
            "insurance_type": "cghs"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "analysis_id" in data
        assert "verdict" in data
```

### Performance Testing

```python
import pytest
import time
from locust import HttpUser, task, between

class BillAnalysisUser(HttpUser):
    """Load testing user for bill analysis."""
    
    wait_time = between(1, 3)
    
    @task
    def analyze_bill(self):
        """Simulate bill analysis request."""
        with open("tests/fixtures/sample_bill.pdf", "rb") as f:
            content = f.read()
        
        response = self.client.post("/analyze", json={
            "file_content": base64.b64encode(content).decode(),
            "filename": "load_test_bill.pdf"
        })
        
        assert response.status_code == 200

@pytest.mark.performance
class TestPerformance:
    """Performance tests with timing constraints."""
    
    @pytest.mark.timeout(30)
    async def test_analysis_performance(self):
        """Test analysis completes within time limit."""
        start_time = time.time()
        
        # Run analysis
        result = await analyze_bill(sample_content)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert performance requirements
        assert processing_time < 30.0  # 30 second max
        assert result["success"] is True
```

## Security Guidelines

### Security Checklist

- [ ] **Input Validation**: Validate all inputs, especially file uploads
- [ ] **API Key Management**: Never commit API keys to version control
- [ ] **SQL Injection**: Use parameterized queries (though we use NoSQL)
- [ ] **XSS Prevention**: Sanitize user inputs in frontend
- [ ] **CSRF Protection**: Implement CSRF tokens for state-changing operations
- [ ] **Rate Limiting**: Implement rate limiting on API endpoints
- [ ] **Authentication**: Secure API endpoints with proper authentication
- [ ] **Error Handling**: Don't expose sensitive information in error messages

### Secure Coding Practices

```python
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Good: Environment variable for API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable required")

# Good: Input validation
def validate_file_content(content: bytes, max_size: int = 10_000_000) -> bool:
    """Validate uploaded file content."""
    if not content:
        raise ValueError("File content cannot be empty")
    
    if len(content) > max_size:
        raise ValueError(f"File too large: {len(content)} bytes > {max_size}")
    
    return True

# Good: Secure error handling
async def process_document(content: bytes) -> Dict[str, Any]:
    """Process document with secure error handling."""
    try:
        validate_file_content(content)
        result = await internal_processing(content)
        return {"success": True, "result": result}
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return {"success": False, "error": "Invalid input"}
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return {"success": False, "error": "Processing failed"}
```

## Release Process

### Version Management

We use semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. [ ] Update version in `pyproject.toml`
2. [ ] Update `CHANGELOG.md`
3. [ ] Run full test suite
4. [ ] Update documentation
5. [ ] Create release PR
6. [ ] Tag release after merge
7. [ ] Deploy to staging
8. [ ] Deploy to production
9. [ ] Monitor post-deployment

### Git Workflow

```bash
# Feature development
git checkout -b feature/new-analysis-type
git commit -m "feat: add new analysis type for insurance claims"
git push origin feature/new-analysis-type

# Bug fixes
git checkout -b fix/duplicate-detection-accuracy
git commit -m "fix: improve duplicate detection accuracy by 5%"
git push origin fix/duplicate-detection-accuracy

# Hotfixes
git checkout -b hotfix/critical-security-fix
git commit -m "fix: patch security vulnerability in file upload"
git push origin hotfix/critical-security-fix
```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

Examples:
```
feat(agents): add smart data agent for web scraping
fix(ocr): resolve text extraction issues with Hindi documents
docs(api): update API documentation for new endpoints
test(integration): add comprehensive E2E test suite
```

## Getting Help

### Community Resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/ashish-frozo/VivaranAI/issues)
- **GitHub Discussions**: [Ask questions and discuss ideas](https://github.com/ashish-frozo/VivaranAI/discussions)
- **Documentation**: [Full project documentation](../README.md)

### Development Support

If you're stuck or need help:

1. Check existing issues and discussions
2. Review the documentation
3. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Relevant code snippets or logs

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to VivaranAI! 🙏

---

**Happy Coding!** 🚀 