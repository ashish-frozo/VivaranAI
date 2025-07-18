# VivaranAI Multi-Vertical Architecture

## System Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend"
        A[Web Dashboard] --> B[API Gateway]
    end

    subgraph "API Layer"
        B --> C[FastAPI Backend]
        C --> D[Authentication]
        C --> E[Document Upload]
    end

    subgraph "Router Layer"
        E --> F[RouterAgent]
        F --> G{Document Classification}
        G -->|Medical Bill| H[Medical Domain]
        G -->|Loan Document| I[Loan Domain]
        G -->|Insurance Claim| J[Insurance Domain]
    end

    subgraph "Medical Domain"
        H --> K[MedicalBillAgent]
        K --> L[RateValidatorTool]
        K --> M[DuplicateDetectorTool]
        K --> N[ProhibitedItemTool]
        L --> O[SmartDataTool]
        O --> P[AI Web Scraper]
    end

    subgraph "Loan Domain"
        I --> Q[LoanRiskAgent]
        Q --> R[LoanPackTool]
        Q --> S[DuplicateDetectorTool]
        Q --> T[ComplianceCheckerTool]
    end

    subgraph "Insurance Domain"
        J --> U[InsuranceClaimAgent]
        U --> V[ClaimValidatorTool]
        U --> W[DuplicateDetectorTool]
    end

    subgraph "Shared Infrastructure"
        X[ToolManager]
        Y[ProductionIntegration]
        Z[HorizontalScalingManager]
        AA[LoadBalancer]
        AB[CircuitBreaker]
        AC[MetricsCollector]
        
        X --> Y
        Y --> Z
        Y --> AA
        Y --> AB
        Y --> AC
    end

    subgraph "Data Layer"
        AD[PostgreSQL]
        AE[Redis Cache]
        AF[Rule Packs]
        AG[Golden Fixtures]
        
        AF -->|Medical Rules| K
        AF -->|Loan Rules| Q
        AF -->|Insurance Rules| U
        AG -->|Regression Tests| AH
    end

    subgraph "Testing & CI/CD"
        AH[Test Suite]
        AI[CI Pipeline]
        AJ[Deployment]
        
        AH --> AI
        AI --> AJ
    end
```

## Multi-Vertical Pack-Driven Architecture

The VivaranAI system implements a sophisticated multi-vertical, pack-driven architecture that enables seamless analysis across different document domains:

### Key Components

1. **RouterAgent**: Central intelligence that:
   - Performs document classification
   - Routes documents to appropriate domain agents
   - Orchestrates cross-vertical workflows
   - Maintains health checks and agent registry

2. **Domain Agents**:
   - **MedicalBillAgent**: Specialized for medical bill analysis
   - **LoanRiskAgent**: Handles loan application risk assessment
   - **InsuranceClaimAgent**: Processes insurance claims

3. **Rule Packs**:
   - Domain-specific rule collections in YAML format
   - Centralized validation logic separate from agent code
   - Versioned and easily updatable without code changes

4. **Shared Infrastructure**:
   - **ToolManager**: Centralized tool registration and management
   - **ProductionIntegration**: Coordinates horizontal scaling and load balancing
   - **HorizontalScalingManager**: Manages tool instance pools
   - **LoadBalancer**: Intelligent request routing with health awareness

5. **Testing Framework**:
   - Golden fixtures for regression testing
   - Parametrized pytest cases
   - CI recall checks for production readiness

### Data Flow

1. Document is uploaded through the API
2. RouterAgent classifies the document type
3. Document is routed to the appropriate domain agent
4. Domain agent loads relevant rule packs
5. Analysis is performed using domain-specific tools
6. Results are returned to the user

### Cross-Vertical Workflows

The system supports complex workflows that span multiple domains:

1. RouterAgent orchestrates the workflow
2. Steps can be sequential or parallel
3. Data can flow between different domain agents
4. Results are aggregated and returned to the user

### Production Infrastructure

- Horizontal scaling for high-demand tools
- Health-aware load balancing
- Circuit breakers for resilience
- Comprehensive metrics collection
- Automated testing and deployment
