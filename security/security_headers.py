"""
Security Headers Middleware for VivaranAI Production

Implements comprehensive security headers for production deployment:
- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- X-Frame-Options, X-Content-Type-Options
- Referrer Policy and Permissions Policy
- HTTPS redirect enforcement
"""

import os
import time
from typing import Dict, List, Optional, Callable
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)

class SecurityConfig:
    """Security configuration"""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.force_https = os.getenv("FORCE_HTTPS", "true").lower() == "true"
        self.allowed_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
        self.csp_report_uri = os.getenv("CSP_REPORT_URI", "")
        self.hsts_max_age = int(os.getenv("HSTS_MAX_AGE", "31536000"))  # 1 year
        
        # Content Security Policy directives
        self.csp_directives = {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
            "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'", "https://fonts.gstatic.com"],
            "connect-src": ["'self'", "https://api.openai.com"],
            "frame-ancestors": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
            "upgrade-insecure-requests": []
        }
        
        # Override CSP for development
        if self.environment == "development":
            self.csp_directives["script-src"].append("'unsafe-eval'")
            self.csp_directives["connect-src"].extend(["http://localhost:*", "ws://localhost:*"])

security_config = SecurityConfig()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    def __init__(self, app):
        super().__init__(app)
        self.config = security_config
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add security headers to response"""
        
        # Check if HTTPS is enforced
        if self.config.force_https and self.config.environment == "production":
            if not self._is_secure_request(request):
                return self._redirect_to_https(request)
        
        # Validate host header
        if not self._is_allowed_host(request):
            logger.warning(f"Blocked request from unauthorized host: {request.headers.get('host')}")
            return Response(
                content="Forbidden", 
                status_code=403,
                headers={"Content-Type": "text/plain"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, request)
        
        return response
    
    def _is_secure_request(self, request: Request) -> bool:
        """Check if request is secure (HTTPS)"""
        # Check various headers that indicate HTTPS
        if request.url.scheme == "https":
            return True
        
        # Check proxy headers
        forwarded_proto = request.headers.get("x-forwarded-proto")
        if forwarded_proto and forwarded_proto.lower() == "https":
            return True
        
        # Check for load balancer headers
        if request.headers.get("x-forwarded-ssl") == "on":
            return True
        
        return False
    
    def _is_allowed_host(self, request: Request) -> bool:
        """Check if host is allowed"""
        if not self.config.allowed_hosts or not self.config.allowed_hosts[0]:
            return True  # No restrictions configured
        
        host = request.headers.get("host", "").lower()
        if not host:
            return False
        
        # Remove port if present
        if ":" in host:
            host = host.split(":")[0]
        
        return host in [h.strip().lower() for h in self.config.allowed_hosts]
    
    def _redirect_to_https(self, request: Request) -> Response:
        """Redirect HTTP to HTTPS"""
        https_url = str(request.url).replace("http://", "https://", 1)
        return Response(
            status_code=301,
            headers={
                "Location": https_url,
                "Strict-Transport-Security": f"max-age={self.config.hsts_max_age}; includeSubDomains"
            }
        )
    
    def _add_security_headers(self, response: Response, request: Request):
        """Add comprehensive security headers"""
        
        # Content Security Policy
        csp_header = self._build_csp_header()
        response.headers["Content-Security-Policy"] = csp_header
        
        # HTTP Strict Transport Security (HSTS)
        if self._is_secure_request(request):
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.config.hsts_max_age}; includeSubDomains; preload"
            )
        
        # X-Frame-Options
        response.headers["X-Frame-Options"] = "DENY"
        
        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=(), "
            "usb=(), magnetometer=(), gyroscope=(), accelerometer=()"
        )
        
        # Cross-Origin policies
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        # Server identification
        response.headers["Server"] = "VivaranAI"
        
        # Cache control for sensitive endpoints
        if self._is_sensitive_endpoint(request):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # Custom security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        
        # API-specific headers
        if request.url.path.startswith("/api/"):
            response.headers["X-API-Version"] = "1.0"
            response.headers["X-RateLimit-Limit"] = "100"
    
    def _build_csp_header(self) -> str:
        """Build Content Security Policy header"""
        csp_parts = []
        
        for directive, sources in self.config.csp_directives.items():
            if sources:
                csp_parts.append(f"{directive} {' '.join(sources)}")
            else:
                csp_parts.append(directive)
        
        # Add report URI if configured
        if self.config.csp_report_uri:
            csp_parts.append(f"report-uri {self.config.csp_report_uri}")
        
        return "; ".join(csp_parts)
    
    def _is_sensitive_endpoint(self, request: Request) -> bool:
        """Check if endpoint contains sensitive data"""
        sensitive_paths = [
            "/api/analyze",
            "/api/admin",
            "/api/auth",
            "/api/users",
            "/api/keys"
        ]
        
        return any(request.url.path.startswith(path) for path in sensitive_paths)

class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for input validation and sanitization"""
    
    def __init__(self, app):
        super().__init__(app)
        self.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", "10485760"))  # 10MB
        self.blocked_user_agents = [
            "python-requests",  # Block basic automated requests
            "curl",
            "wget",
            "nikto",
            "sqlmap",
            "nmap"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate and sanitize request"""
        
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return Response(
                content="Request too large",
                status_code=413,
                headers={"Content-Type": "text/plain"}
            )
        
        # Check user agent
        user_agent = request.headers.get("user-agent", "").lower()
        if any(blocked_ua in user_agent for blocked_ua in self.blocked_user_agents):
            # Allow if it's from a legitimate source with API key
            if not request.headers.get("x-api-key"):
                logger.warning(f"Blocked suspicious user agent: {user_agent}")
                return Response(
                    content="Forbidden",
                    status_code=403,
                    headers={"Content-Type": "text/plain"}
                )
        
        # Validate file uploads
        if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
            if not self._is_valid_file_upload(request):
                return Response(
                    content="Invalid file upload",
                    status_code=400,
                    headers={"Content-Type": "text/plain"}
                )
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _is_valid_file_upload(self, request: Request) -> bool:
        """Validate file upload requests"""
        # This is a basic check - in production, you'd want more sophisticated validation
        allowed_types = ["image/jpeg", "image/png", "image/gif", "application/pdf"]
        
        # Check if it's an allowed endpoint
        if not request.url.path.startswith("/api/analyze"):
            return False
        
        return True

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.redis_client = None
        self.default_rate_limit = int(os.getenv("DEFAULT_RATE_LIMIT", "100"))
        self.burst_limit = int(os.getenv("BURST_RATE_LIMIT", "20"))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting"""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics", "/ready"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, request):
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Content-Type": "text/plain",
                    "Retry-After": "60"
                }
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try API key first
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host
        
        # Check for forwarded IP
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, client_id: str, request: Request) -> bool:
        """Check if client is within rate limits"""
        if not self.redis_client:
            return True  # Skip if Redis unavailable
        
        current_minute = int(time.time() // 60)
        key = f"rate_limit:{client_id}:{current_minute}"
        
        try:
            current_count = await self.redis_client.get(key)
            if current_count and int(current_count) >= self.default_rate_limit:
                return False
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60)
            await pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow request if Redis fails

# Security middleware factory
def create_security_middleware():
    """Create and configure security middleware stack"""
    return [
        SecurityHeadersMiddleware,
        InputValidationMiddleware,
        RateLimitMiddleware
    ] 