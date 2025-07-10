"""
Authentication and Authorization Middleware for VivaranAI Production

Implements:
- API Key authentication for external clients
- JWT token validation for user sessions
- Role-based access control (RBAC)
- Request rate limiting per user/API key
"""

import os
from jose import jwt
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from functools import wraps

import redis.asyncio as redis
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

logger = structlog.get_logger(__name__)

# User roles and permissions
class Role:
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    VIEWER = "viewer"

class Permission:
    ANALYZE_BILLS = "analyze:bills"
    VIEW_ANALYTICS = "view:analytics"
    MANAGE_USERS = "manage:users"
    ACCESS_ADMIN = "access:admin"
    BULK_PROCESS = "bulk:process"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.ANALYZE_BILLS,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_USERS,
        Permission.ACCESS_ADMIN,
        Permission.BULK_PROCESS
    },
    Role.USER: {
        Permission.ANALYZE_BILLS,
        Permission.VIEW_ANALYTICS
    },
    Role.API_CLIENT: {
        Permission.ANALYZE_BILLS,
        Permission.BULK_PROCESS
    },
    Role.VIEWER: {
        Permission.VIEW_ANALYTICS
    }
}

class AuthConfig:
    """Authentication configuration"""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET_KEY")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
        self.api_key_header = os.getenv("API_KEY_HEADER", "X-API-Key")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/3")
        
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET_KEY environment variable is required")

auth_config = AuthConfig()
security = HTTPBearer(auto_error=False)

class AuthenticationError(HTTPException):
    """Authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

class AuthorizationError(HTTPException):
    """Authorization error"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

class User:
    """User model for authentication"""
    
    def __init__(self, user_id: str, email: str, role: str, permissions: Set[str]):
        self.user_id = user_id
        self.email = email
        self.role = role
        self.permissions = permissions
        self.authenticated_at = datetime.utcnow()

class APIKey:
    """API Key model for external clients"""
    
    def __init__(self, api_key: str, client_name: str, role: str, permissions: Set[str]):
        self.api_key = api_key
        self.client_name = client_name
        self.role = role
        self.permissions = permissions
        self.last_used = datetime.utcnow()

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._api_keys_cache: Dict[str, APIKey] = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(auth_config.redis_url)
            await self.redis_client.ping()
            logger.info("AuthManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AuthManager: {e}")
            raise
    
    async def create_jwt_token(self, user: User) -> str:
        """Create JWT token for user"""
        payload = {
            "user_id": user.user_id,
            "email": user.email,
            "role": user.role,
            "permissions": list(user.permissions),
            "exp": datetime.utcnow() + timedelta(minutes=auth_config.jwt_expire_minutes),
            "iat": datetime.utcnow(),
            "type": "access_token"
        }
        
        token = jwt.encode(payload, auth_config.jwt_secret, algorithm=auth_config.jwt_algorithm)
        
        # Store token in Redis for revocation capability
        await self.redis_client.setex(
            f"jwt_token:{user.user_id}:{hashlib.md5(token.encode()).hexdigest()}",
            auth_config.jwt_expire_minutes * 60,
            "valid"
        )
        
        return token
    
    async def verify_jwt_token(self, token: str) -> User:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, auth_config.jwt_secret, algorithms=[auth_config.jwt_algorithm])
            
            # Check if token is revoked
            token_hash = hashlib.md5(token.encode()).hexdigest()
            token_status = await self.redis_client.get(f"jwt_token:{payload['user_id']}:{token_hash}")
            
            if not token_status or token_status.decode() != "valid":
                raise AuthenticationError("Token has been revoked")
            
            return User(
                user_id=payload["user_id"],
                email=payload["email"],
                role=payload["role"],
                permissions=set(payload["permissions"])
            )
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    async def verify_api_key(self, api_key: str) -> APIKey:
        """Verify API key"""
        # Check cache first
        if api_key in self._api_keys_cache:
            cached_key = self._api_keys_cache[api_key]
            if (datetime.utcnow() - cached_key.last_used).seconds < self._cache_ttl:
                cached_key.last_used = datetime.utcnow()
                return cached_key
        
        # Check Redis
        key_data = await self.redis_client.hgetall(f"api_key:{api_key}")
        if not key_data:
            raise AuthenticationError("Invalid API key")
        
        # Convert bytes to strings
        key_data = {k.decode(): v.decode() for k, v in key_data.items()}
        
        api_key_obj = APIKey(
            api_key=api_key,
            client_name=key_data["client_name"],
            role=key_data["role"],
            permissions=set(key_data["permissions"].split(","))
        )
        
        # Update cache
        self._api_keys_cache[api_key] = api_key_obj
        
        # Update last used timestamp
        await self.redis_client.hset(f"api_key:{api_key}", "last_used", datetime.utcnow().isoformat())
        
        return api_key_obj
    
    async def revoke_jwt_token(self, user_id: str, token: str):
        """Revoke JWT token"""
        token_hash = hashlib.md5(token.encode()).hexdigest()
        await self.redis_client.delete(f"jwt_token:{user_id}:{token_hash}")
    
    async def create_api_key(self, client_name: str, role: str) -> str:
        """Create new API key for client"""
        import secrets
        
        api_key = f"viva_{secrets.token_urlsafe(32)}"
        permissions = ROLE_PERMISSIONS.get(role, set())
        
        key_data = {
            "client_name": client_name,
            "role": role,
            "permissions": ",".join(permissions),
            "created_at": datetime.utcnow().isoformat(),
            "last_used": datetime.utcnow().isoformat()
        }
        
        await self.redis_client.hset(f"api_key:{api_key}", mapping=key_data)
        
        logger.info(f"Created API key for client: {client_name}")
        return api_key

# Global auth manager instance
auth_manager = AuthManager()

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current authenticated user from JWT token or API key"""
    
    # Try API Key authentication first
    api_key = request.headers.get(auth_config.api_key_header)
    if api_key:
        try:
            api_key_obj = await auth_manager.verify_api_key(api_key)
            # Convert API key to User-like object
            return User(
                user_id=f"api_client_{api_key_obj.client_name}",
                email=f"{api_key_obj.client_name}@api.vivaranai.com",
                role=api_key_obj.role,
                permissions=api_key_obj.permissions
            )
        except AuthenticationError:
            pass
    
    # Try JWT authentication
    if credentials:
        try:
            return await auth_manager.verify_jwt_token(credentials.credentials)
        except AuthenticationError:
            pass
    
    return None

def require_auth(permissions: Optional[List[str]] = None):
    """Decorator to require authentication and optional permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from dependency injection
            user = kwargs.get('current_user')
            if not user:
                raise AuthenticationError("Authentication required")
            
            # Check permissions if specified
            if permissions:
                missing_permissions = set(permissions) - user.permissions
                if missing_permissions:
                    raise AuthorizationError(
                        f"Missing required permissions: {', '.join(missing_permissions)}"
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(required_role: str):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if not user:
                raise AuthenticationError("Authentication required")
            
            if user.role != required_role:
                raise AuthorizationError(f"Required role: {required_role}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting per user
async def check_rate_limit(user: User, endpoint: str, limit: int = 100) -> bool:
    """Check rate limit for user/endpoint combination"""
    if not auth_manager.redis_client:
        return True  # Skip rate limiting if Redis unavailable
    
    key = f"rate_limit:{user.user_id}:{endpoint}:{int(time.time() // 60)}"  # Per minute
    
    current_count = await auth_manager.redis_client.get(key)
    if current_count and int(current_count) >= limit:
        return False
    
    # Increment counter
    pipe = auth_manager.redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)  # Expire after 1 minute
    await pipe.execute()
    
    return True 