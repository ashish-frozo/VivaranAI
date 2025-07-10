"""
OAuth2 Authentication Service for VivaranAI

Supports Google and GitHub OAuth2 authentication
"""

import os
import uuid
from typing import Dict, Optional, Any, Union
from datetime import datetime, timedelta
from urllib.parse import urlencode
import secrets
import hashlib

from authlib.integrations.starlette_client import OAuth
from authlib.integrations.base_client import OAuthError
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
import httpx
import structlog

from security.auth_middleware import auth_manager, User
from database.models import User as UserModel, UserRole

logger = structlog.get_logger(__name__)

# OAuth2 configuration
class OAuth2Config:
    """OAuth2 configuration"""
    
    def __init__(self):
        # Google OAuth2
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8001/auth/google/callback")
        
        # GitHub OAuth2
        self.github_client_id = os.getenv("GITHUB_CLIENT_ID")
        self.github_client_secret = os.getenv("GITHUB_CLIENT_SECRET")
        self.github_redirect_uri = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8001/auth/github/callback")
        
        # General OAuth2 settings
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.oauth_state_secret = os.getenv("OAUTH_STATE_SECRET", secrets.token_urlsafe(32))
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate OAuth2 configuration"""
        if not self.google_client_id or not self.google_client_secret:
            logger.warning("Google OAuth2 not configured - Google login will be disabled")
        
        if not self.github_client_id or not self.github_client_secret:
            logger.warning("GitHub OAuth2 not configured - GitHub login will be disabled")
        
        if not self.google_client_id and not self.github_client_id:
            logger.warning("No OAuth2 providers configured - OAuth login will be disabled")

oauth2_config = OAuth2Config()

# OAuth2 client models
class OAuth2UserInfo(BaseModel):
    """OAuth2 user information"""
    id: str
    email: EmailStr
    name: str
    picture: Optional[str] = None
    provider: str
    raw_data: Dict[str, Any] = {}

class OAuth2State(BaseModel):
    """OAuth2 state for CSRF protection"""
    state: str
    provider: str
    redirect_url: Optional[str] = None
    created_at: datetime

# OAuth2 service
class OAuth2Service:
    """OAuth2 authentication service"""
    
    def __init__(self):
        self.oauth = OAuth()
        self.state_storage: Dict[str, OAuth2State] = {}
        self._setup_oauth_clients()
    
    def _setup_oauth_clients(self):
        """Setup OAuth2 clients"""
        
        # Google OAuth2
        if oauth2_config.google_client_id and oauth2_config.google_client_secret:
            self.oauth.register(
                name='google',
                client_id=oauth2_config.google_client_id,
                client_secret=oauth2_config.google_client_secret,
                server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
                client_kwargs={
                    'scope': 'openid email profile'
                }
            )
            logger.info("Google OAuth2 client configured")
        
        # GitHub OAuth2
        if oauth2_config.github_client_id and oauth2_config.github_client_secret:
            self.oauth.register(
                name='github',
                client_id=oauth2_config.github_client_id,
                client_secret=oauth2_config.github_client_secret,
                access_token_url='https://github.com/login/oauth/access_token',
                authorize_url='https://github.com/login/oauth/authorize',
                api_base_url='https://api.github.com/',
                client_kwargs={'scope': 'user:email'},
            )
            logger.info("GitHub OAuth2 client configured")
    
    def generate_state(self, provider: str, redirect_url: Optional[str] = None) -> str:
        """Generate OAuth2 state for CSRF protection"""
        state = secrets.token_urlsafe(32)
        
        oauth2_state = OAuth2State(
            state=state,
            provider=provider,
            redirect_url=redirect_url,
            created_at=datetime.utcnow()
        )
        
        self.state_storage[state] = oauth2_state
        
        # Clean up old states (older than 1 hour)
        self._cleanup_old_states()
        
        return state
    
    def validate_state(self, state: str, provider: str) -> bool:
        """Validate OAuth2 state"""
        if state not in self.state_storage:
            return False
        
        oauth2_state = self.state_storage[state]
        
        # Check if state matches provider
        if oauth2_state.provider != provider:
            return False
        
        # Check if state is not expired (1 hour)
        if (datetime.utcnow() - oauth2_state.created_at).total_seconds() > 3600:
            del self.state_storage[state]
            return False
        
        return True
    
    def get_redirect_url(self, state: str) -> Optional[str]:
        """Get redirect URL from state"""
        if state in self.state_storage:
            return self.state_storage[state].redirect_url
        return None
    
    def _cleanup_old_states(self):
        """Clean up old OAuth2 states"""
        current_time = datetime.utcnow()
        expired_states = [
            state for state, oauth2_state in self.state_storage.items()
            if (current_time - oauth2_state.created_at).total_seconds() > 3600
        ]
        
        for state in expired_states:
            del self.state_storage[state]
    
    async def get_authorization_url(self, request: Request, provider: str, redirect_url: Optional[str] = None) -> str:
        """Get OAuth2 authorization URL"""
        
        if provider not in ['google', 'github']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported OAuth2 provider: {provider}"
            )
        
        # Check if provider is configured
        client = getattr(self.oauth, provider, None)
        if not client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{provider.title()} OAuth2 not configured"
            )
        
        # Generate state for CSRF protection
        state = self.generate_state(provider, redirect_url)
        
        # Get redirect URI
        redirect_uri = (
            oauth2_config.google_redirect_uri if provider == 'google' 
            else oauth2_config.github_redirect_uri
        )
        
        # Generate authorization URL
        try:
            authorization_url = await client.authorize_redirect(request, redirect_uri, state=state)
            return authorization_url.headers['location']
        except Exception as e:
            logger.error(f"Failed to generate authorization URL for {provider}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate authorization URL for {provider}"
            )
    
    async def handle_callback(self, request: Request, provider: str) -> OAuth2UserInfo:
        """Handle OAuth2 callback"""
        
        if provider not in ['google', 'github']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported OAuth2 provider: {provider}"
            )
        
        # Get state from query parameters
        state = request.query_params.get('state')
        if not state or not self.validate_state(state, provider):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OAuth2 state"
            )
        
        # Get OAuth2 client
        client = getattr(self.oauth, provider, None)
        if not client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{provider.title()} OAuth2 not configured"
            )
        
        try:
            # Exchange code for token
            token = await client.authorize_access_token(request)
            
            # Get user info
            if provider == 'google':
                user_info = await self._get_google_user_info(token)
            elif provider == 'github':
                user_info = await self._get_github_user_info(token)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported provider: {provider}"
                )
            
            # Clean up state
            if state in self.state_storage:
                del self.state_storage[state]
            
            return user_info
            
        except OAuthError as e:
            logger.error(f"OAuth2 error for {provider}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OAuth2 authentication failed: {e.description}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during {provider} callback: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication failed"
            )
    
    async def _get_google_user_info(self, token: Dict[str, Any]) -> OAuth2UserInfo:
        """Get Google user information"""
        
        # Get user info from Google
        async with httpx.AsyncClient() as client:
            response = await client.get(
                'https://www.googleapis.com/oauth2/v2/userinfo',
                headers={'Authorization': f'Bearer {token["access_token"]}'}
            )
            response.raise_for_status()
            user_data = response.json()
        
        return OAuth2UserInfo(
            id=user_data['id'],
            email=user_data['email'],
            name=user_data['name'],
            picture=user_data.get('picture'),
            provider='google',
            raw_data=user_data
        )
    
    async def _get_github_user_info(self, token: Dict[str, Any]) -> OAuth2UserInfo:
        """Get GitHub user information"""
        
        async with httpx.AsyncClient() as client:
            # Get user info
            user_response = await client.get(
                'https://api.github.com/user',
                headers={'Authorization': f'Bearer {token["access_token"]}'}
            )
            user_response.raise_for_status()
            user_data = user_response.json()
            
            # Get user email (GitHub doesn't always provide email in user endpoint)
            email = user_data.get('email')
            if not email:
                email_response = await client.get(
                    'https://api.github.com/user/emails',
                    headers={'Authorization': f'Bearer {token["access_token"]}'}
                )
                email_response.raise_for_status()
                emails = email_response.json()
                
                # Get primary email
                primary_email = next(
                    (e['email'] for e in emails if e['primary']), 
                    emails[0]['email'] if emails else None
                )
                email = primary_email
            
            if not email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="GitHub account must have a verified email address"
                )
        
        return OAuth2UserInfo(
            id=str(user_data['id']),
            email=email,
            name=user_data['name'] or user_data['login'],
            picture=user_data.get('avatar_url'),
            provider='github',
            raw_data=user_data
        )
    
    async def create_or_update_user(self, oauth_user: OAuth2UserInfo, db_session) -> UserModel:
        """Create or update user from OAuth2 information"""
        
        # Check if user exists by email
        existing_user = await db_session.execute(
            "SELECT * FROM users WHERE email = :email",
            {"email": oauth_user.email}
        )
        existing_user = existing_user.fetchone()
        
        if existing_user:
            # Update existing user
            await db_session.execute(
                """UPDATE users 
                   SET full_name = :full_name, 
                       last_login = :last_login,
                       is_verified = TRUE
                   WHERE email = :email""",
                {
                    "full_name": oauth_user.name,
                    "last_login": datetime.utcnow(),
                    "email": oauth_user.email
                }
            )
            await db_session.commit()
            
            # Fetch updated user
            updated_user = await db_session.execute(
                "SELECT * FROM users WHERE email = :email",
                {"email": oauth_user.email}
            )
            user_data = updated_user.fetchone()
            
        else:
            # Create new user
            user_id = str(uuid.uuid4())
            username = self._generate_username(oauth_user.email, oauth_user.provider)
            
            await db_session.execute(
                """INSERT INTO users 
                   (id, email, username, full_name, role, is_active, is_verified, created_at, last_login)
                   VALUES (:id, :email, :username, :full_name, :role, :is_active, :is_verified, :created_at, :last_login)""",
                {
                    "id": user_id,
                    "email": oauth_user.email,
                    "username": username,
                    "full_name": oauth_user.name,
                    "role": UserRole.USER,
                    "is_active": True,
                    "is_verified": True,
                    "created_at": datetime.utcnow(),
                    "last_login": datetime.utcnow()
                }
            )
            await db_session.commit()
            
            # Fetch created user
            created_user = await db_session.execute(
                "SELECT * FROM users WHERE id = :id",
                {"id": user_id}
            )
            user_data = created_user.fetchone()
        
        # Convert to UserModel
        user_model = UserModel(
            id=user_data.id,
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            role=user_data.role,
            is_active=user_data.is_active,
            is_verified=user_data.is_verified,
            created_at=user_data.created_at,
            last_login=user_data.last_login
        )
        
        logger.info(f"OAuth2 user authenticated: {oauth_user.email} via {oauth_user.provider}")
        
        return user_model
    
    def _generate_username(self, email: str, provider: str) -> str:
        """Generate unique username from email and provider"""
        base_username = email.split('@')[0]
        # Add provider suffix to make it unique
        username = f"{base_username}_{provider}"
        
        # Add hash suffix if needed to ensure uniqueness
        hash_suffix = hashlib.md5(f"{email}{provider}{secrets.token_urlsafe(8)}".encode()).hexdigest()[:8]
        username = f"{username}_{hash_suffix}"
        
        return username

# Global OAuth2 service instance
oauth2_service = OAuth2Service() 