"""
OAuth2 Authentication Endpoints for VivaranAI

Provides OAuth2 login/logout endpoints for Google and GitHub
"""

from typing import Optional
from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import structlog

from security.oauth2_service import oauth2_service, oauth2_config
from security.auth_middleware import auth_manager, User
from database.models import get_db_session

logger = structlog.get_logger(__name__)

# OAuth2 router
oauth2_router = APIRouter(prefix="/auth", tags=["authentication"])

# Response models
class OAuth2LoginResponse(BaseModel):
    """OAuth2 login response"""
    authorization_url: str
    state: str

class OAuth2CallbackResponse(BaseModel):
    """OAuth2 callback response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict

class OAuth2ProvidersResponse(BaseModel):
    """Available OAuth2 providers"""
    providers: dict

# OAuth2 endpoints
@oauth2_router.get("/providers", response_model=OAuth2ProvidersResponse)
async def get_oauth2_providers():
    """Get available OAuth2 providers"""
    providers = {}
    
    # Check Google
    if oauth2_config.google_client_id and oauth2_config.google_client_secret:
        providers['google'] = {
            'name': 'Google',
            'login_url': '/auth/google/login',
            'icon': 'https://developers.google.com/identity/images/g-logo.png'
        }
    
    # Check GitHub
    if oauth2_config.github_client_id and oauth2_config.github_client_secret:
        providers['github'] = {
            'name': 'GitHub',
            'login_url': '/auth/github/login',
            'icon': 'https://github.com/favicon.ico'
        }
    
    return OAuth2ProvidersResponse(providers=providers)

@oauth2_router.get("/google/login")
async def google_login(
    request: Request,
    redirect_url: Optional[str] = None
):
    """Initiate Google OAuth2 login"""
    
    if not oauth2_config.google_client_id or not oauth2_config.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth2 not configured"
        )
    
    try:
        authorization_url = await oauth2_service.get_authorization_url(
            request, 'google', redirect_url
        )
        return RedirectResponse(authorization_url)
    except Exception as e:
        logger.error(f"Google OAuth2 login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate Google login"
        )

@oauth2_router.get("/github/login")
async def github_login(
    request: Request,
    redirect_url: Optional[str] = None
):
    """Initiate GitHub OAuth2 login"""
    
    if not oauth2_config.github_client_id or not oauth2_config.github_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth2 not configured"
        )
    
    try:
        authorization_url = await oauth2_service.get_authorization_url(
            request, 'github', redirect_url
        )
        return RedirectResponse(authorization_url)
    except Exception as e:
        logger.error(f"GitHub OAuth2 login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate GitHub login"
        )

@oauth2_router.get("/google/callback")
async def google_callback(
    request: Request,
    db_session=Depends(get_db_session)
):
    """Handle Google OAuth2 callback"""
    
    try:
        # Handle OAuth2 callback
        oauth_user = await oauth2_service.handle_callback(request, 'google')
        
        # Create or update user
        user_model = await oauth2_service.create_or_update_user(oauth_user, db_session)
        
        # Create JWT token
        auth_user = User(
            user_id=str(user_model.id),
            email=user_model.email,
            role=user_model.role,
            permissions=set(['analyze:bills', 'view:analytics'])  # Default permissions
        )
        
        jwt_token = await auth_manager.create_jwt_token(auth_user)
        
        # Get redirect URL
        state = request.query_params.get('state')
        redirect_url = oauth2_service.get_redirect_url(state) if state else None
        
        if redirect_url:
            # Redirect to frontend with token
            response = RedirectResponse(
                f"{redirect_url}?token={jwt_token}&provider=google"
            )
        else:
            # Default redirect to frontend
            response = RedirectResponse(
                f"{oauth2_config.frontend_url}?token={jwt_token}&provider=google"
            )
        
        # Set secure cookie
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600  # 1 hour
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth2 callback failed: {e}")
        
        # Redirect to frontend with error
        error_url = f"{oauth2_config.frontend_url}?error=oauth_failed&provider=google"
        return RedirectResponse(error_url)

@oauth2_router.get("/github/callback")
async def github_callback(
    request: Request,
    db_session=Depends(get_db_session)
):
    """Handle GitHub OAuth2 callback"""
    
    try:
        # Handle OAuth2 callback
        oauth_user = await oauth2_service.handle_callback(request, 'github')
        
        # Create or update user
        user_model = await oauth2_service.create_or_update_user(oauth_user, db_session)
        
        # Create JWT token
        auth_user = User(
            user_id=str(user_model.id),
            email=user_model.email,
            role=user_model.role,
            permissions=set(['analyze:bills', 'view:analytics'])  # Default permissions
        )
        
        jwt_token = await auth_manager.create_jwt_token(auth_user)
        
        # Get redirect URL
        state = request.query_params.get('state')
        redirect_url = oauth2_service.get_redirect_url(state) if state else None
        
        if redirect_url:
            # Redirect to frontend with token
            response = RedirectResponse(
                f"{redirect_url}?token={jwt_token}&provider=github"
            )
        else:
            # Default redirect to frontend
            response = RedirectResponse(
                f"{oauth2_config.frontend_url}?token={jwt_token}&provider=github"
            )
        
        # Set secure cookie
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=3600  # 1 hour
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub OAuth2 callback failed: {e}")
        
        # Redirect to frontend with error
        error_url = f"{oauth2_config.frontend_url}?error=oauth_failed&provider=github"
        return RedirectResponse(error_url)

@oauth2_router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: Optional[User] = Depends(auth_manager.get_current_user)
):
    """Logout user"""
    
    if current_user:
        # Get token from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            # Revoke JWT token
            await auth_manager.revoke_jwt_token(current_user.user_id, token)
    
    # Clear cookie
    response.delete_cookie("access_token")
    
    return JSONResponse(
        content={"message": "Logged out successfully"},
        status_code=status.HTTP_200_OK
    )

@oauth2_router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(auth_manager.get_current_user)
):
    """Get current user information"""
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "role": current_user.role,
        "permissions": list(current_user.permissions),
        "authenticated_at": current_user.authenticated_at.isoformat()
    }

# Additional utility endpoints
@oauth2_router.get("/status")
async def auth_status():
    """Get authentication system status"""
    
    return {
        "oauth2_enabled": True,
        "google_enabled": bool(oauth2_config.google_client_id and oauth2_config.google_client_secret),
        "github_enabled": bool(oauth2_config.github_client_id and oauth2_config.github_client_secret),
        "jwt_auth_enabled": True,
        "api_key_auth_enabled": True
    } 