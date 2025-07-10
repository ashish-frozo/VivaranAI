# OAuth2 Authentication Setup Guide

This guide explains how to set up OAuth2 authentication with Google and GitHub for VivaranAI.

## Overview

VivaranAI now supports OAuth2 authentication with:
- **Google OAuth2**: Sign in with Google accounts
- **GitHub OAuth2**: Sign in with GitHub accounts
- **JWT Tokens**: Secure session management
- **Role-based Access Control**: User permissions management

## Prerequisites

1. **Google Cloud Console Account** (for Google OAuth2)
2. **GitHub Account** (for GitHub OAuth2)
3. **VivaranAI Server** running with database setup

## 1. Google OAuth2 Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API for your project

### Step 2: Create OAuth2 Credentials

1. Navigate to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth 2.0 Client IDs**
3. Select **Web application**
4. Configure:
   - **Name**: `VivaranAI OAuth2`
   - **Authorized JavaScript origins**: 
     - `http://localhost:8001` (development)
     - `https://your-domain.com` (production)
   - **Authorized redirect URIs**:
     - `http://localhost:8001/auth/google/callback` (development)
     - `https://your-domain.com/auth/google/callback` (production)

### Step 3: Configure Environment Variables

Add to your `.env` file:
```env
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
GOOGLE_REDIRECT_URI=http://localhost:8001/auth/google/callback
```

## 2. GitHub OAuth2 Setup

### Step 1: Create GitHub OAuth App

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **New OAuth App**
3. Configure:
   - **Application name**: `VivaranAI`
   - **Homepage URL**: `http://localhost:8001` (development)
   - **Authorization callback URL**: `http://localhost:8001/auth/github/callback`

### Step 2: Configure Environment Variables

Add to your `.env` file:
```env
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
GITHUB_REDIRECT_URI=http://localhost:8001/auth/github/callback
```

## 3. General OAuth2 Configuration

Add these additional variables to your `.env` file:

```env
# Frontend URL for redirects
FRONTEND_URL=http://localhost:3000

# OAuth2 state secret for CSRF protection
OAUTH_STATE_SECRET=your_oauth_state_secret_here

# JWT configuration (should already exist)
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
```

## 4. Database Migration

The OAuth2 system uses the existing user database schema. Ensure your database is up to date:

```bash
# Run database migrations
alembic upgrade head
```

## 5. Testing OAuth2 Setup

### Backend Testing

1. Start your VivaranAI server:
```bash
python agents/server.py
```

2. Test OAuth2 status endpoint:
```bash
curl http://localhost:8001/auth/status
```

Expected response:
```json
{
  "oauth2_enabled": true,
  "google_enabled": true,
  "github_enabled": true,
  "jwt_auth_enabled": true,
  "api_key_auth_enabled": true
}
```

3. Test available providers:
```bash
curl http://localhost:8001/auth/providers
```

### Frontend Testing

1. Open the OAuth2 frontend:
```bash
# Navigate to frontend directory
cd frontend

# Serve the OAuth2 frontend
python -m http.server 3000
```

2. Visit `http://localhost:3000` and open `index-oauth.html`

3. Test login with Google or GitHub

## 6. API Endpoints

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/auth/providers` | GET | Get available OAuth2 providers |
| `/auth/google/login` | GET | Initiate Google OAuth2 login |
| `/auth/github/login` | GET | Initiate GitHub OAuth2 login |
| `/auth/google/callback` | GET | Handle Google OAuth2 callback |
| `/auth/github/callback` | GET | Handle GitHub OAuth2 callback |
| `/auth/logout` | POST | Logout user |
| `/auth/me` | GET | Get current user info |
| `/auth/status` | GET | Get auth system status |

### Using JWT Tokens

After successful OAuth2 login, you'll receive a JWT token. Use it in API requests:

```bash
# Using Authorization header
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8001/analyze

# Using cookie (set automatically)
curl -b "access_token=YOUR_JWT_TOKEN" \
     http://localhost:8001/analyze
```

## 7. Production Configuration

### Environment Variables

Update your production `.env` file:

```env
# Production URLs
GOOGLE_REDIRECT_URI=https://your-domain.com/auth/google/callback
GITHUB_REDIRECT_URI=https://your-domain.com/auth/github/callback
FRONTEND_URL=https://your-frontend-domain.com

# Security settings
JWT_SECRET_KEY=your_production_jwt_secret_key
OAUTH_STATE_SECRET=your_production_oauth_state_secret

# CORS settings
CORS_ORIGINS=https://your-frontend-domain.com,https://your-domain.com
```

### OAuth2 App Configuration

Update your OAuth2 app configurations:

**Google Cloud Console:**
- Add production domains to authorized origins
- Add production callback URLs

**GitHub OAuth App:**
- Update Homepage URL to production domain
- Update Authorization callback URL to production domain

## 8. Security Considerations

### CSRF Protection
- OAuth2 state parameters provide CSRF protection
- State tokens expire after 1 hour
- Random state generation with secure secrets

### JWT Security
- Tokens are signed with strong secrets
- Tokens expire after configured time (default: 1 hour)
- Tokens can be revoked server-side

### HTTPS Requirements
- **Always use HTTPS in production**
- OAuth2 providers require HTTPS for production apps
- Secure cookies require HTTPS

### Rate Limiting
- OAuth2 endpoints are rate-limited
- Failed authentication attempts are logged
- Suspicious activity monitoring

## 9. Troubleshooting

### Common Issues

**1. "Google OAuth2 not configured" error**
- Check `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` environment variables
- Verify Google Cloud Console configuration

**2. "GitHub OAuth2 not configured" error**
- Check `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` environment variables
- Verify GitHub OAuth App configuration

**3. "Invalid OAuth2 state" error**
- Ensure `OAUTH_STATE_SECRET` is set
- Check that callback URLs match exactly

**4. "Authentication failed" error**
- Verify redirect URIs match exactly
- Check OAuth2 app permissions
- Ensure email is available from OAuth2 provider

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

Check server logs for detailed OAuth2 flow information.

### Testing Without Frontend

You can test OAuth2 flow directly:

1. Visit login URL in browser:
```
http://localhost:8001/auth/google/login?redirect_url=http://localhost:8001
```

2. Complete OAuth2 flow
3. Check callback URL for token

## 10. User Management

### User Roles
- New OAuth2 users get `USER` role by default
- Users can be promoted to `ADMIN` role in database
- Role-based permissions are enforced

### User Data
- OAuth2 users are stored in the same `users` table
- Email is used as unique identifier
- Profile information is updated on login

### Database Queries

```sql
-- View OAuth2 users
SELECT id, email, username, full_name, role, created_at, last_login 
FROM users 
WHERE hashed_password IS NULL;

-- Promote user to admin
UPDATE users 
SET role = 'admin' 
WHERE email = 'user@example.com';
```

## 11. Monitoring and Analytics

### Metrics
- OAuth2 login attempts are tracked
- Authentication success/failure rates
- User session duration
- Provider usage statistics

### Logging
- All OAuth2 events are logged
- Failed authentication attempts
- Security events and warnings

### Health Checks
- OAuth2 provider connectivity
- JWT token validation
- Database connectivity

## Support

For issues with OAuth2 setup:
1. Check server logs for detailed error messages
2. Verify all environment variables are set correctly
3. Test OAuth2 providers configuration
4. Ensure database migrations are current

## Next Steps

After setting up OAuth2 authentication:
1. Configure role-based permissions
2. Set up user analytics
3. Implement password reset flows
4. Add multi-factor authentication
5. Set up audit logging 