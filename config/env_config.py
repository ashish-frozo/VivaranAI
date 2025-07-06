"""
Environment Configuration Module for VivaranAI

This module handles loading environment variables from .env files
and provides default values for all configuration options.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")


class Config:
    """Configuration class that loads from .env files with fallbacks"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration by loading from .env file
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        if DOTENV_AVAILABLE:
            # Find project root
            project_root = Path(__file__).parent.parent
            
            # Load from specified file or default .env
            if env_file:
                env_path = Path(env_file)
            else:
                env_path = project_root / ".env"
            
            # Load .env file if it exists
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✅ Loaded environment from: {env_path}")
            else:
                # Try .env.example as fallback (but warn user)
                example_path = project_root / ".env.example"
                if example_path.exists():
                    print(f"⚠️  .env file not found. Copy .env.example to .env and configure your settings:")
                    print(f"   cp {example_path} {project_root / '.env'}")
                else:
                    print("⚠️  No .env file found. Using environment variables only.")
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Create a .env file with your API key or set the environment variable."
            )
        return key
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL"""
        return os.getenv("REDIS_URL", "redis://localhost:6379")
    
    @property
    def host(self) -> str:
        """Get server host"""
        return os.getenv("HOST", "0.0.0.0")
    
    @property
    def port(self) -> int:
        """Get server port"""
        return int(os.getenv("PORT", "8001"))
    
    @property
    def debug(self) -> bool:
        """Get debug mode"""
        return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")
    
    @property
    def max_workers(self) -> int:
        """Get max workers"""
        return int(os.getenv("MAX_WORKERS", "4"))
    
    @property
    def timeout_seconds(self) -> int:
        """Get timeout in seconds"""
        return int(os.getenv("TIMEOUT_SECONDS", "30"))
    
    @property
    def api_rate_limit(self) -> int:
        """Get API rate limit"""
        return int(os.getenv("API_RATE_LIMIT", "100"))
    
    @property
    def log_level(self) -> str:
        """Get log level"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def log_file(self) -> str:
        """Get log file path"""
        return os.getenv("LOG_FILE", "agent_server.log")


# Global config instance
config = Config()


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get configuration instance
    
    Args:
        env_file: Optional path to specific .env file
        
    Returns:
        Config instance
    """
    if env_file:
        return Config(env_file)
    return config


def check_required_config():
    """
    Check that all required configuration is available
    
    Raises:
        ValueError: If required configuration is missing
    """
    try:
        # This will raise ValueError if API key is missing
        config.openai_api_key
        print("✅ Configuration validated successfully")
        return True
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False


if __name__ == "__main__":
    # Test configuration loading
    print("🔧 Testing Configuration...")
    if check_required_config():
        print(f"🌐 Server will run on {config.host}:{config.port}")
        print(f"📊 Log level: {config.log_level}")
        print(f"🔗 Redis URL: {config.redis_url}")
    else:
        print("❌ Configuration test failed") 