"""
Configuration utility module that loads environment variables
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Load environment variables from .env file
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)


# Configuration class to access environment variables
class Config:
    """Configuration class that provides access to environment variables"""

    # Environment
    APP_MODE = os.getenv("ENV", os.getenv("APP_MODE", "development"))

    # The LLM model to use for generation
    LLM = os.getenv("LLM", "llama2")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    # LLM settings
    LLM_REQ_TIMEOUT_SECONDS = float(os.getenv("LLM_REQ_TIMEOUT_SECONDS", "120.0"))

    # Application settings
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
