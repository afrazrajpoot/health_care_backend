import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        self.location = os.getenv("LOCATION")
        self.processor_id = os.getenv("PROCESSOR_ID")
        self.credentials_path = os.getenv("CREDENTIALS_PATH")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", 40 * 1024 * 1024))  # Default 40MB
        self.upload_dir = os.getenv("UPLOAD_DIR")
        self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")  # New: GCS bucket name
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Validate credentials file
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
        
        # Set Google Cloud credentials environment variable
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path

# Create singleton config
CONFIG: Dict[str, Any] = Settings().__dict__

# Ensure upload directory exists
os.makedirs(CONFIG["upload_dir"], exist_ok=True)