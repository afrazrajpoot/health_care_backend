import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


# ... (your existing imports and class)

class YourService:  # Assuming this is in a class
    def __init__(self):
        # Load encryption key from env (generate once: Fernet.generate_key())
        self.encryption_key = os.getenv('ENCRYPTION_KEY').encode()
        self.cipher_suite = Fernet(self.encryption_key)

  
    # Add this helper method for decryption (call from a FastAPI endpoint)
    def decrypt_patient_token(self, token: str) -> Dict[str, Any]:
        """
        Decrypts the token and returns patient data.
        Use this in a FastAPI route, e.g., @app.post("/decrypt-patient")
        """
        try:
            # Pad the token if needed for base64
            padded_token = token + '=' * (4 - len(token) % 4)
            encrypted_bytes = base64.urlsafe_b64decode(padded_token)
            decrypted_json = self.cipher_suite.decrypt(encrypted_bytes).decode('utf-8')
            patient_data = json.loads(decrypted_json)
            patient_data["dob"] = datetime.fromisoformat(patient_data["dob"])
            patient_data["doi"] = datetime.fromisoformat(patient_data["doi"])
            return patient_data
        except Exception as e:
            logger.error(f"❌ Decryption failed: {str(e)}")
            raise ValueError("Invalid or expired token")