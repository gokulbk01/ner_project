# app.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets
import joblib
import spacy
from typing import List, Dict, Any
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ner-api")

# Initialize FastAPI app
app = FastAPI(
    title="Named Entity Recognition API",
    description="API for recognizing named entities in text",
    version="1.0.0"
)

# Security setup
security = HTTPBasic()

# Define credentials (in a production environment, these should be stored securely)
USERNAME = os.getenv("API_USERNAME", "admin")
PASSWORD = os.getenv("API_PASSWORD", "password")

# Load the trained NER model
try:
    model = spacy.load("final_model")  
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define request and response models
class TextRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    start: int
    end: int
    label: str

class NERResponse(BaseModel):
    entities: List[Entity]
    processed_text: str

# Authentication function
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (correct_username and correct_password):
        logger.warning(f"Failed authentication attempt from user: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Main prediction endpoint
@app.post("/predict", response_model=NERResponse)
def predict(text_request: TextRequest, username: str = Depends(authenticate)):
    """
    Extract named entities from the input text.
    
    The endpoint returns recognized entities along with their positions and labels.
    """
    try:
        text = text_request.text
        logger.info(f"Processing request from user: {username}, text length: {len(text)}")
        
        # Process the text with the NER model
       
        doc = model(text)
        entities = [
            Entity(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_
            )
            for ent in doc.ents
        ]
        
        
        logger.info(f"Found {len(entities)} entities")
        return NERResponse(entities=entities, processed_text=text)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )

# Documentation endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Named Entity Recognition API",
        "endpoints": {
            "/predict": "POST endpoint to recognize entities in text",
            "/health": "GET endpoint to check API health",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)