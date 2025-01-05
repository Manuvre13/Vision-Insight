from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Dict
from pydantic import BaseModel
import logging
from datetime import datetime
import os
import traceback

# Configure TensorFlow for memory optimization
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vision Insight API",
    description="API for detecting eye conditions from fundus images",
    version="1.0.0"
)

# Add CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.keras")
TARGET_SIZE = (224, 224)
DISEASE_COLS = ['N', 'D', 'G', 'C', 'A']
MAX_FILE_SIZE = 5 * 1024 * 1024  # Reduced to 5MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Model loading with proper error handling and memory optimization
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Added compile=False
    model.make_predict_function()  # Eager execution
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None

# Pydantic models for request/response validation
class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    confidence: float
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    model_path: str
    version: str = "1.0.0"

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    method = request.method
    url = request.url.path
    
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"{method} {url} - Status: {response.status_code} - Duration: {duration:.2f}s"
        )
        return response
    except Exception as e:
        logger.error(f"Request failed: {method} {url}")
        logger.error(traceback.format_exc())
        raise

async def validate_image(file: UploadFile) -> bytes:
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.1f}MB"
            )
        
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format or corrupted file"
            )
        
        return contents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=400,
            detail="Error validating image file"
        )

def preprocess_image(image_data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=400,
            detail="Error preprocessing image"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = "healthy" if model is not None else "model not loaded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = datetime.now()
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        logger.info(f"Processing file: {file.filename}, content_type: {file.content_type}")
        
        image_data = await validate_image(file)
        logger.info(f"Image validated successfully, size: {len(image_data)} bytes")
        
        preprocessed_image = preprocess_image(image_data)
        logger.info("Image preprocessed successfully")
        
        with tf.device('/CPU:0'):  # Force CPU usage
            predictions = model.predict(preprocessed_image, verbose=0)
        
        prediction_dict = {
            disease: float(pred)
            for disease, pred in zip(DISEASE_COLS, predictions[0])
        }
        
        confidence = float(max(predictions[0]))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Prediction completed in {processing_time:.2f}s")
        
        return {
            "predictions": prediction_dict,
            "confidence": confidence,
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )