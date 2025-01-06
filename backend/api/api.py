from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Dict, List
from pydantic import BaseModel
import logging
from datetime import datetime
import os
import traceback
import gc
from tensorflow.keras.mixed_precision import set_global_policy # type: ignore

# Memory optimization configurations
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_tensor_float_32_execution(False)
set_global_policy('float16')

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)

# Create a root logger to capture all logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add new Pydantic model for logs
class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str

class LogsResponse(BaseModel):
    logs: List[LogEntry]
    count: int

# Add a custom handler to store logs in memory
class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity=5000):
        super().__init__()
        self.capacity = capacity
        self.logs = []

    def emit(self, record):
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            message=record.getMessage()
        )
        self.logs.append(log_entry)
        if len(self.logs) > self.capacity:
            self.logs.pop(0)

# Initialize memory log handler
memory_handler = MemoryLogHandler()
root_logger.addHandler(memory_handler)

# Capture uvicorn logs
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addHandler(memory_handler)

uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.addHandler(memory_handler)

# Initialize FastAPI app
app = FastAPI(
    title="Vision Insight API",
    description="API for detecting eye conditions from fundus images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.keras")
TARGET_SIZE = (224, 224)
DISEASE_COLS = ['N', 'D', 'G', 'C', 'A']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Environment variables for configuration
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '1'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1'))
WORKER_CONNECTIONS = int(os.getenv('WORKER_CONNECTIONS', '100'))

# Model loading with memory optimization
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.make_predict_function()  # Eager execution
    tf.keras.backend.clear_session()
    gc.collect()
    logger = logging.getLogger(__name__)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None

# Pydantic models for predictions
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
    memory_usage: float

def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@app.on_event("startup")
async def startup_event():
    logger.info("=== Application Starting Up ===")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Worker Connections: {WORKER_CONNECTIONS}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=== Application Shutting Down ===")

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
            img.close()
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
        # Use a context manager for the BytesIO object
        with io.BytesIO(image_data) as bio:
            img = Image.open(bio)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use BILINEAR instead of LANCZOS for faster resizing
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            
            # Use float16 for reduced memory usage
            img_array = np.array(img, dtype=np.float16) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Clear PIL image from memory
            img.close()
            del img
            
            return img_array
            
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=400,
            detail="Error preprocessing image"
        )

def cleanup_memory():
    """Function to clean up memory after prediction"""
    gc.collect()
    tf.keras.backend.clear_session()

@app.get("/", response_model=LogsResponse)
async def root(logs: str = None):
    """
    Root endpoint that handles log requests via query parameters
    """
    if logs in ["build", "container"]:
        try:
            return {
                "logs": memory_handler.logs,
                "count": len(memory_handler.logs)
            }
        except Exception as e:
            logger.error(f"Error retrieving logs: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error retrieving application logs"
            )
    else:
        raise HTTPException(
            status_code=404,
            detail="Not found"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    status = "healthy" if model is not None else "model not loaded"
    memory_usage = get_memory_usage()
    
    return {
        "status": status,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "version": "1.0.0",
        "memory_usage": memory_usage
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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
        
        with tf.device('/CPU:0'):  # Force CPU usage
            preprocessed_image = preprocess_image(image_data)
            predictions = model.predict(
                preprocessed_image,
                verbose=0,
                batch_size=BATCH_SIZE
            )
        
        # Clean up preprocessing data
        del preprocessed_image
        del image_data
        
        prediction_dict = {
            disease: float(pred)
            for disease, pred in zip(DISEASE_COLS, predictions[0])
        }
        
        confidence = float(max(predictions[0]))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_memory)
        
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
    port = int(os.environ.get("PORT", 7860))
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=MAX_WORKERS,
        limit_concurrency=WORKER_CONNECTIONS,
        log_level="info",
        log_config=log_config
    )