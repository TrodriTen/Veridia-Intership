import pickle
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    VECTORIZER_PATH,
    MODEL_PATH,
    ALLOWED_ORIGINS,
    MIN_TEXT_LENGTH,
    ACCEPTED_FILE_TYPES,
    MAX_FILE_SIZE,
    BASE_DIR,
    FRONTEND_DIR
)
from app.schemas import (
    ResumeText,
    PredictionResponse,
    HealthResponse,
    CategoriesResponse,
    ErrorResponse
)
from app.preprocessing import preprocess_text
from app.utils import extract_text_from_pdf, setup_logging, validate_file_size

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

vectorizer = None
model = None

def load_models():
    global vectorizer, model
    
    try:
        if not VECTORIZER_PATH.exists():
            logger.error(f"Vectorizer not found at {VECTORIZER_PATH}")
            return False
        
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
            return False
        
        logger.info("Loading vectorizer...")
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        
        logger.info("Loading model...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    logger.info("="*60)
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info("="*60)
    
    success = load_models()
    
    if not success:
        logger.warning("Application started without models. Run train_pipeline.py first.")
    else:
        logger.info(f"Application ready to receive requests")
    
    logger.info("="*60)


@app.get("/", tags=["General"])
def read_root():
    """Serve the frontend application"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return {
        "message": f"{API_TITLE} v{API_VERSION}",
        "status": "running",
        "models_loaded": vectorizer is not None and model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_file": "/predict/file",
            "categories": "/categories"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    models_loaded = vectorizer is not None and model is not None
    
    if not models_loaded:
        return HealthResponse(
            status="degraded",
            models_loaded=False,
            message="Models not loaded. Run train_pipeline.py"
        )
    
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        message="All systems operational"
    )


@app.post("/predict", 
          response_model=PredictionResponse, 
          tags=["Predicción"],
          status_code=status.HTTP_200_OK)
def predict_category(resume: ResumeText):

    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not available. Contact administrator."
        )
    
    try:
        logger.info("Preprocessing text...")
        processed_text = preprocess_text(resume.text)
        
        if len(processed_text) < MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Processed text is too short (minimum {MIN_TEXT_LENGTH} characters). Provide a resume with more content."
            )
        
        logger.info("Vectorizing text...")
        X = vectorizer.transform([processed_text])

        logger.info("Making prediction...")
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        prob_dict = {
            label: float(prob) 
            for label, prob in zip(model.classes_, probabilities)
        }

        confidence = float(max(probabilities))
        
        logger.info(f"Prediction successful: {prediction} (confidence: {confidence:.2%})")
        
        return PredictionResponse(
            category=prediction,
            confidence=confidence,
            probabilities=prob_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/predict/file", 
          response_model=PredictionResponse, 
          tags=["Predicción"],
          status_code=status.HTTP_200_OK)
async def predict_from_file(file: UploadFile = File(...)):

    if vectorizer is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not available. Contact administrator."
        )
    
    if file.content_type not in ACCEPTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Only accepted: {', '.join(ACCEPTED_FILE_TYPES)}"
        )
    
    try:
        logger.info(f"Processing file: {file.filename}")
        content = await file.read()
        
        if not validate_file_size(len(content), MAX_FILE_SIZE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB"
            )
        
        if file.content_type == "application/pdf":
            logger.info("Extracting text from PDF...")
            text = extract_text_from_pdf(content)
        else:
            logger.info("Decoding plain text...")
            text = content.decode("utf-8")
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from file"
            )
        
        resume = ResumeText(text=text)
        return predict_category(resume)
        
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error decoding file. Make sure it's UTF-8"
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@app.get("/categories", 
         response_model=CategoriesResponse, 
         tags=["Information"])
def get_categories():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    categories = sorted(model.classes_.tolist())
    
    return CategoriesResponse(
        categories=categories,
        total=len(categories)
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
