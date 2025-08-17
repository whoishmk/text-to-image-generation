"""
Model Server for Text-to-Image Generation

This module provides a FastAPI-based server for serving fine-tuned
Stable Diffusion models with REST API endpoints.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import json

import torch
import numpy as np
from PIL import Image
import io
import base64

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import (
    load_base_model,
    load_lora_weights,
    optimize_model_for_inference,
    get_model_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Text-to-Image Generation API",
    description="API for generating images from text using fine-tuned Stable Diffusion models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
model_config = {}


class GenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Text description of the image to generate")
    negative_prompt: str = Field("", description="Text description of what not to include")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of denoising steps")
    height: int = Field(1024, ge=512, le=1280, description="Image height")
    width: int = Field(1024, ge=512, le=1280, description="Image width")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    return_base64: bool = Field(False, description="Return image as base64 string")


class GenerationResponse(BaseModel):
    """Response model for image generation."""
    success: bool
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    generation_time: float
    parameters: Dict[str, Any]
    message: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_loaded: bool
    model_type: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    device: Optional[str] = None
    lora_loaded: bool = False


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    uptime: float


# Global variables for tracking
start_time = time.time()
generation_count = 0
total_generation_time = 0


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Text-to-Image Generation API server...")
    
    # Load environment variables
    global model_config
    model_config = {
        "base_model": os.getenv("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
        "model_type": os.getenv("MODEL_TYPE", "sdxl"),
        "lora_path": os.getenv("LORA_PATH"),
        "device": os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "1"))
    }
    
    logger.info(f"Model configuration: {model_config}")


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text-to-Image Generation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "generate": "/generate",
            "batch_generate": "/generate/batch",
            "load_model": "/model/load"
        },
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        model_loaded=model is not None,
        uptime=uptime
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        return ModelInfoResponse(
            model_loaded=False,
            message="No model loaded"
        )
    
    try:
        info = get_model_info(model)
        
        return ModelInfoResponse(
            model_loaded=True,
            model_type=model_config.get("model_type"),
            model_info=info,
            device=model_config.get("device"),
            lora_loaded=model_config.get("lora_path") is not None
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
async def load_model():
    """Load the model into memory."""
    global model
    
    try:
        logger.info("Loading model...")
        
        # Load base model
        base_model = load_base_model(
            model_config["base_model"],
            model_config["model_type"]
        )
        
        # Load LoRA weights if specified
        if model_config.get("lora_path"):
            base_model = load_lora_weights(base_model, model_config["lora_path"])
            logger.info("LoRA weights loaded successfully")
        
        # Optimize for inference
        model = optimize_model_for_inference(base_model, model_config["device"])
        
        logger.info("Model loaded successfully")
        
        return {
            "success": True,
            "message": "Model loaded successfully",
            "model_type": model_config["model_type"],
            "device": model_config["device"]
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate a single image from text prompt."""
    global model, generation_count, total_generation_time
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load the model first.")
    
    try:
        start_time_gen = time.time()
        
        # Set seed for reproducibility
        if request.seed != -1:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(request.seed)
            np.random.seed(request.seed)
        
        # Generate image
        with torch.no_grad():
            result = model(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                height=request.height,
                width=request.width
            )
        
        generation_time = time.time() - start_time_gen
        
        # Update statistics
        generation_count += 1
        total_generation_time += generation_time
        
        # Convert PIL image to base64 if requested
        image_base64 = None
        if request.return_base64:
            image_base64 = pil_to_base64(result.images[0])
        
        # Save image to disk (optional)
        image_path = save_generated_image(result.images[0], request.prompt)
        
        return GenerationResponse(
            success=True,
            image_base64=image_base64,
            image_url=f"/images/{os.path.basename(image_path)}" if image_path else None,
            generation_time=generation_time,
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "height": request.height,
                "width": request.width,
                "seed": request.seed
            },
            message="Image generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/batch")
async def generate_batch_images(requests: List[GenerationRequest]):
    """Generate multiple images from a batch of requests."""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load the model first.")
    
    if len(requests) > model_config.get("max_batch_size", 1):
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(requests)} exceeds maximum allowed {model_config.get('max_batch_size', 1)}"
        )
    
    try:
        results = []
        
        for i, request in enumerate(requests):
            try:
                # Generate single image
                result = await generate_image(request)
                results.append({
                    "index": i,
                    "success": True,
                    "result": result.dict()
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "batch_size": len(requests),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get generation statistics."""
    return {
        "total_generations": generation_count,
        "total_generation_time": total_generation_time,
        "average_generation_time": total_generation_time / max(generation_count, 1),
        "uptime": time.time() - start_time,
        "model_loaded": model is not None
    }


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def save_generated_image(image: Image.Image, prompt: str) -> Optional[str]:
    """Save generated image to disk."""
    try:
        # Create images directory if it doesn't exist
        images_dir = Path("generated_images")
        images_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt[:50]  # Limit length
        filename = f"{timestamp}_{safe_prompt}.png"
        
        # Save image
        image_path = images_dir / filename
        image.save(image_path)
        
        logger.info(f"Image saved to {image_path}")
        return str(image_path)
        
    except Exception as e:
        logger.warning(f"Failed to save image: {e}")
        return None


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
