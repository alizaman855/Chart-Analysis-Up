#!/usr/bin/env python3
"""
FastAPI Chart Analysis System
Main API file that integrates computer vision and improved GPT vision analysis
"""

from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import json
import shutil
from pathlib import Path
import time
import logging
from typing import Optional
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2

# Import our analysis modules
from src.chart_similarity_cv import find_most_similar_charts_in_video, prepare_results_for_json
from check import GPTVisionAnalyzer  # Import the improved GPT Vision Analyzer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chart Analysis System")

# Setup directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create required directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("historical", exist_ok=True)
os.makedirs("results", exist_ok=True)

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Simple user database (in production, use proper database)
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# Analysis progress tracking
analysis_progress = {}

def update_progress(task_id: str, progress: int):
    """Update analysis progress"""
    analysis_progress[task_id] = progress

# Authentication functions
def get_auth_token_from_cookie(auth_token: Optional[str] = Cookie(None)):
    """Get authentication token from cookie"""
    return auth_token

def verify_user(auth_token: Optional[str] = Depends(get_auth_token_from_cookie)):
    """Verify user from auth token"""
    if not auth_token:
        return None
    
    # Simple token verification (in production, use JWT)
    for username, data in USERS.items():
        expected_token = hashlib.md5(f"{username}:{data['password']}".encode()).hexdigest()
        if auth_token == expected_token:
            return {"username": username, "role": data["role"]}
    return None

def get_current_user(user=Depends(verify_user)):
    """Get current authenticated user"""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

def admin_required(user=Depends(get_current_user)):
    """Require admin role"""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    """Login page"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error
    })

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Handle user login"""
    if username in USERS and USERS[username]["password"] == password:
        token = hashlib.md5(f"{username}:{password}".encode()).hexdigest()
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie("auth_token", token, httponly=True)
        return response
    else:
        return RedirectResponse(url="/?error=invalid", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(get_current_user)):
    """Main dashboard"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "user": user
    })

@app.get("/computer-vision", response_class=HTMLResponse)
async def computer_vision_page(request: Request, user=Depends(get_current_user)):
    """Computer vision analysis page"""
    # Get recent results
    results_dir = Path("results")
    recent_results = []
    
    if results_dir.exists():
        for result_file in results_dir.glob("*_cv_results.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    recent_results.append({
                        'filename': result_file.stem.replace('_cv_results', ''),
                        'timestamp': result_file.stat().st_mtime,
                        'summary': data.get('summary', {})
                    })
            except:
                continue
    
    # Sort by timestamp (newest first)
    recent_results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return templates.TemplateResponse("computer_vision.html", {
        "request": request,
        "user": user,
        "recent_results": recent_results[:5]  # Show last 5 results
    })

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...), 
    analysis_type: str = Form("cv"),
    fps: float = Form(1.0),
    user=Depends(get_current_user)
):
    """Upload video and start analysis"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        # Generate unique task ID
        task_id = f"{int(time.time())}_{file.filename}"
        video_filename = f"{task_id}.mp4"
        video_path = f"uploads/{video_filename}"
        
        # Save uploaded video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize progress
        analysis_progress[task_id] = 0
        
        # Start analysis in background
        if analysis_type == "cv":
            # Computer Vision Analysis
            asyncio.create_task(run_cv_analysis(video_path, task_id, fps))
        else:
            # GPT Vision Analysis
            asyncio.create_task(run_gpt_analysis(video_path, task_id, fps))
        
        return JSONResponse({
            "status": "success",
            "message": f"Video uploaded successfully. {analysis_type.upper()} analysis started.",
            "task_id": task_id,
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_cv_analysis(video_path: str, task_id: str, fps: float):
    """Run computer vision analysis in background"""
    try:
        output_dir = f"results/{task_id}_cv"
        
        def progress_callback(progress):
            update_progress(task_id, progress)
        
        # Run CV analysis
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor, 
            find_most_similar_charts_in_video,
            video_path, 
            output_dir, 
            fps, 
            progress_callback
        )
        
        # Save results
        results_file = f"results/{task_id}_cv_results.json"
        with open(results_file, 'w') as f:
            json.dump(prepare_results_for_json(results), f, indent=2)
        
        analysis_progress[task_id] = 100
        logger.info(f"CV analysis completed for {task_id}")
        
    except Exception as e:
        logger.error(f"CV analysis failed for {task_id}: {e}")
        analysis_progress[task_id] = -1  # Error state

def extract_frames_from_video(video_path: str, output_dir: str, fps: float = 1.0):
    """Extract frames from video for GPT analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frames

async def run_gpt_analysis(video_path: str, task_id: str, fps: float):
    """Run GPT vision analysis in background using improved analyzer"""
    try:
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API key not configured")
        
        output_dir = f"results/{task_id}_gpt"
        frames_dir = f"{output_dir}/frames"
        
        # Extract frames from video
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(
            executor,
            extract_frames_from_video,
            video_path,
            frames_dir,
            fps
        )
        
        update_progress(task_id, 20)
        
        # Initialize the improved GPT Vision Analyzer
        analyzer = GPTVisionAnalyzer(OPENAI_API_KEY)
        
        # Analyze each frame against historical charts
        results = []
        total_frames = len(frames)
        
        for i, frame_path in enumerate(frames):
            try:
                # Run analysis for this frame
                frame_results = await loop.run_in_executor(
                    executor,
                    analyzer.analyze_yearly_charts,
                    frame_path,
                    "historical"
                )
                
                # Add frame info to results
                frame_results['frame_number'] = i
                frame_results['frame_path'] = frame_path
                frame_results['timestamp_in_video'] = i / fps
                
                results.append(frame_results)
                
                # Update progress
                progress = 20 + int((i + 1) / total_frames * 75)
                update_progress(task_id, progress)
                
                logger.info(f"Analyzed frame {i+1}/{total_frames} for task {task_id}")
                
            except Exception as e:
                logger.error(f"Error analyzing frame {i}: {e}")
                continue
        
        # Compile final results
        final_results = {
            "timestamp": time.time(),
            "task_id": task_id,
            "video_path": video_path,
            "total_frames_analyzed": len(results),
            "fps": fps,
            "analysis_type": "gpt_vision_improved",
            "frames": results
        }
        
        # Find overall best matches
        if results:
            # Aggregate scores by year across all frames
            year_scores = {}
            for frame_result in results:
                if 'results' in frame_result:
                    for year_data in frame_result['results']:
                        year = year_data['year']
                        score = year_data['similarity_score']
                        if year not in year_scores:
                            year_scores[year] = []
                        year_scores[year].append(score)
            
            # Calculate average scores
            avg_scores = {}
            for year, scores in year_scores.items():
                avg_scores[year] = sum(scores) / len(scores)
            
            # Sort by average score
            best_matches = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            final_results['best_overall_matches'] = [
                {"year": year, "avg_similarity": score} 
                for year, score in best_matches[:10]
            ]
        
        # Save results
        results_file = f"results/{task_id}_gpt_vision_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        analysis_progress[task_id] = 100
        logger.info(f"GPT analysis completed for {task_id}")
        
    except Exception as e:
        logger.error(f"GPT analysis failed for {task_id}: {e}")
        analysis_progress[task_id] = -1  # Error state

@app.get("/analysis-progress/{task_id}")
async def get_analysis_progress(task_id: str, user=Depends(get_current_user)):
    """Get analysis progress"""
    progress = analysis_progress.get(task_id, 0)
    
    if progress == -1:
        return {"status": "error", "progress": 0, "message": "Analysis failed"}
    elif progress == 100:
        # Check if results file exists
        cv_results = Path(f"results/{task_id}_cv_results.json")
        gpt_results = Path(f"results/{task_id}_gpt_vision_results.json")
        
        if cv_results.exists():
            return {"status": "completed", "progress": 100, "results_type": "cv", "task_id": task_id}
        elif gpt_results.exists():
            return {"status": "completed", "progress": 100, "results_type": "gpt", "task_id": task_id}
        else:
            return {"status": "processing", "progress": progress}
    else:
        return {"status": "processing", "progress": progress}

@app.get("/analysis-results/{task_id}")
async def get_analysis_results(task_id: str, user=Depends(get_current_user)):
    """Get analysis results"""
    try:
        # Try CV results first
        cv_results_file = Path(f"results/{task_id}_cv_results.json")
        if cv_results_file.exists():
            with open(cv_results_file, 'r') as f:
                results = json.load(f)
            return {"status": "success", "type": "cv", "results": results}
        
        # Try GPT results
        gpt_results_file = Path(f"results/{task_id}_gpt_vision_results.json")
        if gpt_results_file.exists():
            with open(gpt_results_file, 'r') as f:
                results = json.load(f)
            return {"status": "success", "type": "gpt", "results": results}
        
        raise HTTPException(status_code=404, detail="Results not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/yearly-fractal", response_class=HTMLResponse)
async def yearly_fractal_page(request: Request, user=Depends(get_current_user)):
    """Yearly chart fractal analysis page"""
    # Load existing results if available
    results_file = Path("results/yearly_analysis.json")
    results = None
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    
    return templates.TemplateResponse("yearly_fractal.html", {
        "request": request,
        "user": user,
        "results": results
    })

@app.post("/upload-2025-chart")
async def upload_2025_chart(file: UploadFile = File(...), user=Depends(admin_required)):
    """Upload 2025 chart (admin only)"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save 2025 chart
        chart_path = "uploads/2025.png"
        with open(chart_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "status": "success", 
            "message": "2025 chart uploaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Error uploading 2025 chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-yearly-analysis")
async def run_yearly_analysis(
    analysis_type: str = Form("gpt"),
    user=Depends(admin_required)
):
    """Run yearly analysis (admin only)"""
    try:
        if analysis_type == "gpt" and not OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="OpenAI API key not configured")
        
        chart_2025_path = Path("uploads/2025.png")
        if not chart_2025_path.exists():
            raise HTTPException(status_code=400, detail="2025 chart not found. Please upload it first.")
        
        # Start analysis in background
        task_id = f"yearly_{int(time.time())}"
        analysis_progress[task_id] = 0
        
        asyncio.create_task(run_yearly_gpt_analysis(task_id))
        
        return JSONResponse({
            "status": "success", 
            "message": f"Yearly GPT analysis started",
            "task_id": task_id
        })
        
    except Exception as e:
        logger.error(f"Error starting yearly analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_yearly_gpt_analysis(task_id: str):
    """Run yearly GPT analysis in background using improved analyzer"""
    try:
        # Use the improved GPT Vision Analyzer from check.py
        analyzer = GPTVisionAnalyzer(OPENAI_API_KEY)
        loop = asyncio.get_event_loop()
        
        # Use the improved analyze_yearly_charts method with retry logic
        results = await loop.run_in_executor(
            executor,
            analyzer.analyze_yearly_charts,
            "uploads/2025.png",
            "historical",
            3  # max_retries parameter
        )
        
        # Save results
        with open("results/yearly_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        
        analysis_progress[task_id] = 100
        logger.info(f"Yearly GPT analysis completed with {results['successful_comparisons']} successful comparisons")
        
    except Exception as e:
        logger.error(f"Yearly GPT analysis failed: {e}")
        analysis_progress[task_id] = -1

@app.get("/yearly-progress/{task_id}")
async def get_yearly_progress(task_id: str, user=Depends(get_current_user)):
    """Get yearly analysis progress"""
    progress = analysis_progress.get(task_id, 0)
    
    if progress == -1:
        return {"status": "error", "progress": 0, "message": "Analysis failed"}
    elif progress == 100:
        return {"status": "completed", "progress": 100}
    else:
        return {"status": "processing", "progress": progress}

@app.get("/logout")
async def logout():
    """Logout user"""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("auth_token")
    return response

# API Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "openai_configured": bool(OPENAI_API_KEY),
        "gpt_vision_version": "improved"
    }

# Static file serving for results
@app.get("/results/{path:path}")
async def serve_results(path: str, user=Depends(get_current_user)):
    """Serve result files (images, etc.)"""
    file_path = Path("results") / path
    if file_path.exists() and file_path.is_file():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/uploads/{filename}")
async def serve_uploads(filename: str, user=Depends(get_current_user)):
    """Serve uploaded files"""
    file_path = Path("uploads") / filename
    if file_path.exists() and file_path.is_file():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/historical/{filename}")
async def serve_historical(filename: str, user=Depends(get_current_user)):
    """Serve historical chart files"""
    file_path = Path("historical") / filename
    if file_path.exists() and file_path.is_file():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    
    # Check if OpenAI API key is configured
    if not OPENAI_API_KEY:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        print("   GPT Vision analysis will not work without it")
    else:
        print("✅ OpenAI API key configured")
    
    print("🚀 Starting Chart Analysis System...")
    print("📊 Computer Vision: Available")
    print("🤖 GPT Vision: Available (Improved Version)" if OPENAI_API_KEY else "🤖 GPT Vision: Disabled (no API key)")
    print("🌐 Server will be available at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)