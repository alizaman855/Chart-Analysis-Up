#!/usr/bin/env python3
"""
Improved GPT Vision Analysis - Enhanced accuracy for chart comparison
"""

import os
import base64
from openai import OpenAI
from PIL import Image
import io
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

class GPTVisionAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        # Set up logger for this instance
        self.logger = logging.getLogger(__name__)
    
    def compare_charts(self, image1_path, image2_path):
        """Compare two chart images and return similarity score with reasoning"""
        
        # Add logging and verification
        self.logger.info(f"Comparing: {image1_path} vs {image2_path}")
        
        # Verify files exist
        if not os.path.exists(image1_path):
            error_msg = f"Image 1 not found: {image1_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        if not os.path.exists(image2_path):
            error_msg = f"Image 2 not found: {image2_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        def img_to_b64(path):
            try:
                self.logger.debug(f"Converting image to base64: {path}")
                img = Image.open(path).convert('RGB')
                self.logger.debug(f"Original image size: {img.size}, mode: {img.mode}")
                
                # Use higher resolution and preserve aspect ratio
                img.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized image size: {img.size}")
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG', quality=100)  # Use PNG for better quality
                return base64.b64encode(buffer.getvalue()).decode()
            except Exception as e:
                self.logger.error(f"Error converting image to base64: {e}")
                raise
        
        img1_b64 = img_to_b64(image1_path)
        img2_b64 = img_to_b64(image2_path)
        
        # More specific prompt to avoid false positives
        prompt = """You are analyzing two different candlestick charts. Look carefully at each image.

Compare these charts for PATTERN SIMILARITY:

1. Overall trend shape and direction
2. Volatility patterns (smooth vs choppy)
3. Key turning points and timing
4. Support/resistance levels
5. Market structure (trending vs ranging)

IMPORTANT: Different years will have different patterns. Do NOT assume similarity.

Scoring (0-100):
95-100: Nearly identical patterns
85-94: Very similar structure
75-84: Similar but notable differences  
65-74: Some similarities
55-64: Mixed - some similar, some different
45-54: More differences than similarities
35-44: Few similarities
25-34: Very different patterns
15-24: Opposite structures
0-14: Completely different

Look at the actual SHAPE and PATTERN of each chart.

Respond ONLY: SCORE: [number]"""
        
        try:
            self.logger.info("Sending request to OpenAI GPT-4 Vision API")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
                    ]
                }],
                max_tokens=50,   # Much lower since we only need the score
                temperature=0.1  # Lower for more consistent scoring
            )
            
            text = response.choices[0].message.content
            self.logger.info(f"GPT-4 Vision response: {text}")
            return self._extract_score_and_reasoning(text)
            
        except Exception as e:
            error_msg = f"Error in API call: {e}"
            self.logger.error(error_msg)
            print(error_msg)  # Also print for console visibility
            return 0.0, error_msg
    
    def _extract_score_and_reasoning(self, text):
        """Extract score with simplified extraction for score-only responses"""
        self.logger.debug(f"Extracting score from: {text}")
        
        # Method 1: Look for SCORE: format
        score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1)) / 100.0
            self.logger.debug(f"Extracted score (method 1): {score}")
            return score, f"Score: {score_match.group(1)}"
        
        # Method 2: Look for any standalone number
        numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
        valid_scores = [float(n) for n in numbers if 0 <= float(n) <= 100]
        
        if valid_scores:
            score = valid_scores[0] / 100.0 if valid_scores[0] > 1 else valid_scores[0]
            self.logger.debug(f"Extracted score (method 2): {score}")
            return score, f"Score: {valid_scores[0]}"
        
        # Fallback
        self.logger.warning(f"Could not extract score from: {text}")
        print(f"Could not extract score from: {text}")
        return 0.5, text
    
    def analyze_yearly_charts(self, chart_2025_path, historical_dir, max_retries=3, print_scores=True):
        """Compare 2025 with all historical charts with retry logic"""
        self.logger.info(f"Starting yearly chart analysis: {chart_2025_path} vs {historical_dir}")
        results = []
        
        if not os.path.exists(chart_2025_path):
            raise FileNotFoundError(f"2025 chart not found: {chart_2025_path}")
        
        historical_files = []
        for year in range(2009, 2025):
            historical_path = Path(historical_dir) / f"{year}.png"
            if historical_path.exists():
                historical_files.append((year, historical_path))
        
        if not historical_files:
            raise FileNotFoundError(f"No historical charts found in {historical_dir}")
        
        print(f"Found {len(historical_files)} historical charts to compare")
        self.logger.info(f"Found {len(historical_files)} historical charts to compare")
        
        for year, historical_path in historical_files:
            success = False
            for attempt in range(max_retries):
                try:
                    print(f"🔍 Analyzing 2025 vs {year} (attempt {attempt + 1})...")
                    self.logger.info(f"Analyzing 2025 vs {year} (attempt {attempt + 1})...")
                    
                    score, reasoning = self.compare_charts(chart_2025_path, historical_path)
                    
                    # Validate score
                    if not (0 <= score <= 1):
                        print(f"Invalid score {score} for {year}, retrying...")
                        self.logger.warning(f"Invalid score {score} for {year}, retrying...")
                        time.sleep(2)
                        continue
                    
                    # Print score for each year as requested (both console and logger)
                    if print_scores:
                        percentage = score * 100
                        score_msg = f"📊 Year {year}: Similarity Score = {score:.3f} ({percentage:.1f}%)"
                        print(score_msg)
                        self.logger.info(score_msg)
                    
                    results.append({
                        "year": year,
                        "similarity_score": score,
                        "analysis": f"Similarity score: {score:.3f} ({score*100:.1f}%)",
                        "reasoning": reasoning,
                        "timestamp": time.time()
                    })
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = f"Error comparing with {year}: {e}"
                    print(error_msg)
                    self.logger.error(error_msg)
                    if attempt == max_retries - 1:
                        results.append({
                            "year": year,
                            "similarity_score": 0.0,
                            "analysis": "Analysis failed",
                            "reasoning": f"Error: {str(e)}",
                            "timestamp": time.time()
                        })
                    time.sleep(3)  # Wait before retry
            
            if success:
                time.sleep(1.5)  # Rate limiting between successful calls
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Add summary statistics
        scores = [r["similarity_score"] for r in results if r["similarity_score"] > 0]
        summary = {
            "mean_similarity": sum(scores) / len(scores) if scores else 0,
            "max_similarity": max(scores) if scores else 0,
            "min_similarity": min(scores) if scores else 0,
            "most_similar_year": results[0]["year"] if results and results[0]["similarity_score"] > 0 else None
        }
        
        # Print final summary
        if print_scores:
            summary_msg = f"🏆 Analysis complete. Most similar year: {summary['most_similar_year']} with score {summary['max_similarity']:.3f} ({summary['max_similarity']*100:.1f}%)"
            print(summary_msg)
            self.logger.info(summary_msg)
        
        return {
            "timestamp": time.time(),
            "total_comparisons": len(results),
            "successful_comparisons": len([r for r in results if r["similarity_score"] > 0]),
            "analysis_type": "gpt_vision_yearly_improved",
            "summary": summary,
            "results": results
        }

# Convenience function for frame extraction with consistent processing
def extract_frames_with_consistent_processing(video_path: str, output_dir: str, fps: float = 1.0):
    """Extract frames from video using consistent PIL processing (same as img_to_b64)"""
    import cv2
    
    print(f"Extracting frames from video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    print(f"Video FPS: {video_fps}, Frame interval: {frame_interval}")
    
    frame_count = 0
    saved_count = 0
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert OpenCV frame (BGR) to PIL Image (RGB) for consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply EXACT same processing as in img_to_b64 method
            pil_image = pil_image.convert('RGB')
            pil_image.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            
            frame_filename = f"frame_{saved_count:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save with same quality as img_to_b64 (PNG, quality=100)
            pil_image.save(frame_path, format='PNG', quality=100)
            frames.append(frame_path)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames so far...")
        
        frame_count += 1
    
    cap.release()
    print(f"Frame extraction complete. Total frames saved: {saved_count}")
    return frames

# Usage example
def main():
    # Replace with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    analyzer = GPTVisionAnalyzer(api_key)
    
    # Example usage
    chart_2025_path = "historical/2025.png"
    historical_dir = "historical"
    
    try:
        results = analyzer.analyze_yearly_charts(chart_2025_path, historical_dir, print_scores=True)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Total comparisons: {results['total_comparisons']}")
        print(f"Successful: {results['successful_comparisons']}")
        print(f"\nSummary:")
        print(f"  Most similar year: {results['summary']['most_similar_year']}")
        print(f"  Max similarity: {results['summary']['max_similarity']:.3f}")
        print(f"  Mean similarity: {results['summary']['mean_similarity']:.3f}")
        
        print(f"\nTop 5 most similar years:")
        for i, result in enumerate(results['results'][:5]):
            print(f"  {i+1}. {result['year']}: {result['similarity_score']:.3f} ({result['similarity_score']*100:.1f}%)")
        
        # Save results
        with open('analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to analysis_results.json")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()