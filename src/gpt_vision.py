#!/usr/bin/env python3
"""
Short GPT Vision Analysis - Simple and clean
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

class GPTVisionAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def compare_charts(self, image1_path, image2_path):
        """Compare two chart images and return similarity score with reasoning"""
        
        def img_to_b64(path):
            img = Image.open(path).convert('RGB')
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            return base64.b64encode(buffer.getvalue()).decode()
        
        img1_b64 = img_to_b64(image1_path)
        img2_b64 = img_to_b64(image2_path)
        
        prompt = """Analyze these two financial candlestick charts and compare:

1. Overall trend patterns (upward/downward/sideways direction)
2. Price action structure (highs, lows, consolidations)
3. Volatility patterns (candle sizes and ranges)
4. Support/resistance levels
5. Chart formations (triangles, channels, breakouts)

Scoring Guidelines (0-100):
- 90-100: Nearly identical patterns
- 80-89: Very similar with minor differences
- 70-79: Similar direction, different structure
- 60-69: Some similarities, notable differences
- 50-59: Mixed similarities
- 40-49: Some common elements but mostly different
- 30-39: Few similarities
- 20-29: Very different patterns
- 10-19: Opposite trends
- 0-9: No similarities

Format:
SCORE: [precise number like 67, 84, 23, 91, 45, etc.]
REASONING: [detailed explanation]

IMPORTANT: 
- Give precise, varied scores - avoid round numbers like 80, 70, 60
- Use realistic decimals like 67, 84, 23, 91, 45, 76, 38, 52, etc.
- Each comparison should have a unique score reflecting real analysis
"""
        
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
            max_tokens=250,
            temperature=0.1
        )
        
        text = response.choices[0].message.content
        
        # Extract score and reasoning
        try:
            score_match = re.search(r'SCORE:\s*(\d+)', text)
            score = float(score_match.group(1)) / 100.0 if score_match else 0.5  # Convert to 0-1 scale
            
            reasoning_match = re.search(r'REASONING:\s*(.+)', text, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else text
            
            return score, reasoning
        except:
            # Fallback extraction
            numbers = re.findall(r'\b\d+\b', text)
            return float(numbers[0]) / 100.0 if numbers else 0.5, text

    def analyze_yearly_charts(self, chart_2025_path, historical_dir):
        """Compare 2025 with all historical charts"""
        results = []
        
        for year in range(2009, 2025):
            historical_path = Path(historical_dir) / f"{year}.png"
            if historical_path.exists():
                print(f"Analyzing 2025 vs {year}...")
                score, reasoning = self.compare_charts(chart_2025_path, historical_path)
                results.append({
                    "year": year,
                    "similarity_score": score,
                    "analysis": f"Similarity score: {score:.3f}",
                    "reasoning": reasoning,
                    "timestamp": time.time()
                })
                time.sleep(1)  # Rate limiting
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "timestamp": time.time(),
            "total_comparisons": len(results),
            "analysis_type": "gpt_vision_yearly",
            "results": results
        }