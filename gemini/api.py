import os
import io
import re
import time
import base64
import uuid
from typing import List, Optional, Dict, Any

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

# ---------------------- Utils ----------------------

def slugify(text: str, max_len: int = 40) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] or "prompt"

def load_font(size: int = 40) -> ImageFont.FreeTypeFont:
    for f in [
        "arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_text_with_outline(draw: ImageDraw.ImageDraw, xy, text, font, fill="white", outline="black", stroke=2):
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline)

def add_logo_and_text(image_data: bytes, logo_path: str = None, phone_number: str = None) -> bytes:
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            img = img.convert("RGBA")

            logo_left = None
            logo_top = None

            if logo_path:
                try:
                    with Image.open(logo_path) as logo:
                        logo = logo.convert("RGBA")
                        new_h = max(40, int(img.height * 0.10))
                        ratio = logo.width / max(1, logo.height)
                        new_w = int(new_h * ratio)
                        logo = logo.resize((new_w, new_h), Image.LANCZOS)

                        margin = int(max(10, img.width * 0.01))
                        x = img.width - new_w - margin
                        y = img.height - new_h - margin

                        # Subtle shadow via offset alpha composite
                        img.alpha_composite(Image.new("RGBA", logo.size, (0, 0, 0, 0)), (x + 2, y + 2))
                        img.alpha_composite(logo, (x, y))
                        logo_left, logo_top = x, y
                except FileNotFoundError:
                    print(f"[WARN] Logo file not found at {logo_path}, skipping logo.")
                    logo_path = None

            if phone_number:
                draw = ImageDraw.Draw(img)
                font = load_font(size=max(28, int(img.height * 0.035)))
                bbox = draw.textbbox((0, 0), phone_number, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                margin = int(max(10, img.width * 0.01))
                if logo_path and logo_left is not None:
                    x = max(margin, (logo_left - text_w - margin))
                else:
                    x = margin
                y = img.height - text_h - margin

                # Semi-transparent background for readability
                pad = int(text_h * 0.35)
                bg = Image.new("RGBA", (text_w + 2 * pad, text_h + 2 * pad), (0, 0, 0, 120))
                img.alpha_composite(bg, (x - pad, y - pad))
                draw_text_with_outline(draw, (x, y), phone_number, font=font)

            out = io.BytesIO()
            img.convert("RGB").save(out, format="PNG", optimize=True)
            return out.getvalue()
    except Exception as e:
        print(f"[WARN] Overlay failed: {e}")
        return image_data

def save_image(data: bytes, logo_path: str = None, phone_number: str = None, out_dir: str = "outputs") -> str:
    """Save image and return its file path"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())[:8]
    file_name = f"image_{unique_id}.png"
    file_path = os.path.join(out_dir, file_name)
    
    # Add logo and text if needed
    if logo_path or phone_number:
        data = add_logo_and_text(data, logo_path, phone_number)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(data)
    
    return file_path

# ---------------------- Gemini helpers ----------------------

def build_contents(prompt: str, system_prompt: str = None) -> List[types.Content]:
    # For image generation, we need to combine system and user prompts into one user prompt
    # as the system role is not supported for image generation
    combined_prompt = prompt
    if system_prompt:
        combined_prompt = f"{system_prompt}\n\nUser request: {prompt}"
    
    # Single-turn prompt with user role only
    return [types.Content(role="user", parts=[types.Part.from_text(text=combined_prompt)])]

def generate_image(client: genai.Client, model: str, prompt: str, system_prompt: str = None, 
                  logo_path: str = None, phone_number: str = None, 
                  out_dir: str = "outputs", max_retries: int = 3) -> Dict[str, Any]:
    """Generate an image and return its details"""
    contents = build_contents(prompt, system_prompt)
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    attempt = 0
    while True:
        try:
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            
            # Process response to extract image data
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data.data:
                    # Save the image and get the file path
                    image_data = part.inline_data.data
                    file_path = save_image(image_data, logo_path, phone_number, out_dir)
                    
                    # Generate URL (for a real API, this would be your domain)
                    base_url = request.host_url.rstrip('/')
                    image_url = f"{base_url}/{file_path}"
                    
                    return {
                        "success": True,
                        "image_path": file_path,
                        "image_url": image_url
                    }
            
            # If no image was found in the response
            return {
                "success": False,
                "error": "No image generated in response"
            }
            
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                return {
                    "success": False,
                    "error": f"Failed after {max_retries} retries: {str(e)}"
                }
            wait = min(8, 2 ** attempt)
            print(f"[WARN] Error: {e} — retrying in {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

# ---------------------- API Routes ----------------------

@app.route('/generate', methods=['POST'])
def generate_image_api():
    try:
        data = request.json
        
        # Get API key from environment or request
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return jsonify({"error": "Missing GOOGLE_API_KEY in environment"}), 500
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        # Get parameters from request
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
            
        # Optional parameters
        system_prompt = data.get('system_prompt', """You are a professional marketing image generator AI. 
Your goal is to produce clean, eye-catching, and brand-ready images that can be used as advertising templates.

## Guidelines:
1. **Focus on marketing aesthetics**:
   - Modern, minimal, and visually appealing layouts.
   - High contrast and clear background areas for overlaying text and logos.
   - Balanced composition with empty safe zones where branding information can be placed.

2. **Brand integration**:
   - Leave enough uncluttered space in the top, bottom, or side areas for a company logo, phone number, or email address.
   - Do NOT generate phone numbers, emails, or logos yourself; leave placeholders.

3. **Style consistency**:
   - Use vibrant colors, gradients, or product-centric visuals depending on the industry.
   - Keep text areas plain and not noisy (avoid unnecessary details that would obstruct branding).

4. **Output requirements**:
   - High resolution (minimum 1080x1080).
   - Print and digital friendly, scalable without quality loss.
   - Realistic lighting and sharpness suitable for online ads and posters.

5. **Variations by industry**:
   - Food & Beverage: appetizing dishes, fresh ingredients, bright colors.
   - Real Estate: clean modern houses/apartments, warm lighting, professional look.
   - Beauty/Wellness: soft tones, spa or skincare scenes, calming backgrounds.

6. **Strict rules**:
   - Never embed actual text into the image (besides natural signs in the scene if needed).
   - Never invent fake logos, brands, or contact info.
   - Always provide visually clear areas where real brand elements can be added later.""")
        
        model = data.get('model', 'gemini-2.0-flash-preview-image-generation')
        logo_path = data.get('logo_path', './logo.png')
        phone_number = data.get('phone_number', '0909 123 456')
        out_dir = data.get('out_dir', 'outputs')
        
        # Generate the image
        result = generate_image(
            client=client,
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            logo_path=logo_path,
            phone_number=phone_number,
            out_dir=out_dir
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    # Import already handled in the serve_image function
    
    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8080)
