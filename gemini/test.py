# generate.py
# Dependencies:
#   pip install google-genai pillow python-dotenv
#
# Usage (defaults embedded prompts; no flags needed):
#   python generate.py
#
# Optional overrides:
#   python generate.py -f my_prefix -o out -l path/to/logo.png -t "0909 123 456" --model gemini-2.0-flash-preview-image-generation
#   python generate.py -p "Custom prompt 1" -p "Custom prompt 2"

import os
import io
import re
import time
import argparse
import mimetypes
from typing import List, Iterable

from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# ---------------------- Embedded default prompts ----------------------
DEFAULT_PROMPTS: List[str] = [
    "Design a futuristic poster for iPhone 16 Black Friday Sale 50% OFF, dark neon background, golden glowing text, sleek product highlight.",
    "Create an iPhone 16 Summer Vacation Sale banner, beach theme with palm trees and ocean waves, bright blue and yellow colors, bold text ‘Half Price Special’.",
    "Christmas promotion poster for iPhone 16, snowy background, Christmas tree and gifts, red and green color palette, headline ‘Holiday Sale 50% OFF’.",
    "Valentine’s Day iPhone 16 Sale 50% OFF design, pink and red romantic style, heart decorations, tagline ‘The Perfect Gift for Love’.",
    "Back to School iPhone 16 banner, chalkboard and books illustration, youthful vibrant style, text ‘Student Discounts Up to 50%’.",
    "New Year 2025 iPhone 16 Celebration Sale, fireworks and champagne theme, black and gold luxury design, text ‘Kick Off the Year with 50% OFF’.",
    "Easter holiday promo for iPhone 16, pastel tones with eggs and bunny illustrations, headline ‘Spring Sale 50% OFF’.",
    "Travel Vacation iPhone 16 Summer Sale, airplane and suitcase graphics, bright orange and turquoise palette, text ‘Get 50% OFF Before You Fly’.",
    "Lunar New Year 2025 iPhone 16 sale poster, red and gold festival theme, lanterns and blossom flowers, text ‘Special 50% OFF New Year Deals’.",
    "Cyber Monday iPhone 16 flash sale design, futuristic digital grid background, blue and black color scheme, headline ‘One Day Only – 50% OFF’.",
]

# ----------------------------- Utils -----------------------------

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

def save_binary_file(file_path: str, data: bytes, logo_path: str = None, phone_number: str = None):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    if logo_path or phone_number:
        data = add_logo_and_text(data, logo_path, phone_number)
    with open(file_path, "wb") as f:
        f.write(data)
    print(f"[OK] Saved: {file_path}")

# ----------------------------- Gemini helpers -----------------------------

def build_contents(prompt: str, system_prompt: str = None) -> List[types.Content]:
    # For image generation, we need to combine system and user prompts into one user prompt
    # as the system role is not supported for image generation
    combined_prompt = prompt
    if system_prompt:
        combined_prompt = f"{system_prompt}\n\nUser request: {prompt}"
    
    # Single-turn prompt with user role only
    return [types.Content(role="user", parts=[types.Part.from_text(text=combined_prompt)])]

def iter_image_parts_from_stream(stream_resp) -> Iterable[types.Part]:
    for chunk in stream_resp:
        cands = getattr(chunk, "candidates", None)
        if not cands:
            continue
        cand = cands[0]
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                yield part
            txt = getattr(part, "text", None)
            if txt:
                print(txt.strip())

def guess_extension(mime_type: str, default=".png") -> str:
    ext = mimetypes.guess_extension(mime_type or "") or default
    if ext == ".jpe":
        ext = ".jpg"
    return ext

# ----------------------------- Generation -----------------------------

def generate_one_prompt(client: genai.Client, model: str, prompt: str, out_prefix: str, out_dir: str,
                        logo_path: str, phone_number: str, system_prompt: str = None, max_retries: int = 3):
    contents = build_contents(prompt, system_prompt)
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    attempt = 0
    while True:
        try:
            stream = client.models.generate_content_stream(model=model, contents=contents, config=cfg)
            idx = 0
            slug = slugify(prompt)
            for part in iter_image_parts_from_stream(stream):
                data = part.inline_data.data
                mime = part.inline_data.mime_type
                ext = guess_extension(mime)
                file_name = f"{out_prefix}_{slug}_{idx}{ext}"
                file_path = os.path.join(out_dir, file_name) if out_dir else file_name
                save_binary_file(file_path, data, logo_path, phone_number)
                idx += 1
            if idx == 0:
                print(f"[INFO] No image parts returned for prompt: {prompt!r}")
            break
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                print(f"[ERROR] Failed for prompt after {max_retries} retries: {prompt!r}\n{e}")
                break
            wait = min(8, 2 ** attempt)
            print(f"[WARN] Error: {e} — retrying in {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

def generate_batch(prompts: List[str], file_name_prefix: str, logo_path: str, phone_number: str,
                   out_dir: str, model: str, system_prompt: str = None):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment. Set it via .env or environment variable.")
    client = genai.Client(api_key=api_key)
    for p in prompts:
        if not p or not p.strip():
            continue
        print(f"\n=== Generating for prompt ===\n{p}\n=============================")
        generate_one_prompt(client, model, p.strip(), file_name_prefix, out_dir, logo_path, phone_number, system_prompt)

# ----------------------------- CLI -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Batch-generate iPhone marketing images (Gemini).")
    parser.add_argument("-p", "--prompt", action="append", help="Prompt text. Use multiple -p for multiple prompts.")
    parser.add_argument("--prompts_file", type=str, help="Path to a .txt file containing prompts (one per line).")
    parser.add_argument("-f", "--file_name", type=str, default="iphone_campaign", help="Prefix for output image files.")
    parser.add_argument("-o", "--out_dir", type=str, default="outputs", help="Directory to save images.")
    parser.add_argument("-l", "--logo", type=str, help="Path to a logo image (PNG with transparency recommended).")
    parser.add_argument("-t", "--phone", type=str, help="Phone number to print on the image.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-preview-image-generation",
                        help="Model name.")
    parser.add_argument("--system_prompt", type=str, 
                        default="""You are a professional marketing image generator AI. 
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
   - Always provide visually clear areas where real brand elements can be added later.""",
                        help="System prompt to guide image generation.")
    return parser.parse_args()

def read_prompts_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

if __name__ == "__main__":
    args = parse_args()

    # Gán mặc định ở đây nếu bạn muốn
    if not args.logo:
        args.logo = "./image.png"     # 👈 thay bằng đường dẫn logo mặc định
    if not args.phone:
        args.phone = "0909 123 456"         # 👈 thay bằng số điện thoại mặc định
    if not args.out_dir:
        args.out_dir = "./outputs"         # 👈 thay bằng đường dẫn output mặc định

    # Build prompt list priority: CLI -p > file > embedded defaults
    prompts: List[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    elif args.prompts_file:
        prompts.extend(read_prompts_from_file(args.prompts_file))
    else:
        prompts = DEFAULT_PROMPTS[:]  # use embedded

    generate_batch(
        prompts=prompts,
        file_name_prefix=args.file_name,
        logo_path=args.logo,
        phone_number=args.phone,
        out_dir=args.out_dir,
        model=args.model,
        system_prompt=args.system_prompt
    )