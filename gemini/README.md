# Image Generation API

This API uses Google's Gemini model to generate marketing images and returns URLs to the generated images.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Google API key:
   - Create a `.env` file in the project root
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

3. Run the API server:
```bash
python api.py
```

## API Usage

### Generate an Image

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
  "prompt": "Design a futuristic poster for iPhone 16 Black Friday Sale 50% OFF",
  "system_prompt": "(Optional) Custom system prompt to guide image generation",
  "model": "gemini-2.0-flash-preview-image-generation",
  "logo_path": "./logo.png",
  "phone_number": "0909 123 456",
  "out_dir": "outputs"
}
```

**Response:**
```json
{
  "success": true,
  "image_path": "outputs/image_12345678.png",
  "image_url": "http://localhost:5000/outputs/image_12345678.png"
}
```

## Example Usage

Using curl:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Design a futuristic poster for iPhone 16 Black Friday Sale 50% OFF"}'
```

Using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:5000/generate",
    json={
        "prompt": "Design a futuristic poster for iPhone 16 Black Friday Sale 50% OFF"
    }
)

print(response.json())
```

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| prompt | Yes | - | The description of the image you want to generate |
| system_prompt | No | (Built-in marketing prompt) | Instructions for the AI on how to generate the image |
| model | No | gemini-2.0-flash-preview-image-generation | The Gemini model to use |
| logo_path | No | ./logo.png | Path to a logo to overlay on the image |
| phone_number | No | 0909 123 456 | Phone number to overlay on the image |
| out_dir | No | outputs | Directory to save generated images |
