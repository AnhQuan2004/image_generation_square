# Image Generation Square

A FastAPI-based application that combines chat functionality with AI image generation, leveraging OpenAI's GPT-4o for conversational AI and DALL-E for image creation.

## Features

- **AI Chat Interface**: Engage with GPT-4o through both traditional HTTP endpoints and WebSocket for real-time streaming responses
- **Image Generation**: Create AI-generated images from text prompts using DALL-E
- **Dual Implementation**:
  - API-only backend (api.py)
  - Full-stack implementation with templates (chat.py)

## Prerequisites

- Python 3.x
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image_generation_square
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_SECRET_KEY=your_api_key_here
```

## Usage

### API-Only Mode

Run the FastAPI application:
```bash
uvicorn api:app --reload
```

#### API Endpoints:
- **POST /chat**: Send messages to the AI assistant
  ```json
  {
    "message": "Can you help me create an image of a sunset?"
  }
  ```

- **POST /image**: Generate images based on text prompts
  ```json
  {
    "prompt": "A beautiful sunset over mountains"
  }
  ```

### Full-Stack Mode

Run the application with templates:
```bash
uvicorn chat:app --reload
```

Then access the application in your browser at `http://localhost:8000`.

## Project Structure

- `api.py` - API-only implementation with FastAPI endpoints
- `chat.py` - Full implementation with web templates and WebSocket support
- `requirements.txt` - Project dependencies
- `templates/` - HTML templates for the web interface

## Dependencies

- FastAPI - Web framework
- OpenAI - AI models integration
- Jinja2 - Template rendering
- python-dotenv - Environment variable management
- uvicorn - ASGI server
- websockets - WebSocket support

## Notes

- Images are generated at 256x256 resolution
- The chat system maintains context for a more coherent conversation
- WebSocket implementation provides real-time streaming responses