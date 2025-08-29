from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_SECRET_KEY"),
)

class ChatRequest(BaseModel):
    message: str

class ImageRequest(BaseModel):
    prompt: str

chat_log = [{'role': 'system', 'content': 'You are a helpful and creative assistant that helps users generate beautiful images.'}]

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    chat_log.append({'role': 'user', 'content': chat_request.message})
    try:
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=chat_log,
            temperature=0.6
        )
        bot_response = response.choices[0].message.content
        chat_log.append({'role': 'assistant', 'content': bot_response})
        return {"response": bot_response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/image")
async def create_image(image_request: ImageRequest):
    try:
        response = openai.images.generate(
            prompt=f"A high-quality, detailed, and photorealistic image of: {image_request.prompt}",
            n=1,
            size="256x256"
        )
        image_url = response.data[0].url
        return {"image_url": image_url}
    except openai.AuthenticationError as e:
        return {"error": "Authentication failed. Please check your OpenAI API key in the .env file."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}