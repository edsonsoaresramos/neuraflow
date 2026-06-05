from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="NeuraFlow API")


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "NeuraFlow API is running"}


@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent document analysis assistant."
            },
            {
                "role": "user",
                "content": f"Analyze the following text and provide a concise summary:\n\n{request.text}"
            }
        ]
    )

    return {
        "analysis": response.choices[0].message.content
    }
