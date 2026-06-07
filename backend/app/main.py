from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="NeuraFlow API")


class AnalyzeRequest(BaseModel):
    text: str


class AnalysisResponse(BaseModel):
    summary: str
    key_points: List[str]
    risks: List[str]
    recommendations: List[str]


class ProcessAnalysisResponse(BaseModel):
    summary: str
    pain_points: List[str]
    automation_opportunities: List[str]
    risks: List[str]
    estimated_benefits: List[str]
    recommended_technologies: List[str]
    automation_score: int
    roi_potential: str
    implementation_complexity: str
    priority: str


@app.get("/")
async def root():
    return {"message": "NeuraFlow API is running"}


@app.post( "/analyze-process", response_model=ProcessAnalysisResponse)
async def analyze_process(request: AnalyzeRequest):
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a senior business process analyst.

Analyze the process description and identify:

- Summary
- Pain points
- Automation opportunities
- Risks
- Estimated benefits
- Recommended technologies

Additionally provide:

Automation Score (integer from 0 to 100)

Scoring criteria:

0-20:
Very difficult to automate.

21-40:
Limited automation opportunities.

41-60:
Moderate automation opportunities.

61-80:
Strong automation opportunities with repetitive and rule-based activities.

81-100:
Excellent automation candidate.
Highly repetitive process, heavy manual work, frequent validations, data processing and significant ROI potential.

ROI Potential:
Low, Medium or High.

Implementation Complexity:
Low, Medium or High.
Priority:

Immediate
Short Term
Medium Term
Long Term

Immediate:
High ROI and high automation potential.

Short Term:
Good ROI but requires some preparation.

Medium Term:
Moderate ROI or complexity.

Long Term:
Low ROI or high complexity.

Focus on operational efficiency, automation and business value.
"""
            },
            {
                "role": "user",
                "content": request.text
            }
        ],
        response_format=ProcessAnalysisResponse,
    )

    analysis = response.choices[0].message.parsed

    return analysis
