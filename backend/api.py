from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

from agent.workflow import get_fixed_code

app = FastAPI()

class DebugRequest(BaseModel):
    code: str
    error_description: str

@app.post("/debug/")
def translate(req: DebugRequest):
    results = {}

    result = get_fixed_code(
        source_code=req.code,
        error_description=req.error_description
    )

    return {"result": result}
