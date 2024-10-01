
from fastapi import FastAPI

from medical_term_pipeline.main import MedicalTermPipeline

app = FastAPI()
pipeline = MedicalTermPipeline()
pipeline.load_data("./data")

@app.get("/query")
def query_medical_terms(query: str):
    responses = []
    for response in pipeline.stream_medical_terms(query):
        responses.append(response)
    return {"responses": responses}