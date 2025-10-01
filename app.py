from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from utils.tests import run_tests
from utils.predict import predict_relation
from utils.mlp import load_model_and_metadata
from utils.processor import ArgumentDataProcessor
from exemples.claims import test_cases

app = FastAPI(title="Argument Mining API")

# CORS middleware
origins = [
    "http://localhost:3000", # local frontend
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # GET, POST, PUT, DELETE...
    allow_headers=["*"],
)

PYTORCH_MODEL_PATH = "models/model.pth"

# Load the model once when the app starts
model_type = "pytorch"  # or "sklearn"
model, embedding_model, best_threshold, label_encoder = load_model_and_metadata(
    PYTORCH_MODEL_PATH, model_type
)

processor = ArgumentDataProcessor()

@app.get("/")
def root():
    return {"message": "Argument Mining API is running..."}

@app.post("/predict-test")
def predict_test():
    """
    Run predefined test cases to validate the model.
    Returns a summary of the test results.
    """
    run_tests(model, embedding_model, processor, best_threshold, label_encoder, model_type, test_cases)
    return {"message": "Test cases executed. Check server logs for details."}

@app.post("/predict-text")
def predict_text(arg1: str = Form(...), arg2: str = Form(...)):
    """
    Take two text arguments and predict their relation.
    Returns the predicted relation with details.
    """
    relation = predict_relation(arg1, arg2, model, embedding_model, processor, best_threshold, label_encoder, model_type)
    return {"arg1": arg1, "arg2": arg2, "relation": relation}


@app.post("/predict-csv")
async def predict_csv(file: UploadFile):
    """
    Take a CSV file with 'parent' and 'child' columns and predict relations for each row.
    The CSV should have two columns: 'parent' and 'child'.
    Returns a list of predictions (max 100 rows).
    """
    df = pd.read_csv(file.file)

    # Limit to 100 rows max
    if len(df) > 100:
        df = df.head(100)

    results = []
    for _, row in df.iterrows():
        relation = predict_relation(
            row["parent"],
            row["child"],
            model,
            embedding_model,
            processor,
            best_threshold,
            label_encoder,
            model_type
        )
        results.append({
            "parent": row["parent"],
            "child": row["child"],
            "relation": relation
        })

    return {
        "results": results,
        "note": "Limited to 100 rows max"
    }
