from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

from relations.tests import run_tests
from relations.predict import predict_relation
from relations.mlp import load_model_and_metadata
from relations.processor import ArgumentDataProcessor
from exemples.claims import test_cases

# ABA imports
from aba.aba_builder import build_aba_framework, prepare_aba_plus_framework, build_aba_framework_from_text

app = FastAPI(title="Argument Mining API")

# CORS middleware
origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXAMPLES_DIR = Path("./aba/exemples")

# Load ML model at startup
PYTORCH_MODEL_PATH = "models/model.pth"
model_type = "pytorch"
model, embedding_model, best_threshold, label_encoder = load_model_and_metadata(
    PYTORCH_MODEL_PATH, model_type
)
processor = ArgumentDataProcessor()

@app.get("/")
def root():
    return {"message": "Argument Mining API is running..."}


# ---------------- ML Prediction Endpoints ---------------- #

@app.post("/predict-test")
def predict_test():
    """Run predefined test cases for model validation."""
    run_tests(model, embedding_model, processor, best_threshold, label_encoder, model_type, test_cases)
    return {"message": "Test cases executed. Check server logs for details."}


@app.post("/predict-text")
def predict_text(arg1: str = Form(...), arg2: str = Form(...)):
    """Predict relation between two text arguments."""
    relation = predict_relation(arg1, arg2, model, embedding_model, processor, best_threshold, label_encoder, model_type)
    return {"arg1": arg1, "arg2": arg2, "relation": relation}


@app.post("/predict-csv")
async def predict_csv(file: UploadFile):
    """Predict relations for pairs of arguments from a CSV file (max 100 rows)."""
    df = pd.read_csv(file.file)

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

    return {"results": results, "note": "Limited to 100 rows max"}


# ---------------- ABA API ---------------- #

@app.post("/aba-upload")
async def aba_upload(file: UploadFile = File(...)):
    """
    Upload a .txt file containing an ABA framework definition
    and return the generated ABA+ framework.
    """
    # Read file contents
    content = await file.read()
    text = content.decode("utf-8")  # assume UTF-8 encoding

    # Build ABA framework
    aba_framework = build_aba_framework_from_text(text)
    aba_framework = prepare_aba_plus_framework(aba_framework)
    aba_framework.make_aba_plus()

    results = {
        "assumptions": [str(a) for a in aba_framework.assumptions],
        "arguments": [str(arg) for arg in aba_framework.arguments],
        "attacks": [str(att) for att in aba_framework.attacks],
        "reverse_attacks": [str(ratt) for ratt in aba_framework.reverse_attacks],
    }
    return results

@app.get("/aba-examples")
def list_aba_examples():
    """Lists all sample files available on the server side."""
    examples = [f.name for f in EXAMPLES_DIR.glob("*.txt")]
    return {"examples": examples}

@app.get("/aba-examples/{filename}")
def get_aba_example(filename: str):
    """Returns the contents of a specific ABA sample file."""
    file_path = EXAMPLES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="text/plain", filename=filename)