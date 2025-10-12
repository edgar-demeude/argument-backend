import os

cache_dir = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import torch
import io

from relations.predict_bert import predict_relation

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ABA imports
from aba.aba_builder import prepare_aba_plus_framework, build_aba_framework_from_text

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
SAMPLES_DIR = Path("./relations/exemples/samples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at startup once
model_name = "edgar-demeude/bert-argument"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)


@app.get("/")
def root():
    return {"message": "Argument Mining API is running..."}


# ---------------- BERT Prediction Endpoints ---------------- #

@app.post("/predict-text")
def predict_text(arg1: str = Form(...), arg2: str = Form(...)):
    """Predict relation between two text arguments using BERT."""
    result = predict_relation(arg1, arg2, model, tokenizer, device)
    return {
        "arg1": arg1,
        "arg2": arg2,
        "relation": result
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile):
    """Predict relations for pairs of arguments from a CSV file (max 250 rows)."""
    content = await file.read()
    # Utiliser StringIO + quotechar='"'
    df = pd.read_csv(io.StringIO(content.decode("utf-8")), quotechar='"')
    
    if len(df) > 250:
        df = df.head(250)

    results = []
    for _, row in df.iterrows():
        result = predict_relation(
            row["parent"],
            row["child"],
            model,
            tokenizer,
            device
        )
        results.append({
            "parent": row["parent"],
            "child": row["child"],
            "relation": result
        })

    return {"results": results, "note": "Limited to 250 rows max"}


@app.get("/samples")
def list_samples():
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".csv")]
    return {"samples": files}


@app.get("/samples/{filename}")
def get_sample(filename: str):
    file_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Sample not found"}
    return FileResponse(file_path, media_type="text/csv")

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
