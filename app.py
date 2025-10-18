import os

cache_dir = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

from gradual.models import GradualInput, GradualOutput
# from gradual.computations import compute_gradual_semantics
from gradual.computations import compute_gradual_space
from aba.aba_builder import prepare_aba_plus_framework, build_aba_framework_from_text
from relations.predict_bert import predict_relation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
import pandas as pd
from pathlib import Path
import asyncio
import json
import io


# -------------------- Config -------------------- #

ABA_EXAMPLES_DIR = Path("./aba/examples")
SAMPLES_DIR = Path("./relations/examples/samples")
GRADUAL_EXAMPLES_DIR = Path("./gradual/examples")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "edgar-demeude/bert-argument"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

# -------------------- App -------------------- #
app = FastAPI(title="Argument Mining API")

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Endpoints -------------------- #


@app.get("/")
def root():
    return {"message": "Argument Mining API is running..."}


# --- Predictions --- #

@app.post("/predict-text")
def predict_text(arg1: str = Form(...), arg2: str = Form(...)):
    """Predict relation between two text arguments using BERT."""
    result = predict_relation(arg1, arg2, model, tokenizer, device)
    return {"arg1": arg1, "arg2": arg2, "relation": result}


@app.post("/predict-csv-stream")
async def predict_csv_stream(file: UploadFile):
    """Stream CSV predictions progressively using SSE."""
    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")), quotechar='"')
    if len(df) > 250:
        df = df.head(250)

    async def event_generator():
        total = len(df)
        completed = 0
        for _, row in df.iterrows():
            try:
                result = predict_relation(
                    row["parent"], row["child"], model, tokenizer, device)
                completed += 1
                payload = {
                    "parent": row["parent"],
                    "child": row["child"],
                    "relation": result,
                    "progress": completed / total
                }
                yield f"data: {json.dumps(payload)}\n\n"
                # FORCER flush
                await asyncio.sleep(0)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'parent': row.get('parent'), 'child': row.get('child')})}\n\n"
                await asyncio.sleep(0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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


# --- ABA --- #

@app.post("/aba-upload")
async def aba_upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    aba_framework = build_aba_framework_from_text(text)
    aba_framework.generate_arguments()
    aba_framework.generate_attacks()

    results = {
        "assumptions": [str(a) for a in aba_framework.assumptions],
        "arguments": [str(arg) for arg in aba_framework.arguments],
        "attacks": [str(att) for att in aba_framework.attacks],
    }
    return results


@app.post("/aba-plus-upload")
async def aba_upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

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
    examples = [f.name for f in ABA_EXAMPLES_DIR.glob("*.txt")]
    return {"examples": examples}


@app.get("/aba-examples/{filename}")
def get_aba_example(filename: str):
    file_path = ABA_EXAMPLES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="text/plain", filename=filename)


# --- Gradual semantics --- #

# @app.post("/gradual", response_model=GradualOutput)
# def compute_gradual(input_data: GradualInput):
#     """API endpoint to compute Weighted h-Categorizer samples and convex hull."""
#     return compute_gradual_semantics(
#         A=input_data.A,
#         R=input_data.R,
#         n_samples=input_data.n_samples,
#         max_iter=input_data.max_iter
#     )

@app.post("/gradual", response_model=GradualOutput)
def compute_gradual(input_data: GradualInput):
    """
    API endpoint to compute Weighted h-Categorizer samples
    and their convex hull (acceptability degree space).
    """
    num_args, hull_volume, hull_area, hull_points, samples, axes = compute_gradual_space(
        num_args=input_data.num_args,
        R=input_data.R,
        n_samples=input_data.n_samples,
        axes=input_data.axes,
        controlled_args=input_data.controlled_args,
    )

    return GradualOutput(
        num_args=num_args,
        hull_volume=hull_volume,
        hull_area=hull_area,
        hull_points=hull_points,
        samples=samples,
        axes=axes,
    )


@app.get("/gradual-examples")
def list_gradual_examples():
    """
    List all available gradual semantics example files.
    Each example must be a JSON file with structure:
    {
        # "args": ["A", "B", "C"],
        # "relations": [["A", "B"], ["B", "C"]]
        "num_args": 3,
        "R": [["A", "B"], ["B", "C"], ["C", "A"]],
    }
    """
    if not GRADUAL_EXAMPLES_DIR.exists():
        return {"examples": []}

    examples = []
    for file in GRADUAL_EXAMPLES_DIR.glob("*.json"):
        examples.append({
            "name": file.stem,
            "path": file.name,
            "content": None
        })
    return {"examples": examples}


@app.get("/gradual-examples/{example_name}")
def get_gradual_example(example_name: str):
    """
    Return the content of a specific gradual example file.
    Example: GET /gradual-examples/simple.json
    """
    file_path = GRADUAL_EXAMPLES_DIR / example_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Example not found")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return JSONResponse(content=content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format in example file")
