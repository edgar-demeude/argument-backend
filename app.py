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
from aba.models import (
    RuleDTO,
    FrameworkSnapshot,
    TransformationStep,
    ABAApiResponseModel,
    ABAPlusDTO,
    MetaInfo,
)
from copy import deepcopy
from datetime import datetime


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

def _make_snapshot(fw) -> FrameworkSnapshot:
    return FrameworkSnapshot(
        language=[str(l) for l in sorted(fw.language, key=str)],
        assumptions=[str(a) for a in sorted(fw.assumptions, key=str)],
        rules=[
            RuleDTO(
                id=r.rule_name,
                head=str(r.head),
                body=[str(b) for b in sorted(r.body, key=str)],
            )
            for r in sorted(fw.rules, key=lambda r: r.rule_name)
        ],
        contraries=[
            (str(c.contraried_literal), str(c.contrary_attacker))
            for c in sorted(fw.contraries, key=str)
        ],
        preferences={
            str(k): [str(v) for v in sorted(vals, key=str)]
            for k, vals in (fw.preferences or {}).items()
        } if getattr(fw, "preferences", None) else None,
    )


def _format_set(s) -> str:
    # s may be a Python set/frozenset of Literal or strings.
    try:
        items = sorted([str(x) for x in s], key=str)
    except Exception:
        # fallback if s is already a string like "{a,b}"
        return str(s)
    return "{" + ",".join(items) + "}"


async def _process_aba_framework(
    text: str,
    enable_aba_plus: bool = False,
) -> dict:
    """
    Core processing logic for ABA frameworks.
    
    Args:
        text: The uploaded file content as text
        enable_aba_plus: If True, compute ABA+ elements
        
    Returns:
        Complete response with before/after snapshots and all computations
    """
    # === 1. Build original framework ===
    base_framework = build_aba_framework_from_text(text)
    base_framework.generate_arguments()
    base_framework.generate_attacks()
    original_snapshot = _make_snapshot(base_framework)

    # --- Classical (argument-level) data ---
    original_arguments = [str(arg) for arg in sorted(base_framework.arguments, key=str)]
    original_attacks = [str(att) for att in sorted(base_framework.attacks, key=str)]
    original_reverse_attacks = []

    # === 2. Transform framework ===
    transformed_framework = deepcopy(base_framework).transform_aba()
    transformations = _detect_transformations(base_framework, transformed_framework)

    # --- Initialize containers ---
    original_assumption_sets = []
    final_assumption_sets = []
    original_aba_plus_attacks = []
    final_aba_plus_attacks = []
    original_reverse_attacks = []
    final_reverse_attacks = []
    warnings = []

    # === 3. ABA+ computations ===
    if enable_aba_plus:
        # --- ABA+ on original framework ---
        fw_plus_original = prepare_aba_plus_framework(deepcopy(base_framework))
        fw_plus_original.generate_arguments()
        fw_plus_original.generate_attacks()
        fw_plus_original.make_aba_plus()

        original_assumption_sets = sorted(
            [_format_set(s) for s in getattr(fw_plus_original, "assumption_combinations", [])],
            key=lambda x: (len(x), x),
        )

        original_aba_plus_attacks = [
            f"{_format_set(src)} → {_format_set(dst)}"
            for (src, dst) in sorted(
                getattr(fw_plus_original, "normal_attacks", []),
                key=lambda p: (str(p[0]), str(p[1])),
            )
        ]

        original_reverse_attacks = [
            f"{_format_set(src)} → {_format_set(dst)}"
            for (src, dst) in sorted(
                getattr(fw_plus_original, "reverse_attacks", []),
                key=lambda p: (str(p[0]), str(p[1])),
            )
        ]

        # --- Ensure transformed framework is consistent before ABA+ ---
        transformed_framework.generate_arguments()
        transformed_framework.generate_attacks()

        # --- Compute ABA+ on transformed framework ---
        fw_plus_transformed = prepare_aba_plus_framework(deepcopy(transformed_framework))
        fw_plus_transformed.generate_arguments()
        fw_plus_transformed.generate_attacks()
        fw_plus_transformed.make_aba_plus()

        final_assumption_sets = sorted(
            [_format_set(s) for s in getattr(fw_plus_transformed, "assumption_combinations", [])],
            key=lambda x: (len(x), x),
        )

        # Debug sanity checks
        print("DEBUG: fw_plus_transformed.assumptions =", getattr(fw_plus_transformed, "assumptions", []))
        print("DEBUG: fw_plus_transformed.normal_attacks =", getattr(fw_plus_transformed, "normal_attacks", []))

        final_aba_plus_attacks = [
            f"{_format_set(src)} → {_format_set(dst)}"
            for (src, dst) in sorted(
                getattr(fw_plus_transformed, "normal_attacks", []),
                key=lambda p: (str(p[0]), str(p[1])),
            )
        ]

        final_reverse_attacks = [
            f"{_format_set(src)} → {_format_set(dst)}"
            for (src, dst) in sorted(
                getattr(fw_plus_transformed, "reverse_attacks", []),
                key=lambda p: (str(p[0]), str(p[1])),
            )
        ]

        warnings = _validate_aba_plus_framework(fw_plus_transformed)
    else:
        warnings = _validate_framework(transformed_framework)

    # === 4. Classical ABA computations (arguments + attacks) ===
    base_framework.generate_arguments()
    base_framework.generate_attacks()

    transformed_framework.generate_arguments()
    transformed_framework.generate_attacks()

    original_arguments = [str(arg) for arg in sorted(base_framework.arguments, key=str)]
    original_arguments_attacks = [str(att) for att in sorted(base_framework.attacks, key=str)]

    final_arguments = [str(arg) for arg in sorted(transformed_framework.arguments, key=str)]
    final_arguments_attacks = [str(att) for att in sorted(transformed_framework.attacks, key=str)]

    # === 5. Snapshots ===
    original_snapshot = _make_snapshot(base_framework)
    final_snapshot = _make_snapshot(transformed_framework)

    # === 6. Build response ===
    response = {
        "meta": {
            "request_id": f"req-{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
            "transformed": any(t["applied"] for t in [_transform_to_dict(t) for t in transformations]),
            "transformations_applied": [
                t["step"] for t in [_transform_to_dict(t) for t in transformations] if t["applied"]
            ],
            "warnings": warnings,
            "errors": [],
        },
        "original_framework": {
            "framework": original_snapshot,
            "arguments": original_arguments,
            "arguments_attacks": original_arguments_attacks,
            "normal_attacks": original_aba_plus_attacks if enable_aba_plus else [],
            "reverse_attacks": original_reverse_attacks if enable_aba_plus else [],
            "assumption_sets": original_assumption_sets if enable_aba_plus else [],
        },
        "transformations": [_transform_to_dict(t) for t in transformations],
        "final_framework": {
            "framework": final_snapshot,
            "arguments": final_arguments,
            "arguments_attacks": final_arguments_attacks,
            "normal_attacks": final_aba_plus_attacks if enable_aba_plus else [],
            "reverse_attacks": final_reverse_attacks if enable_aba_plus else [],
            "assumption_sets": final_assumption_sets if enable_aba_plus else [],
        },
    }

    return response


def _detect_transformations(
    base_framework,
    transformed_framework,
) -> list:
    """
    Detect and describe which transformations were applied.
    """
    transformations = []
    
    if transformed_framework.language == base_framework.language and \
       transformed_framework.rules == base_framework.rules:
        # No transformation needed
        transformations.append({
            "step": "none",
            "applied": False,
            "reason": "The framework was already non-circular and atomic.",
            "description": "No transformation applied.",
            "result_snapshot": None,
        })
        return transformations
    
    # Determine transformation type
    was_circular = base_framework.is_aba_circular()
    was_atomic = base_framework.is_aba_atomic()
    
    step_name = "non_circular" if was_circular else "atomic"
    reason = "circular dependencies" if was_circular else "non-atomic rules"
    
    transformations.append({
        "step": step_name,
        "applied": True,
        "reason": f"The framework contained {reason}.",
        "description": f"Transformed into a {step_name.replace('_', '-')} version.",
        "result_snapshot": _make_snapshot(transformed_framework),
    })
    
    return transformations


def _transform_to_dict(t):
    """Convert TransformationStep to dict if needed."""
    if isinstance(t, dict):
        return t
    return {
        "step": t.step,
        "applied": t.applied,
        "reason": t.reason,
        "description": t.description,
        "result_snapshot": t.result_snapshot,
    }


def _validate_framework(framework) -> list[str]:
    """
    Validate framework and return any warnings.
    """
    warnings = []
    
    if hasattr(framework, "preferences") and framework.preferences:
        all_assumptions = {str(a) for a in framework.assumptions}
        pref_keys = {str(k) for k in framework.preferences.keys()}
        
        if not pref_keys.issubset(all_assumptions):
            warnings.append(
                "Incomplete preference relation: not all assumptions appear in the preference mapping."
            )
    
    return warnings


def _validate_aba_plus_framework(framework) -> list[str]:
    """
    Validate ABA+ framework and return any warnings.
    """
    return _validate_framework(framework)


@app.post("/aba-upload")
async def aba_upload(file: UploadFile = File(...)):
    """
    Handle classical ABA framework generation.
    
    Returns: original & final frameworks with arguments and attacks (no ABA+ data)
    """
    content = await file.read()
    text = content.decode("utf-8")
    return await _process_aba_framework(text, enable_aba_plus=False)


@app.post("/aba-plus-upload")
async def aba_plus_upload(file: UploadFile = File(...)):
    """
    Handle ABA+ framework generation.
    
    Returns: original & final frameworks with arguments, attacks, AND reverse_attacks for both
    """
    content = await file.read()
    text = content.decode("utf-8")
    return await _process_aba_framework(text, enable_aba_plus=True)


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

