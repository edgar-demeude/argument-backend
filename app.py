import io
import json
import asyncio
from pathlib import Path
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from relations.predict_bert import predict_relation
from aba.aba_builder import prepare_aba_plus_framework, build_aba_framework_from_text
from aba.models import (
    RuleDTO,
    FrameworkSnapshot,
    TransformationStep,
    ABAApiResponseModel,
    ABAPlusDTO,
    MetaInfo,
)
from gradual.computations import compute_gradual_space
from gradual.models import GradualInput, GradualOutput
import os
from copy import deepcopy
from datetime import datetime

cache_dir = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)


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
    """
    Handle classical ABA generation.
    Returns:
        - original_framework: before transformations
        - final_framework: after transformations
        - transformations: steps applied (non-circular / atomic)
        - arguments, attacks
        - empty aba_plus section
    """
    content = await file.read()
    text = content.decode("utf-8")

    # === 1. Build original ABA framework ===
    base_framework = build_aba_framework_from_text(text)
    original_snapshot = _make_snapshot(base_framework)

    # === 2. Transform the framework ===
    copy_framework = deepcopy(base_framework)
    transformed_framework = copy_framework.transform_aba()

    was_circular = base_framework.is_aba_circular()
    was_atomic = base_framework.is_aba_atomic()

    transformed_framework = deepcopy(base_framework).transform_aba()

    # Detect transformation type
    transformations: list[TransformationStep] = []
    if transformed_framework.language != base_framework.language or transformed_framework.rules != base_framework.rules:
        # Some transformation happened
        if was_circular:
            transformations.append(
                TransformationStep(
                    step="non_circular",
                    applied=True,
                    reason="The framework contained circular dependencies.",
                    description="Transformed into a non-circular version.",
                    result_snapshot=_make_snapshot(transformed_framework),
                )
            )
        elif not was_atomic:
            transformations.append(
                TransformationStep(
                    step="atomic",
                    applied=True,
                    reason="The framework contained rules with non-assumption bodies.",
                    description="Transformed into an atomic version.",
                    result_snapshot=_make_snapshot(transformed_framework),
                )
            )
    else:
        # No transformation
        transformations.append(
            TransformationStep(
                step="none",
                applied=False,
                reason="The framework was already non-circular and atomic.",
                description="No transformation applied.",
                result_snapshot=None,
            )
        )

    # === 3. Generate arguments and attacks on transformed ===
    transformed_framework.generate_arguments()
    transformed_framework.generate_attacks()

    final_snapshot = _make_snapshot(transformed_framework)

    # === 4. Build response model ===
    response = ABAApiResponseModel(
        meta=MetaInfo(
            request_id=f"req-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow().isoformat(),
            transformed=any(t.applied for t in transformations),
            transformations_applied=[
                t.step for t in transformations if t.applied
            ],
            warnings=[],
            errors=[],
        ),
        original_framework=original_snapshot,
        transformations=transformations,
        final_framework=final_snapshot,
        arguments=[str(arg) for arg in sorted(
            transformed_framework.arguments, key=str)],
        attacks=[str(att)
                 for att in sorted(transformed_framework.attacks, key=str)],
        aba_plus=ABAPlusDTO(
            assumption_combinations=[],
            normal_attacks=[],
            reverse_attacks=[],
        ),
    )

    return response


@app.post("/aba-plus-upload", response_model=ABAApiResponseModel)
async def aba_plus_upload(file: UploadFile = File(...)):
    """
    Handle ABA+ generation.
    Returns:
      - original_framework / final_framework with snapshots
      - transformations applied (non_circular / atomic)
      - arguments, classical attacks (from transformed framework)
      - aba_plus: assumption_combinations, normal_attacks, reverse_attacks (string lists)
    """
    content = await file.read()
    text = content.decode("utf-8")

    # 1) Build base framework + original snapshot
    base_fw = build_aba_framework_from_text(text)
    original_snapshot = _make_snapshot(base_fw)

    was_circular = base_fw.is_aba_circular()
    was_atomic = base_fw.is_aba_atomic()

    # 2) Transform (deepcopy → transform_aba)
    transformed = deepcopy(base_fw).transform_aba()

    # Track transformation step(s)
    transformations: list[TransformationStep] = []
    if transformed.language != base_fw.language or transformed.rules != base_fw.rules:
        if was_circular:
            transformations.append(
                TransformationStep(
                    step="non_circular",
                    applied=True,
                    reason="The framework contained circular dependencies.",
                    description="Transformed into a non-circular version.",
                    result_snapshot=_make_snapshot(transformed),
                )
            )
        elif not was_atomic:
            transformations.append(
                TransformationStep(
                    step="atomic",
                    applied=True,
                    reason="The framework contained non-atomic rules.",
                    description="Transformed into an atomic version.",
                    result_snapshot=_make_snapshot(transformed),
                )
            )
    else:
        transformations.append(
            TransformationStep(
                step="none",
                applied=False,
                reason="The framework was already non-circular and atomic.",
                description="No transformation applied.",
                result_snapshot=None,
            )
        )

    # 3) Prepare for ABA+ (on the transformed copy) and compute
    # generates arguments + classical attacks
    fw_plus = prepare_aba_plus_framework(transformed)
    fw_plus.make_aba_plus()  # fills assumption_combinations, normal_attacks, reverse_attacks

    warnings = []
    if fw_plus.preferences:
        all_assumpptions = {str(a) for a in fw_plus.assumptions}
        pref_keys = {str(k) for k in fw_plus.preferences.keys()}
        if not pref_keys.issubset(all_assumpptions):
            warnings.append(
                "Incomplete preference relation detected: not all assumptions appear in the preference mapping."
            )

    # 4) Final snapshot
    final_snapshot = _make_snapshot(fw_plus)

    # 5) Serialize ABA+ pieces as strings
    assumption_sets = sorted(
        [_format_set(s)
         for s in getattr(fw_plus, "assumption_combinations", [])],
        key=lambda x: (len(x), x)
    )

    normal_str = [
        f"{_format_set(src)} → {_format_set(dst)}"
        for (src, dst) in sorted(
            getattr(fw_plus, "normal_attacks", []),
            key=lambda p: (str(p[0]), str(p[1])),
        )
    ]

    reverse_str = [
        f"{_format_set(src)} → {_format_set(dst)}"
        for (src, dst) in sorted(
            getattr(fw_plus, "reverse_attacks", []),
            key=lambda p: (str(p[0]), str(p[1])),
        )
    ]
    # arguments/attacks from transformed framework (already prepared)
    arguments = [str(arg) for arg in sorted(fw_plus.arguments, key=str)]
    attacks = [str(att) for att in sorted(fw_plus.attacks, key=str)]

    # 6) Build response
    resp = ABAApiResponseModel(
        meta=MetaInfo(
            request_id=f"req-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow().isoformat(),
            transformed=any(t.applied for t in transformations),
            transformations_applied=[
                t.step for t in transformations if t.applied],
            warnings=warnings,
            errors=[],
        ),
        original_framework=original_snapshot,
        transformations=transformations,
        final_framework=final_snapshot,
        arguments=arguments,
        attacks=attacks,
        aba_plus=ABAPlusDTO(
            assumption_combinations=assumption_sets,
            normal_attacks=normal_str,
            reverse_attacks=reverse_str,
        ),
    )
    return resp


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
