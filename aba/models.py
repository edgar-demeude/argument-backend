from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple, Any


# === Basic DTOs ===

class RuleDTO(BaseModel):
    """Represents a single inference rule in the ABA framework."""
    id: str
    head: str
    body: List[str]


class FrameworkSnapshot(BaseModel):
    """Snapshot of an ABA framework at a specific stage (original or transformed)."""
    language: List[str]
    assumptions: List[str]
    rules: List[RuleDTO]
    contraries: List[Tuple[str, str]]
    preferences: Optional[Dict[str, List[str]]] = None


# === Transformation tracking ===

class TransformationStep(BaseModel):
    """Represents one transformation step (non-circular, atomic, etc.)."""
    step: str  # 'non_circular' | 'atomic' | 'none'
    applied: bool
    reason: Optional[str] = None
    description: Optional[str] = None
    result_snapshot: Optional[FrameworkSnapshot] = None


# === ABA+ details ===

class ABAPlusDTO(BaseModel):
    """Results specific to ABA+ semantics (extended attacks between assumption sets)."""
    assumption_combinations: List[str]
    normal_attacks: List[str]
    reverse_attacks: List[str]


# === Meta info ===

class MetaInfo(BaseModel):
    """Metadata about the ABA computation process."""
    request_id: str
    timestamp: str
    transformed: bool
    transformations_applied: List[str]
    warnings: Optional[List[str]] = []
    errors: Optional[List[str]] = []


# === Full API response ===

class ABAApiResponseModel(BaseModel):
    """
    Represents the full backend response for an ABA/ABA+ computation request.
    Includes both the original and transformed frameworks, all transformation steps,
    and computed results (arguments, attacks, ABA+ extensions).
    """
    meta: MetaInfo
    original_framework: FrameworkSnapshot
    transformations: List[TransformationStep]
    final_framework: FrameworkSnapshot
    arguments: List[str]
    attacks: List[str]
    aba_plus: ABAPlusDTO
