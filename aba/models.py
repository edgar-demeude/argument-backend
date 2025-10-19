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
class ABAPlusAttacks(BaseModel):
    """Attacks in ABA+ with distinction between argument attacks and assumption set attacks."""
    # Arguments attacks (classique ABA - entre les arguments)
    argument_attacks: List[Tuple[str, str]]  # [(attacker_arg, attacked_arg), ...]
    # Assumption set attacks (ABA+ - entre les assumption sets)
    assumption_set_attacks: List[Tuple[List[str], List[str]]]  # [(attacking_set, attacked_set), ...]

class ABAPlusFrameworkResults(BaseModel):
    """ABA+ results for a specific framework state (before or after transformation)."""
    assumption_sets: List[List[str]]  # Liste des assumption sets
    attacks: ABAPlusAttacks

class ABAPlusDTO(BaseModel):
    """Results specific to ABA+ semantics with before/after transformation."""
    before_transformation: ABAPlusFrameworkResults
    after_transformation: ABAPlusFrameworkResults

# === Meta info ===
class MetaInfo(BaseModel):
    """Metadata about the ABA computation process."""
    request_id: str
    timestamp: str
    transformed: bool
    transformations_applied: List[str]
    warnings: Optional[List[str]] = []
    errors: Optional[List[str]] = []

class FrameworkWithArgumentsAndAttacks(BaseModel):
    """Framework snapshot with its computed arguments and attacks."""
    framework: FrameworkSnapshot
    arguments: List[str]
    attacks: List[Tuple[str, str]]  # [(attacker, attacked), ...]

class TransformationResult(BaseModel):
    """Transformation results with before/after snapshots."""
    before_transformation: FrameworkWithArgumentsAndAttacks
    after_transformation: FrameworkWithArgumentsAndAttacks
    transformations: List[TransformationStep]

# === Full API response ===
class ABAApiResponseModel(BaseModel):
    """
    Represents the full backend response for an ABA/ABA+ computation request.
    Includes original and transformed frameworks with before/after structure,
    transformation steps, and computed results (arguments, attacks, ABA+ extensions).
    """
    meta: MetaInfo
    transformation: TransformationResult
    aba_plus: Optional[ABAPlusDTO] = None  # None if not ABA+, populated if ABA+