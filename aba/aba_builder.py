from typing import Dict, Set
from .parsing import parse_doc
from .contrary import Contrary
from .literal import Literal
from .rule import Rule
from .aba_framework import ABAFramework
import tempfile
import os


def build_aba_framework(doc_path: str) -> ABAFramework:
    """
    Build an ABAFramework object from a structured document.

    Args:
        doc_path (str): Path to the document to parse.

    Returns:
        ABAFramework: An ABAFramework instance containing:
            - language (Set[Literal]): Set of Literal objects.
            - rules (Set[Rule]): Set of Rule objects.
            - assumptions (Set[Literal]): Set of assumptions.
            - contraries (Set[Contrary]): Set of contraries.
            - preferences (Dict[Literal, Set[Literal]]): Preference mappings.

    Example:
        >>> aba = build_aba_framework("./backend/doc.txt")
        >>> isinstance(next(iter(aba.language)), Literal)
        True
        >>> isinstance(next(iter(aba.rules)), Rule)
        True
    """
    # Parse the document
    language_parse, assumptions_parse, contraries_parse, rules_parse, preferences_parse = parse_doc(
        doc_path)

    # Initialize containers
    language: Dict[str, Literal] = {}
    rules: Set[Rule] = set()
    contraries: Set[Contrary] = set()
    assumptions: Set[Literal] = set()
    preferences: Dict[Literal, Set[Literal]] = {}

    # Language: build Literal objects
    for lit in language_parse:
        language[lit] = Literal(lit)

    language_set: Set[Literal] = set(language.values())

    # Rules: convert parsed structure into Rule objects
    for rule in rules_parse:
        r_id = next(iter(rule))
        head = next(iter(rule[r_id]))
        body_atoms = rule[r_id][head]
        body_literals = {language[i] for i in body_atoms if i in language}
        rules.add(Rule(r_id, language[head], body_literals))

    # Contraries: build Contrary objects
    for lit1, lit2 in contraries_parse:
        contraries.add(Contrary(language[lit1], language[lit2]))

    # Assumptions: convert to set of Literal
    for lit in assumptions_parse:
        assumptions.add(language[lit])

    # Preferences: merge all preference dictionaries and convert to Literal objects
    for pref_dict in preferences_parse:
        for lit_str, less_preferred_strs in pref_dict.items():
            if lit_str in language:
                lit_obj = language[lit_str]
                # Convert string literals to Literal objects
                less_preferred_objs = {
                    language[lp] for lp in less_preferred_strs if lp in language}

                # Merge with existing preferences for this literal
                if lit_obj in preferences:
                    preferences[lit_obj].update(less_preferred_objs)
                else:
                    preferences[lit_obj] = less_preferred_objs

    # Build ABA framework
    aba_framework = ABAFramework(
        language=language_set,
        rules=rules,
        assumptions=assumptions,
        contraries=contraries,
        preferences=preferences
    )

    return aba_framework


def prepare_aba_plus_framework(aba_framework: ABAFramework) -> ABAFramework:
    """
    Prepare an ABA framework for ABA+ by ensuring it's atomic and generating
    all necessary components.

    Args:
        aba_framework: The ABA framework to prepare

    Returns:
        ABAFramework: The prepared framework (modified in place)
    """

    # Generate arguments for atomic framework
    print("\nGenerating arguments for atomic framework...")
    aba_framework.arguments.clear()
    aba_framework.generate_arguments()
    print(f"Generated {len(aba_framework.arguments)} arguments")

    # Generate standard attacks
    print("\nGenerating standard attacks for atomic framework...")
    aba_framework.attacks.clear()
    aba_framework.generate_attacks()
    print(f"Generated {len(aba_framework.attacks)} attacks\n")

    return aba_framework


def build_aba_framework_from_text(text: str) -> ABAFramework:
    """
    Build an ABAFramework directly from raw text.
    Uses a temporary file internally to leverage build_aba_framework.

    Args:
        text (str): The ABA framework as raw text (e.g., uploaded via frontend).

    Returns:
        ABAFramework: The ABAFramework instance.
    """
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        aba = build_aba_framework(tmp_path)
    finally:
        # Cleanup temporary file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return aba