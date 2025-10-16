import re
from typing import List, Tuple, Dict, Set


def _parse_list(content: str) -> List[str]:
    """
    Parse a string representing a list of literals into a Python list of strings.
    Args:
        content (str): A string in the form "[a,b,c]" or "a,b,c".
    Returns:
        List[str]: A list of literal identifiers as strings.
    Examples:
        >>> _parse_list("[a,b,c]")
        ['a', 'b', 'c']
        >>> _parse_list("x,y,z")
        ['x', 'y', 'z']
        >>> _parse_list("[]")
        []
    """
    content = content.strip()
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1].strip()
    if not content:
        return []
    return [x.strip() for x in content.split(",") if x.strip()]


def _parse_rule_line(line: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Parse a single rule line of the form [rX]: head <- body.
    Args:
        line (str): Rule string, e.g. "[r1]: p <- q,a".
    Returns:
        Dict[str, Dict[str, Set[str]]]: Mapping of rule ID to {head: set of body literals}.
    Examples:
        >>> _parse_rule_line("[r1]: p <- q,a")
        {'r1': {'p': {'q', 'a'}}}
        >>> _parse_rule_line("[r2]: q <- ")
        {'r2': {'q': set()}}
    """
    match = re.match(r"\[(r\d+)\]:\s*(\w+)\s*<-\s*(.*)", line)
    if match:
        r_id = match.group(1)
        head = match.group(2)
        body = match.group(3).strip()
        body_atoms = set(_parse_list(body)) if body else set()
        return {r_id: {head: body_atoms}}
    return {}


def _parse_pref_line(line: str) -> Dict[str, Set[str]]:
    """
    Parse a preference line of the form:
    - PREF: a,b > c,d > e
    - PREF: a,b > c and a > d

    Supports multiple chains joined by "and".

    Returns:
        Dict[str, Set[str]]: Dictionary mapping each element to all elements
                             that come after it in the preference chain(s).
                             Literals with no less-preferred values are omitted.
    """
    content = line.split(":", 1)[1].strip()

    # Split into separate chains by 'and'
    chains = [chain.strip() for chain in content.split("and") if chain.strip()]

    pref_dict: Dict[str, Set[str]] = {}

    for chain in chains:
        groups = [group.strip() for group in chain.split(">")]
        parsed_groups = [_parse_list(group) for group in groups]

        for i, current_group in enumerate(parsed_groups):
            less_preferred = set()
            for j in range(i + 1, len(parsed_groups)):
                less_preferred.update(parsed_groups[j])

            # Only add mapping if there are less preferred items
            if less_preferred:
                for literal in current_group:
                    if literal not in pref_dict:
                        pref_dict[literal] = set()
                    pref_dict[literal].update(less_preferred)

    return pref_dict


def parse_doc(path: str) -> Tuple[
    List[str],
    List[str],
    List[Tuple[str, str]],
    List[Dict[str, Dict[str, Set[str]]]],
    List[Dict[str, Set[str]]]
]:
    """
    Parse a structured document containing literals, assumptions, contraries,
    rules, and preferences.
    Args:
        path (str): Path to the text file to parse.
    Returns:
        Tuple containing:
        - language (List[str]): List of all literals in the language.
        - assumptions (List[str]): List of assumed literals.
        - contraries (List[Tuple[str, str]]): List of contrary pairs (literal, contrary).
        - rules (List[Dict[str, Dict[str, Set[str]]]]): List of rules,
          each rule is {rule_id: {head: set of body literals}}.
        - preferences (List[Dict[str, Set[str]]]): List of preference dictionaries,
          each mapping literals to sets of less preferred literals.
    Examples:
        >>> # doc.txt content:
        >>> # L: [a,b,c]
        >>> # A: [a,b]
        >>> # C(a): b
        >>> # [r1]: p <- q,a
        >>> # PREF: a,b > c,d > e > g > f
        >>> parse_doc("doc.txt")
        (['a','b','c'], ['a','b'], [('a','b')], [{'r1': {'p': {'q','a'}}}], 
         [{'a': {'c','d','e','g','f'}, 'b': {'c','d','e','g','f'}, ...}])
    """
    language: List[str] = []
    assumptions: List[str] = []
    contraries: List[Tuple[str, str]] = []
    rules: List[Dict[str, Dict[str, Set[str]]]] = []
    preferences: List[Dict[str, Set[str]]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("L:"):
                language = _parse_list(line.split(":", 1)[1])
            elif line.startswith("A:"):
                assumptions = _parse_list(line.split(":", 1)[1])
            elif line.startswith("C("):
                match = re.match(r"C\((\w+)\):\s*(\w+)", line)
                if match:
                    contraries.append((match.group(1), match.group(2)))
            elif line.startswith("[r"):
                rule = _parse_rule_line(line)
                if rule:
                    rules.append(rule)
            elif line.startswith("PREF:"):
                pref_dict = _parse_pref_line(line)
                if pref_dict:
                    preferences.append(pref_dict)

    return language, assumptions, contraries, rules, preferences


if __name__ == "__main__":
    language, assumptions, contraries, rules, preferences = parse_doc(
        "./backend/data/simple_plus2AND.txt")
    print("Language:", language)
    print("Assumptions:", assumptions)
    print("Contraries:", contraries)
    print("Rules:", rules)
    print("Preferences:", preferences)