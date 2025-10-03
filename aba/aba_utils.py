from .argument import Argument


def format_argument_leaves(arg: Argument) -> str:
    """
    Format an argument's leaves as a comma-separated string.
    
    Args:
        arg: The argument to format
        
    Returns:
        str: Formatted string like "a,b,c" or "{}" for empty leaves
    """
    if not arg.leaves:
        return "{}"
    sorted_leaves = sorted(arg.leaves, key=str)
    return ",".join(str(leaf) for leaf in sorted_leaves)


def format_attack(attack) -> str:
    """
    Format an attack showing the assumption sets instead of argument IDs.
    
    Args:
        attack: The attack object
        
    Returns:
        str: Formatted string like "{a,b} attacks {c,d}"
    """
    attacker_leaves = format_argument_leaves(attack.attacker)
    target_leaves = format_argument_leaves(attack.target)
    return f"{{{attacker_leaves}}} attacks {{{target_leaves}}}"


def format_assumption_set(S: frozenset) -> str:
    """
    Format an assumption set (frozenset) as a string like "{a,b}" or "{}" if empty.
    """
    if not S:
        return "{}"
    return "{" + ",".join(str(lit) for lit in sorted(S, key=str)) + "}"


def print_aba_plus_results(aba_framework):
    """
    Print ABA+ framework results including:
    - all assumption combinations,
    - normal attacks (between assumption sets),
    - reverse attacks (between assumption sets).
    """
    print("\n======= ABA+ Framework Results =======")

    # --- Assumption combinations ---
    print("\nAll Assumption Combinations:")
    for S in sorted(aba_framework.assumption_combinations, key=lambda x: (len(x), str(x))):
        print(f"  {format_assumption_set(frozenset(S))}")

    # --- Normal attacks ---
    print("\nNormal Attacks (between assumption sets):")
    if not aba_framework.normal_attacks:
        print("  None")
    else:
        for (X, Y) in sorted(aba_framework.normal_attacks, key=lambda p: (str(p[0]), str(p[1]))):
            print(f"  {format_assumption_set(X)}  ->  {format_assumption_set(Y)}")

    # --- Reverse attacks ---
    print("\nReverse Attacks (between assumption sets):")
    if not aba_framework.reverse_attacks:
        print("  None")
    else:
        for (X, Y) in sorted(aba_framework.reverse_attacks, key=lambda p: (str(p[0]), str(p[1]))):
            print(f"  {format_assumption_set(X)}  ->  {format_assumption_set(Y)}")

    print("=====================================\n")

   