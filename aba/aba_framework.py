"""
ABAFramework: Represents an Assumption-Based Argumentation (ABA) framework
with support for ABA+ (preference-based attacks between assumption sets).

Features:
- Supports classical ABA argument and attack generation.
- Transforms to atomic ABA framework if needed.
- Generates all combinations of base assumptions.
- Generates normal and reverse attacks between assumption sets for ABA+.
- Provides visualization of attack graphs using PyVis.
"""

from .argument import Argument
from .contrary import Contrary
from .literal import Literal
from .rule import Rule
from .attacks import Attacks
from collections import deque, defaultdict
from itertools import combinations, product
# from pyvis.network import Network


class ABAFramework:
    """
    Represents an Assumption-Based Argumentation (ABA) framework.

    Attributes:
        language (set[Literal]): The set of all literals in the framework.
        rules (set[Rule]): The set of all rules in the framework.
        assumptions (set[Literal]): The set of all assumptions in the framework.
        base_assumptions (set[Literal]): Original assumptions before atomic transformation.
        contraries (set[Contrary]): The set of all contrary relationships.
        preferences (dict[Literal, set[Literal]]): Maps literals to literals they are preferred over.
        arguments (set[Argument]): Set of all generated arguments.
        attacks (set[Attacks]): Standard attacks between arguments (classical ABA).
        normal_attacks (set[tuple]): Normal ABA+ attacks between assumption sets.
        reverse_attacks (set[tuple]): Reverse ABA+ attacks due to preferences.
        assumption_combinations (list[set[Literal]]): All subsets of base assumptions.
    """

    def __init__(self, language: set[Literal], rules: set[Rule], assumptions: set[Literal],
                 contraries: set[Contrary], preferences: dict[Literal, set[Literal]] = None):
        self.language = language
        self.rules = rules
        self.assumptions = assumptions
        self.base_assumptions = set(assumptions)  # save original assumptions
        self.contraries = contraries
        self.preferences = preferences if preferences is not None else {}
        self.arguments = set()
        self.attacks = set()  # classical argument-to-argument attacks
        self.normal_attacks = set()  # ABA+ normal attacks (assumption sets)
        self.reverse_attacks = set()  # ABA+ reverse attacks (assumption sets)
        self.assumption_combinations = []

    def __eq__(self, other):
        return (
            isinstance(other, ABAFramework) and
            self.language == other.language and
            self.rules == other.rules and
            self.assumptions == other.assumptions and
            self.contraries == other.contraries
        )

    def __str__(self):
        """Return string representation of ABA framework, including preferences and arguments."""
        language_str = ', '.join(str(l) for l in sorted(self.language, key=str))
        rules_str = '\n'.join(str(r) for r in sorted(self.rules, key=str))
        assumptions_str = ', '.join(str(a) for a in sorted(self.assumptions, key=str))
        contraries_str = ', '.join(str(c) for c in sorted(self.contraries, key=str))
        result = [
            f"L = {{{language_str}}}",
            f"R = {{\n{rules_str}\n}}",
            f"A = {{{assumptions_str}}}",
            f"CONTRARIES = {{{contraries_str}}}"
        ]
        if self.preferences:
            sorted_prefs = sorted(self.preferences.items(), key=lambda x: str(x[0]))
            preferences_str = '\n'.join(
                f"  {str(literal)} > {{{', '.join(str(p) for p in sorted(prefs, key=str))}}}"
                for literal, prefs in sorted_prefs
            )
            result.append(f"PREFERENCES:\n{preferences_str}")
        if self.arguments:
            arguments_str = '\n'.join(str(arg) for arg in sorted(self.arguments, key=str))
            result.append(f"ARGS:\n{arguments_str}")
        return '\n'.join(result)

    def __hash__(self):
        return hash((frozenset(self.language), frozenset(self.rules),
                     frozenset(self.assumptions), frozenset(self.contraries)))

    # ------------------------- Core ABA Methods -------------------------

    def is_preferred(self, lit1: Literal, lit2: Literal) -> bool:
        """Return True if lit1 is strictly preferred over lit2."""
        return lit1 in self.preferences and lit2 in self.preferences[lit1]

    def generate_arguments(self) -> None:
        """Generate all possible arguments based on rules and assumptions."""
        arg_count = 1
        arguments_by_claim = defaultdict(set)
        queue = deque()

        # Single-assumption arguments
        for a in self.assumptions:
            arg = Argument(f"A{arg_count}", a, {a})
            arguments_by_claim[a].add(arg)
            queue.append(arg)
            arg_count += 1

        # Rules with empty bodies
        for rule in self.rules:
            if not rule.body:
                arg = Argument(f"A{arg_count}", rule.head, set())
                arguments_by_claim[rule.head].add(arg)
                queue.append(arg)
                arg_count += 1

        # Generate other arguments
        while queue:
            current_arg = queue.popleft()
            for rule in self.rules:
                if current_arg.claim not in rule.body:
                    continue
                if not all(lit in arguments_by_claim for lit in rule.body):
                    continue
                body_arg_lists = [arguments_by_claim[lit] for lit in rule.body]
                for combo in product(*body_arg_lists):
                    new_leaves = set().union(*(arg.leaves for arg in combo))
                    new_arg = Argument(f"A{arg_count}", rule.head, new_leaves)
                    if new_arg not in arguments_by_claim[rule.head]:
                        arguments_by_claim[rule.head].add(new_arg)
                        queue.append(new_arg)
                        arg_count += 1

        self.arguments = set().union(*arguments_by_claim.values())

    def generate_attacks(self) -> None:
        """Generate classical attacks between arguments."""
        for arg1 in self.arguments:
            for arg2 in self.arguments:
                for contrary in self.contraries:
                    if arg1.claim == contrary.contrary_attacker and contrary.contraried_literal in arg2.leaves:
                        self.attacks.add(Attacks(arg1, arg2))

    # ------------------------- ABA+ Methods -------------------------

    def save_base_assumptions(self):
        """Save current assumptions as base_assumptions (before atomic transformation)."""
        self.base_assumptions = set(self.assumptions)

    def _generate_assumption_combinations(self) -> list[set[Literal]]:
        """Generate all subsets of base assumptions (including empty set)."""
        source_assumptions = getattr(self, "base_assumptions", self.assumptions)
        all_combos = []
        for r in range(len(source_assumptions) + 1):
            for combo in combinations(source_assumptions, r):
                all_combos.append(set(combo))
        return all_combos

    def arguments_from_assumptions(self, S: set[Literal]) -> set[Argument]:
        """Return arguments whose leaves are subsets of S."""
        return {arg for arg in self.arguments if arg.leaves.issubset(S) or (not arg.leaves and set() <= S)}

    def generate_normal_reverse_attacks(self) -> None:
        """
        Generate ABA+ attacks between assumption sets.

        - Normal: X -> Y if there exists ax from X attacking ay from Y
                without preference-based reversal.
        - Reverse: Y -> X if SOME y in Y is strictly preferred over SOME x in X.
        """
        self.normal_attacks.clear()
        self.reverse_attacks.clear()
        if not self.assumption_combinations:
            self.assumption_combinations = self._generate_assumption_combinations()

        for X in self.assumption_combinations:
            for Y in self.assumption_combinations:
                    
                args_X = self.arguments_from_assumptions(X)
                args_Y = self.arguments_from_assumptions(Y)
                if not args_X or not args_Y:
                    continue
                
                # Check for attacks from X to Y
                for ax in args_X:
                    for ay in args_Y:
                        for contrary in self.contraries:
                            if ax.claim == contrary.contrary_attacker and contrary.contraried_literal in ay.leaves:
                                reverse = any(self.is_preferred(y, x) for y in Y for x in X)
                                if reverse:
                                    self.reverse_attacks.add((frozenset(Y), frozenset(X)))
                                else:
                                    self.normal_attacks.add((frozenset(X), frozenset(Y)))
        
        # Now add the subset attacks: if (ab) attacks c, then (abc) should also attack c
        additional_normal = set()
        additional_reverse = set()
        
        # For normal attacks: if X attacks Y, then any superset of X should attack Y
        for X_att, Y_att in self.normal_attacks:
            X = set(X_att)
            Y = set(Y_att)
            for Z in self.assumption_combinations:
                if X.issubset(Z) and Z != X:
                    additional_normal.add((frozenset(Z), Y_att))
        
        # For reverse attacks: if Y attacks X, then any superset of Y should attack X
        for Y_att, X_att in self.reverse_attacks:
            Y = set(Y_att)
            X = set(X_att)
            for Z in self.assumption_combinations:
                if Y.issubset(Z) and Z != Y:
                    additional_reverse.add((frozenset(Z), X_att))
        
        # Add the additional attacks
        self.normal_attacks.update(additional_normal)
        self.reverse_attacks.update(additional_reverse)


    def make_aba_plus(self) -> None:
        """Generate ABA+ framework: assumption combinations and ABA+ attacks."""
        if not getattr(self, "base_assumptions", None):
            self.base_assumptions = set(self.assumptions)
        self.assumption_combinations = self._generate_assumption_combinations()
        print(f"Generated {len(self.assumption_combinations)} assumption combinations")
        self.generate_normal_reverse_attacks()
        print(f"Generated {len(self.normal_attacks)} normal attacks (assumption sets)")
        print(f"Generated {len(self.reverse_attacks)} reverse attacks (assumption sets)")

    # # ------------------------- Visualization -------------------------

    # def plot_aba_plus_graph(self, output_html="aba_plus_graph.html"):
    #     """
    #     Visualize ABA+ framework with PyVis.
    #     - Normal attacks: solid black lines
    #     - Reverse attacks: dashed red lines
    #     Nodes: labeled with assumption sets
    #     """
    #     if not hasattr(self, "normal_attacks") or not hasattr(self, "reverse_attacks"):
    #         print("Warning: ABA+ attacks not generated. Call make_aba_plus() first.")
    #         return

    #     net = Network(directed=True, notebook=False, height="750px", width="100%")
    #     net.set_options("""
    #     {
    #     "physics": {"enabled": true, "stabilization": {"enabled": true, "iterations": 200}}
    #     }
    #     """)

    #     # Add nodes labeled by assumption sets
    #     for S in self.assumption_combinations:
    #         label = "{" + ",".join(str(l) for l in sorted(S, key=str)) + "}" if S else "{}"
    #         net.add_node(str(frozenset(S)), label=label)

    #     # Normal attacks
    #     for att in self.normal_attacks:
    #         net.add_edge(str(att[0]), str(att[1]), color='black', dashes=False, width=2, title="Normal Attack")

    #     # Reverse attacks
    #     for att in self.reverse_attacks:
    #         net.add_edge(str(att[0]), str(att[1]), color='red', dashes=True, width=2, title="Reverse Attack")

    #     net.write_html(output_html)
    #     print(f"ABA+ attack graph saved to {output_html}")
    #     print(f"  - Normal attacks: {len(self.normal_attacks)}")
    #     print(f"  - Reverse attacks: {len(self.reverse_attacks)}")
