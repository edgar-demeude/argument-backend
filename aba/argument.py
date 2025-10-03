from .literal import Literal


class Argument:
    """
    Represents an argument in the argumentation framework.

    Attributes:
        argument_name (str): The name of the argument.
        claim (Literal): The claim/root literal of the argument.
        leaves (set[Literal]): The set of leaf literals supporting the claim.
    """

    def __init__(self, argument_name: str, claim: Literal, leaves: set[Literal]):
        self.argument_name: str = argument_name
        self.claim: Literal = claim
        self.leaves: set[Literal] = leaves

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Argument):
            return False
        return (
            self.claim == other.claim
            and self.leaves == other.leaves
        )

    def __str__(self) -> str:
        leaves_str = ','.join(str(literal) for literal in self.leaves)
        return f"[{self.argument_name}]={{{leaves_str}}} ⊢ {self.claim}"

    def __hash__(self) -> int:
        return hash((self.claim, frozenset(self.leaves)))

    def __repr__(self) -> str:
        leaves_str = ','.join(sorted(str(l) for l in self.leaves))
        return f"[{self.argument_name}]{{{leaves_str}}} ⊢ {self.claim}"
